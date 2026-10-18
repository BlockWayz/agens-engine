//! Tensor parallelism across processes, one per GPU (`--tp N`, N = 2 or 4).
//!
//! The leader (rank 0, the process the user started) re-launches itself N - 1 times with AGENS_TP_RANK = r (same
//! arguments, GPU `--device` + r). Every rank loads its own shard of the checkpoint (m1::load, tp_rule): 1 / N of
//! every KDA layer's heads, attention layer's q / kv heads and MLP's intermediate rows; the rest is replicated. The
//! ranks then set up the all-reduce exchange (CUDA IPC when every pair of GPUs is a peer pair, else POSIX shared
//! memory) and run the same command's setup. From the workload on (`start`), the leader forwards every
//! state-changing engine call to the followers (agens_cuda::tp), which execute them in order (`follow`); the GPUs pair
//! their all-reduces on device. Outputs are identical on every rank; only the leader writes results.

use anyhow::{bail, Context, Result};
use std::io::BufRead;
use std::time::{Duration, Instant};

pub fn rank() -> usize {
    std::env::var("AGENS_TP_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}
pub fn size() -> usize {
    std::env::var("AGENS_TP_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}
fn id() -> String {
    std::env::var("AGENS_TP_ID").unwrap_or_else(|_| "0".into())
}

/// `--tp N` on the leader: export the TP environment and start the followers (before the load, so the loads overlap).
pub fn launch(tp: usize) -> Result<()> {
    if tp <= 1 || rank() > 0 || std::env::var("AGENS_TP_SIZE").is_ok() {
        return Ok(());
    }
    if ![2, 4].contains(&tp) {
        bail!("--tp {tp}: 2 or 4 (4 KV heads split evenly)");
    }
    std::env::set_var("AGENS_TP_SIZE", tp.to_string());
    std::env::set_var("AGENS_TP_RANK", "0");
    std::env::set_var("AGENS_TP_ID", std::process::id().to_string());
    let exe = std::env::current_exe()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ins = Vec::new();
    for r in 1..tp {
        let log = std::fs::File::create(format!("tp_rank{r}.log")).context("tp_rank log")?;
        let mut c = std::process::Command::new(&exe);
        c.args(&args).env("AGENS_TP_RANK", r.to_string()).stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null()).stderr(log);
        let mut child = c.spawn().context("tp follower")?;
        ins.push(child.stdin.take().unwrap());
        std::mem::forget(child); // followers exit when the leader closes their stdin
    }
    agens_cuda::tp::set_followers(ins);
    eprintln!("tp: {tp} ranks (followers log to tp_rank<r>.log)");
    Ok(())
}

fn wait_file(p: &str, what: &str) -> Result<String> {
    let t0 = Instant::now();
    loop {
        if let Ok(s) = std::fs::read_to_string(p) {
            if s.ends_with('\n') {
                return Ok(s.trim().to_string());
            }
        }
        if t0.elapsed() > Duration::from_secs(1800) {
            bail!("tp: timed out waiting for {what} ({p})");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// After m1::init on every rank: exchange regions + ak_tp_setup. `slot_elems`: the largest all-reduce (prefill chunk x
/// hidden size).
pub fn setup_allreduce(slot_elems: usize) -> Result<()> {
    let (r, n, id) = (rank(), size(), id());
    if n <= 1 {
        return Ok(());
    }
    let bytes = agens_cuda::m1::tp_region_bytes(slot_elems);
    let peer = (0..n).all(|a| (0..n).all(|b| a == b || agens_cuda::m1::tp_can_peer(a, b)))
        && std::env::var("AGENS_TP_SHM").map(|v| v != "1").unwrap_or(true);
    let f = |q: usize, s: &str| format!("/dev/shm/agens_tp_{id}_{q}.{s}");
    let mut regions = vec![std::ptr::null_mut(); n];
    if peer {
        let (p, h) = agens_cuda::m1::tp_region_alloc(bytes)?;
        regions[r] = p;
        let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
        std::fs::write(f(r, "h"), format!("{hex}\n"))?;
        for q in (0..n).filter(|&q| q != r) {
            let hex = wait_file(&f(q, "h"), "peer handle")?;
            let mut h = [0u8; 64];
            for (i, b) in h.iter_mut().enumerate() {
                *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
            }
            regions[q] = agens_cuda::m1::tp_region_open(&h)?;
        }
    } else {
        let shm = |q: usize| format!("/agens_tp_{id}_{q}");
        regions[r] = agens_cuda::m1::tp_region_shm(&shm(r), bytes, true)?;
        std::fs::write(f(r, "h"), "shm\n")?;
        for q in (0..n).filter(|&q| q != r) {
            wait_file(&f(q, "h"), "peer region")?;
            regions[q] = agens_cuda::m1::tp_region_shm(&shm(q), bytes, false)?;
        }
    }
    agens_cuda::m1::tp_setup(r, &regions, slot_elems)?;
    if let Some(it) = std::env::var("AGENS_TP_AR_SELFTEST").ok().and_then(|v| v.parse::<usize>().ok()) {
        let bad = agens_cuda::m1::tp_selftest(it, slot_elems.min(1 << 22));
        eprintln!("tp: rank {r} all-reduce self-test: {it} rounds, {bad} wrong elements");
    }
    // every rank holds every region: the handle files and shm names can go
    std::fs::write(f(r, "ok"), "ok\n")?;
    for q in 0..n {
        wait_file(&f(q, "ok"), "peer setup")?;
    }
    let _ = std::fs::remove_file(f(r, "h"));
    if !peer {
        let _ = std::fs::remove_file(format!("/dev/shm/agens_tp_{id}_{r}"));
    }
    std::thread::sleep(Duration::from_millis(200));
    let _ = std::fs::remove_file(f(r, "ok"));
    eprintln!("tp: rank {r}/{n} all-reduce over {} ({:.0} MB per rank)", if peer { "CUDA IPC (peer GPUs)" } else { "host shared memory" },
              bytes as f64 / 1e6);
    Ok(())
}

/// The workload starts: the leader forwards from here; a follower runs the forwarded calls until stdin closes.
pub fn start() -> Result<()> {
    if size() <= 1 {
        return Ok(());
    }
    if rank() == 0 {
        agens_cuda::tp::activate();
        return Ok(());
    }
    follow()?;
    std::process::exit(0);
}

fn follow() -> Result<()> {
    use agens_cuda::m1;
    use agens_cuda::tp::split;
    let stdin = std::io::stdin();
    let mut parked: std::collections::HashMap<i32, *mut std::ffi::c_void> = Default::default();
    for line in stdin.lock().lines() {
        let line = line?;
        let a: Vec<&str> = line.split(' ').collect();
        let int = |i: usize| -> Result<i32> { Ok(a.get(i).context("tp: short line")?.parse()?) };
        let r: Result<()> = (|| {
            match a[0] {
                "reset" => m1::reset()?,
                "prefill" => drop(m1::prefill(&split(a[1]))?),
                "decode" => drop(m1::decode(int(1)?)?),
                "decode_batch" => drop(m1::decode_batch(&split(a[1]), &split(a[2]))?),
                "spec_step" => drop(m1::spec_step(int(1)?)?),
                "spec_step_hint" => drop(m1::spec_step_hint(int(1)?, &split(a[2]))?),
                "spec_batch" => drop(m1::spec_batch(&split(a[1]), &split(a[2]))?),
                "spec_batch_hint" => {
                    let h: Vec<Vec<i32>> = a[3].split('/').map(split).collect();
                    drop(m1::spec_batch_hint(&split(a[1]), &split(a[2]), &h)?)
                }
                "spec_sampling" => m1::spec_sampling_filtered(a[1].parse()?, int(3)?, a[4].parse()?, a[5].parse()?, int(2)?)?,
                "session_sampling" => m1::session_sampling(int(1)?, a[2].parse()?, int(3)?, a[4].parse()?, a[5].parse()?, int(6)?)?,
                "session_new" => drop(m1::session_new()?),
                "session_use" => m1::session_use(int(1)?)?,
                "session_fork" => drop(m1::session_fork(int(1)?)?),
                "session_free" => m1::session_free(int(1)?)?,
                "session_park" => {
                    let id = int(1)?;
                    let buf = *parked.entry(id).or_insert_with(|| agens_cuda::host_alloc(m1::park_bytes()).unwrap());
                    m1::session_park(id, buf)?
                }
                "session_unpark" => {
                    let id = int(1)?;
                    let buf = *parked.get(&id).context("tp: unpark of a session this rank never parked")?;
                    m1::session_unpark(id, buf)?
                }
                "verify" => drop(m1::verify(&split(a[1]))?),
                "profiler" => m1::profiler(int(1)? != 0)?,
                op => bail!("tp: unknown op {op}"),
            }
            Ok(())
        })();
        if let Err(e) = r {
            eprintln!("tp rank {}: {line:.80}: {e:#}", rank());
            std::process::exit(2);
        }
    }
    Ok(())
}
