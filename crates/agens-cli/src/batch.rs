//! `agens m1-batch`: gate for batched decode (server mode). Every row of a batched decode must be bit-identical to the
//! bs = 1 decode of the same session: same greedy tokens and the same logits bits at every step.
//! Reference: each prompt alone (prefill, then N bs = 1 steps). Batched: B sessions (row r = prompt r mod P), each
//! prefilled alone, then decoded together with continuous-batching churn: row r joins `r * stagger` steps late, rows
//! leave when they have N tokens, and with --park one row is parked (KDA state + rings to pinned host) for a few
//! steps and unparked into whatever slot is free. Compares tokens and an FNV-64 hash of each step's logits row.
//! Set AGENS_M1_BATCH_INVARIANT=1 for the server-mode numerics (engine bf16 GEMV for decode, also at bs = 1).

use anyhow::{bail, Context, Result};
use std::time::Instant;

fn fnv(v: &[u16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in v {
        h = (h ^ x as u64).wrapping_mul(0x100000001b3);
    }
    h
}

pub fn run(args: &[String]) -> Result<()> {
    use agens_cuda::m1;
    let usage = "agens m1-batch <ckpt_dir> <prompts.json> [--batch 2,4,8] [--new-tokens N] [--stagger S] [--park 1] \
                 [m1 flags: --gpu-gib --wctx --pool --sessions --devices]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let (mut batches, mut n_new, mut stagger, mut park, mut rest) = (vec![2usize, 4, 8], 32usize, 2usize, true, Vec::new());
    let mut i = 2;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?.clone();
        match args[i].as_str() {
            "--batch" => batches = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--new-tokens" => n_new = v.parse()?,
            "--stagger" => stagger = v.parse()?,
            "--park" => park = v != "0",
            _ => {
                rest.push(args[i].clone());
                rest.push(v);
            }
        }
        i += 2;
    }
    let bmax = *batches.iter().max().unwrap_or(&1);
    if !rest.iter().any(|a| a == "--sessions") {
        rest.push("--sessions".into());
        rest.push((bmax + 1).to_string());
    }
    let o = crate::m1::parse_opts(&rest, usage)?;
    crate::m1::load(dir, &o)?;
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let plist: Vec<Vec<i32>> = prompts
        .as_array()
        .context("prompts")?
        .iter()
        .map(|p| p["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect())
        .collect();
    // bs = 1 reference on the main session
    let main = m1::session_current();
    let mut refs: Vec<(Vec<i32>, Vec<u64>)> = Vec::new();
    let mut t_ref = 0f64;
    let mut ref_steps = 0usize;
    for ids in &plist {
        m1::session_use(main)?;
        m1::reset()?;
        let mut tok = m1::prefill(ids)?;
        let (mut toks, mut hs) = (vec![tok], vec![fnv(&m1::logits_row(0)?)]);
        while toks.len() < n_new {
            let td = Instant::now();
            tok = m1::decode(tok)?;
            t_ref += td.elapsed().as_secs_f64();
            toks.push(tok);
            hs.push(fnv(&m1::logits_row(0)?));
            ref_steps += 1;
        }
        refs.push((toks, hs));
    }
    eprintln!("m1-batch: bs=1 reference for {} prompts x {n_new} tokens", plist.len());
    let mut all_ok = true;
    let pbytes = m1::park_bytes();
    let pbuf = if park { agens_cuda::host_alloc(pbytes)? } else { std::ptr::null_mut() };
    for &b in &batches {
        // sessions, each prefilled alone
        let mut sess = Vec::new();
        for r in 0..b {
            let id = m1::session_new()?;
            m1::session_use(id)?;
            let tok = m1::prefill(&plist[r % plist.len()])?;
            sess.push((id, vec![tok], vec![fnv(&m1::logits_row(0)?)]));
        }
        m1::session_use(main)?;
        let (mut step, mut row_steps, mut parked) = (0usize, 0usize, None::<usize>);
        let t = Instant::now();
        let mut dec_time = 0f64;
        let mut full: Vec<f64> = Vec::new(); // step times with all b rows decoding
        loop {
            // churn: park row 1 for steps [3, 6) of the run
            if park && b >= 2 {
                if step == 3 && sess[1].1.len() < n_new {
                    m1::session_park(sess[1].0, pbuf)?;
                    parked = Some(1);
                }
                if step == 6 {
                    if let Some(r) = parked.take() {
                        m1::session_unpark(sess[r].0, pbuf)?;
                    }
                }
            }
            let active: Vec<usize> = (0..b)
                .filter(|&r| step >= r * stagger && sess[r].1.len() < n_new && parked != Some(r))
                .collect();
            if active.is_empty() {
                if parked.is_none() && sess.iter().all(|s| s.1.len() >= n_new) {
                    break;
                }
                step += 1;
                continue;
            }
            let ids: Vec<i32> = active.iter().map(|&r| sess[r].0).collect();
            let toks: Vec<i32> = active.iter().map(|&r| *sess[r].1.last().unwrap()).collect();
            let td = Instant::now();
            let next = m1::decode_batch(&ids, &toks)?;
            let dt = td.elapsed().as_secs_f64();
            dec_time += dt;
            if active.len() == b {
                full.push(dt);
            }
            for (j, &r) in active.iter().enumerate() {
                let h = fnv(&m1::logits_row(j)?);
                sess[r].1.push(next[j]);
                sess[r].2.push(h);
            }
            row_steps += active.len();
            step += 1;
        }
        let wall = t.elapsed().as_secs_f64();
        let mut ok = 0;
        for (r, s) in sess.iter().enumerate() {
            let (rt, rh) = &refs[r % plist.len()];
            let same = &s.1 == rt && &s.2 == rh;
            ok += same as usize;
            if !same {
                let dt = s.1.iter().zip(rt).position(|(a, b)| a != b);
                let dh = s.2.iter().zip(rh).position(|(a, b)| a != b);
                eprintln!("  B={b} row {r} (prompt {}): tokens diverge at {dt:?}, logits at {dh:?}", r % plist.len());
            }
        }
        println!(
            "batched decode B={b}: {ok}/{b} rows bit-identical to bs=1 (tokens + logits, {n_new} steps, stagger {stagger}{}); \
             {row_steps} row-steps in {:.2} s decode ({:.1} ms/step, {:.1} row-tok/s; bs=1 reference {:.1} tok/s)",
            if park && b >= 2 { ", row 1 parked for 3 steps" } else { "" },
            dec_time,
            1e3 * dec_time / step.max(1) as f64,
            row_steps as f64 / dec_time,
            ref_steps as f64 / t_ref,
        );
        if !full.is_empty() {
            full.sort_by(|a, c| a.partial_cmp(c).unwrap());
            let med = full[full.len() / 2];
            println!("  B={b} steady state (median of {} all-row steps): {:.1} ms/step = {:.1} tok/s aggregate",
                     full.len(), 1e3 * med, b as f64 / med);
        }
        let _ = wall;
        all_ok &= ok == b;
        for s in &sess {
            m1::session_free(s.0)?;
        }
    }
    if !pbuf.is_null() {
        agens_cuda::host_free(pbuf)?;
    }
    if !all_ok {
        bail!("batched decode is not bit-identical to bs=1");
    }
    Ok(())
}
