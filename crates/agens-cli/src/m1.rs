//! `agens m1`: run the M1 standalone forward (bf16 or W4, tp1, one request) on prompts given as token ids and write
//! the greedy continuations. Weights beyond --gpu-gib go to pinned host memory mapped into the device address space
//! (slow but exact), so the bf16 model can be checked on a 48 GB card before the H200 run.
//! Load-time conversions mirror sglv23's loader: conv1d / A_log / dt_bias widened to fp32, gate_proj + up_proj
//! concatenated into gate_up_proj (MergedColumnParallelLinear order), Engram tables and the token embedding table
//! host-resident (neither is read in full per token).
//! W4 checkpoints (compressed-tensors int4 container, group 64 or 128 per linear): W4 linears are repacked on the GPU
//! to the k-tiled layout (always GPU-resident); W4 linears with k < 1024 (f_up, k = 128: the k-tile would pad it 8x)
//! are dequantized to bf16 at load instead; linears the recipe left bf16 (lm_head, mhc.dyn, in_proj_b (fake-quantized,
//! stored bf16), idx_q/idx_k, engram) stay bf16.

use agens_core::reference::bf16_from_bytes;
use agens_core::Checkpoint;
use anyhow::{bail, Context, Result};
use std::time::Instant;

struct Placer {
    gpu_budget: usize,  // per device
    devs: Vec<i32>,
    on_gpu: Vec<usize>,
    on_host: usize,
    cur: usize,  // index into devs of the device the next tensor belongs to
}

impl Placer {
    fn on(&mut self, dev_index: usize) -> Result<()> {
        self.cur = dev_index;
        agens_cuda::set_device(self.devs[dev_index] as usize)
    }
    fn place(&mut self, bytes: &[u8], force_host: bool) -> Result<*mut std::ffi::c_void> {
        if !force_host && self.on_gpu[self.cur] + bytes.len() <= self.gpu_budget {
            self.on_gpu[self.cur] += bytes.len();
            Ok(agens_cuda::leak(agens_cuda::upload(bytes)?))
        } else {
            self.on_host += bytes.len();
            agens_cuda::upload_host_mapped(bytes)
        }
    }
}

/// Tensor parallel shard of one checkpoint tensor for rank `r` of `tp`: the output rows it keeps (ranges of the full
/// tensor's rows, in order) and, for the row-parallel linears, its input columns (a range of k). None: replicated.
/// Column parallel: q / k / v projections (by q / kv heads), gate / up (by intermediate rows), the KDA projections
/// in_proj_qkv (q, k, v head segments) / in_proj_z / in_proj_b / f_up, conv1d channels, A_log / dt_bias per v head.
/// Row parallel: o_proj, out_proj, down_proj.
pub(crate) struct TpShard {
    pub(crate) rows: Vec<(usize, usize)>,
    pub(crate) cols: Option<(usize, usize)>,
}
pub(crate) fn tp_rule(name: &str, tp: usize, r: usize, n_rows: usize) -> Option<TpShard> {
    if tp <= 1 {
        return None;
    }
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let what = rest.split_once('.')?.1;
    let what = what.strip_suffix(".weight_packed").or_else(|| what.strip_suffix(".weight_scale"))
        .or_else(|| what.strip_suffix(".weight")).unwrap_or(what);
    let (qh, kvh, hd, inter, kh, hv, kk) = (24 / tp, 4 / tp, 256usize, 17408 / tp, 16 / tp, 48 / tp, 128usize);
    let rows = |v: Vec<(usize, usize)>| Some(TpShard { rows: v, cols: None });
    let cols = |c0: usize, n: usize| Some(TpShard { rows: vec![(0, n_rows)], cols: Some((c0, c0 + n)) });
    match what {
        "self_attn.q_proj" => rows(vec![(r * qh * 2 * hd, qh * 2 * hd)]),
        "self_attn.k_proj" | "self_attn.v_proj" => rows(vec![(r * kvh * hd, kvh * hd)]),
        "self_attn.o_proj" => cols(r * qh * hd, qh * hd),
        "mlp.gate_proj" | "mlp.up_proj" => rows(vec![(r * inter, inter)]),
        "mlp.down_proj" => cols(r * inter, inter),
        "linear_attn.in_proj_qkv" | "linear_attn.conv1d" => {
            let hk = 16 * kk;  // full q (and k) segment
            rows(vec![(r * kh * kk, kh * kk), (hk + r * kh * kk, kh * kk), (2 * hk + r * hv * kk, hv * kk)])
        }
        "linear_attn.in_proj_z" | "linear_attn.f_up" => rows(vec![(r * hv * kk, hv * kk)]),
        "linear_attn.in_proj_b" => rows(vec![(r * hv, hv)]),
        "linear_attn.A_log" | "linear_attn.dt_bias" => {
            let per = n_rows / 48;  // values per v head
            rows(vec![(r * hv * per, hv * per)])
        }
        "linear_attn.out_proj" => cols(r * hv * kk, hv * kk),
        _ => None,
    }
}
/// The shard's bytes: `data` is n_rows rows of `row_bytes` covering `k_full` input columns.
pub(crate) fn tp_apply(sh: &TpShard, data: &[u8], row_bytes: usize, k_full: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for &(r0, n) in &sh.rows {
        for row in r0..r0 + n {
            let rb = &data[row * row_bytes..(row + 1) * row_bytes];
            match sh.cols {
                Some((c0, c1)) => out.extend_from_slice(&rb[c0 * row_bytes / k_full..c1 * row_bytes / k_full]),
                None => out.extend_from_slice(rb),
            }
        }
    }
    out
}
/// (rows, k) after the shard
pub(crate) fn tp_dims(sh: &TpShard, k: usize) -> (usize, usize) {
    (sh.rows.iter().map(|x| x.1).sum(), sh.cols.map(|c| c.1 - c.0).unwrap_or(k))
}

fn to_f32_bytes(bf16: &[u8]) -> Vec<u8> {
    bf16_from_bytes(bf16).iter().flat_map(|x| x.to_f32().to_le_bytes()).collect()
}

pub(crate) struct Opts {
    pub(crate) gpu_gib: f64,
    pub(crate) device: usize,
    pub(crate) devices: Vec<i32>,
    pub(crate) new_tokens: usize,
    pub(crate) wctx: usize,
    pub(crate) cuts: usize,
    pub(crate) tier: String,
    pub(crate) seed: u64,
    pub(crate) pool: usize,     // KV pool rows shared by all sessions (0 = wctx)
    pub(crate) sessions: usize, // max live sessions (KDA state slots)
    pub(crate) budget_gib: f64, // > 0: size the KV pool from this GPU budget (plan.rs) instead of --pool
    pub(crate) drafter: Option<String>, // DFlash2 drafter dir (speculative decoding): loaded inside the budget
    pub(crate) tp: usize,               // tensor parallel ranks (processes, one GPU each: --device + rank)
    pub(crate) mtp: Option<String>,     // MTP head dir (docs/mtp.md): a draft source of speculative decoding
}

pub(crate) fn parse_opts(args: &[String], usage: &str) -> Result<Opts> {
    let mut o = Opts { gpu_gib: 40.0, device: 0, devices: vec![], new_tokens: 64, wctx: 36868, cuts: 50, tier: "host".into(), seed: 1, pool: 0, sessions: 1, budget_gib: 0.0, drafter: None, tp: 1, mtp: None };
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).context(usage.to_string())?;
        match args[i].as_str() {
            "--gpu-gib" => o.gpu_gib = v.parse()?,
            "--device" => o.device = v.parse()?,
            "--devices" => o.devices = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--new-tokens" => o.new_tokens = v.parse()?,
            "--wctx" => o.wctx = v.parse()?,
            "--cuts" => o.cuts = v.parse()?,
            "--tier" => o.tier = v.clone(),
            "--seed" => o.seed = v.parse()?,
            "--pool" => o.pool = v.parse()?,
            "--sessions" => o.sessions = v.parse()?,
            "--budget-gib" => o.budget_gib = v.parse()?,
            "--tp" => o.tp = v.parse()?,
            "--mtp" => o.mtp = Some(v.clone()),
            f => bail!("unknown flag {f}\n{usage}"),
        }
        i += 2;
    }
    Ok(o)
}

pub fn run(args: &[String]) -> Result<()> {
    let usage = "agens m1 <ckpt_dir> <prompts.json> <out.json> [--gpu-gib N] [--device D] [--new-tokens N] [--wctx N]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let out_path = args.get(2).context(usage)?;
    let o = parse_opts(&args[3..], usage)?;
    load(dir, &o)?;
    crate::tp::start()?;
    let new_tokens = o.new_tokens;
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let mut results = Vec::new();
    let mut hc = HeadCmp::default();
    for p in prompts.as_array().context("prompts must be a JSON list")? {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let ids: Vec<i32> = p["ids"].as_array().context("ids")?.iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let n_new = p.get("new_tokens").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(new_tokens);
        agens_cuda::m1::reset()?;
        if let Some(force) = p.get("force").and_then(|x| x.as_array()) {
            // teacher-forced: feed the reference continuation, record the engine's greedy pick at every position
            let force: Vec<i32> = force.iter().map(|x| x.as_i64().unwrap() as i32).collect();
            let cmp = std::env::var("AGENS_M1_LMHEAD_INT8").map(|v| v == "cmp").unwrap_or(false);
            // AGENS_M1_LOGITS_DUMP=file: every forced position's logits (bf16, VOCAB_VALID each) appended (quality
            // gates of engine variants: tools/compare_logits.py over two runs)
            let mut dump = match std::env::var("AGENS_M1_LOGITS_DUMP") {
                Ok(f) => Some(std::fs::OpenOptions::new().create(true).append(true).open(f)?),
                Err(_) => None,
            };
            let mut dump_row = |d: &mut Option<std::fs::File>| -> Result<()> {
                if let Some(f) = d {
                    use std::io::Write;
                    let v = agens_cuda::m1::logits_row(0)?;
                    f.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                }
                Ok(())
            };
            let mut out = vec![agens_cuda::m1::prefill(&ids)?];
            dump_row(&mut dump)?;
            if cmp {
                head_cmp(&mut hc)?;
            }
            for &t in &force[..force.len().saturating_sub(1)] {
                out.push(agens_cuda::m1::decode(t)?);
                dump_row(&mut dump)?;
                if cmp {
                    head_cmp(&mut hc)?;
                }
            }
            let same = out.iter().zip(&force).filter(|(a, b)| a == b).count();
            eprintln!("{name}: teacher-forced top-1 agreement {same}/{}", force.len());
            results.push(serde_json::json!({"name": name, "n_prompt": ids.len(), "tokens": out, "force": force}));
            std::fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
            continue;
        }
        let t = Instant::now();
        let mut tok = agens_cuda::m1::prefill(&ids)?;
        let t_prefill = t.elapsed().as_secs_f64();
        if std::env::var("AGENS_M1_MEM_TRACE").is_ok() {
            report_mem(&format!("after the prefill of {name} ({} tokens)", ids.len()));
        }
        let mut out = vec![tok];
        let td = Instant::now();
        let mut st = [0f64; 4];
        // AGENS_M1_PROFILE_DECODE=N: profiler capture range over decode steps 8..8+N of each prompt (nsys)
        let prof = std::env::var("AGENS_M1_PROFILE_DECODE").ok().and_then(|v| v.parse::<usize>().ok());
        while out.len() < n_new && tok != EOT {
            if let Some(n) = prof {
                if out.len() == 8 || out.len() == 8 + n {
                    agens_cuda::m1::profiler(out.len() == 8)?;
                }
            }
            tok = agens_cuda::m1::decode(tok)?;
            out.push(tok);
            for (a, b) in st.iter_mut().zip(agens_cuda::m1::step_stats()) {
                *a += b;
            }
        }
        let t_dec = td.elapsed().as_secs_f64();
        let steps = (out.len() - 1).max(1) as f64;
        eprintln!(
            "{name}: {} prompt tokens, prefill {:.1} s ({:.0} tok/s), {} new tokens in {:.1} s ({:.2} tok/s decode)",
            ids.len(),
            t_prefill,
            ids.len() as f64 / t_prefill,
            out.len(),
            t_dec,
            (out.len() - 1) as f64 / t_dec.max(1e-9)
        );
        if std::env::var_os("AGENS_M1_STEP_STATS").is_some() {
            eprintln!(
                "{name}: decode step: host enqueue {:.2} ms, then wait {:.2} ms; {:.0} linears/step ({:.0} W4)",
                st[0] / steps,
                st[1] / steps,
                st[2] / steps,
                st[3] / steps
            );
        }
        results.push(serde_json::json!({"name": name, "n_prompt": ids.len(), "tokens": out}));
        std::fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
    }
    if hc.n > 0 {
        hc.report();
    }
    report_mem("after the run");
    Ok(())
}

pub(crate) const EOT: i32 = 248046;
/// Bytes of the hard-cap ballast (0 = no cap): device use reports subtract it.
pub(crate) static CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Device memory in use by this process now (the CUDA context and lazily loaded kernel modules included, the hard-cap
/// ballast excluded): what a 16 GB card would have to hold.
pub(crate) fn report_mem(tag: &str) {
    if let Ok((free, total)) = agens_cuda::mem_info() {
        let b = CAP.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("m1: device use {tag}: {:.2} GB ({:.2} GiB; torch cache {:.2} GB){}", (total - free - b) as f64 / 1e9,
                  (total - free - b) as f64 / (1u64 << 30) as f64, agens_cuda::m1::torch_reserved(false) as f64 / 1e9,
                  if b > 0 { format!(", {:.2} GB left under the cap", free as f64 / 1e9) } else { String::new() });
    }
}

/// Mixed precision (AGENS_M1_MIXED=1, or =w4 for the reference with the same parts and groups packed W4): per-linear
/// GSQ-RCO type (bits, group) by base name, from AGENS_M1_MIXED_ASSIGN (a JSON file), else the checkpoint's
/// quantization_audit.json (`gsq_type` per quantized matrix, written by tools/gsq/build_ckpt.py), else its
/// rco_assignment.json (RCO's assignment.json: {"assignment": {name: "w3g128", ...}}).
fn mixed_types(dir: &str) -> Result<std::collections::BTreeMap<String, (usize, usize)>> {
    let parse = |t: &str| -> Result<(usize, usize)> {
        let t = t.to_ascii_lowercase();
        let (b, g) = t.strip_prefix('w').and_then(|r| r.split_once('g')).context(format!("type {t}"))?;
        let (b, g): (usize, usize) = (b.parse()?, g.parse()?);
        if !(2..=4).contains(&b) || (g != 64 && g != 128) {
            bail!("type {t}: bits 2-4, group 64 or 128");
        }
        Ok((b, g))
    };
    let mut m = std::collections::BTreeMap::new();
    let path = std::env::var("AGENS_M1_MIXED_ASSIGN").ok();
    let audit = format!("{dir}/quantization_audit.json");
    if path.is_none() && std::path::Path::new(&audit).exists() {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&audit)?)?;
        for r in j["quantized_matrices"].as_array().into_iter().flatten() {
            if let (Some(n), Some(t)) = (r["name"].as_str(), r["gsq_type"].as_str()) {
                m.insert(n.to_string(), parse(t)?);
            }
        }
        if !m.is_empty() {
            eprintln!("m1: mixed precision: {} linear types from {audit}", m.len());
            return Ok(m);
        }
    }
    let p = path.unwrap_or_else(|| format!("{dir}/rco_assignment.json"));
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).context(p.clone())?)?;
    for (n, t) in j["assignment"].as_object().context("no \"assignment\" object")? {
        m.insert(n.clone(), parse(t.as_str().context("type")?)?);
    }
    eprintln!("m1: mixed precision: {} linear types from {p}", m.len());
    Ok(m)
}

/// A g`group` linear stored in the int4 container at group `cg` (build_ckpt.py writes g128 as g64 with each scale
/// repeated): its own scales, checking the repetition (bf16 bytes, row-major [n][k / cg]).
fn collapse_scales(sc: &[u8], n: usize, k: usize, cg: usize, group: usize, what: &str) -> Result<Vec<u8>> {
    if group == cg {
        return Ok(sc.to_vec());
    }
    if group % cg != 0 {
        bail!("{what}: type group {group} from container group {cg}");
    }
    let (r, cols) = (group / cg, k / cg);
    let mut out = Vec::with_capacity(sc.len() / r);
    for row in 0..n {
        let s = &sc[row * cols * 2..(row + 1) * cols * 2];
        for g in 0..cols / r {
            let first = &s[g * r * 2..g * r * 2 + 2];
            if (1..r).any(|j| &s[(g * r + j) * 2..(g * r + j) * 2 + 2] != first) {
                bail!("{what}: row {row} group {g}: container scales are not repeated, not a g{group} linear");
            }
            out.extend_from_slice(first);
        }
    }
    Ok(out)
}

pub(crate) fn load(dir: &str, o: &Opts) -> Result<()> {
    let (gpu_gib, wctx) = (o.gpu_gib, o.wctx);
    crate::tp::launch(o.tp)?;
    let (tp, rank) = (crate::tp::size(), crate::tp::rank());
    // AGENS_TP_SAME_GPU=1: every rank on --device (correctness runs on one GPU; the ranks time-share it)
    let same = std::env::var("AGENS_TP_SAME_GPU").map(|v| v == "1").unwrap_or(false);
    let devs: Vec<i32> = if o.devices.is_empty() { vec![(o.device + if same { 0 } else { rank }) as i32] } else { o.devices.clone() };
    if tp > 1 && devs.len() != 1 {
        bail!("--tp: one device per rank (no --devices)");
    }
    agens_cuda::set_device(devs[0] as usize)?;
    // --budget-gib B is also a hard cap (unless AGENS_M1_NO_CAP=1): everything but B GiB of the device (the CUDA
    // context included) is taken by a ballast allocation, so any overshoot -- weights, pool, scratch, a lazily
    // allocated workspace -- fails with cudaErrorMemoryAllocation instead of silently using memory a 16 GB card lacks.
    let mut ballast = 0usize;
    // every mode: the far branch's score + attend halves as fused kernels (AGENS_FAR_ATTEND=fused, product numerics:
    // decode and every verify query take the same kernels, so spec == non-spec greedy); AGENS_FAR_ATTEND=bridge keeps
    // the cuBLAS-einsum numerics of the ATen bridge.
    if std::env::var("AGENS_FAR_ATTEND").is_err() {
        std::env::set_var("AGENS_FAR_ATTEND", "fused");
    }
    if o.budget_gib > 0.0 {
        // budget mode: torch's allocator (ATen far branch) in expandable segments, the far extend in 1.25e7-element steps
        // of >= 8 query rows (A1d: 5e7 -> torch cache 0.28 GB after a 33K prefill instead of 2.9 GB; A1e: 1.25e7 / 8 rows
        // -> peak 17.20 instead of 17.35 GB with 8K prompts, 0.24 GB of torch cache after a 240K prefill instead of 1.47),
        // unless set. The step changes einsum batch shapes: greedy tokens can move at near-ties (128K document: token 108
        // of 128 vs the default steps); tiered == all-GPU at equal steps. AGENS_FAR_ATTEND=fused: the far branch's attend half
        // in one fused kernel (product numerics, fp32 order differs from the cuBLAS einsums; decode and verify identical).
        // + the consumer-GPU product config (release defaults; each passed the relaxed-numerics gates: spec == non-spec,
        // tiered == all-GPU, INT8 head / W3 drafter within the quality rule): KV tiered to host RAM with block means
        // and the indexer keys on the host, INT8 lm_head, prefill chunk 1024, the drafter at W3 g128.
        for (k, v) in [("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"), ("AGENS_FAR_EXTEND_ELEMS", "1.25e7"),
                       ("AGENS_FAR_EXTEND_MIN_ROWS", "8"), ("AGENS_FAR_EXTEND_ENGINE", "1"), ("AGENS_KV_TIER", "2"),
                       ("AGENS_KV_IDX_HOST", "1"), ("AGENS_M1_LMHEAD_INT8", "1"), ("AGENS_M1_PREFILL_CHUNK", "1024"),
                       ("AGENS_DRAFT_W4", "1"), ("AGENS_DRAFT_BITS", "3")] {
            if std::env::var(k).is_err() {
                std::env::set_var(k, v);
            }
        }
    }
    if o.budget_gib > 0.0 && std::env::var("AGENS_M1_NO_CAP").map(|v| v != "1").unwrap_or(true) {
        agens_cuda::synchronize()?;
        let (free, total) = agens_cuda::mem_info()?;
        let cap = (o.budget_gib * (1u64 << 30) as f64) as usize;
        if total - free + (64 << 20) < cap && cap < total {
            ballast = free - (cap - (total - free));
            agens_cuda::leak(agens_cuda::device_alloc(ballast)?);
            eprintln!("m1: hard cap {:.2} GiB: {:.2} GiB of the device held by a ballast (context etc. {:.2} GiB)",
                      o.budget_gib, ballast as f64 / (1u64 << 30) as f64, (total - free) as f64 / (1u64 << 30) as f64);
        }
        CAP.store(ballast, std::sync::atomic::Ordering::Relaxed);
    }
    let ck = Checkpoint::open(dir)?;
    let n_layers = ck.config.n_layers;
    let split = std::env::var("AGENS_M1_SPLIT").ok().and_then(|v| v.parse().ok())
        .unwrap_or(if devs.len() == 2 { n_layers / 2 } else { n_layers });
    let layer_of = |name: &str| -> Option<usize> {
        let rest = name.strip_prefix("model.language_model.layers.").or_else(|| name.strip_prefix("model.language_model.mhc."))?;
        rest.split('.').next()?.parse().ok()
    };
    let rep = ck.validate()?;
    if rep.w4_linears > 0 {
        eprintln!("m1: {} W4 linears (by group size {:?}), {} bf16", rep.w4_linears, rep.w4_groups, rep.bf16_linears);
    }
    let t0 = Instant::now();
    let (mut w4_gpu, mut w4_dense, mut w4_max_nk, mut w3_gpu) = (0usize, 0usize, 0usize, 0usize);
    let use_w3 = std::env::var("AGENS_M1_W3").map(|v| v == "1").unwrap_or(false);
    // AGENS_M1_MIXED unset: per-linear W2/W3/W4 when the checkpoint carries sub-4-bit GSQ types (quantization_audit.json
    // gsq_type entries other than W4, or rco_assignment.json)
    let gsq = std::fs::read_to_string(format!("{dir}/quantization_audit.json")).ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .map(|j| j["quantized_matrices"].as_array().into_iter().flatten()
             .any(|r| r["gsq_type"].as_str().map(|t| !t.to_ascii_lowercase().starts_with("w4")).unwrap_or(false)))
        .unwrap_or(false)
        || std::path::Path::new(&format!("{dir}/rco_assignment.json")).exists();
    let mixed_mode = std::env::var("AGENS_M1_MIXED").unwrap_or_else(|_| if gsq { "1".into() } else { String::new() });
    let mixed = match mixed_mode.as_str() {
        "" | "0" => None,
        "1" | "w4" => Some(mixed_types(dir)?),
        v => bail!("AGENS_M1_MIXED={v}: 1 (per-linear W2/W3/W4 KT) or w4 (same parts and groups, all W4 KT)"),
    };
    let mut by_type: std::collections::BTreeMap<String, (usize, usize)> = Default::default(); // type -> (bytes, params)
    let mut gpu_kinds: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut pl = Placer {
        gpu_budget: (gpu_gib * (1u64 << 30) as f64) as usize,
        devs: devs.clone(),
        on_gpu: vec![0; devs.len()],
        on_host: 0,
        cur: 0,
    };
    let mut engram = [std::ptr::null_mut(); 4];
    // =draft: the INT8 head (1.27 GB more) for the drafter's candidate logits while the target decodes with the bf16
    // head (drafter numerics move acceptance only). Not the default: ak_i8_gemv at 7 rows took 1.16 ms on H200 vs
    // 0.62 for the bf16 head.
    let lmh_int8 = std::env::var("AGENS_M1_LMHEAD_INT8").unwrap_or_else(|_| "0".into());
    let lm = "model.language_model";
    for (name, _) in ck.index.iter() {
        if name.starts_with("model.visual.") || name.contains(".mlp.up_proj.") || name.ends_with(".weight_scale")
            || name.ends_with(".weight_shape")
        {
            continue;
        }
        let b = ck.bytes(name)?;
        // device of this tensor: its layer's device; lm_head and the final norm on the last device
        let di = match layer_of(name) {
            Some(l) if l >= split => 1,
            Some(_) => 0,
            None if name == "lm_head.weight" || name == "model.language_model.norm.weight" => devs.len() - 1,
            None => 0,
        };
        pl.on(di)?;
        if let Some(base) = name.strip_suffix(".weight_packed") {
            // W4: gate_proj + up_proj merged row-wise (packed rows and scale rows concatenate), registered under the
            // name of the bf16 weight the forward asks for.
            let Some(agens_core::LinearFormat::W4 { group }) = ck.linear_format(base) else { bail!("{base}: not W4") };
            let (mut k, mut n) = (ck.info(name).unwrap().shape[1] * 8, ck.info(name).unwrap().shape[0]);
            let (mut packed, mut scales) = (b.to_vec(), ck.bytes(&format!("{base}.weight_scale"))?.to_vec());
            // tensor parallel: this rank's rows / columns of the packed weight and of its scales
            let shard = |nm: &str, n: usize, k: usize, p: Vec<u8>, sc: Vec<u8>| -> (usize, usize, Vec<u8>, Vec<u8>) {
                match tp_rule(nm, tp, rank, n) {
                    Some(sh) => {
                        let (nn, kk) = tp_dims(&sh, k);
                        let (pr, sr) = (p.len() / n, sc.len() / n);
                        (nn, kk, tp_apply(&sh, &p, pr, k), tp_apply(&sh, &sc, sr, k))
                    }
                    None => (n, k, p, sc),
                }
            };
            (n, k, packed, scales) = shard(name, n, k, packed, scales);
            let mut reg = format!("{base}.weight");
            let mut parts = vec![(base.to_string(), n)]; // (linear, rows) in output row order
            if let Some(rest) = base.strip_suffix(".mlp.gate_proj") {
                let up = format!("{rest}.mlp.up_proj");
                if ck.linear_format(&up) != Some(agens_core::LinearFormat::W4 { group }) {
                    bail!("{up}: not W4 with group {group} like its gate_proj");
                }
                let nu0 = ck.info(&format!("{up}.weight_packed")).unwrap().shape[0];
                let ku0 = ck.info(&format!("{up}.weight_packed")).unwrap().shape[1] * 8;
                let (nu, _, up_p, up_s) = shard(&format!("{up}.weight_packed"), nu0, ku0,
                                                ck.bytes(&format!("{up}.weight_packed"))?.to_vec(),
                                                ck.bytes(&format!("{up}.weight_scale"))?.to_vec());
                packed.extend_from_slice(&up_p);
                scales.extend_from_slice(&up_s);
                n += nu;
                parts.push((up, nu));
                reg = format!("{rest}.mlp.gate_up_proj.weight");
            }
            if k < 1024 {
                let p: Vec<i32> = packed.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let sc = bf16_from_bytes(&scales);
                let d = agens_core::reference::bf16_to_bytes(&agens_core::reference::w4_dequant_bf16(&p, &sc, n, k, group));
                w4_dense += d.len();
                agens_cuda::m1::register(&reg, pl.place(&d, false)?)?;
            } else if let Some(types) = &mixed {
                // mixed precision: each part at its GSQ-RCO type; parts of one type (all but some gate/up pairs) are one
                // KT linear, as the W4 container merges them. AGENS_M1_MIXED=w4: the same parts and groups, W4 KT.
                w4_max_nk = w4_max_nk.max(n * k);
                let mut tys = Vec::new();
                for (pb, pn) in &parts {
                    let &(bits, g) = types.get(pb.as_str()).with_context(|| format!("{pb}: no GSQ-RCO type"))?;
                    tys.push((*pn, if mixed_mode == "w4" { 4 } else { bits }, g, bits));
                }
                if tys.iter().all(|t| (t.1, t.2) == (tys[0].1, tys[0].2)) {
                    tys = vec![(n, tys[0].1, tys[0].2, tys.iter().map(|t| t.3).max().unwrap())];
                }
                let (prow, srow) = (k / 2, k / group * 2);
                let mut r0 = 0;
                for (pn, bits, g, _) in tys {
                    let sc = collapse_scales(&scales[r0 * srow..(r0 + pn) * srow], pn, k, group, g, &reg)?;
                    let (wt, st) = agens_cuda::wq_upload_kt(bits, &packed[r0 * prow..(r0 + pn) * prow], &sc, pn, k, g)
                        .with_context(|| format!("{reg} rows {r0}..{}", r0 + pn))?;
                    pl.on_gpu[pl.cur] += wt.len() + st.len();
                    let e = by_type.entry(format!("W{bits}g{g}")).or_insert((0, 0));
                    e.0 += wt.len() + st.len();
                    e.1 += pn * k;
                    agens_cuda::m1::register_wq(&reg, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, r0, pn, g, bits)?;
                    r0 += pn;
                }
            } else {
                // AGENS_M1_W3=1: linears whose values all fit 3 bits go to the W3 layout (the 3-bit probe), others W4
                let w3 = if use_w3 { agens_cuda::w3_upload_kt(&packed, &scales, n, k, group)? } else { None };
                w4_max_nk = w4_max_nk.max(n * k);
                if let Some((wt, st)) = w3 {
                    pl.on_gpu[pl.cur] += wt.len() + st.len();
                    w3_gpu += wt.len() + st.len();
                    agens_cuda::m1::register_w3(&reg, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, group)?;
                } else {
                    let (wt, st) = agens_cuda::w4_upload_kt(&packed, &scales, n, k, group)?;
                    pl.on_gpu[pl.cur] += wt.len() + st.len();
                    w4_gpu += wt.len() + st.len();
                    agens_cuda::m1::register_w4(&reg, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, group)?;
                }
            }
            continue;
        }
        // tensor parallel: this rank's rows / columns of a bf16 tensor (and of the KDA fp32 conv / A_log / dt_bias)
        let b_tp: Vec<u8>;
        let b: &[u8] = match tp_rule(name, tp, rank, ck.info(name).and_then(|i| i.shape.first().copied()).unwrap_or(1)) {
            Some(sh) => {
                let sh0 = &ck.info(name).unwrap().shape;
                let k = if sh0.len() >= 2 { sh0[1..].iter().product() } else { 1 };
                b_tp = tp_apply(&sh, b, b.len() / sh0[0], k);
                &b_tp
            }
            None => b,
        };
        if let Some(rest) = name.strip_suffix(".mlp.gate_proj.weight") {
            let upn = format!("{rest}.mlp.up_proj.weight");
            let up0 = ck.bytes(&upn)?;
            let up_tp;
            let up: &[u8] = match tp_rule(&upn, tp, rank, ck.info(&upn).unwrap().shape[0]) {
                Some(sh) => {
                    up_tp = tp_apply(&sh, up0, up0.len() / ck.info(&upn).unwrap().shape[0], ck.info(&upn).unwrap().shape[1]);
                    &up_tp
                }
                None => up0,
            };
            let mut merged = Vec::with_capacity(b.len() + up.len());
            merged.extend_from_slice(b);
            merged.extend_from_slice(up);
            agens_cuda::m1::register(&format!("{rest}.mlp.gate_up_proj.weight"), pl.place(&merged, false)?)?;
            continue;
        }
        if name.contains(".engram.tables.") {
            let p = pl.place(b, true)?;
            let first = ck.config.engram.layers.first().map(|l| name.contains(&format!(".layers.{l}."))).unwrap_or(false);
            let order = if name.contains(".tables.0.") { 0 } else { 1 };
            engram[if first { 0 } else { 2 } + order] = p;
            continue;
        }
        if name.ends_with(".linear_attn.conv1d.weight") || name.ends_with(".linear_attn.A_log")
            || name.ends_with(".linear_attn.dt_bias")
        {
            let f = to_f32_bytes(b);
            *gpu_kinds.entry("KDA conv / A_log / dt_bias (fp32)").or_insert(0) += f.len();
            let reg = format!("{name}.f32");
            agens_cuda::m1::register(&reg, pl.place(&f, false)?)?;
            continue;
        }
        if name == "lm_head.weight" && std::env::var("AGENS_M1_LMHEAD_W4").map(|v| v == "draft").unwrap_or(false) {
            // AGENS_M1_LMHEAD_W4=draft: a W4 g128 copy of the head for the drafter's candidate logits only (the target
            // keeps the bf16 head below, so the output is unchanged; the drafter reads a quarter of the bytes)
            let (n, k) = (ck.info(name).unwrap().shape[0], ck.info(name).unwrap().shape[1]);
            let (pk, sc) = crate::spec::quantize_w4(b, n, k);
            let (wt, st) = agens_cuda::w4_upload_kt(&pk, &sc, n, k, 128)?;
            pl.on_gpu[pl.cur] += wt.len() + st.len();
            *gpu_kinds.entry("lm_head W4 copy (drafter)").or_insert(0) += wt.len() + st.len();
            agens_cuda::m1::register_w4("lm_head.weight.w4draft", agens_cuda::leak(wt), agens_cuda::leak(st), n, k, 128)?;
        }
        if name == "lm_head.weight" && std::env::var("AGENS_M1_LMHEAD_W4").map(|v| v == "1").unwrap_or(false) {
            // W4 lm_head (gated option; quality vs bf16 by teacher-forced logits): RTN to W4 g128 as the drafter's
            // linears, on the engine's W4 kernels (linear(): W4 registry); 0.66 GB instead of 1.27 (INT8) / 2.54 (bf16)
            let (n, k) = (ck.info(name).unwrap().shape[0], ck.info(name).unwrap().shape[1]);
            let (pk, sc) = crate::spec::quantize_w4(b, n, k);
            let (wt, st) = agens_cuda::w4_upload_kt(&pk, &sc, n, k, 128)?;
            pl.on_gpu[pl.cur] += wt.len() + st.len();
            *gpu_kinds.entry("lm_head (W4)").or_insert(0) += wt.len() + st.len();
            agens_cuda::m1::register_w4(name, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, 128)?;
            continue;
        }
        if name == "lm_head.weight" && lmh_int8 != "0" {
            // INT8 lm_head (AGENS_M1_LMHEAD_INT8=1): per-row symmetric RTN, int8 [vocab][hid] + fp32 scales; =cmp keeps
            // the bf16 head too (it decodes; the INT8 logits are compared per token in teacher-forced runs)
            let (n, k) = (ck.info(name).unwrap().shape[0], ck.info(name).unwrap().shape[1]);
            let (q, s) = quantize_i8_rows(b, n, k);
            agens_cuda::m1::register("lm_head.weight.i8", pl.place(&q, false)?)?;
            agens_cuda::m1::register("lm_head.weight.scale", pl.place(&s, false)?)?;
            *gpu_kinds.entry("lm_head (INT8)").or_insert(0) += q.len() + s.len();
            eprintln!("m1: lm_head INT8 {:.2} GB (bf16 {:.2} GB{})", (q.len() + s.len()) as f64 / 1e9, b.len() as f64 / 1e9,
                      if lmh_int8 == "cmp" { ", kept for comparison" } else if lmh_int8 == "draft" { ", kept: the target decodes with it, the drafter takes INT8" } else { ", dropped" });
            if lmh_int8 != "cmp" && lmh_int8 != "draft" {
                continue;
            }
        }
        let host = *name == format!("{lm}.embed_tokens.weight");
        if !host {
            let kind = if name == "lm_head.weight" {
                "lm_head (bf16)"
            } else if ck.info(name).map(|i| i.shape.len() == 2).unwrap_or(false) && !name.contains("norm") {
                "bf16 linears (recipe-ignored)"
            } else {
                "norms, biases, small tensors"
            };
            *gpu_kinds.entry(kind).or_insert(0) += b.len();
        }
        agens_cuda::m1::register(name, pl.place(b, host)?)?;
    }
    for (k, &d) in devs.iter().enumerate() {
        pl.on(k)?;
        if std::env::var("AGENS_ROPE_CACHE").map(|v| v == "1").unwrap_or(false) {  // default: RoPE on the fly
            let rope = agens_cuda::rope_cache((wctx + 4095) & !4095, 64, 10_000_000.0)?; // positions < wctx
            agens_cuda::m1::register(&format!("rope.cos_sin_cache.{d}"), agens_cuda::leak(rope))?;
        }
    }
    pl.on(0)?;
    let mixers: String = ck
        .config
        .layers
        .iter()
        .map(|l| match l.mixer {
            agens_core::Mixer::Kda => 'k',
            agens_core::Mixer::Bcsa => 'b',
            agens_core::Mixer::Dense => 'd',
        })
        .collect();
    let el = &ck.config.engram.layers;
    let eng_layers = [el.first().map(|&x| x as i32).unwrap_or(-1), el.get(1).map(|&x| x as i32).unwrap_or(-1)];
    let draft = match &o.drafter {
        Some(d) => Some(crate::spec::upload_drafter(d)?),
        None => None,
    };
    let mtp = match &o.mtp {
        Some(d) => Some(crate::spec::upload_mtp(d)?),
        None => None,
    };
    let cap_rows = crate::plan::chunk();
    let n_kda = ck.config.layers.iter().filter(|l| l.mixer == agens_core::Mixer::Kda).count();
    let mut reserve = if draft.is_some() || mtp.is_some() {
        crate::spec::spec_reserve(cap_rows, n_kda, draft.as_ref().map(|d| d.w4).unwrap_or(false), wctx,
                                  ck.config.layers.iter().filter(|l| l.mixer == agens_core::Mixer::Bcsa).count())
    } else {
        Vec::new()
    };
    if mtp.is_some() {
        reserve.push(crate::spec::mtp_reserve(cap_rows, o.sessions));
    }
    let t_weights = t0.elapsed().as_secs_f64();
    let mut pool = o.pool;
    if o.budget_gib > 0.0 {  // size the KV pool from the budget left after weights + context (one device)
        agens_cuda::synchronize()?;
        let (free, total) = agens_cuda::mem_info()?;
        let budget = ((o.budget_gib * (1u64 << 30) as f64) as usize).min(total);
        let p = crate::plan::plan(&ck.config, budget, total - free - ballast, o.sessions, wctx, w4_max_nk, &reserve);
        for l in &p.lines {
            eprintln!("{l}");
        }
        if p.pool_rows < 4096 {
            bail!("budget {} GiB leaves no room for a KV pool", o.budget_gib);
        }
        // at most one full-length context of rows by default (the sessions share the pool): with tiered KV every pool
        // row also pins ~87-91 KB of host RAM (24 GB at 262K), so a budget-sized pool on a big card would pin far more
        // host memory than a full context needs. AGENS_M1_POOL_MAX_ROWS overrides (server mode with many long chats).
        let cap: usize = std::env::var("AGENS_M1_POOL_MAX_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(wctx);
        pool = p.pool_rows.min(cap.max(4096));
        if pool < p.pool_rows {
            eprintln!("m1: KV pool capped at {pool} rows (plan {}; AGENS_M1_POOL_MAX_ROWS)", p.pool_rows);
        }
    }
    if tp > 1 {
        agens_cuda::m1::tp_config(rank, tp)?;
    }
    agens_cuda::m1::init(wctx, &mixers, eng_layers, engram, &devs, split, pool, o.sessions)?;
    // the largest all-reduce: a prefill chunk of hidden rows
    let chunk = std::env::var("AGENS_M1_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4096).min(4096);
    crate::tp::setup_allreduce(chunk * 5120)?;
    if let Some(d) = &draft {
        crate::spec::finish_drafter(d, cap_rows)?;
    }
    if let Some(m) = &mtp {
        if draft.is_none() {
            agens_cuda::m1::spec_init(true, None, cap_rows)?;
        }
        crate::spec::finish_mtp(m, cap_rows)?;
    }
    eprintln!(
        "m1: devices {:?} (split at layer {split}), weights {:?} GB on GPU, {:.1} GB host-mapped, loaded in {:.0} s",
        devs,
        pl.on_gpu.iter().map(|b| (*b as f64 / 1e8).round() / 10.0).collect::<Vec<_>>(),
        pl.on_host as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    if o.budget_gib > 0.0 {
        if !by_type.is_empty() {
            let (tb, tp) = by_type.values().fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));
            eprintln!("m1: mixed-precision KT weights {:.3} GB, {:.4} bits/weight with scales over {:.3} B params: {}",
                      tb as f64 / 1e9, tb as f64 * 8.0 / tp as f64, tp as f64 / 1e9,
                      by_type.iter().map(|(t, v)| format!("{t} {:.3} GB ({:.2} B)", v.0 as f64 / 1e9, v.1 as f64 / 1e9))
                          .collect::<Vec<_>>().join(", "));
            w4_gpu += tb;
        }
        eprintln!("m1: GPU weights by kind (GB): W4 KT {:.3}, W3 KT {:.3}, small-k W4 as bf16 {:.3}, {}",
                  w4_gpu as f64 / 1e9, w3_gpu as f64 / 1e9, w4_dense as f64 / 1e9,
                  gpu_kinds.iter().map(|(k, v)| format!("{k} {:.3}", *v as f64 / 1e9)).collect::<Vec<_>>().join(", "));
    }
    if w4_gpu + w3_gpu + w4_dense > 0 {
        eprintln!("m1: W4 k-tiled {:.2} GB + W3 k-tiled {:.2} GB on GPU, small-k W4 as bf16 {:.3} GB; weights placed in \
                   {t_weights:.0} s", w4_gpu as f64 / 1e9, w3_gpu as f64 / 1e9, w4_dense as f64 / 1e9);
    }
    for (k, &d) in devs.iter().enumerate() {
        pl.on(k)?;
        let (free, total) = agens_cuda::mem_info()?;
        let b = if k == 0 { ballast } else { 0 };
        eprintln!("m1: device {d}: {:.2} GB used of {:.2} GB after init (weights + caches + scratch + CUDA context){}",
                  (total - free - b) as f64 / 1e9, total as f64 / 1e9,
                  if b > 0 { format!(", {:.2} GB free under the cap", free as f64 / 1e9) } else { String::new() });
    }
    pl.on(0)?;
    Ok(())
}

/// bf16 [n][k] -> int8 [n][k] + fp32 scale [n] (per row: s = max|w| / 127, q = round(w / s)), rows over threads.
fn quantize_i8_rows(b: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (mut q, mut s) = (vec![0u8; n * k], vec![0u8; n * 4]);
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8).min(64);
    let rows = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for ((src, qd), sd) in b.chunks(rows * k * 2).zip(q.chunks_mut(rows * k)).zip(s.chunks_mut(rows * 4)) {
            sc.spawn(move || {
                for (r, row) in src.chunks_exact(k * 2).enumerate() {
                    let w: Vec<f32> =
                        row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                    let amax = w.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    let sc = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                    for (i, &x) in w.iter().enumerate() {
                        qd[r * k + i] = (x / sc).round().clamp(-127.0, 127.0) as i8 as u8;
                    }
                    sd[r * 4..r * 4 + 4].copy_from_slice(&sc.to_le_bytes());
                }
            });
        }
    });
    (q, s)
}

/// M2a exactness test: a session snapshotted at a random cut, clobbered, and restored must continue with exactly the
/// tokens of the uninterrupted run. Tiers: gpu (device buffer), host (pinned RAM), nvme (file under --dir out/).
/// With --sessions >= 2 also: fork at a random cut (child and parent must both finish exactly like the reference) and
/// two sessions decoded interleaved token by token (each must equal its solo run).
pub fn session_test(args: &[String]) -> Result<()> {
    let usage = "agens m1-session <ckpt_dir> <prompts.json> [--cuts 50] [--tier gpu|host|nvme] [--new-tokens N] [--gpu-gib N] [--wctx N] [--seed S] [--pool ROWS] [--sessions N]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let o = parse_opts(&args[2..], usage)?;
    load(dir, &o)?;
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let plist = prompts.as_array().context("prompts")?;
    let ids_of = |p: &serde_json::Value| -> Vec<i32> {
        p["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
    };
    // uninterrupted reference per prompt
    let mut refs = Vec::new();
    for p in plist {
        agens_cuda::m1::reset()?;
        let mut tok = agens_cuda::m1::prefill(&ids_of(p))?;
        let mut out = vec![tok];
        while out.len() < o.new_tokens {
            tok = agens_cuda::m1::decode(tok)?;
            out.push(tok);
        }
        refs.push(out);
    }
    let mut rng = o.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) as usize
    };
    let (mut ok, mut total) = (0usize, 0usize);
    let (mut t_snap, mut t_rest, mut bytes_max) = (0f64, 0f64, 0usize);
    let clobber: Vec<i32> = (0..37).map(|i| 1000 + 17 * i).collect();
    // one snapshot buffer per tier, grown on demand (allocation is not part of snapshot cost)
    let (mut cap, mut hbuf, mut dbuf): (usize, *mut std::ffi::c_void, Option<agens_cuda::DeviceBuffer>) = (0, std::ptr::null_mut(), None);
    for c in 0..o.cuts {
        let pi = next() % plist.len();
        let cut = 1 + next() % (o.new_tokens - 1);  // tokens produced before the snapshot
        let ids = ids_of(&plist[pi]);
        agens_cuda::m1::reset()?;
        let mut tok = agens_cuda::m1::prefill(&ids)?;
        let mut out = vec![tok];
        while out.len() < cut {
            tok = agens_cuda::m1::decode(tok)?;
            out.push(tok);
        }
        let nb = agens_cuda::m1::state_bytes();
        bytes_max = bytes_max.max(nb);
        if nb > cap {
            cap = nb + nb / 4;
            if o.tier == "gpu" {
                dbuf = Some(agens_cuda::device_alloc(cap)?);
            } else {
                if !hbuf.is_null() {
                    agens_cuda::host_free(hbuf)?;
                }
                hbuf = agens_cuda::host_alloc(cap)?;
            }
        }
        let blob = if o.tier == "gpu" { dbuf.as_ref().unwrap().as_ptr() } else { hbuf };
        let t = Instant::now();
        agens_cuda::m1::snapshot(blob)?;
        let path = format!("out/session_{c}.agss");
        if o.tier == "nvme" {
            std::fs::create_dir_all("out")?;
            let sl = unsafe { std::slice::from_raw_parts(blob as *const u8, nb) };
            std::fs::write(&path, sl)?;
        }
        t_snap += t.elapsed().as_secs_f64();
        // clobber: a different conversation through the same engine state
        agens_cuda::m1::reset()?;
        let mut x = agens_cuda::m1::prefill(&clobber)?;
        for _ in 0..5 {
            x = agens_cuda::m1::decode(x)?;
        }
        let t = Instant::now();
        if o.tier == "nvme" {
            let data = std::fs::read(&path)?;
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), blob as *mut u8, nb) };
            std::fs::remove_file(&path)?;
        }
        agens_cuda::m1::restore(blob)?;
        t_rest += t.elapsed().as_secs_f64();
        while out.len() < o.new_tokens {
            tok = agens_cuda::m1::decode(tok)?;
            out.push(tok);
        }
        total += 1;
        let same = out == refs[pi];
        ok += same as usize;
        if !same {
            let d = out.iter().zip(&refs[pi]).position(|(a, b)| a != b).unwrap_or(out.len());
            eprintln!("cut {c}: prompt {pi} cut at {cut}: DIVERGES at {d}");
        }
    }
    println!(
        "session exactness ({} tier): {ok}/{total} cuts identical; state up to {:.1} MB; snapshot {:.1} ms, restore {:.1} ms avg",
        o.tier,
        bytes_max as f64 / 1e6,
        1e3 * t_snap / total as f64,
        1e3 * t_rest / total as f64
    );
    if !hbuf.is_null() {
        agens_cuda::host_free(hbuf)?;
    }
    drop(dbuf);
    if ok != total {
        bail!("session exactness failed");
    }
    if o.sessions >= 2 {
        multi_session_checks(plist.iter().map(ids_of).collect(), &refs, &o, &mut next)?;
    }
    Ok(())
}

/// Fork + interleave exactness (needs >= 2 KDA slots). All decoding is greedy, so "exact" = identical token lists.
fn multi_session_checks(prompts: Vec<Vec<i32>>, refs: &[Vec<i32>], o: &Opts, next: &mut dyn FnMut() -> usize) -> Result<()> {
    use agens_cuda::m1;
    let n = o.new_tokens;
    let main = m1::session_current();
    // fork: parent runs to the cut, child = fork; child finishes first, then the parent resumes
    let (mut ok, mut total) = (0, 0);
    let mut t_fork = 0f64;
    for _ in 0..prompts.len().max(5) {
        let pi = next() % prompts.len();
        let cut = 1 + next() % (n - 1);
        m1::reset()?;
        let mut tok = m1::prefill(&prompts[pi])?;
        let mut a = vec![tok];
        while a.len() < cut {
            tok = m1::decode(tok)?;
            a.push(tok);
        }
        let t = Instant::now();
        let child = m1::session_fork(main)?;
        t_fork += t.elapsed().as_secs_f64();
        m1::session_use(child)?;
        let (mut b, mut tb) = (a.clone(), tok);
        while b.len() < n {
            tb = m1::decode(tb)?;
            b.push(tb);
        }
        m1::session_use(main)?;
        while a.len() < n {
            tok = m1::decode(tok)?;
            a.push(tok);
        }
        m1::session_free(child)?;
        for (who, out) in [("child", &b), ("parent", &a)] {
            total += 1;
            if *out == refs[pi] {
                ok += 1;
            } else {
                let d = out.iter().zip(&refs[pi]).position(|(x, y)| x != y).unwrap_or(out.len());
                eprintln!("fork: prompt {pi} cut {cut}: {who} DIVERGES at {d}");
            }
        }
    }
    println!("fork exactness: {ok}/{total} branches identical; fork {:.2} ms avg", 1e3 * t_fork / (total / 2) as f64);
    let fork_ok = ok == total;
    // interleave: two sessions alternate token by token (pointer-swap switches every step)
    let (p0, p1) = (0, 1 % prompts.len());
    m1::reset()?;
    let s1 = m1::session_new()?;
    let mut t0 = m1::prefill(&prompts[p0])?;
    m1::session_use(s1)?;
    let mut t1 = m1::prefill(&prompts[p1])?;
    let (mut o0, mut o1) = (vec![t0], vec![t1]);
    let t = Instant::now();
    let mut switches = 0;
    while o0.len() < n || o1.len() < n {
        if o0.len() < n {
            m1::session_use(main)?;
            t0 = m1::decode(t0)?;
            o0.push(t0);
            switches += 1;
        }
        if o1.len() < n {
            m1::session_use(s1)?;
            t1 = m1::decode(t1)?;
            o1.push(t1);
            switches += 1;
        }
    }
    let dt = t.elapsed().as_secs_f64();
    m1::session_use(main)?;
    m1::session_free(s1)?;
    let il_ok = o0 == refs[p0] && o1 == refs[p1];
    println!(
        "interleave: prompts {p0},{p1} alternating every token ({switches} switches, {:.1} ms/step): {}",
        1e3 * dt / switches as f64,
        if il_ok { "both identical" } else { "DIVERGED" }
    );
    println!("pool rows free after the checks: {}", m1::pool_free_rows());
    if !(fork_ok && il_ok) {
        bail!("multi-session exactness failed");
    }
    Ok(())
}

/// M2a demo: replay real agent sessions turn by turn (tools/make_m2a_transcripts.py output).
///   session   : one resident main session per transcript holds the prompt prefix common to consecutive turns
///               (ids_k[:keep_k]); turn k = advance main by the new tokens (tool results etc.), fork, prefill the
///               generation prompt on the fork, greedy-generate, free the fork. Nothing is ever re-prefilled.
///   reprefill : every turn resets and prefills ids_k from scratch (the public image as shipped, no session cache).
/// Per turn: time to first token (everything before the first generated token) and the greedy tokens (<= --new-tokens,
/// stop at end-of-turn). Compare two runs' token lists with tools/compare_m2a.py.
pub fn m2a_demo(args: &[String]) -> Result<()> {
    use agens_cuda::m1;
    let usage = "agens m2a-demo <ckpt_dir> <transcripts.json> <out.json> --mode session|reprefill|replay [--max-sessions N] [--new-tokens N] [--devices 0,1] [--wctx N] [--pool ROWS] [--sessions N]";
    let dir = args.first().context(usage)?;
    let tpath = args.get(1).context(usage)?;
    let out_path = args.get(2).context(usage)?;
    let (mut mode, mut max_sessions, mut rest) = (String::new(), usize::MAX, Vec::new());
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => mode = args.get(i + 1).context(usage)?.clone(),
            "--max-sessions" => max_sessions = args.get(i + 1).context(usage)?.parse()?,
            _ => {
                rest.push(args[i].clone());
                rest.push(args.get(i + 1).context(usage)?.clone());
            }
        }
        i += 2;
    }
    if !["session", "reprefill", "replay"].contains(&mode.as_str()) {
        bail!("{usage}");
    }
    let mut o = parse_opts(&rest, usage)?;
    o.sessions = o.sessions.max(2);
    load(dir, &o)?;
    let tr: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tpath)?)?;
    let ids_of = |v: &serde_json::Value| -> Vec<i32> { v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect() };
    let main = m1::session_current();
    let mut results = Vec::new();
    for s in tr.as_array().context("transcripts")?.iter().take(max_sessions) {
        let task = s["task"].as_str().unwrap_or("?");
        m1::session_use(main)?;
        m1::reset()?;
        let mut held = 0usize; // tokens in the main session
        let mut cuts = vec![0usize]; // session mode's prefill boundaries so far (replay re-feeds them)
        let turns = s["turns"].as_array().context("turns")?;
        for (k, t) in turns.iter().enumerate() {
            let ids = ids_of(&t["ids"]);
            let keep = (t["keep"].as_u64().unwrap() as usize).min(ids.len() - 1);
            let t0 = Instant::now();
            let (mut tok, fork) = if mode == "session" {
                if keep > held {
                    m1::prefill(&ids[held..keep])?; // the turn's new tokens: tool results, user text
                    held = keep;
                }
                let f = m1::session_fork(main)?;
                m1::session_use(f)?;
                (m1::prefill(&ids[held..])?, Some(f))
            } else if mode == "replay" {
                // fresh state, but the same prefill calls the session mode made: separates session-state errors
                // from chunk-boundary numerics
                if keep > held {
                    held = keep;
                    cuts.push(keep);
                }
                m1::reset()?;
                for w in cuts.windows(2) {
                    m1::prefill(&ids[w[0]..w[1]])?;
                }
                (m1::prefill(&ids[held..])?, None)
            } else {
                m1::reset()?;
                (m1::prefill(&ids)?, None)
            };
            let ttft = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut out = vec![tok];
            while out.len() < o.new_tokens && tok != EOT {
                tok = m1::decode(tok)?;
                out.push(tok);
            }
            let dec = t1.elapsed().as_secs_f64();
            if let Some(f) = fork {
                m1::session_use(main)?;
                m1::session_free(f)?;
            }
            eprintln!(
                "{task} turn {k:2}/{}: prompt {:6} tok (main holds {held:6}), first token {:7.0} ms, {} tok in {:.1} s",
                turns.len(),
                ids.len(),
                1e3 * ttft,
                out.len(),
                dec
            );
            results.push(serde_json::json!({"task": task, "turn": k, "prompt": ids.len(), "held": held,
                                            "ttft_ms": 1e3 * ttft, "decode_s": dec, "tokens": out}));
            std::fs::write(out_path, serde_json::to_string(&results)?)?;
        }
    }
    Ok(())
}

/// INT8 vs bf16 lm_head on the same hidden states (AGENS_M1_LMHEAD_INT8=cmp, teacher-forced prompts): KL(p_bf16 ||
/// p_int8) over the valid vocabulary at temperature 1, top-1 agreement, and the bf16 top-1's rank in the INT8 head.
#[derive(Default)]
pub(crate) struct HeadCmp {
    n: usize,
    top1: usize,
    kl: f64,
    kl_max: f64,
    kls: Vec<f64>,
}

fn head_cmp(h: &mut HeadCmp) -> Result<()> {
    let to_f = |v: &[u16]| -> Vec<f64> { v.iter().map(|&b| f32::from_bits((b as u32) << 16) as f64).collect() };
    let (a, b) = (to_f(&agens_cuda::m1::logits_row(0)?), to_f(&agens_cuda::m1::logits_i8_row(0)?));
    let lse = |x: &[f64]| {
        let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(&a), lse(&b));
    let kl: f64 = a.iter().zip(&b).map(|(x, y)| {
        let p = (x - la).exp();
        if p > 0.0 { p * ((x - la) - (y - lb)) } else { 0.0 }
    }).sum();
    let am = |x: &[f64]| x.iter().enumerate().fold(0, |bi, (i, v)| if *v > x[bi] { i } else { bi });
    h.n += 1;
    h.top1 += (am(&a) == am(&b)) as usize;
    h.kl += kl;
    h.kl_max = h.kl_max.max(kl);
    h.kls.push(kl);
    if h.n % 500 == 0 {
        h.report();
    }
    Ok(())
}

impl HeadCmp {
    fn report(&self) {
        let mut k = self.kls.clone();
        k.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| k[((k.len() as f64 - 1.0) * f) as usize];
        eprintln!(
            "lm_head INT8 vs bf16 over {} tokens: top-1 agreement {}/{} ({:.3}%), KL mean {:.3e} nats, p50 {:.2e}, \
             p99 {:.2e}, max {:.3e}",
            self.n, self.top1, self.n, 100.0 * self.top1 as f64 / self.n as f64, self.kl / self.n as f64, q(0.5),
            q(0.99), self.kl_max
        );
    }
}
