//! GPU memory plan for `agens serve` (and any m1 run with --budget-gib): split a byte budget between what is fixed
//! (weights, CUDA context, the prefill-chunk scratch, workspaces) and what scales with concurrency (KDA state slots,
//! per-session GPU window rings when tiering) and with context (KV pool rows: dense-layer K/V, BCSA K/V unless tiered,
//! indexer side store). Sizes follow docs/kv_tiering.md; the KV mode comes from AGENS_KV_TIER / AGENS_KV_INT8.

use agens_core::{Mixer, VolundrConfig};

pub struct Plan {
    pub pool_rows: usize,
    pub lines: Vec<String>,
}

pub fn chunk() -> usize {  // prefill chunk (kernels/m1/m1_model.cu g_chunk, AGENS_M1_PREFILL_CHUNK)
    std::env::var("AGENS_M1_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096usize).clamp(64, 4096)
}
const WIN: usize = 4096;

fn w4_scratch(w4_max_nk: usize) -> usize {  // AGENS_M1_W4_TILE_MB (default 64; 0 = whole weight)
    let mb: usize = std::env::var("AGENS_M1_W4_TILE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    if mb == 0 { w4_max_nk * 2 } else { (w4_max_nk * 2).min(mb << 20) }
}

fn runtime_reserve() -> usize {
    std::env::var("AGENS_M1_RUNTIME_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(850usize) << 20
}

fn env_on(n: &str) -> bool {
    std::env::var(n).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

/// Bytes of the m1 scratch allocated by init for one CHUNK (init_device), plus workspaces and the prefill
/// temporaries measured on the 4090 (track A1: +0.9 GB peak in an 8K prefill).
pub fn scratch_bytes(w4_max_nk: usize, wctx: usize, tier: bool) -> (usize, Vec<(String, usize)>) {
    let t = chunk();
    let (hid, ns, inter) = (5120usize, 4usize, 17408usize);
    let acts = 2 * (3 * t * ns * hid + 6 * t * hid + t * (24 + 4 + 16) + 2 * t * 6144 + t * 10240 + t * 2 * inter + t * inter
        + t * 12288 + t * 6144 + 3 * t * 6144 + 2 * t * 1024 + t * 512 + t * 128 + t * 1024 + t)
        + 4 * t * 24
        + 8 * 5 * t;
    let parts = vec![
        (format!("activations (prefill chunk {t})"), acts),
        ("W4 dequant scratch (prefill, tiled)".to_string(), w4_scratch(w4_max_nk)),
        ("flashinfer + cuBLAS workspaces".to_string(), 2 * (136 << 20) + (32 << 20)),
        ("prefill temporaries (KDA chunk, ATen far branch)".to_string(), (1usize << 30) * t / 4096),
        // m1_kv far_extend: K + V bf16 rows of one layer for every far position (block means: one row per block)
        ("far-extend staging (pool-length rows, tiered)".to_string(),
         if !tier { 0 } else if std::env::var("AGENS_KV_TIER").map(|v| v == "2").unwrap_or(false) { (wctx / 4 + 1) * 4096 }
         else { wctx * 4096 }),
        // measured on the 4090 (track A1d): device use after init -> after 76 / 12K / 33K-token prefills and decode,
        // beyond the parts above: lazily loaded kernel modules (CUDA_MODULE_LOADING=LAZY), torch's cache of the ATen
        // far-extend temporaries (expandable segments, 5e7-element steps: 0.28 GB), CUDA graphs. AGENS_M1_RUNTIME_MB.
        // A1e (Wctx 262148, drafter, TIER=2): growth after init up to 1.24 GB at 8K-240K prompts vs 1.11 GB of planned
        // reserves at 650 MB -> 850 MB.
        ("runtime growth (lazy modules, torch cache, graphs)".to_string(), runtime_reserve()),
        // m1_dense8 scratch_bytes(): bf16 staging of one 8192-position tile (K + V) + merge buffers of a 1024-row chunk
        // ak_far_extend_engine_ws_bytes: far block keys + one 64-query tile's scores / top-k (engine prefill extend)
        ("far extend (engine) workspace".to_string(),
         if tier && std::env::var("AGENS_FAR_EXTEND_ENGINE").map(|v| v == "1").unwrap_or(false) {
             (wctx / 4 + 512) * (256 + 64 * 4 * 2 + 64 * 2) + 64 * 512 * 8 + 4096 } else { 0 }),
        ("INT8 dense KV: tile staging + merge".to_string(),
         if std::env::var("AGENS_KV_INT8_DENSE").map(|v| v == "1").unwrap_or(false) {
             2 * 8192 * 4 * 256 * 2 + 8192 * 4 + 1024 * 24 * 256 * 6 + 1024 * 24 * 8 + 4096 } else { 0 }),
    ];
    (parts.iter().map(|p| p.1).sum(), parts)
}

pub fn plan(cfg: &VolundrConfig, budget: usize, used_now: usize, slots: usize, wctx: usize, w4_max_nk: usize,
            reserve: &[(String, usize)]) -> Plan {
    let tier = env_on("AGENS_KV_TIER");
    let means = std::env::var("AGENS_KV_TIER").map(|v| v == "2").unwrap_or(false);
    let int8 = env_on("AGENS_KV_INT8");
    let n = |m: Mixer| cfg.layers.iter().filter(|l| l.mixer == m).count();
    let (nk, nb, nd) = (n(Mixer::Kda), n(Mixer::Bcsa), n(Mixer::Dense));
    let kv_tok = 4 * 256 * 2 * 2; // K + V bf16, one layer
    let bcsa_tok = if int8 { 4 * 256 * 2 + 4 * 4 * 2 } else { kv_tok };
    let kda_slot = nk * (3 * 10240 * 2 + 48 * 128 * 128 * 4);
    let ring = if tier { nb * (WIN + 8) * bcsa_tok } else { 0 }; // m1_kv.cu RING = window + a verify block
    // per pool row on the GPU: dense-layer K/V, BCSA block keys (one 256 B key per 4-row page = 64 B per row), indexer
    // keys (256 B; host with AGENS_KV_IDX_HOST=1 + tiering), BCSA K/V unless tiered
    let idx_host = tier && std::env::var("AGENS_KV_IDX_HOST").map(|v| v == "1").unwrap_or(false);
    let idx_row = 128 * 2;
    // AGENS_KV_INT8_DENSE=1: the dense layer's K/V int8 + fp32 scale per (token, head) (kernels/m1/m1_dense8.cu)
    let dense8 = std::env::var("AGENS_KV_INT8_DENSE").map(|v| v == "1").unwrap_or(false);
    let dense_tok = if dense8 { 4 * 256 * 2 + 4 * 4 * 2 } else { kv_tok };
    let row_gpu = nd * dense_tok + nb * 64 + if idx_host { 0 } else { nb * idx_row } + if tier { 0 } else { nb * bcsa_tok };
    let row_host = if tier { nb * bcsa_tok + if means { nb * kv_tok / 4 } else { 0 } + if idx_host { nb * idx_row } else { 0 } }
                   else { 0 }; // means: one row per page
    let (scratch, mut parts) = scratch_bytes(w4_max_nk, wctx, tier);
    let scratch = scratch + reserve.iter().map(|p| p.1).sum::<usize>();
    parts.extend(reserve.iter().cloned());
    let fixed = used_now + scratch + slots * (kda_slot + ring);
    let avail = budget.saturating_sub(fixed);
    let pool_rows = avail / row_gpu;
    let gb = |b: usize| b as f64 / 1e9;
    let mut lines = vec![
        format!("memory plan: budget {:.2} GB; KV mode tier={} means={} int8={}", gb(budget), tier, means, int8),
        format!("  weights + CUDA context (measured)        {:7.2} GB", gb(used_now)),
    ];
    for (name, b) in &parts {
        lines.push(format!("  {:40} {:7.2} GB", name, gb(*b)));
    }
    lines.push(format!("  KDA state {slots} slots x {:.1} MB              {:7.2} GB", kda_slot as f64 / 1e6, gb(slots * kda_slot)));
    if tier {
        lines.push(format!("  window rings {slots} slots x {:.1} MB           {:7.2} GB", ring as f64 / 1e6, gb(slots * ring)));
    }
    lines.push(format!(
        "  KV pool {pool_rows} rows x {} B (dense K/V{}, {})  {:7.2} GB  -> {:.0}K tokens over all sessions",
        row_gpu,
        if tier { "" } else { " + BCSA K/V" },
        if idx_host { "blk; idx on host" } else { "idx+blk" },
        gb(pool_rows * row_gpu),
        pool_rows as f64 / 1e3
    ));
    if tier {
        lines.push(format!("  host (pinned): {} B per pool row -> {:.2} GB", row_host, gb(pool_rows * row_host)));
    }
    if pool_rows < wctx {
        lines.push(format!("  WARNING: pool ({pool_rows} rows) is shorter than one full-length session (wctx {wctx})"));
    }
    Plan { pool_rows, lines }
}

/// `agens m1-plan <ckpt_dir> --weights-gb W [--budget-gib 16] [--sessions N] [--wctx N] [--context-gb 0.45]`: the plan
/// for a GPU we do not have (e.g. the 16 GB laptop target), from the checkpoint's config and an assumed weight size.
pub fn dry_run(args: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    let usage = "agens m1-plan <ckpt_dir> --weights-gb W [--budget-gib 16] [--sessions N] [--wctx N] [--context-gb 0.45]";
    let ck = agens_core::Checkpoint::open(args.first().context(usage)?)?;
    let (mut w, mut budget, mut sessions, mut wctx, mut ctx) = (11.8f64, 16.0f64, 8usize, 32772usize, 0.45f64);
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?;
        match args[i].as_str() {
            "--weights-gb" => w = v.parse()?,
            "--budget-gib" => budget = v.parse()?,
            "--sessions" => sessions = v.parse()?,
            "--wctx" => wctx = v.parse()?,
            "--context-gb" => ctx = v.parse()?,
            f => anyhow::bail!("unknown flag {f}\n{usage}"),
        }
        i += 2;
    }
    let p = plan(&ck.config, (budget * (1u64 << 30) as f64) as usize, ((w + ctx) * 1e9) as usize, sessions, wctx, 34816 * 5120,
                 &[]);
    for l in &p.lines {
        println!("{l}");
    }
    Ok(())
}
