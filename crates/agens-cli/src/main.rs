//! `agens` — engine CLI. Subcommands grow with the milestones; today: inspect / validate / verify / load / devices.

use agens_core::weights::{manifest, Expect};
use agens_core::{Checkpoint, Mixer};
use anyhow::{bail, Context, Result};
use std::time::Instant;

#[cfg(feature = "m1")]
mod batch;
#[cfg(feature = "m1")]
mod plan;
#[cfg(feature = "m1")]
mod sampler;
#[cfg(feature = "m1")]
mod serve;
#[cfg(feature = "m1")]
mod prefix;
#[cfg(feature = "m1")]
mod kv;
mod lookup;
mod tp;
#[cfg(feature = "m1")]
mod m1;
#[cfg(feature = "m1")]
mod spec;
#[cfg(feature = "m1")]
mod mm;

const USAGE: &str = "usage:
  agens inspect  <ckpt_dir>             layer map + byte budget
  agens validate <ckpt_dir>             check every tensor's dtype/shape against the config
  agens verify   <ckpt_dir> [threads]   sha256 shards against SHA256SUMS
  agens load     <ckpt_dir> [--device N] [--engram-gpu]   upload text weights to one GPU, report GB/s
  agens devices
  agens m1 <ckpt_dir> <prompts.json> <out.json> [--gpu-gib N] [--new-tokens N] [--wctx N]   (feature m1)
  agens m1-session <ckpt_dir> <prompts.json> [--cuts N --tier gpu|host|nvme --sessions N --pool ROWS]   (feature m1)
  agens m2a-demo <ckpt_dir> <transcripts.json> <out.json> --mode session|reprefill [...]   (feature m1)
  agens m1-kv <ckpt_dir> <prompts.json> <out.json> [--force ref.json --save-logits D | --cmp-logits D]   (feature m1;
        KV mode from AGENS_KV_TIER / AGENS_KV_INT8)
  agens m1-spec <ckpt_dir> <prompts.json> <out.json> [--oracle ref.json --corrupt K | --drafter DIR] [m1 flags]
        speculative decoding (feature m1)
  agens m1-batch <ckpt_dir> <prompts.json> [--batch 2,4,8 --new-tokens N --stagger S --park 1]   (feature m1;
        gate: batched decode rows == bs=1, AGENS_M1_BATCH_INVARIANT=1 for server numerics)
  agens serve <ckpt_dir> [--mode personal|server --max-batch 8 --budget-gib N --port 30000 ...]   (feature m1)";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let dir = || args.get(1).context(USAGE);
    match cmd {
        "inspect" => inspect(&Checkpoint::open(dir()?)?),
        "validate" => {
            let ck = Checkpoint::open(dir()?)?;
            let r = ck.validate()?;
            println!(
                "OK: {} text tensors, {:.2} GB, linears bf16 {} / w4 {} (by group size {:?})",
                r.tensors,
                r.bytes as f64 / 1e9,
                r.bf16_linears,
                r.w4_linears,
                r.w4_groups
            );
            Ok(())
        }
        "verify" => {
            let ck = Checkpoint::open(dir()?)?;
            let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8);
            let t = Instant::now();
            let v = ck.verify_checksums(threads)?;
            println!("OK: {} files match SHA256SUMS ({:.1} s)", v.len(), t.elapsed().as_secs_f64());
            Ok(())
        }
        "load" => {
            let mut device = 0usize;
            let mut engram_gpu = false;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--device" => {
                        device = args.get(i + 1).context("--device N")?.parse()?;
                        i += 1;
                    }
                    "--engram-gpu" => engram_gpu = true,
                    o => bail!("unknown flag {o}\n{USAGE}"),
                }
                i += 1;
            }
            load(&Checkpoint::open(dir()?)?, device, engram_gpu)
        }
        #[cfg(feature = "m1")]
        "m1" => m1::run(&args[1..]),
        #[cfg(feature = "m1")]
        "m1-session" => m1::session_test(&args[1..]),
        #[cfg(feature = "m1")]
        "m2a-demo" => m1::m2a_demo(&args[1..]),
        #[cfg(feature = "m1")]
        "m1-kv" => kv::run(&args[1..]),
        #[cfg(feature = "m1")]
        "m1-spec" => spec::run(&args[1..]),
        "m1-spec-batch" => spec::run_batch(&args[1..]),
        #[cfg(feature = "m1")]
        "m1-batch" => batch::run(&args[1..]),
        #[cfg(feature = "m1")]
        "serve" => serve::run(&args[1..]),
        #[cfg(feature = "m1")]
        "m1-plan" => plan::dry_run(&args[1..]),
        "devices" => {
            let n = agens_cuda::device_count()?;
            println!("{n} CUDA device(s){}", if agens_cuda::available() { "" } else { " (built without cuda)" });
            for d in 0..n {
                println!("  {d}: {}", agens_cuda::device_name(d)?);
            }
            Ok(())
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2)
        }
    }
}

fn category(name: &str) -> &'static str {
    if name.contains(".engram.tables.") {
        "engram tables"
    } else if name.contains("embed_tokens") {
        "embeddings"
    } else if name.starts_with("lm_head") {
        "lm_head"
    } else if name.contains(".linear_attn.") {
        "kda"
    } else if name.contains(".self_attn.idx_") {
        "bcsa indexer"
    } else if name.contains(".self_attn.") {
        "attention"
    } else if name.contains(".mlp.") {
        "mlp"
    } else if name.contains(".mhc.") {
        "mhc"
    } else {
        "other"
    }
}

fn inspect(ck: &Checkpoint) -> Result<()> {
    let c = &ck.config;
    println!(
        "Volundr: {} layers = {} KDA + {} BCSA + {} dense; hidden {}, ffn {}, vocab {}",
        c.n_layers,
        c.count(Mixer::Kda),
        c.count(Mixer::Bcsa),
        c.count(Mixer::Dense),
        c.hidden,
        c.intermediate,
        c.vocab
    );
    let map: String = c
        .layers
        .iter()
        .map(|l| match (l.mixer, l.engram) {
            (_, true) => 'E',
            (Mixer::Kda, _) => 'k',
            (Mixer::Bcsa, _) => 'B',
            (Mixer::Dense, _) => 'D',
        })
        .collect();
    println!("layer map (k=KDA B=BCSA D=dense E=KDA+Engram): {map}");
    println!(
        "per request: KDA state {:.1} MB fp32; KV {} B/token bf16, {} B/token fp8",
        c.kda_state_bytes() as f64 / 1e6,
        c.kv_bytes_per_token(2),
        c.kv_bytes_per_token(1)
    );
    let mut by: std::collections::BTreeMap<&str, usize> = Default::default();
    for (name, &i) in &ck.index {
        if name.starts_with("model.visual.") {
            continue;
        }
        *by.entry(category(name)).or_default() += ck.shards[i].tensors[name].nbytes();
    }
    let total: usize = by.values().sum();
    println!("on-disk bytes by part:");
    for (k, v) in &by {
        println!("  {k:<14} {:>8.2} GB", *v as f64 / 1e9);
    }
    let per_step = total - by.get("engram tables").unwrap_or(&0) - by.get("embeddings").unwrap_or(&0);
    println!(
        "  total {:.2} GB; read per decode step (excl. Engram tables + embeddings) {:.2} GB",
        total as f64 / 1e9,
        per_step as f64 / 1e9
    );
    let lin = manifest(c).values().filter(|e| matches!(e, Expect::Linear { .. })).count();
    println!("linear layers in manifest: {lin}");
    Ok(())
}

fn load(ck: &Checkpoint, device: usize, engram_gpu: bool) -> Result<()> {
    ck.validate()?;
    agens_cuda::set_device(device)?;
    println!("device {device}: {}", agens_cuda::device_name(device)?);
    let (free0, _) = agens_cuda::mem_info()?;
    let t = Instant::now();
    let mut bufs = Vec::new();
    let mut bytes = 0usize;
    for s in &ck.shards {
        s.advise_sequential();
        for name in s.tensors.keys() {
            if name.starts_with("model.visual.") || (!engram_gpu && name.contains(".engram.tables.")) {
                continue;
            }
            let b = s.bytes(name)?;
            bytes += b.len();
            bufs.push((name.clone(), agens_cuda::upload(b)?));
        }
    }
    agens_cuda::synchronize()?;
    let dt = t.elapsed().as_secs_f64();
    let (free1, _) = agens_cuda::mem_info()?;
    println!(
        "uploaded {} tensors, {:.2} GB in {:.1} s ({:.2} GB/s); device memory used {:.2} GiB",
        bufs.len(),
        bytes as f64 / 1e9,
        dt,
        bytes as f64 / 1e9 / dt,
        (free0 - free1) as f64 / (1u64 << 30) as f64
    );
    Ok(())
}
