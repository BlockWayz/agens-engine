//! `agens m1-kv`: KV tiering / INT8 KV evaluation runs (docs/kv_tiering.md). The KV mode is chosen by the environment
//! (AGENS_KV_TIER=1, AGENS_KV_INT8=1), so the same command runs the bf16 all-GPU reference and the variants.
//! Per prompt: prefill, then N greedy steps (or teacher-forced steps with --force: the input at step k is the reference
//! run's token k, the prediction is recorded), with per-step decode time and GPU memory in use after prefill.
//!   --save-logits DIR  store every step's logits (bf16, the reference run)
//!   --cmp-logits DIR   compare every step's logits with a saved run: KL(ref || this), top-1 agreement
//! Long prompts: tools/make_kv_prompts.py.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::time::Instant;

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn log_softmax(l: &[u16]) -> Vec<f64> {
    let x: Vec<f64> = l.iter().map(|&b| bf16_to_f32(b) as f64).collect();
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z = x.iter().map(|v| (v - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|v| v - z).collect()
}

fn argmax(l: &[u16]) -> usize {
    let mut best = 0;
    for (i, &b) in l.iter().enumerate() {
        if bf16_to_f32(b) > bf16_to_f32(l[best]) {
            best = i;
        }
    }
    best
}

pub fn run(args: &[String]) -> Result<()> {
    use agens_cuda::m1;
    let usage = "agens m1-kv <ckpt_dir> <prompts.json> <out.json> [--new-tokens N] [--force ref.json] \
                 [--save-logits DIR | --cmp-logits DIR] [m1 flags: --devices --gpu-gib --wctx --pool --sessions]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let out_path = args.get(2).context(usage)?;
    let (mut new_tokens, mut force, mut save, mut cmp, mut rest) = (64usize, None, None, None, Vec::new());
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?.clone();
        match args[i].as_str() {
            "--new-tokens" => new_tokens = v.parse()?,
            "--force" => force = Some(v),
            "--save-logits" => save = Some(v),
            "--cmp-logits" => cmp = Some(v),
            _ => {
                rest.push(args[i].clone());
                rest.push(v);
            }
        }
        i += 2;
    }
    let o = crate::m1::parse_opts(&rest, usage)?;
    crate::m1::load(dir, &o)?;
    agens_cuda::synchronize()?;
    let (f, total) = agens_cuda::mem_info()?;
    eprintln!("m1-kv: GPU in use after load + pools (first device): {:.0} MB", (total - f) as f64 / 1e6);
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let refs: Option<serde_json::Value> = match &force {
        Some(p) => Some(serde_json::from_str(&std::fs::read_to_string(p)?)?),
        None => None,
    };
    if let Some(d) = &save {
        std::fs::create_dir_all(d)?;
    }
    let mut results = Vec::new();
    for (pi, p) in prompts.as_array().context("prompts must be a JSON list")?.iter().enumerate() {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let ids: Vec<i32> = p["ids"].as_array().context("ids")?.iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let forced: Option<Vec<i32>> = refs.as_ref().map(|r| {
            r[pi]["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
        });
        let n_new = forced.as_ref().map(|f| f.len()).unwrap_or(new_tokens);
        let mut sink = match &save {
            Some(d) => Some(std::io::BufWriter::new(std::fs::File::create(format!("{d}/{pi}.bin"))?)),
            None => None,
        };
        let mut src = match &cmp {
            Some(d) => Some(std::io::BufReader::new(std::fs::File::open(format!("{d}/{pi}.bin"))?)),
            None => None,
        };
        m1::reset()?;
        agens_cuda::synchronize()?;
        let (free0, _) = agens_cuda::mem_info()?;
        let t = Instant::now();
        let mut tok = m1::prefill(&ids)?;
        let t_prefill = t.elapsed().as_secs_f64();
        let (free1, _) = agens_cuda::mem_info()?;
        let (mut kls, mut top1, mut n_cmp) = (Vec::new(), 0usize, 0usize);
        let mut out = vec![tok];
        let mut step_ms = Vec::new();
        loop {
            if sink.is_some() || src.is_some() {
                let l = m1::logits()?;
                if let Some(w) = sink.as_mut() {
                    let bytes: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
                    w.write_all(&bytes)?;
                }
                if let Some(r) = src.as_mut() {
                    let mut bytes = vec![0u8; l.len() * 2];
                    r.read_exact(&mut bytes)?;
                    let rl: Vec<u16> = bytes.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    let (a, b) = (log_softmax(&rl), log_softmax(&l));
                    kls.push(a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum::<f64>());
                    top1 += (argmax(&rl) == argmax(&l)) as usize;
                    n_cmp += 1;
                }
            }
            if out.len() >= n_new || (forced.is_none() && tok == crate::m1::EOT) {
                break;
            }
            let input = forced.as_ref().map(|f| f[out.len() - 1]).unwrap_or(tok);
            let td = Instant::now();
            tok = m1::decode(input)?;
            step_ms.push(1e3 * td.elapsed().as_secs_f64());
            out.push(tok);
        }
        let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        let mut sorted = step_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted.get(sorted.len() / 2).cloned().unwrap_or(0.0);
        let kl_max = kls.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "{name}: {} prompt tokens, prefill {:.1} s, {} steps, decode median {:.2} ms/step, GPU +{:.0} MB after prefill{}",
            ids.len(),
            t_prefill,
            step_ms.len(),
            median,
            (free0 as f64 - free1 as f64) / 1e6,
            if n_cmp > 0 { format!(", top-1 {top1}/{n_cmp}, KL mean {:.3e} max {:.3e}", mean(&kls), kl_max) } else { String::new() }
        );
        let mut r = serde_json::json!({"name": name, "n_prompt": ids.len(), "tokens": out, "prefill_s": t_prefill,
                                       "decode_ms_median": median, "decode_ms_mean": mean(&step_ms),
                                       "gpu_mb_prefill": (free0 as f64 - free1 as f64) / 1e6,
                                       "forced": forced.is_some()});
        if n_cmp > 0 {
            r["top1"] = serde_json::json!(top1);
            r["n_cmp"] = serde_json::json!(n_cmp);
            r["kl_mean"] = serde_json::json!(mean(&kls));
            r["kl_max"] = serde_json::json!(kl_max);
            r["kl"] = serde_json::json!(kls);
        }
        results.push(r);
        std::fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
    }
    if results.is_empty() {
        bail!("no prompts");
    }
    Ok(())
}
