//! `agens m1-spec`: speculative decoding in the M1 engine (one device, greedy). The target runs batch-invariant
//! (decode and the 8-token verify give each token the same bits), so greedy with speculation must reproduce greedy
//! without it token for token: run `agens m1` with AGENS_M1_BI=1 for the reference.
//! Drafts come from `--oracle ref.json` (the reference continuation itself, every step's chain corrupted at one
//! position unless the step index is a multiple of --corrupt: exercises every accept length and the rollback) or,
//! with `--drafter DIR`, from the DFlash2 drafter.

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::m1::EOT;

pub fn run(args: &[String]) -> Result<()> {
    let usage = "agens m1-spec <ckpt_dir> <prompts.json> <out.json> [--oracle ref.json --corrupt K | --drafter DIR | --mtp DIR] \
                 [--temperature T --seed S] [m1 flags]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let out_path = args.get(2).context(usage)?;
    let (mut oracle, mut corrupt, mut drafter, mut rest) = (None, 3usize, None, Vec::new());
    let (mut temperature, mut seed) = (0f32, 1i32);
    let (mut top_k, mut top_p, mut min_p) = (0i32, 1f32, 0f32); // sampled verify filters (samp_pos), with --temperature
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?.clone();
        match args[i].as_str() {
            "--oracle" => oracle = Some(v),
            "--corrupt" => corrupt = v.parse()?,
            "--drafter" => drafter = Some(v),
            "--temperature" => temperature = v.parse()?,
            "--seed" => seed = v.parse()?,
            "--top-k" => top_k = v.parse()?,
            "--top-p" => top_p = v.parse()?,
            "--min-p" => min_p = v.parse()?,
            _ => {
                rest.push(args[i].clone());
                rest.push(v);
            }
        }
        i += 2;
    }
    let mut o = crate::m1::parse_opts(&rest, usage)?;
    o.drafter = drafter.clone();
    // draft sources: the DFlash2 drafter and / or the MTP head (hybrid drafting with copy-from-context), or the oracle
    let drafting = drafter.is_some() || o.mtp.is_some();
    let copy = o.copy && !drafting; // --spec-copy 1 alone: copy-only drafting (greedy)
    if copy && temperature > 0.0 {
        bail!("--spec-copy without a drafter is greedy only");
    }
    if oracle.is_some() == (drafting || copy) {
        bail!("{usage}: pass --oracle, or --drafter and / or --mtp, or --spec-copy 1");
    }
    crate::m1::load(dir, &o)?;
    if !drafting && !copy {
        agens_cuda::m1::spec_init(true, None, 0)?;
    }
    agens_cuda::m1::spec_sampling_filtered(temperature, top_k, top_p, min_p, seed)?;
    crate::tp::start()?; // tensor parallel: followers execute the leader's calls from here
    // reference tokens (oracle drafts; with the drafter: optional comparison, AGENS_SPEC_REF=ref.json)
    let ref_path = oracle.clone().or_else(|| std::env::var("AGENS_SPEC_REF").ok());
    let refs: BTreeMap<String, Vec<i32>> = match &ref_path {
        Some(rp) => serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(rp)?)?
            .as_array()
            .context("reference json")?
            .iter()
            .map(|r| {
                let t = r["tokens"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
                (r["name"].as_str().unwrap().to_string(), t)
            })
            .collect(),
        None => BTreeMap::new(),
    };
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let base = std::path::Path::new(prompts_path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let mut results = Vec::new();
    let (mut all_steps, mut all_tok) = (0usize, 0usize);
    let mut by_class: BTreeMap<String, Vec<(usize, f64, usize, usize)>> = BTreeMap::new();  // (new, s, steps, prompt)
    // "append": true continues the session instead of starting over (long-context runs: a long document once, then
    // each case as a follow-up turn): the previous case's last token (its bonus, not yet in the KV), an end of turn
    // unless it was one, then the case's ids (which start with the newline after the end-of-turn token).
    let mut last_full: Option<i32> = None;
    let mut lk = crate::lookup::Lookup::new(); // copy-from-context proposals (hybrid drafting)
    if copy {
        lk = lk.copy_only();
    }
    let mut by_class_lk: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new(); // lookup steps, steps, acc
    for p in prompts.as_array().context("prompts must be a JSON list")? {
        let name = p["name"].as_str().unwrap_or("?").to_string();
        let class = p["class"].as_str().unwrap_or("all").to_string();
        let mut ids: Vec<i32> = p["ids"].as_array().context("ids")?.iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let append = p.get("append").and_then(|x| x.as_bool()).unwrap_or(false);
        if append {
            let l = last_full.context("\"append\" on the first prompt")?;
            let mut pre = vec![l];
            if l != EOT {
                pre.push(EOT);
            }
            pre.extend_from_slice(&ids);
            ids = pre;
        }
        let n_new = p.get("new_tokens").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(o.new_tokens);
        let reference = refs.get(&name).cloned().unwrap_or_default();
        if !append {
            agens_cuda::m1::reset()?;
        }
        agens_cuda::m1::spec_sampling_filtered(temperature, top_k, top_p, min_p, seed)?; // per prompt: the same prompt samples the same way
        let t0 = Instant::now();
        let (mut bonus, fed) = crate::mm::case_prefill(p, &ids, &base)?;  // "images": image cases
        ids = fed;
        let t_prefill = t0.elapsed().as_secs_f64();
        if std::env::var("AGENS_M1_MEM_TRACE").is_ok() {
            crate::m1::report_mem(&format!("after the prefill of {name} ({} tokens)", ids.len()));
        }
        let mut out = vec![bonus];
        if append { lk.extend(&ids) } else { lk.reset(&ids) }
        lk.extend(&[bonus]);
        let lk0 = (lk.used, lk.steps, lk.acc_lookup, lk.acc_draft);
        let (mut steps, mut hist) = (0usize, [0usize; 9]);
        let td = Instant::now();
        // AGENS_M1_PROFILE_DECODE=N: profiler capture range over verify steps 8..8+N of each prompt (nsys)
        let prof = std::env::var("AGENS_M1_PROFILE_DECODE").ok().and_then(|v| v.parse::<usize>().ok());
        while out.len() < n_new && bonus != EOT {
            if let Some(n) = prof {
                if steps == 8 || steps == 8 + n {
                    agens_cuda::m1::profiler(steps == 8)?;
                }
            }
            let emitted = if copy {
                let hint = lk.propose();
                let em = crate::lookup::copy_step(bonus, &hint)?;
                lk.record(&em);
                em
            } else if drafting {
                let hint = lk.propose();
                let em = if hint.is_empty() {
                    agens_cuda::m1::spec_step(bonus)?.0
                } else {
                    agens_cuda::m1::spec_step_hint(bonus, &hint)?.0
                };
                lk.record(&em);
                em
            } else {
                // oracle chain: the reference's next 7 tokens after the bonus, one corrupted unless step % corrupt == 0
                let mut chain = vec![bonus];
                for j in 1..8 {
                    chain.push(*reference.get(out.len() - 1 + j).unwrap_or(&0));
                }
                if corrupt > 0 && steps % corrupt != 0 {
                    let j = 1 + steps % 7;
                    chain[j] = (chain[j] + 1) % 248000;
                }
                agens_cuda::m1::verify(&chain)?.0
            };
            hist[emitted.len()] += 1;
            steps += 1;
            out.extend_from_slice(&emitted);
            bonus = *emitted.last().unwrap();
            if let Some(e) = out.iter().position(|&t| t == EOT) {
                out.truncate(e + 1);  // end of turn inside the accepted chain: stop there
                break;
            }
        }
        last_full = out.last().copied();
        let ctx = agens_cuda::m1::seq();
        out.truncate(n_new);
        let t_dec = td.elapsed().as_secs_f64();
        let same = reference.iter().zip(&out).take_while(|(a, b)| a == b).count();
        eprintln!(
            "{name}: {} prompt tokens{}, prefill {:.1} s; {} tokens in {} steps ({:.2} tok/step, {:.2} tok/s); \
             identical prefix vs reference {same}/{}; commits per step {:?}",
            ids.len(),
            if append { format!(" appended (context {ctx})") } else { String::new() },
            t_prefill,
            out.len(),
            steps,
            (out.len() - 1) as f64 / steps.max(1) as f64,
            (out.len() - 1) as f64 / t_dec.max(1e-9),
            out.len().min(reference.len()),
            &hist[1..]
        );
        all_steps += steps;
        all_tok += out.len() - 1;
        if drafting || copy {
            let e = by_class_lk.entry(class.clone()).or_default();
            e.0 += lk.used - lk0.0;
            e.1 += lk.steps - lk0.1;
            e.2 += lk.acc_lookup - lk0.2;
            e.3 += lk.acc_draft - lk0.3;
            if std::env::var("AGENS_SPEC_LOOKUP_LOG").is_ok() {
                eprintln!("  {name}: {}", lk.summary());
            }
        }
        by_class.entry(class.clone()).or_default().push((out.len() - 1, t_dec, steps, ids.len()));
        results.push(serde_json::json!({"name": name, "class": class, "n_prompt": ids.len(), "context": ctx,
                                        "prefill_s": t_prefill, "tokens": out, "steps": steps,
                                        "decode_s": t_dec, "tok_s": (out.len() - 1) as f64 / t_dec.max(1e-9)}));
        std::fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
    }
    crate::m1::report_mem("after the run");
    eprintln!("spec: {all_tok} tokens in {all_steps} verify steps ({:.2} tok/step)", all_tok as f64 / all_steps.max(1) as f64);
    // per class, as tools/spec_bench.py prints it (tok/s = mean over prompts of new tokens / decode wall; accepted per
    // verify step = mean over prompts of new tokens / steps)
    eprintln!("class        n  prompt(avg)  tok/s(mean)  accepted/step(mean)");
    for (c, v) in &by_class {
        let n = v.len() as f64;
        let tps = v.iter().map(|x| x.0 as f64 / x.1.max(1e-9)).sum::<f64>() / n;
        let acc = v.iter().map(|x| x.0 as f64 / x.2.max(1) as f64).sum::<f64>() / n;
        let pr = v.iter().map(|x| x.3 as f64).sum::<f64>() / n;
        eprintln!("{c:10} {:3} {pr:11.0} {tps:12.1} {acc:12.2}", v.len());
    }
    if !by_class_lk.is_empty() {
        eprintln!("hybrid drafting: class, lookup steps / steps, accepted drafts per step on lookup / drafter steps");
        for (c, (u, st, al, ad)) in &by_class_lk {
            eprintln!("{c:10} {u:5}/{st:<5} {:6.2} {:6.2}", *al as f64 / (*u).max(1) as f64,
                      *ad as f64 / (st - u).max(1) as f64);
        }
    }
    Ok(())
}

/// DFlash2 drafter checkpoint (sglang DFlash2DraftModel export): bf16 weights to the GPU (gate + up merged), the
/// capture layers (dflash_config.target_layer_ids + 1: sglang captures the residual stream at layer entry) and the
/// context buffers. Prints weight and buffer bytes (the 16 GB budget).
/// AGENS_DRAFT_W4=1: every linear with k % 128 == 0 and k >= 1024 (projections, MLP, conv kernel projections, fc, the
/// selector's hidden projection) is quantized at load to W4 g128 (symmetric round-to-nearest, the compressed-tensors
/// format the target uses) and runs on the engine's W4 kernels; norms, conv base kernels and the selector codebooks
/// stay bf16 (the codebooks host-mapped: a draft step gathers 23 rows). The drafter only proposes, so this moves
/// acceptance, never the output.
pub(crate) struct Drafter {
    dir: String,
    ids: [i32; 5],
    mask: i32,
    bytes: usize,
    pub(crate) w4: bool,
    secs: f64,
}

/// The drafter's weights to the GPU (before the target's memory plan, which then sees them as used).
pub(crate) fn upload_drafter(dir: &str) -> Result<Drafter> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let dc = &cfg["dflash_config"];
    let ids: Vec<i32> = dc["target_layer_ids"]
        .as_array()
        .context("target_layer_ids")?
        .iter()
        .map(|x| x.as_i64().unwrap() as i32 + 1)
        .collect();
    if ids.len() != 5 || dc["block_size"].as_u64() != Some(8) {
        bail!("drafter: expected 5 target layers and block size 8, got {ids:?} / {}", dc["block_size"]);
    }
    let mask = dc["mask_token_id"].as_i64().context("mask_token_id")? as i32;
    let mut names: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    names.sort();
    let t0 = Instant::now();
    let w4 = std::env::var("AGENS_DRAFT_W4").map(|v| v == "1").unwrap_or(false);
    let mut bytes = 0usize;
    // -> true when registered as W4
    let put_w4 = |name: &str, b: &[u8], n: usize, k: usize, bytes: &mut usize| -> Result<bool> {
        let g = agens_core::reference::w4::GROUP;
        if !w4 || k % g != 0 || k < 1024 {
            return Ok(false);
        }
        // AGENS_DRAFT_BITS (default 4): 3 or 2 = RTN to that many bits (q in [-2^(b-1), 2^(b-1) - 1], scale = amax /
        // ((2^b - 1) / 2)), packed W3 / W2 KT; the drafter only proposes, so this moves acceptance, never the output
        let bits: usize = std::env::var("AGENS_DRAFT_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let (packed, scales) = if bits == 4 { quantize_w4(b, n, k) } else { quantize_wq(b, n, k, bits) };
        let (wt, st) = agens_cuda::wq_upload_kt(bits, &packed, &scales, n, k, g)?;
        *bytes += wt.len() + st.len();
        agens_cuda::m1::draft_register_wq(name, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, g, bits)?;
        Ok(true)
    };
    for f in &names {
        let st = agens_core::safetensors::SafeTensorsFile::open(f)?;
        for (name, _) in st.tensors.iter() {
            if name.contains(".mlp.up_proj.") {
                continue;
            }
            let b = st.bytes(name)?;
            let shape = &st.tensors[name].shape;
            let linear = shape.len() == 2 && name.ends_with(".weight");
            if let Some(rest) = name.strip_suffix(".mlp.gate_proj.weight") {
                let mut m = b.to_vec();
                m.extend_from_slice(st.bytes(&format!("{rest}.mlp.up_proj.weight"))?);
                let reg = format!("{rest}.mlp.gate_up_proj.weight");
                if !put_w4(&reg, &m, 2 * shape[0], shape[1], &mut bytes)? {
                    bytes += m.len();
                    agens_cuda::m1::draft_register(&reg, agens_cuda::leak(agens_cuda::upload(&m)?))?;
                }
            } else if name.ends_with("_codebook") {
                // selector codebooks [vocab][256]: 23 rows gathered per draft step -> host-mapped (0.25 GB off the GPU)
                agens_cuda::m1::draft_register(name, agens_cuda::upload_host_mapped(b)?)?;
            } else if !(linear && put_w4(name, b, shape[0], shape[1], &mut bytes)?) {
                bytes += b.len();
                agens_cuda::m1::draft_register(name, agens_cuda::leak(agens_cuda::upload(b)?))?;
            }
        }
    }
    Ok(Drafter { dir: dir.to_string(), ids: [ids[0], ids[1], ids[2], ids[3], ids[4]], mask, bytes, w4,
                 secs: t0.elapsed().as_secs_f64() })
}

/// MTP head checkpoint (docs/mtp.md: train_mtp.py export, `mtp.*` tensor names): bf16 weights to
/// the GPU (gate + up merged) as W4 g128 RTN at load (linears with k % 128 == 0; the default), or bf16 with AGENS_MTP_W4=0.
pub(crate) struct Mtp {
    dir: String,
    bytes: usize,
    pub(crate) window: usize,
    w4: bool,
}
pub(crate) fn upload_mtp(dir: &str) -> Result<Mtp> {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let window = cfg["_volundr_mtp"]["window"].as_u64().unwrap_or(2048) as usize;
    if cfg["hidden_size"].as_u64() != Some(5120) || cfg["num_hidden_layers"].as_u64().unwrap_or(1) != 1 {
        bail!("mtp {dir}: expected hidden 5120 and one layer");
    }
    let flag = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
    // W4 by default (1.2: the measured configuration; 0.23 GB instead of 0.85); AGENS_MTP_W4=0 keeps bf16
    let w4 = std::env::var("AGENS_MTP_W4").map(|v| v != "0").unwrap_or(true) || flag("AGENS_DRAFT_W4");
    let mut names: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    names.sort();
    let mut bytes = 0usize;
    let put = |name: &str, b: &[u8], n: usize, k: usize, linear: bool, bytes: &mut usize| -> Result<()> {
        let g = agens_core::reference::w4::GROUP;
        if w4 && linear && k % g == 0 && k >= 1024 {
            let (packed, scales) = quantize_w4(b, n, k);
            let (wt, st) = agens_cuda::wq_upload_kt(4, &packed, &scales, n, k, g)?;
            *bytes += wt.len() + st.len();
            agens_cuda::m1::mtp_register_wq(name, agens_cuda::leak(wt), agens_cuda::leak(st), n, k, g, 4)
        } else {
            *bytes += b.len();
            agens_cuda::m1::mtp_register(name, agens_cuda::leak(agens_cuda::upload(b)?))
        }
    };
    let mut seen = 0;
    for f in &names {
        let st = agens_core::safetensors::SafeTensorsFile::open(f)?;
        for (full, _) in st.tensors.iter() {
            let Some(name) = full.strip_prefix("mtp.") else { continue };
            if name.contains(".mlp.up_proj.") {
                continue;
            }
            if st.tensors[full].dtype != agens_core::safetensors::Dtype::BF16 {
                bail!("mtp {full}: {:?} (bf16 expected)", st.tensors[full].dtype);
            }
            let b = st.bytes(full)?;
            let shape = st.tensors[full].shape.clone();
            seen += 1;
            if let Some(rest) = name.strip_suffix(".mlp.gate_proj.weight") {
                let mut m = b.to_vec();
                m.extend_from_slice(st.bytes(&format!("mtp.{rest}.mlp.up_proj.weight"))?);
                put(&format!("{rest}.mlp.gate_up_proj.weight"), &m, 2 * shape[0], shape[1], true, &mut bytes)?;
            } else {
                let linear = shape.len() == 2 && name.ends_with(".weight") && !name.contains("norm");
                put(name, b, shape[0], *shape.get(1).unwrap_or(&1), linear, &mut bytes)?;
            }
        }
    }
    if seen < 14 {
        bail!("mtp {dir}: {seen} mtp.* tensors (expected 15)");
    }
    Ok(Mtp { dir: dir.to_string(), bytes, window, w4 })
}
/// The MTP chain's shortlist head (1.2): its first AGENS_MTP_VOCAB (default 32768; 0 = the whole head) token ids, by
/// frequency in Volundr V1's own generations (tools/mtp_vocab.py; 6.1% of held-out output tokens fall outside the first
/// 32768, and the chain also scores the DFlash2 drafter's 16 candidates of each slot). A chain step then reads 0.17 GB
/// of the INT8 head instead of 1.27 GB.
static MTP_VOCAB: &[u8] = include_bytes!("mtp_vocab.bin");
pub(crate) fn mtp_vocab_n() -> usize {
    std::env::var("AGENS_MTP_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(32768).min(MTP_VOCAB.len() / 4)
}
pub(crate) fn finish_mtp(m: &Mtp, cap_rows: usize) -> Result<()> {
    let n = mtp_vocab_n();
    let ids: Vec<i32> = MTP_VOCAB.chunks_exact(4).take(n).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32).collect();
    agens_cuda::m1::mtp_shortlist(&ids)?;
    let own = agens_cuda::m1::mtp_setup(cap_rows, m.window)?;
    eprintln!("mtp {}: weights {:.2} GB ({}), buffers {:.2} GB (window {}, context in passes of <= 256 rows), head {}", m.dir,
              m.bytes as f64 / 1e9, if m.w4 { "W4 g128 linears + bf16" } else { "bf16" }, own as f64 / 1e9, m.window,
              if n > 0 { format!("shortlist {n} ids + the drafter's candidates (INT8 head)") } else { "whole".to_string() });
    Ok(())
}
/// Device bytes finish_mtp allocates (memory plan, kernels/m1/m1_mtp.cu setup): a KV ring + pending row per KDA slot
/// (sessions + 1), context-row buffers for a 256-row pass of a prefill chunk (or a batched verify's 8 B rows), chain buffers per
/// sequence of a batched verify (AGENS_SPEC_BATCH_MAX; 1 without), the W4 dequant tile.
pub(crate) fn mtp_reserve(cap_rows: usize, sessions: usize) -> (String, usize) {
    let ns = batch_max().max(1);
    let rows = cap_rows.clamp(8, 256).max(8 * ns);  // m1_mtp.cu kMatRows
    let hid = 5120usize;
    let per_row = 7 * hid * 2 + 2 * 1024 * 2 + 8;
    let per_seq = 2 * hid * 2 + 2 * 6144 * 2 + 3 * 6144 * 2 + 7 * 2 * 1024 * 2 + 3 * 17408 * 2 + 248320 * 2
        + 24 * 16 * (256 + 2) * 4 + 32;
    (format!("mtp: KV rings x {} + {rows}-row context passes + chain x {ns}", sessions + 1),
     (sessions + 1) * (2 * 4096 * 1024 * 2 + hid * 2 + 4) + rows * per_row + ns * per_seq + (16 << 20)
         + mtp_vocab_n() * 4 * (1 + ns) + ns * 16 * 4)
}

/// After the target's init: speculative buffers (capture of cap_rows tokens = the prefill chunk, KDA stash) and the
/// drafter's own (KV ring, rope, context buffers).
pub(crate) fn finish_drafter(d: &Drafter, cap_rows: usize) -> Result<()> {
    agens_cuda::m1::spec_init(true, Some(d.ids), cap_rows)?;
    // AGENS_SPEC_BATCH_MAX=B: batched speculative decoding of up to B sequences per step (server mode)
    let bmax = batch_max();
    if bmax > 0 {
        agens_cuda::m1::spec_batch_init(bmax)?;
    }
    let own = agens_cuda::m1::draft_setup(d.mask)?;
    eprintln!(
        "drafter {}: weights {:.2} GB ({}), buffers {:.2} GB (KV ring 4096 x 5 layers, {cap_rows}-token context, \
         logits), capture layers {:?}, loaded in {:.0} s",
        d.dir,
        d.bytes as f64 / 1e9,
        if d.w4 { "W4 g128 linears + bf16" } else { "bf16" },
        own as f64 / 1e9,
        d.ids,
        d.secs
    );
    Ok(())
}

pub(crate) fn batch_max() -> usize {
    std::env::var("AGENS_SPEC_BATCH_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(16)
}
/// Device bytes finish_drafter will allocate (for the memory plan): kernels/m1 spec_init + m1_draft.cu setup.
/// `drafter` false (copy-only drafting, MTP alone): no drafter ring / buffers and no target capture.
pub(crate) fn spec_reserve(cap_rows: usize, kda_layers: usize, w4: bool, wctx: usize, bcsa_layers: usize, drafter: bool)
                            -> Vec<(String, usize)> {
    let (hid, kdim) = (5120usize, 2 * 16 * 128 + 48 * 128);
    let nb = wctx / 4;
    let union = std::env::var("AGENS_SPEC_FAR_UNION").map(|v| v != "0").unwrap_or(true);
    // AGENS_SPEC_FAR_CACHE blocks per BCSA layer (default 2048) with TIER=2 (block means): ak_far_cache_bytes
    let cap: usize = std::env::var("AGENS_SPEC_FAR_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let means = std::env::var("AGENS_KV_TIER").map(|v| v == "2").unwrap_or(false);
    let cache = if union && means && cap > 0 { bcsa_layers * (cap * (8 + 2 * 2048) + nb * 4 + 4096) } else { 0 };
    let mut v = vec![
        (format!("spec: far-branch union cache ({cap} blocks x {bcsa_layers} layers)"), cache),
        // ak_far_decode_multi_ws_bytes(wctx, 8): block keys + marks + union block means + one query's scratch
        ("spec: far-branch union workspace".to_string(),
         if union { nb * (128 * 2 + 1 + 4 + 8 * (4 * 2 + 2)) + 2 * 8 * 512 * 8 + 2 * 8 * 512 * 2048 + 2 * 512 * 2048 + (1 << 20) }
         else { 0 }),
        (format!("spec: target capture ({cap_rows} x 5 x 5120)"), cap_rows * 5 * hid * 2),
        ("spec: KDA verify stash".to_string(), kda_layers * 8 * (2 * kdim + 48 * 128 + 48) * 2),
        ("drafter: KV ring + rope".to_string(), 2 * 5 * 4096 * 1024 * 2
            + if std::env::var("AGENS_ROPE_CACHE").map(|v| v == "1").unwrap_or(false) {
                ((wctx + 4095) & !4095).max(65536) * 128 * 4 } else { 0 }),
        (format!("drafter: context + block buffers"), cap_rows * (hid + 2048) * 2 + 8 * 248320 * 2 + (8 << 20)),
        ("drafter: W4 dequant scratch".to_string(), if w4 { 16 << 20 } else { 0 }),
        // batched speculation (AGENS_SPEC_BATCH_MAX = B): KDA stash for 8 B rows; drafter rings, one per KDA slot (the
        // plan's sessions + 1 is approximated by 2 B + 1) and 8 B-row buffers; the two batched attention slots
        ("spec: batched KDA stash".to_string(), kda_layers * 8 * batch_max() * (2 * kdim + 48 * 128 + 48) * 2),
        ("drafter: batched rings + buffers".to_string(), if batch_max() > 0 {
            2 * batch_max() * 2 * 5 * 4096 * 1024 * 2 + 8 * batch_max() * (248320 * 2 + 2 * 17408 * 2 + 32 * 9 * 128 * 4)
        } else { 0 }),
        // the batched verify's logits ([8 B][VOCAB] bf16) beyond the prefill chunk's scratch (1024 x 24 x 512 bf16)
        ("spec: batched verify logits".to_string(), (8 * batch_max() * 248320 * 2).saturating_sub(1024 * 24 * 512 * 2)),
        ("attention: batched verify slots".to_string(), if batch_max() > 0 {
            let r = 8 * batch_max().max(2);
            r * (4104 + wctx) * 4 + 2 * r * 66 * (24 * 256 * 2 + 24 * 4)
        } else { 0 }),
    ];
    if !drafter {
        v.retain(|(n, _)| !n.starts_with("drafter") && !n.starts_with("spec: target capture"));
    }
    v
}

/// bf16 [n][k] -> W4 g128 in the compressed-tensors layout (packed I32 [n][k/8], scales BF16 [n][k/128]) by
/// symmetric round-to-nearest (agens_core::reference::w4::quantize_row), rows split over the host's threads.
/// bf16 [n][k] -> `bits`-bit values (2 or 3) g128 in the int4 container (nibble q + 8): symmetric RTN, q in
/// [-2^(bits-1), 2^(bits-1) - 1], scale = bf16(amax / ((2^bits - 1) / 2)) (the GSQ sglang-eval convention).
fn quantize_wq(b: &[u8], n: usize, k: usize, bits: usize) -> (Vec<u8>, Vec<u8>) {
    let g = agens_core::reference::w4::GROUP;
    let (lo, hi, div) = (-(1i32 << (bits - 1)) as f32, ((1i32 << (bits - 1)) - 1) as f32, ((1u32 << bits) - 1) as f32 / 2.0);
    let (mut packed, mut scales) = (vec![0u8; n * k / 2], vec![0u8; n * k / g * 2]);
    for r in 0..n {
        let row = &b[r * k * 2..(r + 1) * k * 2];
        let w: Vec<f32> = row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        for (gi, grp) in w.chunks_exact(g).enumerate() {
            let amax = grp.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let sb = (if amax == 0.0 { 1.0 } else { amax / div }).to_bits();
            let sb16 = ((sb + 0x7FFF + ((sb >> 16) & 1)) >> 16) as u16;  // bf16 round to nearest even
            let sf = f32::from_bits((sb16 as u32) << 16);
            scales[(r * k / g + gi) * 2..][..2].copy_from_slice(&sb16.to_le_bytes());
            for (j, &x) in grp.iter().enumerate() {
                let q = (x / sf).round().clamp(lo, hi) as i32;
                let i = r * k + gi * g + j;  // nibble i of the row-major container
                packed[i / 2] |= (((q + 8) as u8) & 0xF) << (4 * (i % 2));
            }
        }
    }
    (packed, scales)
}

pub(crate) fn quantize_w4(b: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    use agens_core::reference::w4;
    let (mut packed, mut scales) = (vec![0u8; n * k / 2], vec![0u8; n * k / w4::GROUP * 2]);
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8).min(64);
    let rows = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for ((src, pk), sl) in b
            .chunks(rows * k * 2)
            .zip(packed.chunks_mut(rows * k / 2))
            .zip(scales.chunks_mut(rows * k / w4::GROUP * 2))
        {
            sc.spawn(move || {
                for (r, row) in src.chunks_exact(k * 2).enumerate() {
                    let w: Vec<f32> =
                        row.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
                    let (q, s) = w4::quantize_row(&w);
                    for (j, v) in w4::pack_row(&q).iter().enumerate() {
                        pk[(r * k / 8 + j) * 4..][..4].copy_from_slice(&v.to_le_bytes());
                    }
                    for (j, v) in s.iter().enumerate() {
                        sl[(r * k / w4::GROUP + j) * 2..][..2].copy_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
            });
        }
    });
    (packed, scales)
}

/// `agens m1-spec-batch`: batched speculative decoding gate + bench. Prompts are taken B at a time; each gets its own
/// session (prefilled alone), then all B decode together with ak_m1_spec_batch until each has its new tokens or ends
/// its turn (a finished sequence leaves the batch). Every sequence must equal its own non-spec BI greedy (`agens m1`
/// with AGENS_M1_BI=1: compare with tools/compare_tokens.py). Prints per class the per-sequence tok/s and acceptance,
/// and the aggregate tok/s of each batch.
pub fn run_batch(args: &[String]) -> Result<()> {
    let usage = "agens m1-spec-batch <ckpt_dir> <prompts.json> <out.json> --drafter DIR and / or --mtp DIR [--batch B] \
                 [m1 flags]";
    let dir = args.first().context(usage)?;
    let prompts_path = args.get(1).context(usage)?;
    let out_path = args.get(2).context(usage)?;
    let (mut drafter, mut bsz, mut rest) = (None, 4usize, Vec::new());
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context(usage)?.clone();
        match args[i].as_str() {
            "--drafter" => drafter = Some(v),
            "--batch" => bsz = v.parse()?,
            _ => {
                rest.push(args[i].clone());
                rest.push(v);
            }
        }
        i += 2;
    }
    if std::env::var("AGENS_SPEC_BATCH_MAX").is_err() {
        std::env::set_var("AGENS_SPEC_BATCH_MAX", bsz.to_string());
    }
    if !rest.iter().any(|a| a == "--sessions") {
        rest.push("--sessions".into());
        rest.push((bsz + 1).to_string());
    }
    let mut o = crate::m1::parse_opts(&rest, usage)?;
    if drafter.is_none() && o.mtp.is_none() {
        bail!("{usage}");
    }
    o.drafter = drafter;
    crate::m1::load(dir, &o)?;
    agens_cuda::m1::spec_sampling(0.0, 1)?;
    crate::tp::start()?;
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts_path)?)?;
    let base = std::path::Path::new(prompts_path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let prompts = prompts.as_array().context("prompts must be a JSON list")?;
    let mut results = vec![serde_json::Value::Null; prompts.len()];
    let mut by_class: BTreeMap<String, Vec<(usize, f64, usize)>> = BTreeMap::new();
    let (mut agg_tok, mut agg_s) = (0usize, 0f64);
    for chunk in (0..prompts.len()).collect::<Vec<_>>().chunks(bsz) {
        // sessions + prefill, one prompt at a time
        let mut sess = Vec::new();
        let (mut out, mut bonus, mut steps, mut done, mut t_end): (Vec<Vec<i32>>, Vec<i32>, Vec<usize>, Vec<bool>, Vec<f64>) =
            (vec![], vec![], vec![], vec![], vec![]);
        let mut lks: Vec<crate::lookup::Lookup> = Vec::new();
        for &pi in chunk {
            let p = &prompts[pi];
            let ids: Vec<i32> = p["ids"].as_array().context("ids")?.iter().map(|x| x.as_i64().unwrap() as i32).collect();
            let id = agens_cuda::m1::session_new()?;
            agens_cuda::m1::session_use(id)?;
            let (b, ids) = crate::mm::case_prefill(p, &ids, &base)?;
            sess.push(id);
            let mut l = crate::lookup::Lookup::new();
            l.reset(&ids);
            l.extend(&[b]);
            lks.push(l);
            out.push(vec![b]);
            bonus.push(b);
            steps.push(0);
            done.push(b == EOT);
            t_end.push(0.0);
        }
        let n_new: Vec<usize> = chunk.iter().map(|&pi| prompts[pi].get("new_tokens").and_then(|x| x.as_u64())
                                                 .map(|x| x as usize).unwrap_or(o.new_tokens)).collect();
        for k in 0..chunk.len() { done[k] |= out[k].len() >= n_new[k]; }
        let td = Instant::now();
        let prof = std::env::var("AGENS_M1_PROFILE_DECODE").ok().and_then(|v| v.parse::<usize>().ok());
        let mut bsteps = 0usize;
        loop {
            let act: Vec<usize> = (0..chunk.len()).filter(|&k| !done[k]).collect();
            if act.is_empty() { break; }
            if let Some(n) = prof {  // nsys capture range over batched steps 8 .. 8 + n of each batch
                if bsteps == 8 || bsteps == 8 + n {
                    agens_cuda::m1::profiler(bsteps == 8)?;
                }
            }
            bsteps += 1;
            let ss: Vec<i32> = act.iter().map(|&k| sess[k]).collect();
            let bb: Vec<i32> = act.iter().map(|&k| bonus[k]).collect();
            let hints: Vec<Vec<i32>> = act.iter().map(|&k| lks[k].propose()).collect();
            let em = if hints.iter().all(|h| h.is_empty()) {
                agens_cuda::m1::spec_batch(&ss, &bb)?
            } else {
                agens_cuda::m1::spec_batch_hint(&ss, &bb, &hints)?
            };
            for (j, &k) in act.iter().enumerate() {
                lks[k].record(&em[j]);
                steps[k] += 1;
                out[k].extend_from_slice(&em[j]);
                bonus[k] = *em[j].last().unwrap();
                if let Some(e) = out[k].iter().position(|&t| t == EOT) { out[k].truncate(e + 1); done[k] = true; }
                if out[k].len() >= n_new[k] { done[k] = true; }
                if done[k] { t_end[k] = td.elapsed().as_secs_f64(); }
            }
        }
        let wall = td.elapsed().as_secs_f64();
        let mut tot = 0;
        for (j, &pi) in chunk.iter().enumerate() {
            let p = &prompts[pi];
            let name = p["name"].as_str().unwrap_or("?").to_string();
            let class = p["class"].as_str().unwrap_or("all").to_string();
            out[j].truncate(n_new[j]);
            let n = out[j].len() - 1;
            tot += n;
            eprintln!("{name}: {n} tokens in {} steps ({:.2} tok/step, {:.1} tok/s in a batch of {})", steps[j],
                      n as f64 / steps[j].max(1) as f64, n as f64 / t_end[j].max(1e-9), chunk.len());
            by_class.entry(class.clone()).or_default().push((n, t_end[j], steps[j]));
            results[pi] = serde_json::json!({"name": name, "class": class, "tokens": out[j], "steps": steps[j],
                                             "decode_s": t_end[j], "batch": chunk.len()});
        }
        agens_cuda::m1::session_use(0)?;  // session 0 (never freed) is current while the batch's sessions go
        for &id in &sess {
            agens_cuda::m1::session_free(id)?;
        }
        eprintln!("batch of {}: {tot} tokens in {wall:.2} s = {:.1} tok/s aggregate", chunk.len(), tot as f64 / wall);
        agg_tok += tot;
        agg_s += wall;
        std::fs::write(out_path, serde_json::to_string_pretty(&results)?)?;
    }
    eprintln!("aggregate: {agg_tok} tokens in {agg_s:.2} s = {:.1} tok/s (batches of {bsz})", agg_tok as f64 / agg_s.max(1e-9));
    eprintln!("class        n  tok/s per sequence(mean)  accepted/step(mean)");
    for (c, v) in &by_class {
        let n = v.len() as f64;
        eprintln!("{c:10} {:3} {:12.1} {:12.2}", v.len(), v.iter().map(|x| x.0 as f64 / x.1.max(1e-9)).sum::<f64>() / n,
                  v.iter().map(|x| x.0 as f64 / x.2.max(1) as f64).sum::<f64>() / n);
    }
    Ok(())
}
