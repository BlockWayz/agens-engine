//! Checkpoint = release folder (config.json + model.safetensors.index.json + shards + SHA256SUMS).
//! Builds the expected tensor manifest from the config, checks every tensor's dtype/shape, and verifies shard
//! checksums. Linear weights may be bf16 or compressed-tensors pack-quantized W4 with group 128 (the INT4 release
//! format) or 64 (the 3-bit probe's grid in the same container); the group size is read per linear from its scales.

use crate::config::{Mixer, VolundrConfig};
use crate::safetensors::{Dtype, SafeTensorsFile, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Group size of the INT4 release; the engine also runs 64 (W4_GROUPS).
pub const W4_GROUP: usize = 128;
pub const W4_GROUPS: [usize; 2] = [64, 128];

/// How one linear layer's weight is stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearFormat {
    Bf16,
    /// `weight_packed` I32 [out, in/8] (8 nibbles, low first, value = q + 8), `weight_scale` BF16 [out, in/group].
    W4 { group: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect {
    /// Plain tensor with exact shape (bf16 on disk).
    Tensor(Vec<usize>),
    /// Linear `[out, in]`; format detected from the files.
    Linear { out: usize, inp: usize },
}

/// Expected text-model tensors (vision tower and MTP excluded), keyed by base name (linear: without `.weight`).
pub fn manifest(c: &VolundrConfig) -> BTreeMap<String, Expect> {
    use Expect::*;
    let mut m = BTreeMap::new();
    let h = c.hidden;
    let lm = "model.language_model";
    let lin = |m: &mut BTreeMap<String, Expect>, name: String, out: usize, inp: usize| {
        m.insert(name, Linear { out, inp });
    };
    m.insert(format!("{lm}.embed_tokens.weight"), Tensor(vec![c.vocab, h]));
    m.insert(format!("{lm}.norm.weight"), Tensor(vec![h]));
    lin(&mut m, "lm_head".into(), c.vocab, h);
    let n = c.mhc_streams;
    let k = &c.kda;
    for l in &c.layers {
        let p = format!("{lm}.layers.{}", l.idx);
        if n > 1 {
            let q = format!("{lm}.mhc.{}", l.idx);
            m.insert(format!("{q}.res_static"), Tensor(vec![n, n]));
            m.insert(format!("{q}.pre_static"), Tensor(vec![n]));
            m.insert(format!("{q}.post_static"), Tensor(vec![n]));
            m.insert(format!("{q}.norm.weight"), Tensor(vec![h]));
            lin(&mut m, format!("{q}.dyn"), 2 * n + n * n, h);
        }
        m.insert(format!("{p}.input_layernorm.weight"), Tensor(vec![h]));
        m.insert(format!("{p}.post_attention_layernorm.weight"), Tensor(vec![h]));
        lin(&mut m, format!("{p}.mlp.gate_proj"), c.intermediate, h);
        lin(&mut m, format!("{p}.mlp.up_proj"), c.intermediate, h);
        lin(&mut m, format!("{p}.mlp.down_proj"), h, c.intermediate);
        match l.mixer {
            Mixer::Kda => {
                let a = format!("{p}.linear_attn");
                lin(&mut m, format!("{a}.in_proj_qkv"), k.conv_dim(), h);
                lin(&mut m, format!("{a}.in_proj_z"), k.value_dim(), h);
                lin(&mut m, format!("{a}.in_proj_b"), k.n_v_heads, h);
                lin(&mut m, format!("{a}.f_down"), k.gate_rank, h);
                lin(&mut m, format!("{a}.f_up"), k.n_v_heads * k.k_dim, k.gate_rank);
                lin(&mut m, format!("{a}.out_proj"), h, k.value_dim());
                m.insert(format!("{a}.conv1d.weight"), Tensor(vec![k.conv_dim(), 1, k.conv_kernel]));
                m.insert(format!("{a}.A_log"), Tensor(vec![k.n_v_heads]));
                m.insert(format!("{a}.dt_bias"), Tensor(vec![k.n_v_heads * k.k_dim]));
                m.insert(format!("{a}.norm.weight"), Tensor(vec![k.v_dim]));
            }
            Mixer::Bcsa | Mixer::Dense => {
                let a = format!("{p}.self_attn");
                let hd = c.head_dim;
                lin(&mut m, format!("{a}.q_proj"), c.n_heads * hd * 2, h); // query + output gate
                lin(&mut m, format!("{a}.k_proj"), c.n_kv_heads * hd, h);
                lin(&mut m, format!("{a}.v_proj"), c.n_kv_heads * hd, h);
                lin(&mut m, format!("{a}.o_proj"), h, c.n_heads * hd);
                m.insert(format!("{a}.q_norm.weight"), Tensor(vec![hd]));
                m.insert(format!("{a}.k_norm.weight"), Tensor(vec![hd]));
                if l.mixer == Mixer::Bcsa {
                    let b = &c.bcsa;
                    lin(&mut m, format!("{a}.idx_q"), b.idx_heads * b.idx_dim, h);
                    lin(&mut m, format!("{a}.idx_k"), b.idx_dim, h);
                    m.insert(format!("{a}.idx_norm.weight"), Tensor(vec![b.idx_dim]));
                }
            }
        }
        if l.engram {
            let e = format!("{p}.engram");
            let cat = c.engram.dim * c.engram.orders.len();
            for i in 0..c.engram.orders.len() {
                m.insert(format!("{e}.tables.{i}.weight"), Tensor(vec![c.engram.rows, c.engram.dim]));
            }
            m.insert(format!("{e}.norm.weight"), Tensor(vec![cat]));
            m.insert(format!("{e}.gate.weight"), Tensor(vec![1, h]));
            m.insert(format!("{e}.gate.bias"), Tensor(vec![1]));
            lin(&mut m, format!("{e}.value"), h, cat);
        }
    }
    m
}

fn is_text_model(name: &str) -> bool {
    !name.starts_with("model.visual.") && !name.contains(".mtp.") && !name.starts_with("mtp.")
}

pub struct Checkpoint {
    pub dir: PathBuf,
    pub config: VolundrConfig,
    pub shards: Vec<SafeTensorsFile>,
    /// tensor name -> shard index
    pub index: BTreeMap<String, usize>,
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub tensors: usize,
    pub bytes: usize,
    pub bf16_linears: usize,
    pub w4_linears: usize,
    /// W4 linears per group size
    pub w4_groups: BTreeMap<usize, usize>,
    pub unexpected: Vec<String>,
}

impl Checkpoint {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let config = VolundrConfig::from_file(dir.join("config.json"))?;
        let mut names: Vec<String> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".safetensors"))
            .collect();
        names.sort();
        ensure!(!names.is_empty(), "no .safetensors files in {}", dir.display());
        let shards = names.iter().map(|n| SafeTensorsFile::open(dir.join(n))).collect::<Result<Vec<_>>>()?;
        let mut index = BTreeMap::new();
        for (i, s) in shards.iter().enumerate() {
            for name in s.tensors.keys() {
                if let Some(prev) = index.insert(name.clone(), i) {
                    bail!("tensor {name} in both {} and {}", names[prev], names[i]);
                }
            }
        }
        // Cross-check the HF index if present.
        let idx_path = dir.join("model.safetensors.index.json");
        if idx_path.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx_path)?)?;
            let wm = v["weight_map"].as_object().context("weight_map")?;
            for (t, f) in wm {
                let f = f.as_str().context("weight_map value")?;
                let i = *index.get(t).with_context(|| format!("index lists {t} but no shard has it"))?;
                ensure!(names[i] == f, "index puts {t} in {f}, found in {}", names[i]);
            }
            ensure!(wm.len() == index.len(), "index has {} tensors, shards have {}", wm.len(), index.len());
        }
        Ok(Checkpoint { dir, config, shards, index })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.shards[i].tensors[name])
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = *self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        self.shards[i].bytes(name)
    }

    pub fn linear_format(&self, base: &str) -> Option<LinearFormat> {
        if self.index.contains_key(&format!("{base}.weight")) {
            Some(LinearFormat::Bf16)
        } else if let Some(p) = self.info(&format!("{base}.weight_packed")) {
            // group = in / scale columns (0 when the scales are missing or malformed: validate reports it)
            let inp = p.shape.get(1).copied().unwrap_or(0) * 8;
            let cols = self.info(&format!("{base}.weight_scale")).and_then(|s| s.shape.get(1).copied()).unwrap_or(0);
            Some(LinearFormat::W4 { group: if cols > 0 && inp % cols == 0 { inp / cols } else { 0 } })
        } else {
            None
        }
    }

    /// Check every manifest entry against the files. Errors list all problems, not just the first.
    pub fn validate(&self) -> Result<ValidationReport> {
        let man = manifest(&self.config);
        let mut errs = Vec::new();
        let mut seen = BTreeSet::new();
        let mut rep = ValidationReport::default();
        let check = |name: &str, dtype: Dtype, shape: &[usize], errs: &mut Vec<String>, seen: &mut BTreeSet<String>| {
            match self.info(name) {
                None => errs.push(format!("missing {name}")),
                Some(t) => {
                    seen.insert(name.to_string());
                    if t.dtype != dtype || t.shape != shape {
                        errs.push(format!("{name}: {:?} {:?}, expected {:?} {:?}", t.dtype, t.shape, dtype, shape));
                    }
                }
            }
        };
        for (name, e) in &man {
            match e {
                Expect::Tensor(shape) => check(name, Dtype::BF16, shape, &mut errs, &mut seen),
                Expect::Linear { out, inp } => match self.linear_format(name) {
                    Some(LinearFormat::Bf16) => {
                        rep.bf16_linears += 1;
                        check(&format!("{name}.weight"), Dtype::BF16, &[*out, *inp], &mut errs, &mut seen)
                    }
                    Some(LinearFormat::W4 { group }) => {
                        rep.w4_linears += 1;
                        *rep.w4_groups.entry(group).or_default() += 1;
                        if !W4_GROUPS.contains(&group) || inp % group != 0 {
                            errs.push(format!("{name}: group {group} (in={inp}) not one of {W4_GROUPS:?}"));
                        }
                        let g = group.max(1);
                        check(&format!("{name}.weight_packed"), Dtype::I32, &[*out, inp / 8], &mut errs, &mut seen);
                        check(&format!("{name}.weight_scale"), Dtype::BF16, &[*out, inp / g], &mut errs, &mut seen);
                        let shp = format!("{name}.weight_shape");
                        if self.info(&shp).is_some() {
                            seen.insert(shp);
                        }
                    }
                    None => errs.push(format!("missing linear {name}")),
                },
            }
        }
        for (name, &i) in &self.index {
            if !is_text_model(name) {
                continue;
            }
            if !seen.contains(name) {
                rep.unexpected.push(name.clone());
            } else {
                rep.tensors += 1;
                rep.bytes += self.shards[i].tensors[name].nbytes();
            }
        }
        if !rep.unexpected.is_empty() {
            errs.push(format!("{} unexpected text-model tensors, e.g. {}", rep.unexpected.len(), rep.unexpected[0]));
        }
        if !errs.is_empty() {
            let shown: Vec<_> = errs.iter().take(20).cloned().collect();
            bail!("{} problems:\n  {}", errs.len(), shown.join("\n  "));
        }
        Ok(rep)
    }

    /// Verify shards against `SHA256SUMS` (sha256sum format). Hashes shards in parallel threads.
    pub fn verify_checksums(&self, threads: usize) -> Result<Vec<(String, String)>> {
        verify_sha256sums(&self.dir, threads)
    }
}

pub fn parse_sha256sums(text: &str) -> Result<Vec<(String, String)>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (h, f) = l.split_once(char::is_whitespace).with_context(|| format!("bad SHA256SUMS line {l:?}"))?;
            let f = f.trim_start().trim_start_matches('*');
            ensure!(h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()), "bad hash in {l:?}");
            Ok((f.to_string(), h.to_ascii_lowercase()))
        })
        .collect()
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Returns (file, hash) for every verified entry; errors on any mismatch or missing file.
pub fn verify_sha256sums(dir: &Path, threads: usize) -> Result<Vec<(String, String)>> {
    let sums = parse_sha256sums(&std::fs::read_to_string(dir.join("SHA256SUMS")).context("read SHA256SUMS")?)?;
    let threads = threads.max(1);
    let results: Vec<Result<(String, String)>> = std::thread::scope(|s| {
        let chunks: Vec<_> = sums.chunks(sums.len().div_ceil(threads).max(1)).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|(f, want)| {
                            let got = sha256_file(&dir.join(f))?;
                            ensure!(&got == want, "{f}: sha256 {got} != {want}");
                            Ok((f.clone(), got))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VolundrConfig;
    use crate::safetensors::serialize;

    /// A 4-layer toy Volundr (kda, kda+engram, bcsa, attn) with tiny dims (hidden 128 so W4 g128 applies).
    pub(crate) fn toy_config_json() -> String {
        serde_json::json!({"text_config": {
            "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 10, "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 1,
            "linear_num_value_heads": 2, "linear_value_head_dim": 4, "kda_gate_rank": 2,
            "mixer_types": ["kda", "kda", "bcsa", "attn"], "bcsa_window": 8, "bcsa_compress": 4, "bcsa_topk": 2,
            "engram_layers": [1], "engram_orders": [2, 3], "engram_rows": 8, "engram_dim": 4, "mhc_streams": 4,
            "mhc_sinkhorn_iters": 10, "eos_token_id": 9,
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25}
        }})
        .to_string()
    }

    fn write_toy(dir: &Path, drop: Option<&str>, w4: bool) -> Result<()> {
        write_toy_g(dir, drop, w4, W4_GROUP)
    }

    fn write_toy_g(dir: &Path, drop: Option<&str>, w4: bool, group: usize) -> Result<()> {
        let c = VolundrConfig::from_json(&toy_config_json())?;
        let man = manifest(&c);
        let mut tensors = Vec::new();
        for (name, e) in &man {
            if Some(name.as_str()) == drop {
                continue;
            }
            match e {
                Expect::Tensor(s) => tensors.push((name.clone(), Dtype::BF16, s.clone())),
                Expect::Linear { out, inp } if w4 && inp % group == 0 => {
                    tensors.push((format!("{name}.weight_packed"), Dtype::I32, vec![*out, inp / 8]));
                    tensors.push((format!("{name}.weight_scale"), Dtype::BF16, vec![*out, inp / group]));
                }
                Expect::Linear { out, inp } => tensors.push((format!("{name}.weight"), Dtype::BF16, vec![*out, *inp])),
            }
        }
        tensors.push(("model.visual.blocks.0.attn.qkv.weight".into(), Dtype::BF16, vec![2, 2]));
        let (a, b) = tensors.split_at(tensors.len() / 2);
        let mut sums = String::new();
        for (i, part) in [a, b].iter().enumerate() {
            let ser: Vec<_> = part
                .iter()
                .map(|(n, d, s)| (n.as_str(), *d, s.clone(), vec![i as u8; s.iter().product::<usize>() * d.size()]))
                .collect();
            let bytes = serialize(&ser);
            let f = format!("model-{i:05}.safetensors");
            std::fs::write(dir.join(&f), &bytes)?;
            let h: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            sums.push_str(&format!("{h}  {f}\n"));
        }
        std::fs::write(dir.join("SHA256SUMS"), sums)?;
        std::fs::write(dir.join("config.json"), toy_config_json())?;
        Ok(())
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("agens-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn production_manifest_size() {
        let c = VolundrConfig::from_json(&crate::config::tests::volundr_json()).unwrap();
        let man = manifest(&c);
        // Count from the Preview release's text-model tensors (vision excluded): see refs/ header dump.
        let linears = man.values().filter(|e| matches!(e, Expect::Linear { .. })).count();
        // per layer: mhc.dyn + 3 mlp; kda: 6; attn: 4 (+2 indexer on bcsa); engram value x2; lm_head.
        assert_eq!(linears, 72 * 4 + 54 * 6 + 18 * 4 + 17 * 2 + 2 + 1);
        // Parameter count of the text model, bf16 (brief: 32.7B incl. vision; Engram 2.15B).
        let params: usize = man
            .values()
            .map(|e| match e {
                Expect::Tensor(s) => s.iter().product::<usize>(),
                Expect::Linear { out, inp } => out * inp,
            })
            .sum();
        assert!((32_000_000_000..33_000_000_000).contains(&params), "params = {params}");
    }

    #[test]
    fn toy_checkpoint_validates_and_verifies() {
        let d = tmpdir("ok");
        write_toy(&d, None, false).unwrap();
        let ck = Checkpoint::open(&d).unwrap();
        let rep = ck.validate().unwrap();
        assert!(rep.unexpected.is_empty());
        assert_eq!(rep.w4_linears, 0);
        assert_eq!(ck.verify_checksums(2).unwrap().len(), 2);
    }

    #[test]
    fn detects_missing_tensor_and_bad_hash() {
        let d = tmpdir("bad");
        write_toy(&d, Some("model.language_model.layers.1.linear_attn.A_log"), false).unwrap();
        let ck = Checkpoint::open(&d).unwrap();
        let err = ck.validate().unwrap_err().to_string();
        assert!(err.contains("missing model.language_model.layers.1.linear_attn.A_log"), "{err}");
        let f = d.join("model-00001.safetensors");
        let mut b = std::fs::read(&f).unwrap();
        let last = b.len() - 1;
        b[last] ^= 1;
        std::fs::write(&f, b).unwrap();
        assert!(ck.verify_checksums(1).is_err());
    }

    #[test]
    fn w4_linears_detected() {
        let d = tmpdir("w4");
        write_toy(&d, None, true).unwrap();
        let ck = Checkpoint::open(&d).unwrap();
        let rep = ck.validate().unwrap();
        assert!(rep.w4_linears > 0 && rep.bf16_linears > 0);
        assert_eq!(ck.linear_format("model.language_model.layers.2.self_attn.idx_q"), Some(LinearFormat::W4 { group: 128 }));
        assert_eq!(rep.w4_groups.get(&128), Some(&rep.w4_linears));
    }

    #[test]
    fn w4_group64_detected() {
        let d = tmpdir("w4g64");
        write_toy_g(&d, None, true, 64).unwrap();
        let ck = Checkpoint::open(&d).unwrap();
        let rep = ck.validate().unwrap();
        assert!(rep.w4_linears > 0);
        assert_eq!(rep.w4_groups.get(&64), Some(&rep.w4_linears));
        assert_eq!(ck.linear_format("model.language_model.layers.0.mlp.down_proj"), Some(LinearFormat::W4 { group: 64 }));
    }

    #[test]
    fn sha256sums_parser() {
        let s = format!("{}  a.safetensors\n{} *b.bin\n", "a".repeat(64), "B".repeat(64));
        let v = parse_sha256sums(&s).unwrap();
        assert_eq!(v[1], ("b.bin".to_string(), "b".repeat(64)));
        assert!(parse_sha256sums("xyz  f").is_err());
    }
}
