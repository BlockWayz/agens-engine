//! Volundr `config.json` parser and the per-layer map (mixer kind, Engram, block expansion).

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Kimi Delta Attention (linear, fp32 recurrent state).
    Kda,
    /// Blockway compressed-sparse attention: sliding window + top-k far blocks.
    Bcsa,
    /// Dense gated attention (last layer).
    Dense,
}

impl Mixer {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "kda" => Mixer::Kda,
            "bcsa" => Mixer::Bcsa,
            "attn" => Mixer::Dense,
            other => bail!("unknown mixer type {other:?}"),
        })
    }
    pub fn has_kv(self) -> bool {
        !matches!(self, Mixer::Kda)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    pub idx: usize,
    pub mixer: Mixer,
    pub engram: bool,
    pub new_block: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    kda_gate_rank: usize,
    mixer_types: Vec<String>,
    #[serde(default)]
    new_block_indices: Vec<usize>,
    bcsa_window: usize,
    bcsa_compress: usize,
    bcsa_topk: usize,
    #[serde(default)]
    engram_layers: Vec<usize>,
    engram_orders: Vec<usize>,
    engram_rows: usize,
    engram_dim: usize,
    mhc_streams: usize,
    mhc_sinkhorn_iters: usize,
    #[serde(default)]
    mtp_num_hidden_layers: usize,
    eos_token_id: u32,
    rope_parameters: RawRope,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRope {
    rope_theta: f64,
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_section: Vec<usize>,
    #[serde(default)]
    mrope_interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    text_config: RawTextConfig,
}

/// Text-model hyper-parameters the engine needs. Vision tower is out of scope (M0–M4).
#[derive(Debug, Clone)]
pub struct VolundrConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    pub kda: KdaDims,
    pub bcsa: BcsaDims,
    pub engram: EngramDims,
    pub mhc_streams: usize,
    pub mhc_sinkhorn_iters: usize,
    pub mtp_layers: usize,
    pub eos_token_id: u32,
    pub layers: Vec<LayerSpec>,
}

#[derive(Debug, Clone, Copy)]
pub struct KdaDims {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub gate_rank: usize,
}

impl KdaDims {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.k_dim
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.v_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// fp32 recurrent state per layer per request: HV x K x V.
    pub fn state_bytes(&self) -> usize {
        self.n_v_heads * self.k_dim * self.v_dim * 4
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BcsaDims {
    pub window: usize,
    pub compress: usize,
    pub topk: usize,
    /// Indexer heads/dim are not in config.json; fixed by `bcsa.py` defaults (checked against weight shapes).
    pub idx_heads: usize,
    pub idx_dim: usize,
}

#[derive(Debug, Clone)]
pub struct EngramDims {
    pub layers: Vec<usize>,
    pub orders: Vec<usize>,
    pub rows: usize,
    pub dim: usize,
}

impl VolundrConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(text).context("parse Volundr config.json")?;
        let t = raw.text_config;
        if t.mixer_types.len() != t.num_hidden_layers {
            bail!("mixer_types has {} entries, num_hidden_layers = {}", t.mixer_types.len(), t.num_hidden_layers);
        }
        let layers = t
            .mixer_types
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                Ok(LayerSpec {
                    idx,
                    mixer: Mixer::parse(m)?,
                    engram: t.engram_layers.contains(&idx),
                    new_block: t.new_block_indices.contains(&idx),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for &l in &t.engram_layers {
            if l >= t.num_hidden_layers {
                bail!("engram layer {l} out of range");
            }
        }
        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor).round() as usize;
        Ok(VolundrConfig {
            hidden: t.hidden_size,
            intermediate: t.intermediate_size,
            n_layers: t.num_hidden_layers,
            n_heads: t.num_attention_heads,
            n_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            vocab: t.vocab_size,
            rms_eps: t.rms_norm_eps,
            rope_theta: t.rope_parameters.rope_theta,
            rotary_dim,
            mrope_section: t.rope_parameters.mrope_section,
            mrope_interleaved: t.rope_parameters.mrope_interleaved,
            kda: KdaDims {
                n_k_heads: t.linear_num_key_heads,
                n_v_heads: t.linear_num_value_heads,
                k_dim: t.linear_key_head_dim,
                v_dim: t.linear_value_head_dim,
                conv_kernel: t.linear_conv_kernel_dim,
                gate_rank: t.kda_gate_rank,
            },
            bcsa: BcsaDims {
                window: t.bcsa_window,
                compress: t.bcsa_compress,
                topk: t.bcsa_topk,
                idx_heads: 4,
                idx_dim: 128,
            },
            engram: EngramDims { layers: t.engram_layers, orders: t.engram_orders, rows: t.engram_rows, dim: t.engram_dim },
            mhc_streams: t.mhc_streams,
            mhc_sinkhorn_iters: t.mhc_sinkhorn_iters,
            mtp_layers: t.mtp_num_hidden_layers,
            eos_token_id: t.eos_token_id,
            layers,
        })
    }

    pub fn count(&self, m: Mixer) -> usize {
        self.layers.iter().filter(|l| l.mixer == m).count()
    }

    pub fn kv_layers(&self) -> impl Iterator<Item = &LayerSpec> {
        self.layers.iter().filter(|l| l.mixer.has_kv())
    }

    /// KV bytes per token across all KV layers at `bytes_per_elem` (2 = bf16, 1 = fp8).
    pub fn kv_bytes_per_token(&self, bytes_per_elem: usize) -> usize {
        self.kv_layers().count() * 2 * self.n_kv_heads * self.head_dim * bytes_per_elem
    }

    /// fp32 KDA recurrent state for one request, all KDA layers.
    pub fn kda_state_bytes(&self) -> usize {
        self.count(Mixer::Kda) * self.kda.state_bytes()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn volundr_json() -> String {
        // Layer pattern of the shipped Preview (attention at 3,8,12,17,21,26,30,31,35,40,44,49,53,58,62,63,67,71).
        let attn = [3, 8, 12, 17, 21, 26, 30, 31, 35, 40, 44, 49, 53, 58, 62, 63, 67];
        let mixers: Vec<String> = (0..72)
            .map(|i| if i == 71 { "attn" } else if attn.contains(&i) { "bcsa" } else { "kda" }.to_string())
            .collect();
        serde_json::json!({
            "text_config": {
                "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 72,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_value_head_dim": 128,
                "kda_gate_rank": 128, "mixer_types": mixers, "new_block_indices": [7,15,23,31,39,47,55,63],
                "bcsa_window": 4096, "bcsa_compress": 4, "bcsa_topk": 512, "engram_layers": [2, 22],
                "engram_orders": [2, 3], "engram_rows": 1048576, "engram_dim": 512, "mhc_streams": 4,
                "mhc_sinkhorn_iters": 10, "mtp_num_hidden_layers": 1, "eos_token_id": 248046,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                                    "mrope_section": [11, 11, 10], "mrope_interleaved": true, "rope_type": "default"},
                "some_unknown_field": 1
            }
        })
        .to_string()
    }

    #[test]
    fn layer_map_matches_brief() {
        let c = VolundrConfig::from_json(&volundr_json()).unwrap();
        assert_eq!(c.n_layers, 72);
        assert_eq!(c.count(Mixer::Kda), 54);
        assert_eq!(c.count(Mixer::Bcsa), 17);
        assert_eq!(c.count(Mixer::Dense), 1);
        assert_eq!(c.layers[71].mixer, Mixer::Dense);
        let kv: Vec<usize> = c.kv_layers().map(|l| l.idx).collect();
        assert_eq!(kv, vec![3, 8, 12, 17, 21, 26, 30, 31, 35, 40, 44, 49, 53, 58, 62, 63, 67, 71]);
        assert!(c.layers[2].engram && c.layers[22].engram && !c.layers[3].engram);
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.kda.conv_dim(), 10240);
    }

    #[test]
    fn memory_budget_numbers() {
        let c = VolundrConfig::from_json(&volundr_json()).unwrap();
        // Brief: FP8 KV = 37 KB/token, KDA state ~170 MB/request.
        assert_eq!(c.kv_bytes_per_token(1), 36_864);
        assert_eq!(c.kda_state_bytes(), 169_869_312);
    }

    #[test]
    fn rejects_bad_mixer() {
        let bad = volundr_json().replace("\"bcsa\"", "\"mamba\"");
        assert!(VolundrConfig::from_json(&bad).is_err());
    }
}
