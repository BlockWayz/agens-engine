//! Sampling parameters and model defaults.
//!
//! Defaults come from `generation_config.json` (temperature 1.0, top-k 20, top-p 0.95 for Volundr) and apply only
//! where the request leaves a field unset, as in the reference chat endpoint. `/v1/completions` uses the OpenAI
//! defaults instead (temperature 1, top-p 1, no top-k), also as the reference does.
//!
//! Token ids the tokenizer does not define (`len(tokenizer) .. vocab_size`, 248077..248319 for Volundr) are padding
//! rows of the embedding / lm_head; the reference masks their logits. They are listed in
//! [`SamplingParams::suppress_tokens`] together with any `suppress_tokens` from `generation_config.json`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    /// EOS ids from `generation_config.json` and `config.json` (the reference stops on any of them).
    pub eos_token_ids: Vec<u32>,
    pub suppress_tokens: Vec<u32>,
    /// `config.json` vocab_size (logit rows), when known.
    pub vocab_size: Option<usize>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Deserialize, Default)]
struct RawGen {
    temperature: Option<f64>,
    top_k: Option<i64>,
    top_p: Option<f64>,
    min_p: Option<f64>,
    repetition_penalty: Option<f64>,
    eos_token_id: Option<OneOrMany>,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
}

#[derive(Deserialize, Default)]
struct RawConfig {
    vocab_size: Option<usize>,
    eos_token_id: Option<OneOrMany>,
    text_config: Option<Box<RawConfig>>,
}

impl GenerationDefaults {
    /// Read `generation_config.json` and `config.json` from a model directory (either may be missing).
    pub fn from_model_dir(dir: impl AsRef<Path>, tokenizer_len: usize) -> Result<Self> {
        let dir = dir.as_ref();
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
        let gen: RawGen = match read("generation_config.json") {
            Some(s) => serde_json::from_str(&s).context("generation_config.json")?,
            None => RawGen::default(),
        };
        let cfg: RawConfig = match read("config.json") {
            Some(s) => serde_json::from_str(&s).context("config.json")?,
            None => RawConfig::default(),
        };
        Ok(Self::from_parts(gen, cfg, tokenizer_len))
    }

    fn from_parts(gen: RawGen, cfg: RawConfig, tokenizer_len: usize) -> Self {
        let ids = |o: Option<OneOrMany>| match o {
            Some(OneOrMany::One(i)) => vec![i],
            Some(OneOrMany::Many(v)) => v,
            None => vec![],
        };
        let vocab_size = cfg.vocab_size.or_else(|| cfg.text_config.as_ref().and_then(|t| t.vocab_size));
        let mut eos = ids(cfg.eos_token_id);
        if let Some(t) = cfg.text_config {
            eos.extend(ids(t.eos_token_id));
        }
        eos.extend(ids(gen.eos_token_id));
        eos.sort_unstable();
        eos.dedup();
        let mut suppress = gen.suppress_tokens;
        if let Some(v) = vocab_size {
            suppress.extend(tokenizer_len as u32..v as u32);
        }
        suppress.sort_unstable();
        suppress.dedup();
        GenerationDefaults {
            temperature: gen.temperature,
            top_k: gen.top_k,
            top_p: gen.top_p,
            min_p: gen.min_p,
            repetition_penalty: gen.repetition_penalty,
            eos_token_ids: eos,
            suppress_tokens: suppress,
            vocab_size,
        }
    }
}

/// What the backend needs to sample one request. Built by the server; stop strings, stop ids, EOS and
/// `max_new_tokens` are also enforced by the server (see `engine`), so a backend may treat them as hints.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// 1.0 when greedy (`top_k == 1` carries greediness, as in the reference).
    pub temperature: f64,
    pub top_p: f64,
    /// `None` = whole vocabulary.
    pub top_k: Option<u32>,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: usize,
    pub max_new_tokens: usize,
    pub seed: Option<u64>,
    /// Logit bias by token id.
    pub logit_bias: Vec<(u32, f64)>,
    /// Ids whose logits must be -inf (unused vocabulary rows + `generation_config.json` suppress_tokens).
    pub suppress_tokens: Vec<u32>,
    /// Ids that end generation (EOS + request `stop_token_ids`), empty with `ignore_eos`.
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    /// Reasoning budget (Volundr): after this many thinking tokens force `<|/think|>`.
    pub reasoning_budget: Option<u32>,
    pub reasoning_budget_soft_bias: f64,
}

/// The reference's `_SAMPLING_EPS`: temperatures below it mean greedy.
pub const SAMPLING_EPS: f64 = 1e-6;

/// Raw request values after the default chain (request > generation_config > OpenAI default), before the
/// greedy rewrite and validation.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestSampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: i64,
    pub max_new_tokens: Option<i64>,
    pub seed: Option<i64>,
    pub logit_bias: Vec<(String, f64)>,
}

impl RequestSampling {
    /// `SamplingParams.__post_init__` + `verify`: greedy rewrite and range checks with the reference's messages.
    pub fn finalize(&self, vocab_size: usize) -> Result<(f64, Option<u32>), String> {
        let (mut temperature, mut top_k) = (self.temperature, self.top_k);
        if (0.0..SAMPLING_EPS).contains(&temperature) {
            temperature = 1.0;
            top_k = 1;
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(format!("temperature must be a non-negative finite number, got {}.", py_num(self.temperature)));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}.", py_num(self.top_p)));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}.", py_num(self.min_p)));
        }
        if top_k != -1 && top_k < 1 {
            return Err(format!("top_k must be -1 (disable) or at least 1, got {top_k}."));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(format!("frequency_penalty must be in [-2, 2], got {}.", py_num(self.frequency_penalty)));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(format!("presence_penalty must be in [-2, 2], got {}.", py_num(self.presence_penalty)));
        }
        if !(self.repetition_penalty > 0.0 && self.repetition_penalty <= 2.0) {
            return Err(format!(
                "repetition_penalty must be in (0, 2] (1.0 = no penalty), got {}.",
                py_num(self.repetition_penalty)
            ));
        }
        if self.min_new_tokens < 0 {
            return Err(format!("min_new_tokens must be in [0, max_new_tokens], got {}.", self.min_new_tokens));
        }
        if let Some(m) = self.max_new_tokens {
            if m < 0 {
                return Err(format!("max_new_tokens must be at least 0, got {m}."));
            }
            if self.min_new_tokens > m {
                return Err(format!(
                    "min_new_tokens must be in [0, max_new_tokens({m})], got {}.",
                    self.min_new_tokens
                ));
            }
        }
        for (k, _) in &self.logit_bias {
            match k.trim().parse::<i64>() {
                Ok(id) if id >= 0 && (id as usize) < vocab_size => {}
                _ => return Err(format!("logit_bias must has keys in [0, {}], got {k}.", vocab_size - 1)),
            }
        }
        Ok((temperature, if top_k == -1 { None } else { Some(top_k as u32) }))
    }
}

/// Python `str(float)`.
fn py_num(f: f64) -> String {
    if f.is_nan() {
        "nan".into()
    } else if f.is_infinite() {
        if f > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        crate::pyjson::float_repr(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_from_volundr_files() {
        let gen: RawGen = serde_json::from_str(
            r#"{"bos_token_id": 248044, "do_sample": true, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
                "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
        )
        .unwrap();
        let cfg: RawConfig = serde_json::from_str(r#"{"vocab_size": 248320, "eos_token_id": 248046}"#).unwrap();
        let d = GenerationDefaults::from_parts(gen, cfg, 248077);
        assert_eq!((d.temperature, d.top_k, d.top_p), (Some(1.0), Some(20), Some(0.95)));
        assert_eq!(d.eos_token_ids, vec![248044, 248046]);
        assert_eq!(d.suppress_tokens.first(), Some(&248077));
        assert_eq!(d.suppress_tokens.last(), Some(&248319));
        assert_eq!(d.suppress_tokens.len(), 243);
    }
}
