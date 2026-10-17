// The detokenizer state machine is ported from SGLang (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! `tokenizer.json` (HF `tokenizers`) plus the reference server's incremental detokenizer.
//!
//! Encoding is `encode(text, add_special_tokens=False)`; the Volundr tokenizer has no post-processor
//! specials, so this equals what the chat path feeds the model. Decoding uses the tokenizer's own
//! `special` flags for `skip_special_tokens` (same as `tokenizer.batch_decode` in transformers 5).

use anyhow::{anyhow, Context, Result};
use std::path::Path;

#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    len: usize,
}

/// transformers 5 `Qwen2Tokenizer.PRETOKENIZE_REGEX`. When `tokenizer_config.json` names `Qwen2Tokenizer`,
/// transformers rebuilds the pre-tokenizer with this pattern instead of the one in `tokenizer.json` (which adds
/// `\p{M}`), so combining marks (Devanagari, Thai, ...) split differently. The reference server tokenizes through
/// transformers, so we do the same.
pub const QWEN2_PRETOKENIZE_REGEX: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

impl Tokenizer {
    /// `tokenizer.json` exactly as written.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading {}", path.display()))?;
        let len = inner.get_vocab_size(true);
        Ok(Tokenizer { inner, len })
    }

    /// The tokenizer transformers builds from a model directory (`AutoTokenizer.from_pretrained`):
    /// `tokenizer.json`, plus the `Qwen2Tokenizer` pre-tokenizer override when `tokenizer_config.json` asks for it.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut t = Self::from_file(dir.join("tokenizer.json"))?;
        let class = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("tokenizer_class").and_then(|c| c.as_str()).map(str::to_string));
        if class.as_deref() == Some("Qwen2Tokenizer") {
            t.use_qwen2_pretokenizer()?;
        }
        Ok(t)
    }

    /// Apply transformers' `Qwen2Tokenizer` pre-tokenizer (see [`QWEN2_PRETOKENIZE_REGEX`]).
    pub fn use_qwen2_pretokenizer(&mut self) -> Result<()> {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        let split = Split::new(
            SplitPattern::Regex(QWEN2_PRETOKENIZE_REGEX.into()),
            tokenizers::SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| anyhow!("{e}"))?;
        let seq = Sequence::new(vec![split.into(), ByteLevel::new(false, true, false).into()]);
        self.inner.with_pre_tokenizer(Some(seq));
        Ok(())
    }

    /// Number of ids the tokenizer defines (base vocab + added tokens): `len(tokenizer)` in Python.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.inner.encode_fast(text, false) {
            Ok(e) => e.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        self.inner.decode(ids, skip_special_tokens).unwrap_or_default()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// Python `s[n:]` on code points.
pub fn chars_from(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

/// Python `s[:n]` on code points.
pub fn chars_to(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// sglang `find_printable_text`: the prefix that is safe to show while a multi-byte character is pending.
pub fn find_printable_text(text: &str) -> &str {
    if text.ends_with('\n') {
        return text;
    }
    let mut it = text.chars().rev();
    let last = it.next();
    let prev = it.next();
    if last.is_some_and(is_cjk) {
        return text;
    }
    if prev.is_some_and(is_cjk) {
        let n = char_len(text);
        return chars_to(text, n - 1);
    }
    match text.rfind(' ') {
        Some(i) => &text[..i + 1],
        None => "",
    }
}

/// Why a request stopped. Serialises like sglang's `finish_reason.to_json()`.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Length(usize),
    /// EOS / stop token id (`matched` is the id).
    MatchedToken(u32),
    /// Stop string (`matched` is the string).
    MatchedStr(String),
    Abort(String),
}

impl FinishReason {
    pub fn kind(&self) -> &'static str {
        match self {
            FinishReason::Length(_) => "length",
            FinishReason::MatchedToken(_) | FinishReason::MatchedStr(_) => "stop",
            FinishReason::Abort(_) => "abort",
        }
    }
}

/// The sglang detokenizer state machine (`DetokenizerManager` + `DecodeStatus`) for one request, including
/// the 5 surrounding prompt tokens it decodes with and the U+FFFD hold-back for split UTF-8 sequences.
pub struct IncrementalDecoder {
    decode_ids: Vec<u32>,
    surr_offset: usize,
    read_offset: usize,
    decoded_text: String,
    decoded_len: usize,
    sent_offset: usize,
    skip_special_tokens: bool,
    no_stop_trim: bool,
}

pub const INIT_INCREMENTAL_DETOKENIZATION_OFFSET: usize = 5;

impl IncrementalDecoder {
    pub fn new(prompt_ids: &[u32], skip_special_tokens: bool, no_stop_trim: bool) -> Self {
        let surr = prompt_ids.len().saturating_sub(INIT_INCREMENTAL_DETOKENIZATION_OFFSET);
        IncrementalDecoder {
            decode_ids: prompt_ids[surr..].to_vec(),
            surr_offset: 0,
            read_offset: prompt_ids.len() - surr,
            decoded_text: String::new(),
            decoded_len: 0,
            sent_offset: 0,
            skip_special_tokens,
            no_stop_trim,
        }
    }

    /// Feed the output ids produced since the last call (already cut at the stop position) and the finish
    /// state; returns the text to emit for this step.
    pub fn step(&mut self, tok: &Tokenizer, new_ids: &[u32], finish: Option<&FinishReason>) -> String {
        self.decode_ids.extend_from_slice(new_ids);
        let mut read_ids = &self.decode_ids[self.surr_offset..];
        // (`if not matched` in the reference: a matched id of 0 is falsy and is not trimmed)
        if let (Some(FinishReason::MatchedToken(id)), false) = (finish, self.no_stop_trim) {
            if !read_ids.is_empty() && *id != 0 {
                read_ids = &read_ids[..read_ids.len() - 1];
            }
        }
        let surr_ids = &self.decode_ids[self.surr_offset..self.read_offset];
        let surr_text =
            if surr_ids.is_empty() { String::new() } else { tok.decode(surr_ids, self.skip_special_tokens) };
        let read_text =
            if read_ids.is_empty() { String::new() } else { tok.decode(read_ids, self.skip_special_tokens) };
        let new_text = chars_from(&read_text, char_len(&surr_text)).to_string();

        match finish {
            None => {
                let pending = self.sent_offset - self.decoded_len;
                if !new_text.is_empty() && !new_text.ends_with('\u{fffd}') {
                    self.decoded_len += char_len(&new_text);
                    self.decoded_text.push_str(&new_text);
                    self.surr_offset = self.read_offset;
                    self.read_offset = self.decode_ids.len();
                    self.sent_offset = self.decoded_len;
                    chars_from(&new_text, pending).to_string()
                } else {
                    let printable = find_printable_text(&new_text);
                    self.sent_offset = self.decoded_len + char_len(printable);
                    chars_from(printable, pending).to_string()
                }
            }
            Some(fr) => {
                let mut out = std::mem::take(&mut self.decoded_text);
                out.push_str(&new_text);
                if let FinishReason::MatchedStr(m) = fr {
                    if !m.is_empty() {
                        if let Some(pos) = out.find(m.as_str()) {
                            let end = if self.no_stop_trim { pos + m.len() } else { pos };
                            out.truncate(end);
                        }
                    }
                }
                let emit = chars_from(&out, self.sent_offset).to_string();
                self.sent_offset = char_len(&out);
                self.decoded_text = out;
                emit
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_text() {
        assert_eq!(find_printable_text("hello wor"), "hello ");
        assert_eq!(find_printable_text("line\n"), "line\n");
        assert_eq!(find_printable_text("你好\u{fffd}"), "你好");
        assert_eq!(find_printable_text("abc\u{fffd}"), "");
        assert_eq!(chars_from("é😀x", 2), "x");
    }
}
