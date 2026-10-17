//! Tokenizer parity with the reference image (transformers 5 / tokenizers 0.22 as loaded by sglang):
//! encode, decode with and without special tokens, and the streaming detokenizer's per-step output.

mod common;
use agens_server::tokenizer::{FinishReason, IncrementalDecoder};
use common::*;

#[test]
fn encode_decode_match_reference() {
    let Some(tok) = tokenizer() else { return };
    let f = fixture("tokenizer_cases.json");
    assert_eq!(tok.len() as i64, f.get("vocab_len").unwrap().as_i64().unwrap());
    let mut fails = Vec::new();
    let enc = f.get("encode").unwrap().as_list().unwrap();
    for c in enc {
        let text = s(c.get("text").unwrap());
        let want = ids(c.get("ids").unwrap());
        let got = tok.encode(text);
        if got != want {
            fails.push(format!("encode {text:?}: {got:?} != {want:?}"));
        }
    }
    let dec = f.get("decode").unwrap().as_list().unwrap();
    for c in dec {
        let i = ids(c.get("ids").unwrap());
        for (skip, key) in [(true, "skip"), (false, "keep")] {
            let got = tok.decode(&i, skip);
            if got != s(c.get(key).unwrap()) {
                fails.push(format!("decode {i:?} skip={skip}: {got:?} != {:?}", s(c.get(key).unwrap())));
            }
        }
    }
    eprintln!(
        "tokenizer parity: encode {}/{}, decode {}/{}",
        enc.len() - fails.iter().filter(|f| f.starts_with("encode")).count(),
        enc.len(),
        2 * dec.len() - fails.iter().filter(|f| f.starts_with("decode")).count(),
        2 * dec.len()
    );
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n"));
}

#[test]
fn streaming_detokenizer_matches_reference() {
    let Some(tok) = tokenizer() else { return };
    let f = fixture("tokenizer_cases.json");
    let cases = f.get("detok").unwrap().as_list().unwrap();
    for c in cases {
        let prompt = ids(c.get("prompt_ids").unwrap());
        let out = ids(c.get("output_ids").unwrap());
        let skip = c.get("skip_special_tokens").unwrap().truthy();
        let want: Vec<String> = c.get("deltas").unwrap().as_list().unwrap().iter().map(|d| s(d).to_string()).collect();
        // One token per step; max_new_tokens == len(output) so the last step finishes with "length" (the
        // length check runs before the EOS check, so a trailing EOS is not trimmed here).
        let mut dec = IncrementalDecoder::new(&prompt, skip, false);
        let mut got = Vec::new();
        for (i, t) in out.iter().enumerate() {
            let finish = if i + 1 == out.len() { Some(FinishReason::Length(out.len())) } else { None };
            got.push(dec.step(&tok, &[*t], finish.as_ref()));
        }
        assert_eq!(got, want, "output {out:?} skip={skip}");
    }
    eprintln!("detokenizer parity: {}/{} streams", cases.len(), cases.len());
}
