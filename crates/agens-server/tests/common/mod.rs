//! Shared helpers for the parity tests. Fixtures under `tests/fixtures` were produced by
//! `tests/reference/gen_ref.py` running the real transformers / sglang code of the reference image.
#![allow(dead_code)]

use agens_server::pyjson::{self, Dialect, PyValue};
use agens_server::template::ChatTemplate;
use agens_server::tokenizer::Tokenizer;
use std::path::PathBuf;

pub fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn fixture(name: &str) -> PyValue {
    let p = fixtures().join(name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    pyjson::loads(&s, Dialect::Python).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

pub fn template() -> ChatTemplate {
    let s = std::fs::read_to_string(fixtures().join("model/chat_template.jinja")).unwrap();
    ChatTemplate::new(s).unwrap()
}

/// `tokenizer.json` is ~10 MB and is not checked in: a directory holding it (and `tokenizer_config.json`) from
/// `$AGENS_MODEL_DIR`, else `refs/volundr/` at the workspace root (`scripts/server_ref14.sh fetch` fills it).
pub fn model_dir() -> Option<PathBuf> {
    let mut cands = Vec::new();
    if let Ok(d) = std::env::var("AGENS_MODEL_DIR") {
        cands.push(PathBuf::from(d));
    }
    cands.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../refs/volundr"));
    cands.into_iter().find(|p| p.join("tokenizer.json").exists())
}

pub fn tokenizer() -> Option<Tokenizer> {
    match model_dir() {
        Some(p) => Some(Tokenizer::from_model_dir(p).unwrap()),
        None => {
            eprintln!("SKIP: tokenizer.json not found (set AGENS_MODEL_DIR or run scripts/server_ref14.sh fetch)");
            None
        }
    }
}

pub fn s(v: &PyValue) -> &str {
    v.as_str().unwrap_or_else(|| panic!("expected a string, got {v:?}"))
}

pub fn ids(v: &PyValue) -> Vec<u32> {
    v.as_list().unwrap().iter().map(|x| x.as_i64().unwrap() as u32).collect()
}

pub fn opt<'a>(v: &'a PyValue, k: &str) -> Option<&'a PyValue> {
    v.get(k).filter(|x| !x.is_null())
}

/// First differing position, for readable failures.
pub fn diff(a: &str, b: &str) -> String {
    let i = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let ctx = |s: &str| s.chars().skip(i.saturating_sub(40)).take(120).collect::<String>();
    format!("first difference at char {i}\n  got:      {:?}\n  expected: {:?}", ctx(a), ctx(b))
}
