//! End-to-end parity through the HTTP layer with a mock backend replaying recorded token ids.
//!
//! The expected bodies / SSE streams come from the reference's own serving code (`OpenAIServingChat` /
//! `OpenAIServingCompletion`) fed the same token ids through its real scheduler finish logic and detokenizer
//! (see `tests/reference/gen_ref.py`). Only `created` and tool-call ids (time / random) are normalised.

mod common;
use agens_server::backend::MockBackend;
use agens_server::pyjson::PyValue;
use agens_server::sampling::GenerationDefaults;
use agens_server::server::{router, AppState};
use axum::body::Body;
use axum::http::Request;
use common::*;
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

/// `"created":<n>` -> `"created":0`, `"id":"call_<24 hex>"` -> `"id":"call_X"`.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let a = rest.find("\"created\":");
        let b = rest.find("\"id\":\"call_");
        let (i, is_created) = match (a, b) {
            (None, None) => break,
            (Some(a), None) => (a, true),
            (None, Some(b)) => (b, false),
            (Some(a), Some(b)) => {
                if a < b {
                    (a, true)
                } else {
                    (b, false)
                }
            }
        };
        out.push_str(&rest[..i]);
        if is_created {
            let after = &rest[i + 10..];
            let n = after.bytes().take_while(|c| c.is_ascii_digit()).count();
            out.push_str("\"created\":0");
            rest = &after[n..];
        } else {
            let after = &rest[i + 11..];
            let n = after.bytes().take_while(|c| c.is_ascii_hexdigit()).count();
            if n == 24 {
                out.push_str("\"id\":\"call_X");
                rest = &after[n..];
            } else {
                out.push_str("\"id\":\"call_");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn state(tok: agens_server::tokenizer::Tokenizer, tokens: Vec<u32>, step: usize) -> Arc<AppState> {
    let defaults = GenerationDefaults::from_model_dir(fixtures().join("model"), tok.len()).unwrap();
    let mut backend = MockBackend::fixed(tokens);
    backend.step_size = step;
    Arc::new(AppState::new(tok, template(), defaults, Arc::new(backend), "volundr".into()))
}

async fn post(st: Arc<AppState>, path: &str, body: &PyValue) -> (u16, String) {
    let req =
        Request::post(path).header("content-type", "application/json").body(Body::from(body.dumps_compact())).unwrap();
    let resp = router(st).oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_events(body: &str) -> Vec<String> {
    body.split_inclusive("\n\n").map(normalize).collect()
}

#[tokio::test]
async fn chat_completions_match_reference() {
    let Some(tok) = tokenizer() else { return };
    let f = fixture("e2e_cases.json");
    let tool_sets = f.get("tool_sets").unwrap();
    let cases = f.get("cases").unwrap().as_list().unwrap();
    let (mut ns_ok, mut st_ok) = (0, 0);
    let mut fails = Vec::new();
    for c in cases {
        let name = s(c.get("name").unwrap()).to_string();
        let mut body = c.get("body").unwrap().clone();
        if let Some(t) = opt(c, "tools") {
            body.set("tools", tool_sets.get(s(t)).unwrap().clone());
        }
        body.set("rid", PyValue::str("RID"));
        let out_ids = ids(c.get("output_ids").unwrap());
        let step = c.get("step").and_then(|v| v.as_i64()).unwrap_or(1) as usize;
        if let Some(err) = opt(c, "error") {
            // request rejected before generation: same status and error body, streaming or not
            for stream in [false, true] {
                let mut b = body.clone();
                b.set("stream", PyValue::Bool(stream));
                let (status, got) = post(state(tok.clone(), out_ids.clone(), step), "/v1/chat/completions", &b).await;
                let want_status = err.get("status").unwrap().as_i64().unwrap() as u16;
                let body_ok = opt(err, "body").is_none_or(|w| s(w) == got);
                if status == want_status && body_ok {
                    if stream {
                        st_ok += 1
                    } else {
                        ns_ok += 1
                    }
                } else {
                    fails.push(format!("{name} [error, stream={stream}] {status} {got} != {}", err.dumps(false)));
                }
            }
            continue;
        }
        // non-streaming
        let mut b = body.clone();
        b.set("stream", PyValue::Bool(false));
        let (status, got) = post(state(tok.clone(), out_ids.clone(), step), "/v1/chat/completions", &b).await;
        let want = c.get("nonstream").unwrap();
        if status == 200 && normalize(&got) == s(want.get("body").unwrap()) {
            ns_ok += 1;
        } else {
            fails.push(format!(
                "{name} [non-stream] status {status}\n{}",
                diff(&normalize(&got), s(want.get("body").unwrap()))
            ));
        }
        // streaming
        let mut b = body.clone();
        b.set("stream", PyValue::Bool(true));
        let (status, got) = post(state(tok.clone(), out_ids, step), "/v1/chat/completions", &b).await;
        let want: Vec<String> = c.get("sse").unwrap().as_list().unwrap().iter().map(|x| s(x).to_string()).collect();
        let got = sse_events(&got);
        if status == 200 && got == want {
            st_ok += 1;
        } else {
            let i = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            fails.push(format!(
                "{name} [stream] status {status}, {} vs {} events, first difference at event {i}:\n  got:      {:?}\n  expected: {:?}",
                got.len(),
                want.len(),
                got.get(i),
                want.get(i)
            ));
        }
    }
    eprintln!("e2e chat parity: non-stream {ns_ok}/{n}, stream {st_ok}/{n}", n = cases.len());
    assert!(fails.is_empty(), "{} mismatches:\n{}", fails.len(), fails.join("\n\n"));
}

#[tokio::test]
async fn completions_match_reference() {
    let Some(tok) = tokenizer() else { return };
    let f = fixture("completion_cases.json");
    let cases = f.get("cases").unwrap().as_list().unwrap();
    for c in cases {
        let name = s(c.get("name").unwrap());
        let mut body = c.get("body").unwrap().clone();
        body.set("rid", PyValue::str("RID"));
        let out_ids = ids(c.get("output_ids").unwrap());
        let mut b = body.clone();
        b.set("stream", PyValue::Bool(false));
        let (status, got) = post(state(tok.clone(), out_ids.clone(), 1), "/v1/completions", &b).await;
        assert_eq!(status, 200, "{name}");
        assert_eq!(normalize(&got), s(c.get("nonstream").unwrap()), "{name} non-stream");
        let mut b = body.clone();
        b.set("stream", PyValue::Bool(true));
        let (_, got) = post(state(tok.clone(), out_ids, 1), "/v1/completions", &b).await;
        let want: Vec<String> = c.get("sse").unwrap().as_list().unwrap().iter().map(|x| s(x).to_string()).collect();
        assert_eq!(sse_events(&got), want, "{name} stream");
    }
    eprintln!("e2e completion parity: {}/{} (stream + non-stream)", cases.len(), cases.len());
}
