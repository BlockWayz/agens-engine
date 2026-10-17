//! What the server hands the backend (sampling defaults, unused-token mask, stop ids, session hint) and the
//! small endpoints.

mod common;
use agens_server::backend::MockBackend;
use agens_server::sampling::GenerationDefaults;
use agens_server::server::{router, AppState};
use axum::body::Body;
use axum::http::Request;
use common::*;
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

async fn call(app: axum::Router, req: Request<Body>) -> (u16, String) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn chat(body: &str) -> Request<Body> {
    Request::post("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()
}

#[tokio::test]
async fn backend_sees_defaults_mask_and_session() {
    let Some(tok) = tokenizer() else { return };
    let reply = {
        let mut v = tok.encode("ok\n<|/think|>\n\nhi");
        v.push(248046);
        v
    };
    let defaults = GenerationDefaults::from_model_dir(fixtures().join("model"), tok.len()).unwrap();
    let backend = Arc::new(MockBackend::fixed(reply));
    let st = Arc::new(AppState::new(tok, template(), defaults, backend.clone(), "volundr".into()));

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-session-id", "conv-42")
        .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#))
        .unwrap();
    let (status, body) = call(router(st.clone()), req).await;
    assert_eq!(status, 200, "{body}");
    let (status, _) = call(
        router(st.clone()),
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0,"seed":7,"session_id":"body-sess","stop_token_ids":[13]}"#),
    )
    .await;
    assert_eq!(status, 200);

    let calls = backend.calls();
    let a = &calls[0].sampling;
    assert_eq!((a.temperature, a.top_k, a.top_p), (1.0, Some(20), 0.95), "generation_config.json defaults");
    assert_eq!(a.suppress_tokens.len(), 243);
    assert_eq!((a.suppress_tokens[0], *a.suppress_tokens.last().unwrap()), (248077, 248319));
    assert!(a.stop_token_ids.contains(&248046) && a.stop_token_ids.contains(&248044));
    assert_eq!(calls[0].session.as_ref().unwrap().session_id, "conv-42");
    let b = &calls[1].sampling;
    assert_eq!((b.temperature, b.top_k, b.seed), (1.0, Some(1), Some(7)), "temperature 0 means greedy");
    assert!(b.stop_token_ids.contains(&13));
    assert_eq!(calls[1].session.as_ref().unwrap().session_id, "body-sess");
    assert_eq!(calls[0].prompt_ids, calls[1].prompt_ids);
}

#[tokio::test]
async fn small_endpoints_and_rejections() {
    let Some(tok) = tokenizer() else { return };
    let defaults = GenerationDefaults::from_model_dir(fixtures().join("model"), tok.len()).unwrap();
    let st = Arc::new(AppState::new(tok, template(), defaults, Arc::new(MockBackend::fixed(vec![248046])), "volundr".into()));
    let (s, b) = call(router(st.clone()), Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!((s, b.as_str()), (200, ""));
    let (s, b) = call(router(st.clone()), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(s, 200);
    assert!(b.starts_with(r#"{"object":"list","data":[{"id":"volundr","object":"model","created":"#), "{b}");
    assert!(b.ends_with(r#""owned_by":"sglang","root":"volundr","parent":null,"max_model_len":262144}]}"#), "{b}");
    let (s, _) = call(router(st.clone()), Request::get("/v1/models/other").body(Body::empty()).unwrap()).await;
    assert_eq!(s, 404);
    for body in [
        r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
        r#"{"messages":[]}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"top_p":0}"#,
        r#"not json"#,
    ] {
        let (s, b) = call(router(st.clone()), chat(body)).await;
        assert_eq!(s, 400, "{body} -> {b}");
        assert!(b.starts_with(r#"{"object":"error","message":"#), "{b}");
    }
    let (_, b) = call(router(st.clone()), chat(r#"{"messages":[{"role":"user","content":"hi"}],"top_p":0}"#)).await;
    assert!(b.contains("top_p must be in (0, 1], got 0.0."), "{b}");
}
