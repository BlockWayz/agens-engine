//! axum routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
//!
//! Session hint: a request may name its conversation with the body field `session_id` or the header
//! `X-Session-Id` (the header wins). It is passed to the backend untouched (see [`crate::backend`]); responses
//! do not change.

use crate::backend::{Backend, SessionHint};
use crate::engine::{self, EngineRequest};
use crate::openai::{self, ApiError, ChatRequest, ChatStream, CompletionPrompt, CompletionRequest, RespMeta, Usage};
use crate::sampling::{GenerationDefaults, RequestSampling, SamplingParams};
use crate::template::ChatTemplate;
use crate::tokenizer::{FinishReason, Tokenizer};
use anyhow::{Context, Result};
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct AppState {
    pub tokenizer: Arc<Tokenizer>,
    pub template: ChatTemplate,
    pub defaults: GenerationDefaults,
    pub backend: Arc<dyn Backend>,
    pub model_name: String,
    pub context_length: usize,
    pub created: u64,
    pub think_end_ids: Vec<u32>,
    /// Logit rows of the model (`config.json` vocab_size), for range checks.
    pub vocab_size: usize,
}

impl AppState {
    /// Load tokenizer, chat template and generation defaults from a model directory.
    pub fn from_model_dir(
        dir: impl Into<PathBuf>,
        backend: Arc<dyn Backend>,
        model_name: Option<String>,
    ) -> Result<Self> {
        let dir = dir.into();
        let tokenizer = Tokenizer::from_model_dir(&dir)?;
        let source = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .with_context(|| format!("{}/chat_template.jinja", dir.display()))?;
        let template = ChatTemplate::new(source).map_err(|e| anyhow::anyhow!("chat template: {e}"))?;
        let defaults = GenerationDefaults::from_model_dir(&dir, tokenizer.len())?;
        Ok(Self::new(tokenizer, template, defaults, backend, model_name.unwrap_or_else(|| dir.display().to_string())))
    }

    pub fn new(
        tokenizer: Tokenizer,
        template: ChatTemplate,
        defaults: GenerationDefaults,
        backend: Arc<dyn Backend>,
        model_name: String,
    ) -> Self {
        let think_end_ids = tokenizer.encode(crate::reasoning::THINK_END);
        let vocab_size = defaults.vocab_size.unwrap_or(tokenizer.len());
        AppState {
            context_length: backend.context_length(),
            tokenizer: Arc::new(tokenizer),
            template,
            defaults,
            backend,
            model_name,
            created: now(),
            think_end_ids,
            vocab_size,
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/health_generate", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{*model}", get(model))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::OK
}

fn json(status: StatusCode, body: String) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

fn error(e: ApiError) -> Response {
    json(StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_REQUEST), e.body())
}

fn sse(rx: mpsc::Receiver<String>) -> Response {
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|s| (Ok::<Bytes, std::convert::Infallible>(Bytes::from(s)), rx))
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    json(StatusCode::OK, openai::models_json(&st.model_name, now(), st.context_length))
}

async fn model(State(st): State<Arc<AppState>>, Path(name): Path<String>) -> Response {
    if name != st.model_name {
        let msg = pyjsonq(&format!("The model '{name}' does not exist"));
        return json(
            StatusCode::NOT_FOUND,
            format!(
                "{{\"error\":{{\"message\":{msg},\"type\":\"invalid_request_error\",\"param\":\"model\",\"code\":\"model_not_found\"}}}}"
            ),
        );
    }
    json(StatusCode::OK, openai::model_card_json(&st.model_name, now(), st.context_length))
}

fn pyjsonq(s: &str) -> String {
    crate::pyjson::quote(s, false)
}

fn check_json(headers: &HeaderMap) -> Result<(), ApiError> {
    let ct = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    if ct.split(';').next().unwrap_or("").trim() != "application/json" {
        return Err(ApiError::bad_request("Unsupported Media Type: Only 'application/json' is allowed"));
    }
    Ok(())
}

fn session(headers: &HeaderMap, body_session: Option<String>) -> Option<SessionHint> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or(body_session)
        .filter(|s| !s.is_empty())
        .map(|session_id| SessionHint { session_id })
}

/// Common tail of both endpoints: validate sampling, check lengths, build the engine request.
#[allow(clippy::too_many_arguments)]
fn engine_request(
    st: &AppState,
    prompt_ids: Vec<u32>,
    rs: &RequestSampling,
    stop: Vec<String>,
    stop_token_ids: Vec<u32>,
    ignore_eos: bool,
    skip_special_tokens: bool,
    no_stop_trim: bool,
    stream: bool,
    require_reasoning: bool,
    budget: (Option<i64>, Option<f64>),
    session: Option<SessionHint>,
) -> Result<EngineRequest, ApiError> {
    let (temperature, top_k) = rs.finalize(st.vocab_size).map_err(ApiError::bad_request)?;
    let n = prompt_ids.len();
    if n >= st.context_length {
        return Err(ApiError::bad_request(format!(
            "The input ({n} tokens) is longer than the model's context length ({} tokens).",
            st.context_length
        )));
    }
    let requested = rs.max_new_tokens.map(|m| m.max(0) as usize).unwrap_or(1 << 30);
    let max_new_tokens = requested.min(st.context_length.saturating_sub(n + 1));
    let logit_bias =
        rs.logit_bias.iter().filter_map(|(k, v)| k.trim().parse::<u32>().ok().map(|id| (id, *v))).collect();
    let sampling = SamplingParams {
        temperature,
        top_p: rs.top_p,
        top_k,
        min_p: rs.min_p,
        presence_penalty: rs.presence_penalty,
        frequency_penalty: rs.frequency_penalty,
        repetition_penalty: rs.repetition_penalty,
        min_new_tokens: rs.min_new_tokens.max(0) as usize,
        max_new_tokens,
        seed: rs.seed.map(|s| s as u64),
        logit_bias,
        suppress_tokens: st.defaults.suppress_tokens.clone(),
        stop_token_ids: Vec::new(),
        ignore_eos,
        reasoning_budget: budget.0.filter(|b| *b > 0).map(|b| b as u32),
        reasoning_budget_soft_bias: budget.1.unwrap_or(0.0),
    };
    Ok(EngineRequest {
        prompt_ids,
        sampling,
        stop_strs: stop,
        stop_token_ids,
        eos_token_ids: st.defaults.eos_token_ids.clone(),
        ignore_eos,
        skip_special_tokens,
        no_stop_trim,
        stream,
        require_reasoning,
        think_end_ids: st.think_end_ids.clone(),
        vocab_size: st.vocab_size,
        session,
    })
}

/// Server-wide defaults of the chat endpoint's reasoning budget, as the reference image reads them.
fn env_num(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| *v != 0.0)
}

async fn chat(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    match chat_inner(st, headers, body).await {
        Ok(r) => r,
        Err(e) => error(e),
    }
}

async fn chat_inner(st: Arc<AppState>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    check_json(&headers)?;
    let mut req = ChatRequest::parse(&body)?;
    openai::validate_chat(&req, st.context_length)?;
    let plan = openai::plan_chat(&mut req, &st.template, &st.tokenizer, &st.defaults)?;
    let er = engine_request(
        &st,
        plan.prompt_ids.clone(),
        &plan.sampling,
        req.stop.clone(),
        req.stop_token_ids.clone(),
        req.ignore_eos,
        plan.skip_special_tokens,
        req.no_stop_trim,
        req.stream,
        plan.require_reasoning,
        (
            req.reasoning_budget.or_else(|| env_num("VOLUNDR_REASONING_BUDGET").map(|v| v as i64)),
            req.reasoning_budget_soft_bias.or_else(|| env_num("VOLUNDR_REASONING_BUDGET_SOFT_BIAS")),
        ),
        session(&headers, req.session_id.clone()),
    )?;
    let meta = RespMeta {
        id: req.rid.clone().unwrap_or_else(openai::new_request_id),
        created: now(),
        model: req.model.clone(),
    };
    let mut outputs = engine::run(st.tokenizer.clone(), st.backend.clone(), er);
    if req.stream {
        let (tx, rx) = mpsc::channel::<String>(64);
        tokio::spawn(async move {
            let mut cs = ChatStream::new(meta, &req, &plan);
            while let Some(item) = outputs.recv().await {
                let o = match item {
                    Ok(o) => o,
                    Err(msg) => {
                        let e = ApiError { status: 500, message: msg, err_type: "InternalServerError".into() };
                        let _ = tx.send(format!("data: {}\n\n", e.stream_body())).await;
                        let _ = tx.send(openai::SSE_DONE.into()).await;
                        return;
                    }
                };
                let usage = Usage {
                    prompt_tokens: o.prompt_tokens,
                    completion_tokens: o.completion_tokens,
                    reasoning_tokens: o.reasoning_tokens,
                };
                let chunks = match cs.on_output(&o.delta, o.finish.as_ref(), usage) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(format!("data: {}\n\n", e.stream_body())).await;
                        let _ = tx.send(openai::SSE_DONE.into()).await;
                        return;
                    }
                };
                for c in chunks {
                    if tx.send(c).await.is_err() {
                        return;
                    }
                }
                if let Some(f) = &o.finish {
                    for c in cs.on_finish(f) {
                        if tx.send(c).await.is_err() {
                            return;
                        }
                    }
                    return;
                }
            }
        });
        return Ok(sse(rx));
    }
    let (text, finish, usage) = collect(&mut outputs).await?;
    Ok(json(StatusCode::OK, openai::chat_response(&meta, &req, &plan, &text, &finish, usage)))
}

async fn collect(
    outputs: &mut mpsc::Receiver<Result<engine::Output, String>>,
) -> Result<(String, FinishReason, Usage), ApiError> {
    let mut text = String::new();
    while let Some(item) = outputs.recv().await {
        let o = item.map_err(|m| ApiError { status: 500, message: m, err_type: "InternalServerError".into() })?;
        text.push_str(&o.delta);
        if let Some(f) = o.finish {
            let usage = Usage {
                prompt_tokens: o.prompt_tokens,
                completion_tokens: o.completion_tokens,
                reasoning_tokens: o.reasoning_tokens,
            };
            return Ok((text, f, usage));
        }
    }
    Err(ApiError {
        status: 500,
        message: "generation ended without a result".into(),
        err_type: "InternalServerError".into(),
    })
}

async fn completions(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    match completions_inner(st, headers, body).await {
        Ok(r) => r,
        Err(e) => error(e),
    }
}

async fn completions_inner(st: Arc<AppState>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    check_json(&headers)?;
    let req = CompletionRequest::parse(&body)?;
    let prompt_ids = match &req.prompt {
        CompletionPrompt::Text(s) => st.tokenizer.encode(s),
        CompletionPrompt::Ids(v) => v.clone(),
    };
    let er = engine_request(
        &st,
        prompt_ids,
        &req.sampling(),
        req.stop.clone(),
        req.stop_token_ids.clone(),
        req.ignore_eos,
        req.skip_special_tokens,
        req.no_stop_trim,
        req.stream,
        false,
        (None, None),
        session(&headers, req.session_id.clone()),
    )?;
    let meta = RespMeta {
        id: req.rid.clone().unwrap_or_else(openai::new_request_id),
        created: now(),
        model: req.model.clone(),
    };
    let mut outputs = engine::run(st.tokenizer.clone(), st.backend.clone(), er);
    if req.stream {
        let (tx, rx) = mpsc::channel::<String>(64);
        tokio::spawn(async move {
            let (include_usage, continuous) = req
                .stream_options
                .as_ref()
                .map(|o| (o.include_usage, o.continuous_usage_stats))
                .unwrap_or((false, false));
            let mut last = Usage { prompt_tokens: 0, completion_tokens: 0, reasoning_tokens: 0 };
            while let Some(item) = outputs.recv().await {
                let o = match item {
                    Ok(o) => o,
                    Err(msg) => {
                        let e = ApiError { status: 500, message: msg, err_type: "InternalServerError".into() };
                        let _ = tx.send(format!("data: {}\n\n", e.stream_body())).await;
                        break;
                    }
                };
                last = Usage {
                    prompt_tokens: o.prompt_tokens,
                    completion_tokens: o.completion_tokens,
                    reasoning_tokens: o.reasoning_tokens,
                };
                let chunk = openai::completion_chunk(&meta, &o.delta, o.finish.as_ref(), continuous.then_some(&last));
                if tx.send(chunk).await.is_err() {
                    return;
                }
                if o.finish.is_some() {
                    break;
                }
            }
            if include_usage {
                let _ = tx.send(openai::completion_usage_chunk(&meta, &last)).await;
            }
            let _ = tx.send(openai::SSE_DONE.into()).await;
        });
        return Ok(sse(rx));
    }
    let (text, finish, usage) = collect(&mut outputs).await?;
    Ok(json(StatusCode::OK, openai::completion_response(&meta, &text, &finish, usage)))
}
