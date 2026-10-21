// Request handling ported from SGLang (Apache-2.0). See THIRD_PARTY_NOTICES.md.
//! OpenAI protocol: request normalisation along the reference's chat path (pydantic `model_dump`, tool-call
//! argument parsing, content-part processing, `reasoning_effort` / `enable_thinking` resolution, template
//! kwargs) and byte-exact response / SSE serialisation (field order, nulls, escaping of the reference's
//! pydantic / msgspec / orjson encoders).

use crate::pyjson::{self, Dialect, PyValue};
use crate::reasoning::ReasoningParser;
use crate::sampling::{GenerationDefaults, RequestSampling};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::{FinishReason, Tokenizer};
use crate::toolcall::{self, ToolCallItem, ToolCallStreamParser, ToolSet};
use std::fmt::Write as _;

/// An HTTP error in the reference's `ErrorResponse` shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub err_type: String,
}

impl ApiError {
    /// `create_error_response(msg)`: request validation failures.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError { status: 400, message: msg.into(), err_type: "BadRequestError".into() }
    }
    /// `ValueError` raised while converting the request (template errors, bad tool-call arguments, ...).
    pub fn value_error(msg: impl Into<String>) -> Self {
        ApiError { status: 400, message: msg.into(), err_type: "BadRequest".into() }
    }
    pub fn not_supported(what: &str) -> Self {
        ApiError::bad_request(format!("{what} is not supported by this server."))
    }
    pub fn body(&self) -> String {
        let mut s = String::from("{\"object\":\"error\",\"message\":");
        pyjson::quote_into(&mut s, &self.message, false);
        s.push_str(",\"type\":");
        pyjson::quote_into(&mut s, &self.err_type, false);
        let _ = write!(s, ",\"param\":null,\"code\":{}}}", self.status);
        s
    }
    /// `create_streaming_error_response`: `json.dumps({"error": ...})` (default separators).
    pub fn stream_body(&self) -> String {
        let v = PyValue::Dict(vec![(
            "error".into(),
            PyValue::Dict(vec![
                ("object".into(), PyValue::str("error")),
                ("message".into(), PyValue::str(self.message.clone())),
                ("type".into(), PyValue::str(self.err_type.clone())),
                ("param".into(), PyValue::Null),
                ("code".into(), PyValue::int(self.status as i64)),
            ]),
        )]);
        v.dumps(true)
    }
}

type R<T> = Result<T, ApiError>;

fn invalid(field: &str, what: &str) -> ApiError {
    ApiError::bad_request(format!("Invalid request: field '{field}' {what}"))
}

// ------------------------------------------------------------------------------------------------ field access

fn get<'a>(b: &'a PyValue, k: &str) -> Option<&'a PyValue> {
    b.get(k).filter(|v| !v.is_null())
}

fn opt_str(b: &PyValue, k: &str) -> R<Option<String>> {
    match get(b, k) {
        None => Ok(None),
        Some(PyValue::Str(s)) => Ok(Some(s.clone())),
        Some(_) => Err(invalid(k, "must be a string")),
    }
}

fn opt_f64(b: &PyValue, k: &str) -> R<Option<f64>> {
    match get(b, k) {
        None => Ok(None),
        Some(PyValue::Int(s)) => Ok(Some(s.parse().unwrap_or(f64::INFINITY))),
        Some(PyValue::Float(f)) => Ok(Some(*f)),
        Some(PyValue::Str(s)) => pyjson::py_float(s).map(Some).ok_or_else(|| invalid(k, "must be a number")),
        Some(_) => Err(invalid(k, "must be a number")),
    }
}

fn opt_i64(b: &PyValue, k: &str) -> R<Option<i64>> {
    match get(b, k) {
        None => Ok(None),
        Some(PyValue::Int(s)) => s.parse().map(Some).map_err(|_| invalid(k, "is out of range")),
        Some(PyValue::Float(f)) if f.fract() == 0.0 && f.is_finite() => Ok(Some(*f as i64)),
        Some(PyValue::Str(s)) => {
            pyjson::py_int(s).and_then(|i| i.parse().ok()).map(Some).ok_or_else(|| invalid(k, "must be an integer"))
        }
        Some(_) => Err(invalid(k, "must be an integer")),
    }
}

fn opt_bool(b: &PyValue, k: &str) -> R<Option<bool>> {
    match get(b, k) {
        None => Ok(None),
        Some(PyValue::Bool(x)) => Ok(Some(*x)),
        Some(PyValue::Int(s)) if s == "0" || s == "1" => Ok(Some(s == "1")),
        Some(PyValue::Str(s)) => match s.to_lowercase().as_str() {
            "true" | "1" | "yes" | "y" | "on" | "t" => Ok(Some(true)),
            "false" | "0" | "no" | "n" | "off" | "f" => Ok(Some(false)),
            _ => Err(invalid(k, "must be a boolean")),
        },
        Some(_) => Err(invalid(k, "must be a boolean")),
    }
}

fn str_or_list(b: &PyValue, k: &str) -> R<Option<Vec<String>>> {
    match get(b, k) {
        None => Ok(None),
        Some(PyValue::Str(s)) => Ok(Some(vec![s.clone()])),
        Some(PyValue::List(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| invalid(k, "must be a string or a list of strings")))
            .collect::<R<Vec<_>>>()
            .map(Some),
        Some(_) => Err(invalid(k, "must be a string or a list of strings")),
    }
}

fn id_list(b: &PyValue, k: &str) -> R<Vec<u32>> {
    match get(b, k) {
        None => Ok(vec![]),
        Some(PyValue::List(items)) => items
            .iter()
            .map(|v| {
                v.as_i64()
                    .filter(|i| *i >= 0)
                    .map(|i| i as u32)
                    .ok_or_else(|| invalid(k, "must be a list of token ids"))
            })
            .collect(),
        Some(_) => Err(invalid(k, "must be a list of token ids")),
    }
}

fn reject_unsupported(b: &PyValue) -> R<()> {
    for (k, what) in [
        ("regex", "regex-constrained decoding"),
        ("ebnf", "ebnf-constrained decoding"),
        ("json_schema", "json_schema-constrained decoding"),
        ("response_format", "response_format"),
        ("stop_regex", "stop_regex"),
        ("input_ids", "input_ids"),
        ("custom_logit_processor", "custom_logit_processor"),
    ] {
        if get(b, k).is_some() {
            if k == "response_format" && get(b, k).and_then(|r| r.get("type")).and_then(|t| t.as_str()) == Some("text")
            {
                continue;
            }
            return Err(ApiError::not_supported(what));
        }
    }
    if opt_i64(b, "n")?.unwrap_or(1) != 1 {
        return Err(ApiError::not_supported("n != 1"));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ chat request

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Named(Option<String>),
}

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub include_usage: bool,
    pub continuous_usage_stats: bool,
}

/// A validated `/v1/chat/completions` request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<PyValue>,
    pub tools: Vec<PyValue>,
    pub tool_choice: ToolChoice,
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub reasoning_effort: Option<PyValue>,
    pub chat_template_kwargs: Option<Vec<(String, PyValue)>>,
    pub separate_reasoning: bool,
    pub stream_reasoning: bool,
    pub continue_final_message: bool,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub max_tokens: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub min_tokens: i64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub logit_bias: Vec<(String, f64)>,
    pub session_id: Option<String>,
    pub rid: Option<String>,
    pub reasoning_budget: Option<i64>,
    pub reasoning_budget_soft_bias: Option<f64>,
}

const GENERIC_ROLES: [&str; 6] = ["system", "assistant", "tool", "function", "developer", "latest_reminder"];
const EFFORT_TIERS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn parse_effort(v: &PyValue) -> R<PyValue> {
    match v {
        PyValue::Str(s) if EFFORT_TIERS.contains(&s.as_str()) => Ok(v.clone()),
        PyValue::Str(s) => match pyjson::py_float(s) {
            Some(f) if (0.0..=0.99).contains(&f) => Ok(PyValue::Float(f)),
            _ => Err(invalid(
                "reasoning_effort",
                "must be one of none/minimal/low/medium/high/xhigh/max or a float in [0, 0.99]",
            )),
        },
        PyValue::Int(_) | PyValue::Float(_) => match v.as_f64() {
            Some(f) if (0.0..=0.99).contains(&f) => Ok(PyValue::Float(f)),
            _ => Err(invalid("reasoning_effort", "must be a float in [0, 0.99]")),
        },
        _ => Err(invalid("reasoning_effort", "must be a string or a number")),
    }
}

impl ChatRequest {
    /// Parse a request body (orjson semantics) and apply the reference's pydantic validators.
    pub fn parse(body: &[u8]) -> R<ChatRequest> {
        let text = std::str::from_utf8(body).map_err(|_| ApiError::bad_request("Request body is not valid UTF-8"))?;
        let mut b =
            pyjson::loads(text, Dialect::Orjson).map_err(|e| ApiError::bad_request(format!("Invalid JSON: {e}")))?;
        if !matches!(b, PyValue::Dict(_)) {
            return Err(ApiError::bad_request("Request body must be a JSON object"));
        }
        reject_unsupported(&b)?;
        // normalize_reasoning_inputs (model_validator, mode="before")
        let mut thinking: Option<bool> = None;
        if let Some(r @ PyValue::Dict(_)) = get(&b, "reasoning").cloned().as_ref() {
            let effort = get(r, "effort").or_else(|| get(r, "reasoning_effort"));
            if let Some(e) = effort {
                let e = parse_effort(e)?;
                b.set("reasoning_effort", e);
            }
            let enabled = get(r, "enabled").or_else(|| get(r, "enable")).cloned().unwrap_or(PyValue::Bool(false));
            let enabled = match enabled {
                PyValue::Str(s) => ["1", "true", "yes", "y", "on"].contains(&s.trim().to_lowercase().as_str()),
                other => other.truthy(),
            };
            if enabled {
                thinking = Some(true);
            }
        }
        let reasoning_effort = match get(&b, "reasoning_effort") {
            Some(PyValue::Bool(_)) => return Err(invalid("reasoning_effort", "must not be a boolean")),
            Some(e) => Some(parse_effort(e)?),
            None => None,
        };
        if let Some(e) = &reasoning_effort {
            thinking = Some(e.as_str() != Some("none"));
        }
        let mut ctk = match get(&b, "chat_template_kwargs") {
            Some(PyValue::Dict(kv)) => Some(kv.clone()),
            Some(_) => return Err(invalid("chat_template_kwargs", "must be an object")),
            None => None,
        };
        if let Some(t) = thinking {
            let kv = ctk.get_or_insert_with(Vec::new);
            for key in ["thinking", "enable_thinking"] {
                if !kv.iter().any(|(k, _)| k == key) {
                    kv.push((key.to_string(), PyValue::Bool(t)));
                }
            }
        }

        let messages = match get(&b, "messages") {
            Some(PyValue::List(m)) => m.clone(),
            _ => return Err(invalid("messages", "is required and must be a list")),
        };
        let mut tools = Vec::new();
        if let Some(t) = get(&b, "tools") {
            let PyValue::List(items) = t else { return Err(invalid("tools", "must be a list")) };
            for it in items {
                tools.push(validate_tool(it)?);
            }
        }
        let has_msg_tools = messages.iter().any(|m| {
            m.get("role")
                .and_then(|r| r.as_str())
                .is_some_and(|r| ["system", "developer"].contains(&r.to_lowercase().as_str()))
                && m.get("tools").is_some_and(|t| t.truthy())
        });
        let tool_choice = match get(&b, "tool_choice") {
            None => {
                if get(&b, "tools").is_none() && !has_msg_tools {
                    ToolChoice::None
                } else {
                    ToolChoice::Auto
                }
            }
            Some(PyValue::Str(s)) => match s.as_str() {
                "auto" => ToolChoice::Auto,
                "required" => ToolChoice::Required,
                "none" => ToolChoice::None,
                _ => return Err(invalid("tool_choice", "must be 'auto', 'required', 'none' or a function")),
            },
            Some(d @ PyValue::Dict(_)) => {
                if d.get("type").is_some_and(|t| t.as_str() != Some("function")) {
                    return Err(invalid("tool_choice", "type must be 'function'"));
                }
                let f = get(d, "function").ok_or_else(|| invalid("tool_choice", "needs a function"))?;
                ToolChoice::Named(opt_str(f, "name")?)
            }
            Some(_) => return Err(invalid("tool_choice", "is invalid")),
        };
        let stream_options = match get(&b, "stream_options") {
            None => None,
            Some(o @ PyValue::Dict(_)) => Some(StreamOptions {
                include_usage: opt_bool(o, "include_usage")?.unwrap_or(false),
                continuous_usage_stats: opt_bool(o, "continuous_usage_stats")?.unwrap_or(false),
            }),
            Some(_) => return Err(invalid("stream_options", "must be an object")),
        };
        let mut logit_bias = Vec::new();
        if let Some(PyValue::Dict(kv)) = get(&b, "logit_bias") {
            for (k, v) in kv {
                logit_bias
                    .push((k.clone(), v.as_f64().ok_or_else(|| invalid("logit_bias", "values must be numbers"))?));
            }
        }
        let model = opt_str(&b, "model")?.unwrap_or_else(|| "default".into());
        let rid = match get(&b, "rid") {
            Some(PyValue::Str(s)) => Some(s.clone()),
            _ => None,
        };
        Ok(ChatRequest {
            model,
            messages,
            tools,
            tool_choice,
            stream: opt_bool(&b, "stream")?.unwrap_or(false),
            stream_options,
            reasoning_effort,
            chat_template_kwargs: ctk,
            separate_reasoning: opt_bool(&b, "separate_reasoning")?.unwrap_or(true),
            stream_reasoning: opt_bool(&b, "stream_reasoning")?.unwrap_or(true),
            continue_final_message: opt_bool(&b, "continue_final_message")?.unwrap_or(false),
            skip_special_tokens: opt_bool(&b, "skip_special_tokens")?.unwrap_or(true),
            no_stop_trim: opt_bool(&b, "no_stop_trim")?.unwrap_or(false),
            ignore_eos: opt_bool(&b, "ignore_eos")?.unwrap_or(false),
            stop: str_or_list(&b, "stop")?.unwrap_or_default(),
            stop_token_ids: id_list(&b, "stop_token_ids")?,
            max_tokens: opt_i64(&b, "max_tokens")?,
            max_completion_tokens: opt_i64(&b, "max_completion_tokens")?,
            min_tokens: opt_i64(&b, "min_tokens")?.unwrap_or(0),
            temperature: opt_f64(&b, "temperature")?,
            top_p: opt_f64(&b, "top_p")?,
            top_k: opt_i64(&b, "top_k")?,
            min_p: opt_f64(&b, "min_p")?,
            presence_penalty: opt_f64(&b, "presence_penalty")?.unwrap_or(0.0),
            frequency_penalty: opt_f64(&b, "frequency_penalty")?.unwrap_or(0.0),
            repetition_penalty: opt_f64(&b, "repetition_penalty")?,
            seed: opt_i64(&b, "seed")?,
            logit_bias,
            session_id: opt_str(&b, "session_id")?,
            rid,
            reasoning_budget: opt_i64(&b, "reasoning_budget")?,
            reasoning_budget_soft_bias: opt_f64(&b, "reasoning_budget_soft_bias")?,
        })
    }

    /// `_collect_tools`: request tools plus tools attached to system / developer messages.
    pub fn all_tools(&self) -> Vec<PyValue> {
        let mut out = self.tools.clone();
        for m in &self.messages {
            let role = m.get("role").and_then(|r| r.as_str()).map(str::to_lowercase);
            if matches!(role.as_deref(), Some("system" | "developer")) {
                if let Some(PyValue::List(ts)) = get(m, "tools") {
                    for t in ts {
                        if let Ok(t) = validate_tool(t) {
                            out.push(t);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn tools_active(&self) -> bool {
        !self.all_tools().is_empty() && self.tool_choice != ToolChoice::None
    }

    pub fn is_required(&self) -> bool {
        matches!(self.tool_choice, ToolChoice::Required | ToolChoice::Named(_))
    }
}

/// `Tool(**t).model_dump()` with `normalize_json_schema_types` applied to `parameters`.
fn validate_tool(t: &PyValue) -> R<PyValue> {
    let PyValue::Dict(_) = t else { return Err(invalid("tools", "entries must be objects")) };
    let ty = match t.get("type") {
        None => "function".to_string(),
        Some(PyValue::Str(s)) => s.clone(),
        Some(_) => return Err(invalid("tools.type", "must be a string")),
    };
    let f = match t.get("function") {
        Some(f @ PyValue::Dict(_)) => f,
        _ => return Err(invalid("tools.function", "is required")),
    };
    let name = match f.get("name") {
        Some(PyValue::Str(s)) => s.clone(),
        _ => return Err(invalid("tools.function.name", "is required")),
    };
    let description = match get(f, "description") {
        None => PyValue::Null,
        Some(PyValue::Str(s)) => PyValue::Str(s.clone()),
        Some(_) => return Err(invalid("tools.function.description", "must be a string")),
    };
    let mut parameters = f.get("parameters").cloned().unwrap_or(PyValue::Null);
    if !parameters.is_null() {
        toolcall::normalize_json_schema_types(&mut parameters);
    }
    let strict = opt_bool(f, "strict")?.unwrap_or(false);
    let tool_defer = opt_bool(t, "defer_loading")?;
    let fn_defer = opt_bool(f, "defer_loading")?.or(tool_defer);
    let mut func = vec![
        ("description".to_string(), description),
        ("name".to_string(), PyValue::Str(name)),
        ("parameters".to_string(), parameters),
        ("strict".to_string(), PyValue::Bool(strict)),
    ];
    if let Some(d) = fn_defer {
        func.push(("defer_loading".into(), PyValue::Bool(d)));
    }
    Ok(PyValue::Dict(vec![
        ("type".into(), PyValue::Str(ty)),
        ("function".into(), PyValue::Dict(func)),
        ("defer_loading".into(), tool_defer.map(PyValue::Bool).unwrap_or(PyValue::Null)),
    ]))
}

// ------------------------------------------------------------------------------------------------ messages

/// pydantic validation + `model_dump()` of one content part, then sglang's openai-format processing.
/// Returns `None` for parts the processing drops (thinking parts).
fn process_part(p: &PyValue, role: &str) -> R<Option<PyValue>> {
    let ty = p.get("type").and_then(|t| t.as_str()).ok_or_else(|| invalid("content", "parts need a 'type'"))?;
    let part = |kv: Vec<(&str, PyValue)>| PyValue::Dict(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
    Ok(match ty {
        "text" => {
            let t =
                p.get("text").and_then(|t| t.as_str()).ok_or_else(|| invalid("content", "text parts need 'text'"))?;
            Some(part(vec![("type", PyValue::str("text")), ("text", PyValue::str(t))]))
        }
        "thinking" | "reasoning" => {
            if role != "assistant" {
                return Err(ApiError::bad_request("thinking content parts are only valid in assistant messages"));
            }
            None
        }
        "image_url" => {
            let iu = p.get("image_url").ok_or_else(|| invalid("content", "image_url parts need 'image_url'"))?;
            iu.get("url").and_then(|u| u.as_str()).ok_or_else(|| invalid("content", "image_url needs 'url'"))?;
            Some(part(vec![("type", PyValue::str("image"))]))
        }
        "video_url" => {
            p.get("video_url").and_then(|v| v.get("url")).ok_or_else(|| invalid("content", "video_url needs 'url'"))?;
            Some(part(vec![("type", PyValue::str("video"))]))
        }
        "audio_url" => {
            p.get("audio_url").and_then(|v| v.get("url")).ok_or_else(|| invalid("content", "audio_url needs 'url'"))?;
            Some(part(vec![("type", PyValue::str("audio"))]))
        }
        "tool_reference" => {
            let n = p
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| invalid("content", "tool_reference needs 'name'"))?;
            Some(part(vec![("type", PyValue::str("tool_reference")), ("name", PyValue::str(n))]))
        }
        _ => return Err(invalid("content", "has an unknown part type")),
    })
}

/// `ToolCall(**tc).model_dump()` with string arguments parsed (`normalize_assistant_tool_call_arguments`).
fn process_tool_call(tc: &PyValue) -> R<PyValue> {
    let f = match tc.get("function") {
        Some(f @ PyValue::Dict(_)) => f,
        _ => return Err(invalid("tool_calls.function", "is required")),
    };
    if tc.get("type").is_some_and(|t| t.as_str() != Some("function")) {
        return Err(invalid("tool_calls.type", "must be 'function'"));
    }
    let name = match get(f, "name") {
        None => PyValue::Null,
        Some(PyValue::Str(s)) => PyValue::Str(s.clone()),
        Some(_) => return Err(invalid("tool_calls.function.name", "must be a string")),
    };
    let arguments = match get(f, "arguments") {
        None => PyValue::Null,
        Some(PyValue::Str(s)) => match pyjson::loads(s, Dialect::Orjson) {
            Ok(v @ PyValue::Dict(_)) => v,
            Ok(_) => {
                return Err(ApiError::value_error("Assistant tool call function.arguments must be a JSON object."))
            }
            Err(_) => return Err(ApiError::value_error("Assistant tool call function.arguments must be valid JSON.")),
        },
        Some(d @ PyValue::Dict(_)) => d.clone(),
        Some(_) => return Err(invalid("tool_calls.function.arguments", "must be a string or an object")),
    };
    let id = match get(tc, "id") {
        Some(PyValue::Str(s)) => PyValue::Str(s.clone()),
        _ => PyValue::Null,
    };
    let index = get(tc, "index").cloned().unwrap_or(PyValue::Null);
    Ok(PyValue::Dict(vec![
        ("id".into(), id),
        ("index".into(), index),
        ("type".into(), PyValue::str("function")),
        ("function".into(), PyValue::Dict(vec![("name".into(), name), ("arguments".into(), arguments)])),
    ]))
}

/// The messages handed to the chat template, plus the `continue_final_message` prefix.
pub struct PreparedMessages {
    pub messages: Vec<PyValue>,
    pub assistant_prefix: Option<String>,
    /// `image_url` urls in message / part order (= the order the template renders their placeholders)
    pub image_urls: Vec<String>,
}

pub fn prepare_messages(req: &ChatRequest) -> R<PreparedMessages> {
    let mut out = Vec::new();
    let mut image_urls = Vec::new();
    for m in &req.messages {
        let PyValue::Dict(_) = m else { return Err(invalid("messages", "entries must be objects")) };
        let raw_role = m.get("role").and_then(|r| r.as_str()).ok_or_else(|| invalid("messages.role", "is required"))?;
        let role = if raw_role == "user" {
            "user".to_string()
        } else {
            let l = raw_role.to_lowercase();
            if !GENERIC_ROLES.contains(&l.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "'role' must be one of 'system', 'assistant', 'tool', 'function', 'developer', 'latest_reminder' (case-insensitive), or 'user'; got {raw_role:?}"
                )));
            }
            l
        };
        let mut kv: Vec<(String, PyValue)> = vec![("role".into(), PyValue::Str(role.clone()))];
        // content: None -> "" (string path keeps the key order of the pydantic dump)
        let content = match m.get("content") {
            None | Some(PyValue::Null) => {
                if role == "user" {
                    return Err(invalid("messages.content", "is required for user messages"));
                }
                PyValue::str("")
            }
            Some(PyValue::Str(s)) => PyValue::Str(s.clone()),
            Some(PyValue::List(parts)) => {
                let mut items = Vec::new();
                for p in parts {
                    if p.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                        if let Some(u) = p.get("image_url").and_then(|i| i.get("url")).and_then(|u| u.as_str()) {
                            image_urls.push(u.to_string());
                        }
                    }
                    if let Some(x) = process_part(p, &role)? {
                        items.push(x);
                    }
                }
                PyValue::List(items)
            }
            Some(_) => return Err(invalid("messages.content", "must be a string or a list of parts")),
        };
        let content_is_list = matches!(content, PyValue::List(_));
        if !content_is_list {
            kv.push(("content".into(), content.clone()));
        }
        if role != "user" {
            for key in ["tool_call_id", "name", "reasoning_content"] {
                match get(m, key) {
                    None => {}
                    Some(PyValue::Str(s)) => kv.push((key.into(), PyValue::Str(s.clone()))),
                    Some(_) => return Err(invalid(key, "must be a string")),
                }
            }
            if let Some(tcs) = get(m, "tool_calls") {
                let PyValue::List(items) = tcs else { return Err(invalid("tool_calls", "must be a list")) };
                let mut calls = Vec::new();
                for tc in items {
                    calls.push(process_tool_call(tc)?);
                }
                kv.push(("tool_calls".into(), PyValue::List(calls)));
            }
            if let Some(PyValue::List(ts)) = get(m, "tools") {
                let mut tools = Vec::new();
                for t in ts {
                    tools.push(validate_tool(t)?);
                }
                kv.push(("tools".into(), PyValue::List(tools)));
            }
        }
        let mut content = content;
        if content_is_list && role == "tool" {
            // normalize_tool_content: all-text parts -> " ".join
            if let PyValue::List(parts) = &content {
                if parts.iter().all(|p| p.get("type").and_then(|t| t.as_str()) == Some("text")) {
                    let texts: Vec<&str> =
                        parts.iter().map(|p| p.get("text").and_then(|t| t.as_str()).unwrap_or("")).collect();
                    content = PyValue::Str(texts.join(" "));
                }
            }
        }
        if content_is_list {
            kv.push(("content".into(), content));
        }
        out.push(PyValue::Dict(kv));
    }
    if out.is_empty() {
        return Err(ApiError::bad_request("Messages cannot be empty."));
    }
    // _handle_last_assistant_message
    let mut assistant_prefix = None;
    let last = out.last().unwrap();
    if last.get("role").and_then(|r| r.as_str()) == Some("assistant") {
        if let Some(PyValue::Str(c)) = last.get("content").cloned() {
            if req.continue_final_message {
                assistant_prefix = Some(c);
                out.pop();
            } else {
                *out.last_mut().unwrap() =
                    PyValue::Dict(vec![("role".into(), PyValue::str("user")), ("content".into(), PyValue::Str(c))]);
            }
        }
    }
    Ok(PreparedMessages { messages: out, assistant_prefix, image_urls })
}

/// Everything the serving path derives from a chat request before generation.
pub struct ChatPlan {
    pub prompt_ids: Vec<u32>,
    /// `image_url` urls in placeholder order (server.rs decodes them and expands `prompt_ids`)
    pub image_urls: Vec<String>,
    pub prompt_text: String,
    pub sampling: RequestSampling,
    pub require_reasoning: bool,
    pub skip_special_tokens: bool,
    pub tools: ToolSet,
    /// `continue_final_message` content (seeds the reasoning parser).
    pub previous_content: Option<String>,
    pub force_nonempty_content: bool,
}

/// `_validate_request`.
pub fn validate_chat(req: &ChatRequest, context_length: usize) -> R<()> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("Messages cannot be empty."));
    }
    let all = req.all_tools();
    match &req.tool_choice {
        ToolChoice::Required if all.is_empty() => {
            return Err(ApiError::bad_request("Tools cannot be empty if tool choice is set to required."))
        }
        ToolChoice::Named(name) => {
            if all.is_empty() {
                return Err(ApiError::bad_request("Tools cannot be empty if tool choice is set to a specific tool."));
            }
            let exists = all
                .iter()
                .any(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) == name.as_deref());
            if !exists {
                return Err(ApiError::bad_request(format!(
                    "Tool '{}' not found in tools list.",
                    name.as_deref().unwrap_or("None")
                )));
            }
        }
        _ => {}
    }
    let max_out = req.max_completion_tokens.filter(|m| *m != 0).or(req.max_tokens);
    if let Some(m) = max_out.filter(|m| *m != 0) {
        if m > context_length as i64 {
            return Err(ApiError::bad_request(format!(
                "max_completion_tokens is too large: {m}.This model supports at most {context_length} completion tokens."
            )));
        }
    }
    Ok(())
}

/// Template kwargs: `reasoning_effort` (if set) then `chat_template_kwargs`, `reasoning_effort` popped out of the
/// latter first (`_convert_to_internal_request`).
fn template_kwargs(req: &mut ChatRequest) -> Vec<(String, PyValue)> {
    if let Some(kv) = req.chat_template_kwargs.as_mut() {
        if let Some(i) = kv.iter().position(|(k, _)| k == "reasoning_effort") {
            let (_, v) = kv.remove(i);
            if !v.is_null() {
                req.reasoning_effort = Some(v);
            }
        }
    }
    let mut out = Vec::new();
    if let Some(e) = &req.reasoning_effort {
        out.push(("reasoning_effort".to_string(), e.clone()));
    }
    if let Some(kv) = &req.chat_template_kwargs {
        for (k, v) in kv {
            pyjson::dict_set(&mut out, k.clone(), v.clone());
        }
    }
    out
}

/// `_get_reasoning_from_request` for the Volundr template (toggle `enable_thinking`, default on).
pub fn reasoning_enabled(req: &ChatRequest) -> bool {
    match &req.chat_template_kwargs {
        None => true,
        Some(kv) if kv.is_empty() => true,
        Some(kv) => {
            kv.iter().find(|(k, _)| k == "enable_thinking").map(|(_, v)| v != &PyValue::Bool(false)).unwrap_or(true)
        }
    }
}

/// Render the prompt and derive sampling / parsing settings (the reference's `_process_messages` + sampling).
pub fn plan_chat(req: &mut ChatRequest, tmpl: &ChatTemplate, tok: &Tokenizer, gen: &GenerationDefaults) -> R<ChatPlan> {
    let kwargs = template_kwargs(req);
    let all_tools = req.all_tools();
    let mut skip_special_tokens = req.skip_special_tokens;
    let mut template_tools: Option<Vec<PyValue>> = None;
    if !all_tools.is_empty() && req.tool_choice != ToolChoice::None {
        skip_special_tokens = false;
        template_tools = match &req.tool_choice {
            ToolChoice::Named(name) => {
                let v: Vec<PyValue> = req
                    .tools
                    .iter()
                    .filter(|t| {
                        t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) == name.as_deref()
                    })
                    .cloned()
                    .collect();
                if v.is_empty() {
                    None
                } else {
                    Some(v)
                }
            }
            _ if !req.tools.is_empty() => Some(req.tools.clone()),
            _ => None,
        };
    }
    let prepared = prepare_messages(req)?;
    let image_urls = prepared.image_urls;
    let messages = PyValue::List(prepared.messages);
    let tools_v = template_tools.map(PyValue::List);
    let rendered = match tmpl.render(&messages, tools_v.as_ref(), true, &kwargs) {
        Ok(s) => s,
        Err(_) => {
            // retry with function-only tools (Mistral-style templates), as the reference does
            let flat = tools_v.as_ref().map(|t| {
                PyValue::List(
                    t.as_list()
                        .unwrap()
                        .iter()
                        .map(|x| x.get("function").cloned().unwrap_or_else(|| x.clone()))
                        .collect(),
                )
            });
            tmpl.render(&messages, flat.as_ref(), true, &kwargs).map_err(|e| match e {
                TemplateError::Raised(m) | TemplateError::Render(m) => ApiError::value_error(m),
            })?
        }
    };
    let mut prompt_ids = tok.encode(&rendered);
    if let Some(prefix) = &prepared.assistant_prefix {
        prompt_ids.extend(tok.encode(prefix));
    }
    let max_new = req.max_completion_tokens.filter(|m| *m != 0).or(req.max_tokens);
    let sampling = RequestSampling {
        temperature: req.temperature.or(gen.temperature).unwrap_or(1.0),
        top_p: req.top_p.or(gen.top_p).unwrap_or(1.0),
        top_k: req.top_k.or(gen.top_k).unwrap_or(-1),
        min_p: req.min_p.or(gen.min_p).unwrap_or(0.0),
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: req.repetition_penalty.or(gen.repetition_penalty).unwrap_or(1.0),
        min_new_tokens: req.min_tokens,
        max_new_tokens: max_new,
        seed: req.seed,
        logit_bias: req.logit_bias.clone(),
    };
    let previous_content = if req.continue_final_message {
        req.messages
            .last()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(str::to_string)
    } else {
        None
    };
    let force_nonempty_content = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|kv| kv.iter().find(|(k, _)| k == "force_nonempty_content"))
        .is_some_and(|(_, v)| v == &PyValue::Bool(true));
    Ok(ChatPlan {
        prompt_ids,
        image_urls,
        prompt_text: rendered,
        sampling,
        require_reasoning: reasoning_enabled(req),
        skip_special_tokens,
        tools: ToolSet::from_openai(&all_tools),
        previous_content,
        force_nonempty_content,
    })
}

// ------------------------------------------------------------------------------------------------ responses

fn q(s: &str) -> String {
    pyjson::quote(s, false)
}

fn q_opt(s: Option<&str>) -> String {
    s.map(q).unwrap_or_else(|| "null".into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
}

impl Usage {
    /// `UsageInfo` as pydantic dumps it (`prompt_tokens_details` null).
    pub fn json(&self) -> String {
        format!(
            "{{\"prompt_tokens\":{},\"total_tokens\":{},\"completion_tokens\":{},\"prompt_tokens_details\":null,\"reasoning_tokens\":{}}}",
            self.prompt_tokens,
            self.prompt_tokens + self.completion_tokens,
            self.completion_tokens,
            self.reasoning_tokens
        )
    }
    /// `model_dump_json(exclude_none=True)` (completions' final usage chunk).
    pub fn json_exclude_none(&self) -> String {
        format!(
            "{{\"prompt_tokens\":{},\"total_tokens\":{},\"completion_tokens\":{},\"reasoning_tokens\":{}}}",
            self.prompt_tokens,
            self.prompt_tokens + self.completion_tokens,
            self.completion_tokens,
            self.reasoning_tokens
        )
    }
}

/// `matched_stop` as JSON.
pub fn matched_json(f: Option<&FinishReason>) -> String {
    match f {
        Some(FinishReason::MatchedToken(t)) => t.to_string(),
        Some(FinishReason::MatchedStr(s)) => q(s),
        _ => "null".into(),
    }
}

pub fn new_tool_call_id() -> String {
    format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..24])
}

pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Header fields shared by every chunk / response.
#[derive(Debug, Clone)]
pub struct RespMeta {
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl RespMeta {
    fn head(&self, object: &str) -> String {
        format!(
            "{{\"id\":{},\"object\":\"{object}\",\"created\":{},\"model\":{}",
            q(&self.id),
            self.created,
            q(&self.model)
        )
    }
}

/// `build_sse_content` (msgspec): `reasoning_content` always present, `role` / `content` only when set, `usage`
/// only when set.
pub fn sse_msgspec(
    m: &RespMeta,
    role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
    finish_reason: Option<&str>,
    matched_stop: &str,
    usage: Option<&Usage>,
) -> String {
    let mut s = m.head("chat.completion.chunk");
    s.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
    s.push_str(&q_opt(reasoning));
    if let Some(r) = role {
        let _ = write!(s, ",\"role\":{}", q(r));
    }
    if let Some(c) = content {
        let _ = write!(s, ",\"content\":{}", q(c));
    }
    let _ = write!(
        s,
        "}},\"logprobs\":null,\"finish_reason\":{},\"matched_stop\":{}}}]",
        q_opt(finish_reason),
        matched_stop
    );
    if let Some(u) = usage {
        let _ = write!(s, ",\"usage\":{}", u.json());
    }
    s.push('}');
    format!("data: {s}\n\n")
}

/// A tool call as pydantic dumps `ToolCall`.
fn tool_call_json(id: Option<&str>, index: Option<i64>, name: Option<&str>, arguments: Option<&str>) -> String {
    format!(
        "{{\"id\":{},\"index\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":{}}}}}",
        q_opt(id),
        index.map(|i| i.to_string()).unwrap_or_else(|| "null".into()),
        q_opt(name),
        q_opt(arguments)
    )
}

/// `ChatCompletionStreamResponse.model_dump_json()` with a content or tool-call delta.
pub fn sse_pydantic(m: &RespMeta, content: Option<&str>, tool_calls: Option<&str>, usage: Option<&Usage>) -> String {
    let mut s = m.head("chat.completion.chunk");
    let _ = write!(
        s,
        ",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":null,\"content\":{},\"reasoning_content\":null,\"tool_calls\":{}}},\"logprobs\":null,\"finish_reason\":null,\"matched_stop\":null}}],\"usage\":{}}}",
        q_opt(content),
        tool_calls.map(|t| format!("[{t}]")).unwrap_or_else(|| "null".into()),
        usage.map(|u| u.json()).unwrap_or_else(|| "null".into())
    );
    format!("data: {s}\n\n")
}

pub fn sse_usage(m: &RespMeta, usage: &Usage) -> String {
    format!("data: {},\"choices\":[],\"usage\":{}}}\n\n", m.head("chat.completion.chunk"), usage.json())
}

pub const SSE_DONE: &str = "data: [DONE]\n\n";

/// Streaming state for one chat completion (`_generate_chat_stream` / `_generate_stream_content`).
pub struct ChatStream {
    pub meta: RespMeta,
    reasoning: Option<ReasoningParser>,
    tools: Option<(ToolSet, ToolCallStreamParser)>,
    first: bool,
    has_tool_calls: bool,
    continuous_usage: bool,
    include_usage: bool,
    last_usage: Usage,
}

impl ChatStream {
    pub fn new(meta: RespMeta, req: &ChatRequest, plan: &ChatPlan) -> Self {
        let reasoning = req.separate_reasoning.then(|| {
            ReasoningParser::new(req.stream_reasoning, plan.require_reasoning, plan.previous_content.as_deref())
                .with_force_nonempty_content(plan.force_nonempty_content)
        });
        let tools = (req.tool_choice != ToolChoice::None && !plan.tools.is_empty())
            .then(|| (plan.tools.clone(), ToolCallStreamParser::new()));
        let (include_usage, continuous_usage) = match &req.stream_options {
            Some(o) => (o.include_usage, o.continuous_usage_stats),
            None => (false, false),
        };
        ChatStream {
            meta,
            reasoning,
            tools,
            first: true,
            has_tool_calls: false,
            continuous_usage,
            include_usage,
            last_usage: Usage { prompt_tokens: 0, completion_tokens: 0, reasoning_tokens: 0 },
        }
    }

    /// SSE chunks for one engine output.
    pub fn on_output(
        &mut self,
        delta: &str,
        finish: Option<&FinishReason>,
        usage: Usage,
    ) -> Result<Vec<String>, ApiError> {
        let mut out = Vec::new();
        self.last_usage = usage;
        let cu = self.continuous_usage.then_some(usage);
        if self.first {
            self.first = false;
            out.push(sse_msgspec(&self.meta, Some("assistant"), Some(""), None, None, "null", None));
        }
        let mut delta = delta.to_string();
        if let Some(rp) = self.reasoning.as_mut() {
            let r = rp.parse_stream_chunk(&delta);
            let (mut rt, mut nt) = (r.reasoning_text, r.normal_text);
            if finish.is_some_and(|f| !matches!(f, FinishReason::Abort(_))) {
                let e = rp.finish();
                rt.push_str(&e.reasoning_text);
                nt.push_str(&e.normal_text);
            }
            if !rt.is_empty() {
                out.push(sse_msgspec(&self.meta, None, None, Some(&rt), None, "null", cu.as_ref()));
            }
            delta = nt;
        }
        if let Some((tools, parser)) = self.tools.as_mut() {
            let r = parser.parse(&delta, tools).map_err(|e| ApiError::value_error(e.0))?;
            if !r.normal_text.is_empty() {
                out.push(sse_pydantic(&self.meta, Some(&r.normal_text), None, cu.as_ref()));
            }
            for c in r.calls {
                self.has_tool_calls = true;
                let (id, name) = match &c.name {
                    Some(n) => (Some(new_tool_call_id()), Some(n.clone())),
                    None => (None, None),
                };
                let tc = tool_call_json(id.as_deref(), Some(c.tool_index), name.as_deref(), Some(&c.parameters));
                out.push(sse_pydantic(&self.meta, None, Some(&tc), cu.as_ref()));
            }
        } else if !delta.is_empty() {
            out.push(sse_msgspec(&self.meta, None, Some(&delta), None, None, "null", cu.as_ref()));
        }
        Ok(out)
    }

    /// Finish chunk, optional usage chunk, `[DONE]`.
    pub fn on_finish(&self, finish: &FinishReason) -> Vec<String> {
        let mut out = Vec::new();
        let mut reason = finish.kind();
        if self.has_tool_calls && reason == "stop" {
            reason = "tool_calls";
        }
        out.push(sse_msgspec(&self.meta, None, None, None, Some(reason), &matched_json(Some(finish)), None));
        if self.include_usage {
            out.push(sse_usage(&self.meta, &self.last_usage));
        }
        out.push(SSE_DONE.to_string());
        out
    }
}

/// `_process_tool_calls` result.
struct ToolOutcome {
    calls: Option<Vec<(String, i64, String, String)>>, // id, index, name, arguments
    text: String,
    finish_kind: &'static str,
    matched: String,
}

fn process_tool_calls(text: &str, tools: &ToolSet, finish: &FinishReason, required: bool) -> ToolOutcome {
    let mut finish_kind = finish.kind();
    let mut matched = matched_json(Some(finish));
    if !tools.is_empty() && toolcall::has_tool_call(text) {
        let r = toolcall::detect_and_parse(text, tools);
        if r.calls.is_empty() {
            return ToolOutcome { calls: None, text: r.normal_text, finish_kind, matched };
        }
        let calls = r
            .calls
            .into_iter()
            .map(|c: ToolCallItem| (new_tool_call_id(), c.tool_index, c.name.unwrap_or_default(), c.parameters))
            .collect();
        if finish_kind == "stop" {
            finish_kind = "tool_calls";
            matched = "null".into();
        }
        return ToolOutcome { calls: Some(calls), text: r.normal_text, finish_kind, matched };
    }
    if required {
        // json_schema fallback: output is a JSON array of {"name", "parameters"}
        let original = finish_kind;
        if finish_kind == "stop" {
            finish_kind = "tool_calls";
            matched = "null".into();
        }
        if let Ok(PyValue::List(items)) = pyjson::loads(text, Dialect::Orjson) {
            let mut calls = Vec::new();
            let mut ok = true;
            for (i, it) in items.iter().enumerate() {
                match (it.get("name"), it.get("parameters")) {
                    (Some(PyValue::Str(n)), Some(p)) => {
                        calls.push((new_tool_call_id(), i as i64, n.clone(), p.dumps(false)))
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return ToolOutcome { calls: Some(calls), text: String::new(), finish_kind, matched };
            }
        }
        return ToolOutcome { calls: None, text: text.to_string(), finish_kind: original, matched };
    }
    ToolOutcome { calls: None, text: text.to_string(), finish_kind, matched }
}

/// Non-streaming chat response body (`_build_chat_response`, serialised like FastAPI's `JSONResponse`).
pub fn chat_response(
    meta: &RespMeta,
    req: &ChatRequest,
    plan: &ChatPlan,
    text: &str,
    finish: &FinishReason,
    usage: Usage,
) -> String {
    let mut text = text.to_string();
    let mut reasoning = String::new();
    if req.separate_reasoning {
        let p = ReasoningParser::new(false, plan.require_reasoning, plan.previous_content.as_deref())
            .with_force_nonempty_content(plan.force_nonempty_content);
        let r = p.parse_non_stream(&text);
        reasoning = r.reasoning_text;
        text = r.normal_text;
    }
    let mut finish_kind = finish.kind();
    let mut matched = matched_json(Some(finish));
    let mut tool_calls = "null".to_string();
    if req.tool_choice != ToolChoice::None && !plan.tools.is_empty() {
        let o = process_tool_calls(&text, &plan.tools, finish, req.is_required());
        text = o.text;
        finish_kind = o.finish_kind;
        matched = o.matched;
        if let Some(calls) = o.calls {
            let items: Vec<String> =
                calls.iter().map(|(id, i, n, a)| tool_call_json(Some(id), Some(*i), Some(n), Some(a))).collect();
            tool_calls = format!("[{}]", items.join(","));
        }
    }
    let mut s = meta.head("chat.completion");
    let _ = write!(
        s,
        ",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{},\"reasoning_content\":{},\"tool_calls\":{}}},\"logprobs\":null,\"finish_reason\":{},\"matched_stop\":{}}}],\"usage\":{},\"metadata\":{{\"weight_version\":\"default\"}}}}",
        q(&text),
        if reasoning.is_empty() { "null".to_string() } else { q(&reasoning) },
        tool_calls,
        q(finish_kind),
        matched,
        usage.json()
    );
    s
}

// ------------------------------------------------------------------------------------------------ completions

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    pub max_tokens: i64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
    pub min_tokens: i64,
    pub seed: Option<i64>,
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub session_id: Option<String>,
    pub rid: Option<String>,
    pub logit_bias: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub enum CompletionPrompt {
    Text(String),
    Ids(Vec<u32>),
}

impl CompletionRequest {
    pub fn parse(body: &[u8]) -> R<CompletionRequest> {
        let text = std::str::from_utf8(body).map_err(|_| ApiError::bad_request("Request body is not valid UTF-8"))?;
        let b =
            pyjson::loads(text, Dialect::Orjson).map_err(|e| ApiError::bad_request(format!("Invalid JSON: {e}")))?;
        reject_unsupported(&b)?;
        if opt_bool(&b, "echo")?.unwrap_or(false) {
            return Err(ApiError::not_supported("echo"));
        }
        if get(&b, "logprobs").is_some() {
            return Err(ApiError::not_supported("logprobs"));
        }
        if get(&b, "suffix").is_some() {
            return Err(ApiError::not_supported("suffix"));
        }
        let prompt = match get(&b, "prompt") {
            Some(PyValue::Str(s)) => CompletionPrompt::Text(s.clone()),
            Some(PyValue::List(items)) if items.len() == 1 && items[0].as_str().is_some() => {
                CompletionPrompt::Text(items[0].as_str().unwrap().to_string())
            }
            Some(PyValue::List(items)) if items.iter().all(|v| v.as_i64().is_some()) => {
                CompletionPrompt::Ids(items.iter().map(|v| v.as_i64().unwrap() as u32).collect())
            }
            Some(PyValue::List(_)) => return Err(ApiError::not_supported("batched prompts")),
            _ => return Err(invalid("prompt", "is required")),
        };
        match &prompt {
            CompletionPrompt::Text(s) if s.is_empty() => return Err(ApiError::bad_request("Prompt cannot be empty")),
            CompletionPrompt::Ids(v) if v.is_empty() => return Err(ApiError::bad_request("Prompt cannot be empty")),
            _ => {}
        }
        let max_tokens = opt_i64(&b, "max_tokens")?.unwrap_or(16);
        if max_tokens <= 0 {
            return Err(ApiError::bad_request("max_tokens must be positive"));
        }
        let mut logit_bias = Vec::new();
        if let Some(PyValue::Dict(kv)) = get(&b, "logit_bias") {
            for (k, v) in kv {
                logit_bias
                    .push((k.clone(), v.as_f64().ok_or_else(|| invalid("logit_bias", "values must be numbers"))?));
            }
        }
        Ok(CompletionRequest {
            model: opt_str(&b, "model")?.unwrap_or_else(|| "default".into()),
            prompt,
            max_tokens,
            temperature: opt_f64(&b, "temperature")?.unwrap_or(1.0),
            top_p: opt_f64(&b, "top_p")?.unwrap_or(1.0),
            top_k: opt_i64(&b, "top_k")?.unwrap_or(-1),
            min_p: opt_f64(&b, "min_p")?.unwrap_or(0.0),
            presence_penalty: opt_f64(&b, "presence_penalty")?.unwrap_or(0.0),
            frequency_penalty: opt_f64(&b, "frequency_penalty")?.unwrap_or(0.0),
            repetition_penalty: opt_f64(&b, "repetition_penalty")?.unwrap_or(1.0),
            min_tokens: opt_i64(&b, "min_tokens")?.unwrap_or(0),
            seed: opt_i64(&b, "seed")?,
            stop: str_or_list(&b, "stop")?.unwrap_or_default(),
            stop_token_ids: id_list(&b, "stop_token_ids")?,
            stream: opt_bool(&b, "stream")?.unwrap_or(false),
            stream_options: match get(&b, "stream_options") {
                Some(o @ PyValue::Dict(_)) => Some(StreamOptions {
                    include_usage: opt_bool(o, "include_usage")?.unwrap_or(false),
                    continuous_usage_stats: opt_bool(o, "continuous_usage_stats")?.unwrap_or(false),
                }),
                _ => None,
            },
            ignore_eos: opt_bool(&b, "ignore_eos")?.unwrap_or(false),
            skip_special_tokens: opt_bool(&b, "skip_special_tokens")?.unwrap_or(true),
            no_stop_trim: opt_bool(&b, "no_stop_trim")?.unwrap_or(false),
            session_id: opt_str(&b, "session_id")?,
            rid: match get(&b, "rid") {
                Some(PyValue::Str(s)) => Some(s.clone()),
                _ => None,
            },
            logit_bias,
        })
    }

    pub fn sampling(&self) -> RequestSampling {
        RequestSampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            min_new_tokens: self.min_tokens,
            max_new_tokens: Some(self.max_tokens),
            seed: self.seed,
            logit_bias: self.logit_bias.clone(),
        }
    }
}

fn completion_choice(text: &str, finish: Option<&FinishReason>) -> String {
    format!(
        "{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":{},\"matched_stop\":{}}}",
        q(text),
        q_opt(finish.map(|f| f.kind())),
        matched_json(finish)
    )
}

pub fn completion_chunk(m: &RespMeta, text: &str, finish: Option<&FinishReason>, usage: Option<&Usage>) -> String {
    format!(
        "data: {},\"choices\":[{}],\"usage\":{}}}\n\n",
        m.head("text_completion"),
        completion_choice(text, finish),
        usage.map(|u| u.json()).unwrap_or_else(|| "null".into())
    )
}

pub fn completion_usage_chunk(m: &RespMeta, usage: &Usage) -> String {
    format!("data: {},\"choices\":[],\"usage\":{}}}\n\n", m.head("text_completion"), usage.json_exclude_none())
}

pub fn completion_response(m: &RespMeta, text: &str, finish: &FinishReason, usage: Usage) -> String {
    format!(
        "{},\"choices\":[{}],\"usage\":{},\"metadata\":{{\"weight_version\":\"default\"}}}}",
        m.head("text_completion"),
        completion_choice(text, Some(finish)),
        usage.json()
    )
}

/// `/v1/models` body (`ModelList` via orjson).
pub fn models_json(name: &str, created: u64, max_model_len: usize) -> String {
    format!("{{\"object\":\"list\",\"data\":[{}]}}", model_card_json(name, created, max_model_len))
}

pub fn model_card_json(name: &str, created: u64, max_model_len: usize) -> String {
    format!(
        "{{\"id\":{},\"object\":\"model\",\"created\":{created},\"agens_version\":\"{}\",\"owned_by\":\"sglang\",\"root\":{},\"parent\":null,\"max_model_len\":{max_model_len}}}",
        q(name),
        env!("CARGO_PKG_VERSION"),
        q(name)
    )
}
