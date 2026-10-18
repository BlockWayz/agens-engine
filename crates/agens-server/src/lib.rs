//! Agens Engine serving front end: OpenAI-compatible HTTP API whose output matches the sglang image as we
//! ship it (`--reasoning-parser agens --tool-call-parser agens`, chat template from the model directory).
//!
//! Layers, bottom up:
//! * [`pyjson`]    Python-compatible JSON (key order, int/float, `json.dumps` formatting, `literal_eval`)
//! * [`tokenizer`] `tokenizer.json` + the reference incremental detokenizer
//! * [`template`]  `chat_template.jinja` rendered like transformers' `apply_chat_template`
//! * [`reasoning`] / [`toolcall`]  ports of the sglang `agens` reasoning and tool-call parsers
//! * [`sampling`]  generation defaults (`generation_config.json`) and per-request sampling params
//! * [`backend`]   the model interface ([`backend::Backend`]) and a mock for tests
//! * [`engine`]    request lifecycle: stop strings / EOS / max tokens, detokenization, usage counts
//! * [`openai`]    request normalisation (sglang's chat path) and response / SSE serialisation
//! * [`server`]    axum routes
//! * [`vision`]    image inputs: data-URL decoding, sizing, placeholder expansion, MRoPE positions

pub mod backend;
pub mod engine;
pub mod openai;
pub mod pyjson;
pub mod reasoning;
pub mod sampling;
pub mod server;
pub mod template;
pub mod tokenizer;
pub mod toolcall;
pub mod vision;
