//! The model interface. The server owns tokenization, templates, stop conditions, detokenization and parsing;
//! a backend only turns prompt ids into sampled token ids.
//!
//! Contract for implementers (the CUDA engine):
//! * [`Backend::generate`] starts one request and returns at once with a [`TokenStream`]. Run the request on your
//!   own thread(s) and push each decode step's accepted tokens with `blocking_send` (one id per step normally,
//!   several with speculative decoding).
//! * Stop when `max_new_tokens` ids were produced, after emitting one of `stop_token_ids` (unless `ignore_eos`),
//!   or as soon as the receiver is dropped (`blocking_send` returns `Err`): the server drops it when a stop string
//!   matches, when the client disconnects, or when it has all the tokens it needs.
//! * Apply the sampling fields of [`SamplingParams`] (temperature / top-k / top-p / min-p / penalties / seed /
//!   logit bias) and set the logits of `suppress_tokens` to -inf. `top_k == Some(1)` means greedy.
//! * `session` is a hint that consecutive requests belong to one conversation, so per-conversation state (KV /
//!   recurrent state) can be kept and the shared prompt prefix skipped. The prompt is always the full prompt;
//!   the backend decides how much of it it already has. Sessions are evicted at the backend's discretion.
//! * Report failures as `Err(BackendError)` items; the server ends the request with an error.

use crate::sampling::SamplingParams;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Per-conversation hint (request field `session_id` or header `X-Session-Id`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionHint {
    pub session_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

/// Tokens of one decode step (or an error). Dropping the stream cancels the request.
pub type TokenStream = mpsc::Receiver<Result<Vec<u32>, BackendError>>;

pub trait Backend: Send + Sync + 'static {
    /// Start generating for `prompt_ids` (the full rendered prompt). Must not block.
    fn generate(&self, prompt_ids: Vec<u32>, sampling: SamplingParams, session: Option<SessionHint>) -> TokenStream;

    /// Longest prompt + completion the backend accepts (`max_model_len` in `/v1/models`).
    fn context_length(&self) -> usize;
}

/// A recorded request, for tests.
#[derive(Debug, Clone)]
pub struct MockCall {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub session: Option<SessionHint>,
}

type Script = Box<dyn FnMut(&[u32]) -> Vec<u32> + Send>;

/// Test backend: replays a fixed token script (one id per step), honouring `max_new_tokens`, stop ids and
/// cancellation like a real backend, and records every call.
pub struct MockBackend {
    script: std::sync::Mutex<Script>,
    calls: Arc<std::sync::Mutex<Vec<MockCall>>>,
    context_length: usize,
    /// Tokens per step (>1 imitates speculative decoding).
    pub step_size: usize,
}

impl MockBackend {
    /// Always answer with `tokens`.
    pub fn fixed(tokens: Vec<u32>) -> Self {
        Self::with_script(move |_| tokens.clone())
    }

    /// Answer with `script(prompt_ids)`.
    pub fn with_script(script: impl FnMut(&[u32]) -> Vec<u32> + Send + 'static) -> Self {
        MockBackend {
            script: std::sync::Mutex::new(Box::new(script)),
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
            context_length: 262_144,
            step_size: 1,
        }
    }

    pub fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().unwrap().clone()
    }
}

impl Backend for MockBackend {
    fn generate(&self, prompt_ids: Vec<u32>, sampling: SamplingParams, session: Option<SessionHint>) -> TokenStream {
        let tokens = (self.script.lock().unwrap())(&prompt_ids);
        self.calls.lock().unwrap().push(MockCall { prompt_ids, sampling: sampling.clone(), session });
        let (tx, rx) = mpsc::channel(16);
        let step = self.step_size.max(1);
        tokio::spawn(async move {
            let mut produced = 0usize;
            let mut i = 0usize;
            while i < tokens.len() && produced < sampling.max_new_tokens {
                let n = step.min(tokens.len() - i).min(sampling.max_new_tokens - produced);
                let mut chunk = Vec::with_capacity(n);
                let mut stop = false;
                for &t in &tokens[i..i + n] {
                    chunk.push(t);
                    if !sampling.ignore_eos && sampling.stop_token_ids.contains(&t) {
                        stop = true;
                        break;
                    }
                }
                i += chunk.len();
                produced += chunk.len();
                if tx.send(Ok(chunk)).await.is_err() || stop {
                    break;
                }
            }
        });
        rx
    }

    fn context_length(&self) -> usize {
        self.context_length
    }
}
