# agens-server

OpenAI-compatible HTTP front end of Agens Engine 1.0. It renders prompts, samples through a `Backend`, and
parses reasoning and tool calls. Its responses are byte-compatible with the serving image we ship today. That
image runs with `--reasoning-parser agens --tool-call-parser agens` and takes its chat template from the model
directory. Everything here runs on the CPU. The model itself sits behind the `Backend` trait.

```
cargo run -p agens-server -- --model-dir <dir> --port 30000 --served-model-name volundr
```

`<dir>` needs `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` and
`config.json`; weights are not read here. Until the CUDA backend is wired in, `--backend mock` (the default)
answers with a canned reply.

## Endpoints

| route | notes |
|---|---|
| `POST /v1/chat/completions` | stream (SSE) and non-stream; `reasoning_content`, `tool_calls`, usage incl. `reasoning_tokens` |
| `POST /v1/completions` | stream and non-stream, single prompt (text or token ids) |
| `GET /v1/models`, `GET /v1/models/{name}` | one model card |
| `GET /health`, `GET /health_generate` | 200, empty body |

Request fields follow the reference:
- Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`.
- Length and stopping: `max_tokens` / `max_completion_tokens`, `min_tokens`, `stop`, `stop_token_ids`, `ignore_eos`, `no_stop_trim`, `skip_special_tokens`.
- Streaming: `stream_options.include_usage` / `continuous_usage_stats`.
- Reasoning: `separate_reasoning`, `stream_reasoning`, `reasoning_effort` (or `reasoning.effort`), and `chat_template_kwargs` (`enable_thinking`, `reasoning_effort` low/medium/xhigh, `preserve_thinking`, `force_nonempty_content`, ...).
- Tools and messages: `tools`, `tool_choice`, `continue_final_message`.
- Extensions: `reasoning_budget` and `rid`.

Defaults come from `generation_config.json`: temperature 1.0, top-k 20 and top-p 0.95 apply when a request leaves
them unset. `/v1/completions` uses the OpenAI defaults, as the reference does. Token ids that the tokenizer does not define (248077..248319,
padding rows of the embedding / lm_head) plus any `suppress_tokens` are passed to the backend as
`SamplingParams::suppress_tokens` (logits -inf).

### Session hint

A request can name its conversation with the body field `session_id` or the header `X-Session-Id` (the header
wins). The server passes it to the backend as `SessionHint`. The prompt is still the full prompt; the backend
may keep per-conversation state (KV / recurrent state) under that id and skip the prefix it already has.
Responses do not depend on it.

## Backend contract

```rust
pub trait Backend: Send + Sync + 'static {
    fn generate(&self, prompt_ids: Vec<u32>, sampling: SamplingParams, session: Option<SessionHint>) -> TokenStream;
    fn context_length(&self) -> usize;
}
pub type TokenStream = tokio::sync::mpsc::Receiver<Result<Vec<u32>, BackendError>>;
```

`generate` returns immediately. Push each decode step's accepted ids with `blocking_send`: one id per step,
or several with speculative decoding. Stop when any of these happens:
- `max_new_tokens` ids are out;
- a `stop_token_ids` id was emitted (EOS ids plus request ids; the list is empty with `ignore_eos`);
- the receiver is dropped (stop string matched, client gone).

The server owns stop strings, detokenization, usage counts and parsing. `top_k == Some(1)` means greedy.
`MockBackend` replays a token script and records calls, for tests.

## Parity tests

`cargo test -p agens-server` checks the crate against fixtures produced by the reference image's own
transformers / sglang code (`tests/reference/gen_ref.py`, run on CPU by `scripts/server_ref14.sh gen`):

| test | what |
|---|---|
| `template_parity` | 51 `apply_chat_template` cases, rendered text and token ids |
| `tokenizer_parity` | encode (special tokens, Cantonese / Chinese, emoji, combining marks, real transcripts), decode with / without specials, streaming detokenizer |
| `request_parity` | OpenAI request -> prompt ids, sampling, reasoning toggle, error messages |
| `parser_parity` | reasoning + tool-call parsers, non-stream and 3 chunkings x 2 reasoning modes |
| `e2e_parity` | full SSE streams and JSON bodies (chat incl. multi-token steps, stop strings, errors; completions) |

The tokenizer-dependent tests need `tokenizer.json`. They look in `$AGENS_MODEL_DIR`, then in `refs/volundr/`,
which `scripts/server_ref14.sh fetch` fills. Without it they print `SKIP` and pass.

Note on tokenization: the reference tokenizes through transformers. For the tokenizer class named in
`tokenizer_config.json`, transformers replaces the `tokenizer.json` pre-tokenizer regex with its own, which has
no `\p{M}`. Combining marks (Devanagari, Thai, Vietnamese, ...) therefore split differently from what
`tokenizer.json` says. `Tokenizer::from_model_dir` applies the same override. Anything else that tokenizes for
this model should do the same.

## Not supported / not byte-identical

- Rejected with 400:
  - constrained decoding (`response_format`, `regex`, `ebnf`, `json_schema`);
  - `n > 1`, `logprobs`, `stop_regex`, `input_ids`, `custom_logit_processor`;
  - on completions: `echo` / `suffix` and batched prompts.
- `tool_choice: "required"` or a named tool: the reference constrains decoding with a JSON-schema grammar. We do
  not; output is parsed the same way (native format first, then the JSON-array fallback).
- Pydantic validation errors (malformed fields) return the same 400 error shape, but the message text differs.
  Tool `parameters` are not meta-validated against JSON Schema 2020-12.
- Streaming chunk boundaries: one chunk per decode step. The reference may coalesce steps when a client reads
  slowly; the concatenated content is the same.
- Tool argument conversion via `ast.literal_eval` covers literals that JSON can represent. Sets, bytes, complex
  numbers and non-string dict keys stay strings (the reference fails on them in `json.dumps`).
- `/health_generate` does not run a one-token generation.
