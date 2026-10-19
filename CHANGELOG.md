# Changelog

## 1.1.1

### Fixes

- **`16gb` / `12gb` profiles: the server crashed with out of memory in a very long generation.** With the INT8
  dense-layer KV cache, the engine captures a new CUDA graph for every 8,192 tokens of context and kept all of them
  until the request ended. A generation of about 100K+ tokens (for example a repetition loop) filled the 16 GiB cap,
  `cudaGraphInstantiate` failed, and the server exited. The engine now keeps only the current graph, so memory stays
  flat however long the generation runs.
- **A failed engine step no longer stops the server.** If a step still runs out of device memory, the engine frees
  its cached CUDA graphs and retries, then runs the step without a graph. If that also fails, only that request ends
  with an error, and the server keeps serving. This applies to every profile and both modes. Only an unrecoverable
  CUDA error ends the process.
- If the KV pool that fits under the budget cannot hold the requested `--wctx`, the context is now cut to what the pool
  holds, and a log line (`--wctx N clamped to M`) says so. Before, the server started at the full context and a
  request that ran past the pool failed inside the engine. No shipped profile is affected: each one's pool holds its
  full context.
- A prefill that runs out of device memory now fails cleanly. Before, the linear-attention (KDA) prefill could launch on a
  temporary buffer whose allocation had failed, which made the CUDA context unusable.
- A request whose client disconnects now stops at the next decode step. Before, it stopped at its next output, which
  for non-streaming requests came every 50 tokens.

### New

- **Default generation cap:** a request without `max_tokens` now generates at most **32,768** tokens. Before, the only
  limit was the context (about 258K tokens). Set it with `agens serve --max-tokens-default N`, or `0` for no cap. An
  explicit `max_tokens` is still honoured up to the context.
- **Thinking repetition guard** (on by default): inside the thinking block, a 32-token sequence repeated 20 times
  within the last 8,192 tokens closes the block and the model answers, as a reasoning budget would. This is a
  **mitigation** for the long-context reasoning loops seen with the 16 GB build, **not a fix** for them. Set it with
  `agens serve --think-guard N,K[,W]` or turn it off with `--think-guard off`. The server logs each time it fires.

### Docs

- README: known limitation of the 16 GB build with thinking on over long documents.

## 1.1.0

### New

- **Image input.** `/v1/chat/completions` accepts images as `image_url` content parts with base64 `data:` URLs (PNG,
  JPEG, WebP, GIF, BMP), several per message and across turns. On by default in the `24gb` and `48gb` profiles, off in
  `16gb`; `--images on|off` overrides. With `--images on` the `16gb` profile runs a ~225K-token context (262K with
  images off). Conversation prefix reuse, sessions and speculative decoding work on image prompts.
- **Speculative sampling.** Speculative decoding now runs for sampled requests too: temperature with top-k, top-p and
  min-p, in one-conversation mode, in server mode (batched, each request on its own sampling parameters and random
  stream) and with context-copy drafting. The accept rule samples the same distribution as plain sampling. Requests
  that send no sampling fields use the model's generation config (temperature 1.0, top-k 20, top-p 0.95) and are now
  speculated as well; in 1.0.0 every sampled request decoded one token per step. Repetition / frequency / presence
  penalties, logit bias and the reasoning budget still decode without speculation.
- **Prebuilt kernels for compute capability 9.0** (validated) and **8.6** (built, not yet validated on hardware), in
  addition to 8.9 and 12.0.

### Fixes

- The `24gb` profile did not start with the 4-bit release weights and the drafter ("budget 23 GiB leaves no room for
  a KV pool"). It now plans 4 conversation slots.
- Server mode: greedy replies with the drafter could differ from the same requests without it. Server mode now uses
  one set of numerics with and without speculation; replies are identical with and without the drafter at every
  concurrency level.
- Server mode with tiered KV: the batched verify step wrote its logits past their buffer at 7 or more concurrent
  sequences (the `48gb` profile serves 16). The buffer is now sized for the largest batch.
- Batched verify with the INT8 dense-layer KV cache could replay a CUDA graph captured for a different context length
  and skip the last attention tile past 16K tokens of context. The shipped profiles do not combine the two (INT8
  dense-layer KV is used in one-conversation mode only).
- Sampling seeds: small seeds (1, 2, 3, ...) gave correlated first samples across requests.
