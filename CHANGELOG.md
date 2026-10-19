# Changelog

## 1.2.0

1.2.0 includes every 1.1.1 fix and limit below.

### New

- **MTP drafting.** `--mtp DIR` loads the model's multi-token-prediction (MTP) head, about one decoder layer that
  drafts from the model's own last hidden state, as a third draft source next to context copy and the companion
  drafter: in each speculative step the context copy fills the slots it covers, the MTP head the next ones (up to 4
  with the drafter, all 7 without it) and the drafter the rest. It works in one-conversation and server mode, greedy
  and sampled; greedy replies stay identical to plain greedy decoding and sampled replies follow the same
  distribution. The head is stored at 4 bits at load (0.22 GB, plus 0.07 GB of context buffers) and each draft step
  scores a 32K-token shortlist of the output layer. On a 16 GB budget, short prompts: chat 108.0 vs 98.4 tok/s, code
  140.7 vs 128.8, tool calls 124.1 vs 122.0 greedy (see the README for every class, sampled rows and 219K context).
  `scripts/agens-run.sh --mtp DIR` and the image entry point (`-v DIR:/mtp:ro --mtp /mtp`) take it in the `16gb`,
  `24gb` and `48gb` profiles. With `--mtp` the `16gb` profile runs a ~241K-token context (images off; with
  `--images on` it ignores `--mtp`), and the `24gb` profile with images on a ~221K-token context.
- **Multi-GPU (tensor parallel).** `--tp 2` or `--tp 4` splits the model across 2 or 4 GPUs of one machine (one
  process per GPU; `scripts/agens-run.sh --tp 2 --gpu 0,1`, or `--tp 2` on the image entry point with
  `--gpus '"device=0,1"' --ipc=host`). The profile's memory budget applies per GPU. Every 1.1 feature works under
  `--tp`: speculative decoding with the drafter (greedy and sampled), image input (the vision encoder runs on the
  first GPU), server mode with concurrent requests, conversation prefix reuse and parking, session snapshot /
  restore, and the INT8 KV cache of the dense attention layer. GPUs with peer access (NVLink-class links, or PCIe
  peer-to-peer) exchange partial results directly; GPUs without peer-to-peer go through host shared memory.
- Under `--tp` the KV cache stays on the GPUs (each GPU holds its share of the attention heads), so a pair of GPUs
  holds more context than one: two 16 GB GPUs run the 4-bit release weights with the drafter and a ~100K-token
  context; two 24 GB GPUs the full 262K context without host tiering; two 48 GB GPUs the bf16 weights with a 262K
  context, which do not fit one 48 GB GPU at all.
- Prebuilt tensor-parallel prefill kernels for compute capability 8.6, 8.9, 9.0 and 12.0 (8.9 and 9.0 validated).

### Limits

- `--mtp` and `--tp` do not combine yet: the engine refuses `--mtp` with `--tp 2|4`, and the image entry point drops
  `--mtp` (with a message) when `--tp` is set. Tensor parallel runs with context copy and the drafter.
- With more than 8 concurrent sequences in one batched step, server mode skips the MTP chain for that step (the
  drafter still runs).
- Under `--tp`, a failed engine step still ends the server, as in 1.1.0: the 1.1.1 recovery (the request fails, the
  server keeps serving) is for one GPU.

### Fixes

- The all-reduce between GPUs reads each GPU's own share locally instead of back through the exchange buffer: 25-45%
  less time per exchange over host shared memory, identical results.

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
