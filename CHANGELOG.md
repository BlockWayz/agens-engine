# Changelog

## 1.3.5

### New

- **Running requests are no longer ended for lack of KV cache.** In 1.3.4 a server under heavy long-context load
  could end a request mid-reply with an HTTP 500 ("KV pool full"). The server now checks, before admitting a request
  and before every prefill chunk and decode step, that every running request can still finish. A request that does
  not fit waits instead of failing, and if the server stalls, the newest request that has not produced a token yet
  goes back to the queue. Output is unchanged: greedy replies are token-identical to 1.3.4.
- **The KV pool sizes itself in server mode.** With `--gpu-gib` and no `--pool`, `agens serve --mode server` now
  gives the KV pool what the `--gpu-gib` budget leaves after the weights, the prefill cache, the prefix cache and
  8 GiB of headroom, instead of one `--wctx` context shared by all sessions. The breakdown is printed at startup.
  On 141 GB compute capability 9.0 GPUs, 2-way tensor parallel, Volundr INT4, `--gpu-gib 128 --max-batch 48
  --max-convs 48 --vision 1` with speculative decoding, the pool holds 1,050,040 tokens per GPU: 33 resident
  sessions of 30K tokens or 22 of 45K, where 1.3.4 held 8 and 5.
- **Long agent sessions under load.** Same setup, 48 concurrent multi-turn agent sessions (session IDs, tool
  schemas):
  - about 30K tokens each: 14 failed requests -> 0, aggregate decode 66 -> 144 tokens/s, median time to first token
    25.5 -> 1.1 s;
  - about 45K tokens each: 5 failed requests -> 0, 45 -> 101 tokens/s, 27.8 -> 1.4 s.

  In mixed-traffic runs (chat, image, 8K-document and agent requests, up to 48 concurrent clients) of 2 hours and
  1 hour, about 7,000 requests in all, no request failed and GPU memory stayed flat. Short-chat median time to first
  token was 0.49 s.
- **Conversation-aware eviction.** When rows run short the server frees, in order: prefix-cache snapshots that
  actually return rows, then conversations idle for more than 10 s with no turn queued (oldest first), then recently
  idle ones. A new long conversation waits (up to 600 s) while the active conversations would not fit. Short
  requests without a session ID skip that wait.
- **Fairer prefill.** The shortest remaining prompt is prefilled first, and every 4th chunk goes to the oldest
  waiting request, so short requests are not stuck behind long documents and long ones still progress.
- **Request log.** Each request's done line now shows its queue time and its time to first token after admission.

### Changed

- **`--gpu-gib` now counts memory used by other processes on the GPU** in server mode when the pool is auto-sized:
  everything in use on the GPU when the engine starts counts against the budget (capped at the device's memory). If
  another process holds GPU memory, the pool is that much smaller than on an empty GPU.
- **Batched-verify CUDA graphs are capped at 64** (least recently used dropped). Without a cap, long runs with many
  different batch shapes kept adding graphs until the GPU ran out of memory.

### Options

- `--pool N`: a fixed KV pool of N tokens per GPU (no auto-sizing).
- `AGENS_SERVE_AUTO_POOL=0`: the 1.3.4 behaviour (a pool of one `--wctx` context).
- `AGENS_SERVE_POOL_HEADROOM_GIB` (default 8): memory the auto-sized pool leaves free.
- `AGENS_SERVE_ADMIT_WAIT_S` (default 600): how long a new conversation waits for room before it may evict recently
  used ones.
- `AGENS_M1_BV_GRAPHS` (default 64): the batched-verify CUDA graph cap.

### Recommended server flags (compute capability 9.0, long agent sessions)

`--tp 2 --gpu-gib 128 --wctx 262148 --mode server --max-batch 48 --max-convs 48 --vision 1`, plus your drafting
flags. Leave `--pool` unset so the pool sizes itself.

### Unchanged

- Greedy output on our test prompts is identical to 1.3.4, alone and batched. Single-stream and 48-request decode
  speed are unchanged.
- Budget mode (`--budget-gib`, all `--profile` settings) is not affected: every profile keeps its 1.3.4 context.

## 1.3.4

### New

- **Faster single-stream decoding on compute capability 9.0.** Measured on 141 GB compute capability 9.0 GPUs, 2-way
  tensor parallel (`--tp 2`), Volundr INT4, server mode, one request at a time, vp_bench decode tokens/s, greedy: with
  `--mtp` 241 -> 315 on a short chat prompt, 242 -> 293 at a 6K-token prompt and 258 -> 312 at a 22K-token prompt
  (+21-31%); with the DFlash2 drafter 194 -> 215 and 224 -> 240. The speculative verify step launches fewer, larger
  kernels and overlaps their start-up. Greedy output is identical to 1.3.3.
- **Server mode batches up to 64 requests (`--max-batch 1..64`, was 16).** On the same setup with `--mtp`, default
  sampling, aggregate decode tokens/s at 16 / 24 / 32 / 48 concurrent requests: 1,273 / 1,288 / 1,288 / 1,289 (1.3.3,
  `--max-batch 16`) -> 1,329 / 1,361 / 1,620 / 1,857 (`--max-batch 48`). One request alone: 234 -> 314. Without
  speculation (no drafter, no `--mtp`), 13 concurrent requests: 727 -> 806.
- **Recommended on compute capability 9.0:** `--mode server --max-batch 48 --max-convs 48 --mtp <dir>`. With
  `--max-batch` above 4 the server keeps a second copy of some weights in a layout for small batches (about 5 GB per
  GPU with Volundr INT4 at `--tp 2`); the context stays at 258,052 tokens with `--gpu-gib 110`. This applies only
  without `--budget-gib`.
- **Budget mode (`--budget-gib`, all `--profile` settings) keeps at most 16 requests per batch:** a larger
  `--max-batch` is lowered to 16, because the memory plan covers batches of up to 16. No profile changes: contexts,
  memory use and speed are as in 1.3.3.

### Unchanged

- Greedy output on our test prompts is identical to 1.3.3, alone and batched (up to 48 requests per batch), with and
  without speculative decoding. Seeded sampled replies are the same alone, four at a time and 48 at a time, with and
  without `--mtp`, and the sampled distribution is unchanged (chi-square against plain decoding).
- The `16gb`, `24gb` and `24gb --tp 2` profiles (compute capability 8.9) keep 1.3.3's context (258,052; with `--mtp`
  on `24gb` 258,052, on `16gb` 237,572; `--tp 2` 245,764) and decode speed (within 1-2%; `--tp 2` within run-to-run
  noise).

## 1.3.3

### New

- **Reproducible sampling.** With a fixed `seed`, a sampled reply (temperature, top-k / top-p / min-p, penalties) is
  now the same token for token whether the request runs alone or batched with others, and with or without speculative
  decoding: every token is drawn from the target distribution at its position by one rule. On a test set of 6 prompts
  x 8 seeds x 2 samplings, 96 of 96 replies were identical alone vs four at a time, and with vs without `--mtp`. The
  sampled distribution is unchanged (chi-square against plain decoding).
- **Sampled requests without speculation sample on the GPU.** In server mode without a drafter, sampled requests no
  longer copy each step's logits to the host: on 141 GB compute capability 9.0 GPUs, 2-way tensor parallel, Volundr
  INT4, default sampling, 13 concurrent requests, aggregate decode rose from 83 to 725 tokens/s.
- **More throughput under concurrency with `--mtp` (compute capability 9.0).** Server mode runs the MTP chain for every
  batch and verifies shorter blocks when several requests decode: same setup with `--mtp`, aggregate decode tokens/s at
  1 / 4 / 13 / 16 concurrent requests 223 / 627 / 750 / 872 -> 241 / 731 / 1,104 / 1,277.
- **Faster sampled requests on the compute capability 8.9 profiles.** The sampling kernels' fast path now fits these
  GPUs: single stream, vp_bench decode tokens/s, temperature 0.7 / top_p 0.95: 12-26% faster in the `16gb` and `24gb`
  profiles (with and without `--mtp`, and `24gb --tp 2`); with a presence penalty 4-30% faster.
- **Version reporting.** `agens --version` prints the engine version, and `/v1/models` reports it as `agens_version`.

### Fixes

- **Speculative greedy output at long contexts.** The verify step's dense attention could differ from plain decoding in
  rare positions of long prompts; it now uses the same per-query plans as decoding on every path, so greedy output with
  speculation is identical to greedy output without it.

### Unchanged

- Greedy output on our test prompts is identical to 1.3.2 (the verify fix above only changes the rare positions where
  speculative greedy output differed from plain decoding). Every profile keeps 1.3.2's context, and greedy decode speed
  is within 1% of 1.3.2.

## 1.3.2

### New

- **Faster decoding on data-center GPUs (compute capability 9.0).** Measured on 141 GB compute capability 9.0 GPUs,
  2-way tensor parallel (`--tp 2`), Volundr INT4, one request at a time, vp_bench decode tokens/s, greedy: with the
  DFlash2 drafter 143 -> 197 on a short chat prompt and 176 -> 224 at a 22K-token prompt (+27-38%); with `--mtp` alone
  238 and 256 (+45-67% over 1.3.1 with the drafter). The verify step takes 10.3 ms instead of 15.0. Greedy output is
  identical to 1.3.1.
- **`--mtp` recommended on compute capability 9.0.** With the same setup the MTP head alone drafts faster than the
  DFlash2 drafter (each draft reads a 32K-token shortlist of the output head instead of the whole head); use `--mtp`
  without `--drafter`.
- **Speculative decoding with penalties, including under `--tp`.** Requests with presence, frequency or repetition
  penalties or a logit bias used to decode one token at a time; they now draft and verify like any other sampled
  request, with the same output distribution. On the compute capability 8.9 profiles (16 GiB and 23 GiB budgets) such
  requests decode 40-110% faster (vp_bench, presence 1.0).
- **Prefix cache.** Requests without a session id reuse the prompt prefix of earlier requests (`--prefix-cache-gib`,
  default 16 GiB per GPU outside budget mode; off in the `16gb` / `24gb` profiles).

### Fixes

- **Long conversations evicted early.** A request reserved KV cache rows for its whole `max_tokens` up front, which
  could evict other conversations' sessions; rows are now taken as tokens are generated.
- **HTTP 500 on long prompts in budget mode.** After short requests, a 4K+-token prompt could fail with an out-of-memory
  error; the prefill's temporary memory is now reserved at load.

### Unchanged

- The `16gb` and `24gb` profiles (compute capability 8.9) decode at 1.3.1's speed (within 2%, vp_bench greedy and
  sampled); greedy output is identical to 1.3.1. The single-GPU `16gb` / `24gb` profiles (with or without `--mtp`) keep
  1.3.1's contexts; `24gb --tp 2` serves 245,764 tokens (1.3.1: 249,860) because prefill memory is now reserved at load.

## 1.3.1

### Fixes

- **Weights placed in host memory under a large `--budget-gib`.** In budget mode the weights' own GPU limit stayed at
  the `--gpu-gib` default of 40 GiB, so weights larger than that (the bf16 build, ~63 GB) put the rest of their layers in
  host memory and decoded at a few tokens per second, while the KV cache took the GPU memory they should have used. The
  weights now go first: in budget mode they may use everything the budget leaves after the CUDA context, and the KV
  cache is planned from what remains. Only the embedding and lookup tables stay in host memory, by design. On 1.3.0,
  passing `--gpu-gib` equal to `--budget-gib` works around it. The 4-bit profiles were not affected.

### Docs

- README: community results for compute capability 8.6 with MTP and a first 12.0 report.

## 1.3.0

### New

- **MTP drafting under `--tp`.** `--mtp` now combines with `--tp 2` and `--tp 4`: every GPU runs the same MTP chain in
  its verify step, and greedy replies stay identical to plain greedy decoding (13 test prompts, single and batched),
  server mode gives the same replies with and without drafting, and sampled replies keep their distribution. The image
  entry point no longer drops `--mtp` under `--tp`; the profile asks for its full context and the engine clamps it to
  the per-GPU KV pool (logged).
- **MTP drafting with image input on, in every profile.** In 1.2 the `16gb` profile ignored `--mtp` with `--images on`.
  The vision encoder's working memory now shares the prefill's temporary memory (an image is encoded between prefill
  steps), so it no longer comes on top of it. On a 48 GB GPU held to 16 GiB, `16gb` with `--images on --mtp` serves a
  217,092-token context (225,284 with images and no MTP); a 219K-token document, a follow-up turn and two images peaked
  at 16.75 GB of device memory, inside the cap. The `24gb` profile with images on and `--mtp` now runs the full 262K
  context (1.2: about 221K). Image prompts: greedy output with MTP is identical to plain greedy decoding (26 test
  prompts), and sampled output keeps its distribution.
- **KV cache on the GPU when it fits.** With `--budget-gib`, 1.2 always tiered the KV cache to host memory on one GPU,
  even with budgets far above the model's needs. The memory plan now keeps all KV on the GPU when the budget holds one
  full context that way, and tiers it otherwise; the log says which. The shipped `16gb`, `24gb` and `48gb` profiles
  still tier. `AGENS_KV_AUTO=0` restores the 1.2 behaviour.
- **Compute capability 8.6 validated.** The prebuilt 8.6 kernels (shipped since 1.1) are now validated on hardware, by a
  community result on a 24 GB GPU (#1; see the README, Community results).

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
  context, which do not fit one 48 GB GPU at all. The 16 GB and 24 GB figures are planned under per-GPU caps on
  48 GB GPUs, not yet run on physical 16/24 GB cards.
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
