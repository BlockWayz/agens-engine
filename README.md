# Agens Engine

Agens Engine is the inference engine for the Agens Volundr model on a single consumer GPU. It serves an
OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`), and it runs a 262K-token context window
(258K tokens per conversation) on a 16 GB GPU.

- **Mixed-precision weights.** Each linear layer is stored at 2, 3 or 4 bits, chosen per layer.
- **KV cache tiered to host memory.** The full context fits without filling the GPU.
- **Speculative decoding with a companion drafter model and the model's MTP head, for greedy and sampled
  requests.** Greedy outputs are identical to plain greedy decoding, and sampled outputs follow the same distribution
  as plain sampling; drafting only changes speed.
- **Image input.** Images in chat messages, on by default on 24 GB and 48 GB GPUs.
- **Multi-GPU.** `--tp 2` / `--tp 4` splits the model across the GPUs of one machine for more context, more
  conversations or bf16 weights.
- **Hard memory budget.** The engine plans weights, context and cache inside the budget you give it, and fails
  loudly rather than overrunning it.

Agens Engine is open source under the MIT License (see `LICENSE`).

Third-party components keep their own licenses (`NOTICE`, `THIRD_PARTY_NOTICES.md`).

What changed in each release: [CHANGELOG.md](CHANGELOG.md).

## Requirements

- A GPU with 16 GB, 24 GB or 48 GB of memory and a CUDA 12.9-capable driver.
  - Prebuilt kernels: compute capability 8.9 and 9.0 (validated), 12.0 and 8.6 (built, not yet validated on
    hardware).
- Host RAM: 32 GB or more for the full 262K context. The tiered KV cache keeps up to ~24 GB in pinned host memory.
- Docker with the NVIDIA container toolkit.
- The model checkpoint for your GPU size and, optionally, the drafter checkpoint and the MTP head
  ([Blockway/Agens-Volundr-32B-v1-MTP](https://huggingface.co/Blockway/Agens-Volundr-32B-v1-MTP)).
- Multi-GPU (`--tp 2|4`): 2 or 4 GPUs of the same size in one machine; the budget is per GPU.

## Quickstart

```bash
# 16 GB GPU: 262K context, one conversation, speculative decoding
scripts/agens-run.sh --profile 16gb --model /path/to/model --drafter /path/to/drafter

# the same with the MTP head as a second draft source (context up to 241K)
scripts/agens-run.sh --profile 16gb --model /path/to/model --drafter /path/to/drafter --mtp /path/to/mtp

# then, from another shell
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "agens", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The same flags work on the image directly:

```bash
docker run --gpus device=0 --ipc=host -p 30000:30000 \
  -v /path/to/model:/model:ro -v /path/to/drafter:/drafter:ro \
  ghcr.io/blockwayz/agens-engine:1.2.0 --profile 16gb --model /model --drafter /drafter
```

## Profiles

| Profile | Budget | Context (up to) | Mode | Drafter | Image input | With `--mtp` |
|---|---|---|---|---|---|---|
| `16gb` | 16 GiB | 262K (225K with images on) | one conversation | yes | off (`--images on`) | up to 241K; ignored with images on |
| `24gb` | 23 GiB | 262K | server, up to 4 concurrent requests | yes | on | up to 221K with images on, 262K with images off |
| `48gb` | 46 GiB | 262K | server, up to 16 concurrent requests | yes | on | unchanged |

The context column is the most a profile asks for. At startup the engine plans the KV pool that fits the budget on
your GPU; if the pool holds fewer tokens than that, the engine clamps the context to what the pool holds and logs
`--wctx N clamped to M`. The pool depends on the GPU: the driver's own memory per GPU differs between cards. The
numbers in this README were measured on a 48 GB GPU of compute capability 8.9 held to the profile's budget; there the
`16gb` profile serves a 258,052-token context per conversation (237,572 with `--mtp`), and the `24gb` profile 217,092
with images on and `--mtp`.

`--tp 2|4` runs any profile across 2 or 4 GPUs, with the budget per GPU (see Multi-GPU below). `--mtp` is not
supported with `--tp` in 1.2: the entry point drops it and says so.

Extra `agens serve` flags go after `--`, for example `-- --served-model-name agens`. The `AGENS_*` environment
variables override the profile's tuning defaults.

In 1.0.0 the `24gb` profile did not start with the 4-bit weights and the drafter. 1.1.0 fixes this (see the
changelog).

### Request limits

- A request that sets no `max_tokens` generates at most 32,768 tokens (1.1.1; before, up to the context).
  `-- --max-tokens-default N` changes it (`0`: up to the context). An explicit `max_tokens` is honoured up to the
  context.
- A request whose client disconnects (streaming or not) stops at the next decode step, so the next request starts at
  once.
- Thinking repetition guard (1.1.1, on by default): inside the thinking block, a 32-token sequence repeated 20 times
  within the last 8,192 tokens closes the block and the model answers, as a reasoning budget would. It is a mitigation
  for the long-context reasoning loops seen with the 16 GB build, not a fix. `-- --think-guard N,K[,W]` sets the
  n-gram length, repeat count and window; `-- --think-guard off` turns it off. Each time it fires the server logs
  `think guard`.

## Image input

The chat API takes images as `image_url` content parts with base64 `data:` URLs (PNG, JPEG, WebP, GIF, BMP):

```bash
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "agens",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 chart.png)"'"}},
    {"type": "text", "text": "What does this chart show?"}]}]}'
```

- Image input is on by default in the `24gb` and `48gb` profiles and off in `16gb`. Turn it on or off with
  `--images on|off` (`scripts/agens-run.sh` and the image entry point).
- With `--images on` the `16gb` profile's context drops to about 225K tokens (from 262K). The vision encoder's working
  memory comes out of the KV cache budget.
- A message can carry several images, and images can appear in several turns of a conversation. Speculative decoding
  and conversation prefix reuse work on image prompts.
- An image costs one token per 32 x 32 pixels. Images over 4.19 megapixels (`AGENS_IMAGE_MAX_PIXELS`) are scaled down
  first, so one image is at most 4,096 tokens.
- http(s) image URLs and video input are not supported.

Accuracy, measured against SGLang serving the same 4-bit weights with images on the same GPU: on 22 image prompts
(charts, documents in English and Chinese, screenshots, photos) the engine's top-1 next-token agreement with SGLang's
greedy answers is 99.18% (teacher-forced), against 99.20% on text-only prompts; 99.59% on 6 multi-image prompts.
Preprocessed pixels are bit-identical to the reference image processor's on all 20 test images. Greedy speculative
decoding gives the same tokens as plain greedy decoding on every image prompt.

## Speculative sampling

Speculative decoding runs for greedy requests and for sampled ones: temperature with any top-k, top-p and min-p, in
one-conversation and server mode. The verify step samples the request's filtered distribution on the GPU, so a sampled
reply follows the same distribution as without the drafter. Requests that send no sampling fields use the model's
generation config (temperature 1.0, top-k 20, top-p 0.95) and are speculated too. Requests with repetition, frequency
or presence penalties, logit bias or a reasoning budget decode one token per step.

Distribution check (the release gate): 6 prompts (chat in English, Chinese and Cantonese, code, math, JSON) x 400
samples x 16 tokens at temperature 0.6, top-p 0.95, with the drafter and without it (two runs without, on different
seeds, as the noise baseline). Summed total-variation distance of the per-position token distributions:

| Mode | with drafter vs without | without vs without (noise) |
|---|---|---|
| one conversation | 4.13 | 4.78 |
| server mode, 4 concurrent requests | 4.06 and 5.21 (vs each run without) | 4.49 |
| image prompts (4 prompts x 300 samples) | 3.36 | 4.33 |

The drafter's distances are inside the run-to-run noise.

## MTP drafting

`--mtp DIR` loads the model's multi-token-prediction head
([Blockway/Agens-Volundr-32B-v1-MTP](https://huggingface.co/Blockway/Agens-Volundr-32B-v1-MTP)), about one decoder
layer that drafts from the model's own last hidden state. It joins context copy and the drafter in one drafting chain: in each verify step of 8 positions
the context copy fills the slots it covers, the MTP head chains the next ones (up to 4 with the drafter, 7 without
it) and the drafter fills the rest. The head is quantized to 4 bits at load (0.22 GB, plus 0.07 GB of context
buffers that come out of the KV cache budget), and each draft step scores a 32K-token shortlist of the output layer
plus the drafter's candidates for that slot.

- It runs in one-conversation and server mode, greedy and sampled. Greedy replies with `--mtp` are identical to plain
  greedy decoding on all 13 test prompts, at one GPU budget of 46 GiB and of 16 GiB and batched at 2, 4, 8 and 16
  sequences; in server mode replies are identical with and without drafting at 1 to 16 concurrent requests.
- Sampled replies follow the same distribution (6 prompts x 400 samples x 16 tokens, T 0.6, top-p 0.95, summed TV):
  4.77 with MTP and the drafter vs 5.02 between two runs without drafting; server mode, 4 concurrent requests, 4.33
  vs 4.37.
- Server-mode steps of more than 8 sequences skip the MTP chain (`AGENS_MTP_BATCH_MAX`); the drafter still runs.

Speed with and without `--mtp`, both with the drafter, one stream, on one 48 GB GPU of compute capability 8.9 held to
the profile's budget; tok/s (accepted tokens per verify step):

**16 GB profile** (Engine-16GB build, 16 GiB, images off)

| Prompt class | short, greedy: MTP | without | short, T 0.6: MTP | without | after 219K, greedy: MTP | without |
|---|---|---|---|---|---|---|
| chat | 108.0 (2.67) | 98.4 (2.24) | 93.2 (2.72) | 85.7 (2.33) | 71.2 (2.46) | 68.0 (2.20) |
| code | 140.7 (3.47) | 128.8 (2.93) | 122.4 (3.57) | 113.9 (3.10) | 103.1 (3.60) | 94.6 (3.11) |
| reasoning | 122.2 (3.04) | 116.8 (2.68) | 97.3 (2.85) | 91.8 (2.51) | 88.4 (3.07) | 82.6 (2.76) |
| tool calls | 124.1 (3.06) | 122.0 (2.77) | 135.7 (3.94) | 120.5 (3.27) | 75.1 (2.56) | 73.9 (2.42) |
| agent session (8K context) | 155.2 (4.24) | 150.3 (3.80) | 131.6 (4.18) | 123.0 (3.66) | 89.2 (3.12) | 83.0 (2.73) |

**24 GB profile** (4-bit weights, 23 GiB)

| Prompt class | short, greedy: MTP | without | short, T 0.6: MTP | without | after 219K, greedy: MTP | without |
|---|---|---|---|---|---|---|
| chat | 92.3 (2.56) | 85.2 (2.20) | 83.1 (2.68) | 73.0 (2.21) | 68.4 (2.40) | 63.8 (2.09) |
| code | 129.0 (3.58) | 115.3 (2.98) | 109.3 (3.52) | 98.4 (2.98) | 105.1 (3.71) | 91.5 (3.00) |
| reasoning | 106.9 (2.98) | 100.4 (2.61) | 91.2 (2.95) | 85.7 (2.61) | 92.7 (3.21) | 90.0 (2.98) |
| tool calls | 122.1 (3.38) | 114.5 (2.95) | 95.9 (3.08) | 99.3 (3.00) | 82.7 (2.84) | 72.9 (2.37) |
| agent session (8K context) | 132.1 (3.99) | 127.1 (3.58) | 112.3 (3.89) | 124.7 (4.08) | 94.3 (3.28) | 84.7 (2.77) |

Short: 5 chat, 5 code, 6 reasoning, 2 tool-call and 4 agent prompts. After 219K: a 219K-token document, then 2
prompts per class as follow-up turns. One run each; sampled rows and the 2-4 prompt classes carry run-to-run noise
(the two sampled losses at 24 GB are tool calls and agent sessions). With `--mtp` the 16 GB run of the 219K document
plus 12 follow-up turns (to ~239K tokens of context) peaked at 16.76 GB of device memory, inside the 16 GiB cap.

Server mode, 19 short prompts, 256 new tokens, greedy, aggregate tok/s with / without `--mtp`: `24gb` profile at 1 /
2 / 4 concurrent requests 104.4 / 149.5 / 159.6 vs 95.8 / 133.7 / 145.0; `48gb` profile at 1 / 4 / 8 104.2 / 164.1 /
188.1 vs 95.6 / 156.4 / 166.0. At 16 concurrent requests the `48gb` profile measured 141.5 with `--mtp` vs 168.2
without (161.6 in a second run); there, leave `--mtp` off for now.

## Multi-GPU (tensor parallel)

`--tp 2` or `--tp 4` splits the model across 2 or 4 GPUs of one machine, one engine process per GPU:

```bash
scripts/agens-run.sh --profile 48gb --tp 2 --gpu 0,1 --model /path/to/model --drafter /path/to/drafter
# or on the image: docker run --gpus '"device=0,1"' --ipc=host ... --profile 48gb --tp 2 --model /model --drafter /drafter
```

- The profile's memory budget is per GPU. Each GPU holds its share of the attention and KDA heads and of the MLP;
  embeddings, the output layer and the drafter are on every GPU.
- Under `--tp` the KV cache stays on the GPUs, not tiered to host memory. Two GPUs hold more context than one:

  | GPUs (per-GPU budget) | weights | context with the drafter |
  |---|---|---|
  | 2 x 16 GB | v1 4-bit | about 100K tokens (the 4-bit weights do not fit one 16 GB GPU) |
  | 2 x 24 GB | v1 4-bit | 262K, all on the GPUs (278K tokens of KV in total) |
  | 2 x 48 GB | v1 4-bit | 262K, with KV for 821K tokens across conversations |
  | 2 x 48 GB | bf16 | 262K (351K tokens of KV); bf16 does not fit one 48 GB GPU |

- Everything else works as on one GPU: speculative decoding with the drafter (greedy and sampled), image input (the
  vision encoder runs on the first GPU), server mode, conversation reuse and parking, the INT8 dense-layer KV cache.
  `--mtp` does not combine with `--tp` in 1.2.
- GPUs with peer access (NVLink-class links, PCIe peer-to-peer) exchange partial sums directly. GPUs without
  peer-to-peer exchange through host shared memory; this needs `--ipc=host` and costs speed (see below).

Correctness (release 4-bit weights, 13 test prompts): next-token agreement with one GPU, teacher-forced, is
98.7-99.1% at `--tp 2` and 98.2% at `--tp 4` (mean KL 6-8 x 10^-4), on image prompts 99.4-99.5% and 98.9%. As on
one GPU, greedy speculative decoding gives the same tokens as plain greedy decoding, batched decoding the same as
single, and server mode the same replies at every concurrency level; sampled replies follow the same distribution
with and without the drafter (summed TV 4.4 vs a 5.1 noise baseline at `--tp 2`, 4.8 vs 5.5 at `--tp 4`). The
greedy text at `--tp 2` is not identical to one GPU's: the sums run in a different order, so near-ties can flip.

Speed. Single stream, decode tok/s, with the drafter, same prompts and weights on both sides, on NVLink-class GPUs
(compute capability 9.0):

| Prompt class | 4-bit: SGLang 1 GPU | Engine 1 GPU | Engine `--tp 2` | Engine `--tp 4` | bf16: SGLang `--tp 2` | Engine `--tp 2` | SGLang `--tp 4` | Engine `--tp 4` |
|---|---|---|---|---|---|---|---|---|
| chat | 124.2 | 146.1 | 158.0 | 166.9 | 108.8 | 130.9 | 136.7 | 159.6 |
| code | 155.1 | 193.9 | 195.6 | 209.2 | 140.2 | 164.3 | 171.4 | 192.9 |
| reasoning | 116.2 | 142.6 | 151.2 | 166.8 | 113.3 | 127.0 | 130.7 | 148.5 |
| tool calls | 142.9 | 230.0 | 208.9 | 200.0 | 152.9 | 164.8 | 171.7 | 192.7 |
| agent session (8K) | 142.0 | 171.5 | 226.3 | 217.3 | 122.5 | 153.6 | 150.8 | 183.5 |

Greedy. SGLang serves this model at `--tp 2` and `--tp 4` only in bf16, so the like-for-like multi-GPU comparison is
bf16; the 4-bit columns are the product configuration. Server mode, 19 short prompts, aggregate tok/s at 1 / 4 / 8 / 16
concurrent requests: Engine 4-bit 164 / 360 / 497 / 616 on one GPU, 173 / 403 / 602 / 750 at `--tp 2`,
180 / 436 / 640 / 792 at `--tp 4`; bf16 at `--tp 4`, SGLang 130 / 339 / 559 / 1023, Engine 167 / 351 / 603 / 740.

On two 48 GB GPUs without peer-to-peer (compute capability 8.9), the `48gb` profile at `--tp 2` vs one GPU: chat 87.8 vs 80.9, code 110.2 vs
102.1, reasoning 85.7 vs 79.3, tool calls 109.2 vs 101.0, agent session (8K) 126.3 vs 91.5 tok/s greedy; server mode
95 / 186 / 191 / 236 vs 87 / 153 / 164 / 166 tok/s. Most of the gain is the untiered KV cache. Without the profile's
KV savings, splitting a model that already fits one GPU is slower at high concurrency over host shared memory (16
concurrent requests: 282 vs 364 tok/s). On such GPUs use `--tp` for capacity (bf16, longer contexts, more
conversations), not for single-GPU-sized speed.

## Performance

All numbers below were measured on one GPU (a 48 GB card of compute capability 8.9), with the engine held to the
stated memory budget by its hard cap, and speculative decoding with the drafter. Tokens per second are decode
throughput per request; "accepted" is the mean number of tokens committed per verify step. Each table states whether
decoding was greedy or sampled. Sampled rows are single runs of a random process: reply lengths and content vary from
run to run, so they are less stable than the greedy rows.

**16 GB profile** (`--budget-gib 16`, model build Agens-Volundr-32B-v1-Engine-16GB: mixed 2/3/4-bit, 2.82 bits per
weight including scales; images off). Short context, tok/s (accepted):

| Prompt class | greedy | T 0.6, top-p 0.95 | model default (T 1.0, top-k 20, top-p 0.95) |
|---|---|---|---|
| chat | 98.4 (2.24) | 85.7 (2.33) | 81.2 (2.19) |
| code | 128.7 (2.93) | 113.9 (3.10) | 110.8 (2.99) |
| reasoning | 116.6 (2.68) | 91.7 (2.51) | 90.4 (2.46) |
| tool calls | 121.9 (2.77) | 120.1 (3.27) | 99.3 (2.68) |
| agent session (8K context) | 150.4 (3.80) | 122.8 (3.66) | 122.7 (3.62) |

After a 219K-token document (one conversation; the same classes asked as follow-up turns), tok/s (accepted):

| Prompt class | greedy | T 0.6, top-p 0.95 | model default |
|---|---|---|---|
| chat | 67.2 (2.20) | 56.5 (2.08) | 51.1 (1.87) |
| code | 93.4 (3.11) | 78.8 (2.93) | 83.3 (3.09) |
| reasoning | 84.8 (2.76) | 70.5 (2.60) | 73.8 (2.70) |
| tool calls | 74.4 (2.42) | 67.3 (2.45) | 67.8 (2.47) |
| agent session (8K context) | 82.9 (2.73) | 71.9 (2.67) | 75.7 (2.80) |

The 219K-token prompt took 148 s to prefill; the conversation then grew to 239K tokens of context. Device memory
peaked at 15.5 GiB (16.64 GB), inside the engine's hard 16 GiB cap. With `--images on` (context 225K), the same
document plus follow-up turns and two images reached 223K tokens of context at 15.4 GiB (16.53 GB). On a 16 GB GPU
that also drives a display, less memory is free; use a shorter context there.

Quality of the 16 GB build against the other v1 builds:

| Build | GPQA Diamond (best of 4 runs) | HumanEval (pass@1) |
|---|---|---|
| bf16 v1 | 80.3 | 78.7 |
| v1 INT4 | 81.3 | 73.8 |
| Engine 16 GB | 73.7 | 73.2 |

GPQA Diamond: best of 4 runs, 6K-token thinking budget with forced close, temperature 0.6. HumanEval: pass@1,
greedy, 1 run.

**Known limitation (16 GB build):** with thinking on and a long document in the prompt (32K+ tokens), this build can
fall into repetition loops or give wrong answers. In our test (a 110K-token document, high thinking effort), 4 of 13
runs never finished thinking; the v1 4-bit build on the same engine finished 10 of 10 with no loops. For long
documents, turn thinking off or use the 4-bit build (`24gb` profile). The thinking repetition guard added in 1.1.1
limits the damage but does not fix this.

**24 GB profile vs SGLang** (`--profile 24gb`, v1 4-bit weights). Both run on the same 48 GB GPU with the same
4-bit weights, the same v1 drafter, the same prompts and token limits and a single stream; one run per side. The
SGLang side is the Agens SGLang build with its README drafter settings; its speculative sampling uses the same
target-only accept rule. Decode tok/s:

| Prompt class | greedy: SGLang | Agens Engine | speed-up | T 0.6, top-p 0.95: SGLang | Agens Engine | speed-up |
|---|---|---|---|---|---|---|
| chat | 58.3 | 85.6 | 1.47x | 59.6 | 70.8 | 1.19x |
| code | 79.0 | 112.7 | 1.43x | 76.3 | 98.2 | 1.29x |
| reasoning | 69.4 | 101.2 | 1.46x | 68.3 | 79.2 | 1.16x |
| tool calls | 79.7 | 110.1 | 1.38x | 75.2 | 101.2 | 1.35x |
| agent session (8K context) | 90.5 | 138.9 | 1.53x | 80.0 | 126.1 | 1.58x |
| agent session (30K context) | 67.1 | 114.3 | 1.70x | 65.0 | 98.2 | 1.51x |
| Cantonese | 67.2 | 104.4 | 1.55x | 65.6 | 70.7 | 1.08x |
| JSON output | 124.7 | 201.5 | 1.62x | 163.2 | 125.4 | 0.77x |
| reasoning, code | 54.6 | 100.4 | 1.84x | 54.0 | 79.6 | 1.47x |

At greedy both sides accept about the same number of draft tokens per step (chat 2.26 vs 2.21, code 2.99 vs 2.92,
reasoning 2.69 vs 2.63); the engine's verify step is faster. Agent rows are 3 (8K) and 4 (30K) prompts, and their
replies differ in length between the two sides; the first 8K prompt is left out on both sides because SGLang's run
has no decode time for it. The JSON, Cantonese and "reasoning, code" rows are one prompt each.

Server mode, 19 short prompts, 256 new tokens each, greedy, aggregate tok/s:

| Concurrent requests | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `24gb` profile | 95.6 | 135.2 | 156.3 | | |
| `48gb` profile | 95.5 | | 156.0 | 166.2 | 168.8 |

Every completion was identical at every concurrency level. In server mode, greedy replies are also identical with and
without the drafter (13 test prompts, 1 to 16 concurrent requests). Above 4
concurrent requests the `48gb` profile gains little: batched verify over the KV cache tiered to host memory is the
limit (see the roadmap).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what comes next: 12 GB GPU support (in development, not yet released), MTP drafting
under `--tp` and on 12 GB, and more measured GPUs.

## Building from source

```bash
cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;90;120" -DAGENS_BUILD_ORACLE=ON \
  -DAGENS_TORCH_PREFIX=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake --build kernels/build -j
cargo build --release --features m1 -p agens-cli
```

The build needs CUDA 12.9, CMake 3.27+, Rust stable and libtorch (PyTorch 2.11).
