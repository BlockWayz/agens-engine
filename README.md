# Agens Engine

Agens Engine is the inference engine for the Agens Volundr model on a single consumer GPU. It serves an
OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`), and it runs a 262K-token context window
(258K tokens per conversation) on a 16 GB GPU.

- **Mixed-precision weights.** Each linear layer is stored at 2, 3 or 4 bits, chosen per layer.
- **KV cache tiered to host memory.** The full context fits without filling the GPU.
- **Speculative decoding with a companion drafter model, for greedy and sampled requests.** Greedy outputs are
  identical to plain greedy decoding, and sampled outputs follow the same distribution as plain sampling; drafting
  only changes speed.
- **Image input.** Images in chat messages, on by default on 24 GB and 48 GB GPUs.
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
- The model checkpoint for your GPU size and, optionally, the drafter checkpoint.

## Quickstart

```bash
# 16 GB GPU: 262K context, one conversation, speculative decoding
scripts/agens-run.sh --profile 16gb --model /path/to/model --drafter /path/to/drafter

# then, from another shell
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "agens", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The same flags work on the image directly:

```bash
docker run --gpus device=0 --ipc=host -p 30000:30000 \
  -v /path/to/model:/model:ro -v /path/to/drafter:/drafter:ro \
  ghcr.io/blockwayz/agens-engine:1.1.0 --profile 16gb --model /model --drafter /drafter
```

## Profiles

| Profile | Budget | Context | Mode | Drafter | Image input |
|---|---|---|---|---|---|
| `16gb` | 16 GiB | 262K (about 225K with images on) | one conversation | yes | off (`--images on`) |
| `24gb` | 23 GiB | 262K | server, up to 4 concurrent requests | yes | on |
| `48gb` | 46 GiB | 262K | server, up to 16 concurrent requests | yes | on |

Extra `agens serve` flags go after `--`, for example `-- --served-model-name agens`. The `AGENS_*` environment
variables override the profile's tuning defaults.

In 1.0.0 the `24gb` profile did not start with the 4-bit weights and the drafter. 1.1.0 fixes this (see the
changelog).

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

See [ROADMAP.md](ROADMAP.md) for what comes next: 12 GB GPU support (in development, not yet released), MTP drafting,
more measured GPUs, and multi-GPU support.

## Building from source

```bash
cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;90;120" -DAGENS_BUILD_ORACLE=ON \
  -DAGENS_TORCH_PREFIX=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake --build kernels/build -j
cargo build --release --features m1 -p agens-cli
```

The build needs CUDA 12.9, CMake 3.27+, Rust stable and libtorch (PyTorch 2.11).
