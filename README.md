# Agens Engine

Agens Engine is the inference engine for the Agens Volundr model on a single consumer GPU. It serves an
OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`), and it runs a 262K-token context window
(258K tokens per conversation) on a 16 GB GPU.

- **Mixed-precision weights.** Each linear layer is stored at 2, 3 or 4 bits, chosen per layer.
- **KV cache tiered to host memory.** The full context fits without filling the GPU.
- **Speculative decoding with a companion drafter model.** Outputs are identical to plain greedy decoding; drafting
  only changes speed.
- **Hard memory budget.** The engine plans weights, context and cache inside the budget you give it, and fails
  loudly rather than overrunning it.

Agens Engine is open source under the MIT License (see `LICENSE`).

Third-party components keep their own licenses (`NOTICE`, `THIRD_PARTY_NOTICES.md`).

## Requirements

- A GPU with 16 GB, 24 GB or 48 GB of memory and a CUDA 12.9-capable driver.
  - Prebuilt kernels: compute capability 8.9 and 12.0.
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
  ghcr.io/blockwayz/agens-engine:1.0.0 --profile 16gb --model /model --drafter /drafter
```

## Profiles

| Profile | Budget | Context | Mode | Drafter |
|---|---|---|---|---|
| `16gb` | 16 GiB | 262K | one conversation | yes |
| `24gb` | 23 GiB | 262K | server (batched requests) | yes |
| `48gb` | 46 GiB | 262K | server (batched requests) | yes |

Extra `agens serve` flags go after `--`, for example `-- --served-model-name agens`. The `AGENS_*` environment
variables override the profile's tuning defaults.

## Performance

All numbers below were measured on one GPU (a 48 GB card of compute capability 8.9), with the engine held to the
stated memory budget by its hard cap, greedy decoding, and speculative decoding with the drafter. Tokens per second
are decode throughput per request; "accepted" is the mean number of tokens committed per verify step.

**16 GB profile** (`--budget-gib 16`, model build Agens-Volundr-32B-v1-Engine-16GB: mixed 2/3/4-bit, 2.82 bits per
weight including scales):

| Prompt class | short context: tok/s | accepted | after a 219K-token document: tok/s | accepted |
|---|---|---|---|---|
| chat | 90.2 | 2.24 | 62.0 | 2.20 |
| code | 118.2 | 2.93 | 85.8 | 3.11 |
| reasoning | 107.2 | 2.68 | 77.2 | 2.76 |
| tool calls | 112.1 | 2.77 | 69.0 | 2.42 |
| agent session (8K context) | 139.0 | 3.80 | 75.5 | 2.73 |

The 219K-token prompt took 151 s to prefill; the conversation then grew to 239K tokens of context. Device memory peaked at 15.5 GiB (16.64 GB), inside the engine's hard 16 GiB cap. On a 16 GB GPU that also drives a display, less memory is free; use a shorter context there.

Quality of the 16 GB build against the other v1 builds:

| Build | GPQA Diamond (best of 4 runs) | HumanEval (pass@1) |
|---|---|---|
| bf16 v1 | 80.3 | 78.7 |
| v1 INT4 | 81.3 | 73.8 |
| Engine 16 GB | 73.7 | 73.2 |

GPQA Diamond: best of 4 runs, 6K-token thinking budget with forced close, temperature 0.6. HumanEval: pass@1,
greedy, 1 run.

**24 GB profile** (`--budget-gib 23`, 4-bit weights):

| Prompt class | tok/s | accepted |
|---|---|---|
| chat | 78.0 | 2.20 |
| code | 105.5 | 2.98 |
| reasoning | 92.0 | 2.61 |
| tool calls | 104.8 | 2.95 |
| agent session (8K context) | 117.0 | 3.58 |

The 24 GB profile in server mode, 19 short prompts, 256 new tokens each: 87.7 tok/s aggregate at 1 concurrent
request, 129.3 at 2, 155.1 at 4. Every completion was identical at every concurrency level.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what comes next: 12 GB GPU support (in development, not yet released), MTP drafting, more
measured GPUs, and multi-GPU support.

## Building from source

```bash
cmake -S kernels -B kernels/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89;120" -DAGENS_BUILD_ORACLE=ON \
  -DAGENS_TORCH_PREFIX=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake --build kernels/build -j
cargo build --release --features m1 -p agens-cli
```

The build needs CUDA 12.9, CMake 3.27+, Rust stable and libtorch (PyTorch 2.11).
