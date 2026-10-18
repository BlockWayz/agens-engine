# Roadmap

What we are working on next. Items are listed by priority, not by date. Each one ships only after it passes the
same quality and speed gates as the current release, and its numbers are measured on the GPU class they describe.

## Next

- **12 GB GPUs.** 12 GB support is in development and is not released yet. It will start with context-copy drafting,
  which needs no extra memory; a lightweight multi-token-prediction (MTP) head, about one decoder layer that shares the
  model's output layer, will then bring model-based drafting to 12 GB.
- **MTP drafting on all profiles.** MTP joins context-copy and the companion drafter in one drafting chain; the
  engine picks whichever proposes the most accepted tokens.
- **Measured results on more GPUs.** Native runs and tuned kernels on 12 GB and 16 GB cards, including laptop GPUs,
  and validation on compute capability 12.0 and 8.6 (prebuilt today, not yet validated on hardware).
- **Smaller runtime image.** Drop the GPU libraries the engine links but never calls.
- **Image input on 16 GB GPUs at the full context.** Today `--images on` costs the 16 GB profile about 37K tokens of
  context.

## Later

- **Multi-GPU (tensor parallel).** Splitting one model across 2 or 4 GPUs. It is implemented and passes the
  correctness gates; it ships once its speed is benchmarked against the alternatives on the same hardware.
- **Faster 4-bit kernels for datacenter GPUs.** Closer to the memory-bandwidth limit on 141 GB-class GPUs.
- **Higher-concurrency serving.** Batched prefill across requests for 8–16 concurrent conversations, and faster
  batched verify with the KV cache tiered to host memory.
- **Smaller, stronger drafters.** Retrained on more model-generated data, including a compact variant for small
  memory budgets.

## Done

- **1.1.0:** image input; speculative sampling (temperature, top-k, top-p, min-p) in one-conversation and server
  mode; prebuilt kernels for compute capability 9.0 and 8.6; the `24gb` profile fix. See [CHANGELOG.md](CHANGELOG.md).

Have a request? Open an issue.
