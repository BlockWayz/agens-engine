# Changelog

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
