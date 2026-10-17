# Vendored FlashInfer headers — ORACLE / TEST TREE ONLY

Upstream: https://github.com/flashinfer-ai/flashinfer, version 0.6.15.post1,
commit de358e829443d2cab2261b4c2b403444f6925ab6 (copied unmodified from the `flashinfer-python` wheel's
`data/include` inside `volundr:sglv23-sm89`). License: Apache-2.0 (LICENSE in this folder).

Why: sglv23 runs these templates (norms via sgl_kernel, BCSA local decode via flashinfer). Instantiating the same
templates gives the M1 parity bridge bit-exact by construction. Rule (Volundr session, 2026-10-08): the M2
persistent kernel must NOT depend on FlashInfer's plan/workspace model at runtime; anything used here is
re-implemented inside the engine's own kernels. Only `kernels/oracle/` may include these headers.
