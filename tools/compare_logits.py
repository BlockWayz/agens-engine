"""Quality gate of an engine variant against a reference run over the same teacher-forced positions: both runs written
with AGENS_M1_LOGITS_DUMP (bf16 logits, VOCAB_VALID = 248077 per position, row-major). Prints top-1 agreement and
KL(p_ref || p_var) at temperature 1 (mean, p50, p99, max), computed exactly over the valid vocabulary.
usage: python3 compare_logits.py <ref.bin> <variant.bin> [label]"""
import sys

import numpy as np

V = 248077
a, b = (np.fromfile(f, dtype=np.uint16) for f in sys.argv[1:3])
label = sys.argv[3] if len(sys.argv) > 3 else "variant"
n = min(len(a), len(b)) // V
if len(a) != len(b):
    print(f"warning: {len(a) // V} vs {len(b) // V} positions; comparing {n}")
f = lambda x: (x.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
top1, kls = 0, []
for i in range(n):
    x, y = f(a[i * V:(i + 1) * V]), f(b[i * V:(i + 1) * V])
    lx = x.max() + np.log(np.exp(x - x.max()).sum())
    ly = y.max() + np.log(np.exp(y - y.max()).sum())
    p = np.exp(x - lx)
    kls.append(float((p * ((x - lx) - (y - ly))).sum()))
    top1 += int(x.argmax() == y.argmax())
k = np.array(kls)
print(f"{label} vs reference over {n} positions: top-1 agreement {top1}/{n} ({100 * top1 / max(n, 1):.2f}%), "
      f"KL mean {k.mean():.3e} nats, p50 {np.median(k):.2e}, p99 {np.quantile(k, 0.99):.2e}, max {k.max():.3e}")
