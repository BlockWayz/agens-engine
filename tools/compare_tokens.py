"""Compare greedy token outputs: engine vs oracle JSON ([{name, tokens}]). Prints identical / first divergence."""
import json, sys
A = {r["name"]: r["tokens"] for r in json.load(open(sys.argv[1]))}
B = {r["name"]: r["tokens"] for r in json.load(open(sys.argv[2]))}
same = 0
for k in A:
    a, b = A[k], B.get(k)
    if b is None:
        print(f"  {k}: missing in B"); continue
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    if d is None and len(a) == len(b):
        same += 1; print(f"  {k}: IDENTICAL ({len(a)} tokens)")
    else:
        print(f"  {k}: diverge at {d if d is not None else n} (len {len(a)} vs {len(b)}): A {a[max(0,(d or n)-2):(d or n)+3]} B {b[max(0,(d or n)-2):(d or n)+3]}")
print(f"identical: {same}/{len(A)}")
