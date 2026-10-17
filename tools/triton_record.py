"""Record every Triton kernel launch made by a piece of sglv23 code: kernel name, the compiled kernel (cubin + metadata),
constexprs incl. the autotuner's chosen config, grid, and the launch arguments (tensors by shape/dtype/stride, scalars
by value). Output is the input to the AOT step (tools/triton_aot.py) and the record of which tiles were pinned.

Usage inside a script:
    from triton_record import Recorder
    with Recorder("out/aot/kda_prefill") as rec:
        ... call sglang code ...
    rec.save()
"""
import json, os

import torch
import triton
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction


def _describe(v):
    if isinstance(v, torch.Tensor):
        return {"tensor": True, "shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype),
                "ptr_mod16": v.data_ptr() % 16}
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if hasattr(v, "value"):  # tl.constexpr
        return _describe(v.value)
    return repr(v)


class Recorder:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.launches = []
        self.kernels = {}  # key -> metadata
        self._orig = None
        self.benchmarking = 0
        self.autotune = {}  # kernel fn -> {key: chosen config}
        self._tuners = []

    def __enter__(self):
        self._orig = JITFunction.run
        self._orig_bench = Autotuner._bench
        self._orig_at_run = Autotuner.run
        rec = self

        def bench(self_at, *a, **kw):  # autotuner trial launches are not part of the recorded program
            rec.benchmarking += 1
            try:
                return rec._orig_bench(self_at, *a, **kw)
            finally:
                rec.benchmarking -= 1

        def at_run(self_at, *a, **kw):
            if self_at not in rec._tuners:
                rec._tuners.append(self_at)
            return rec._orig_at_run(self_at, *a, **kw)

        Autotuner._bench = bench
        Autotuner.run = at_run

        def run(self_fn, *args, grid, warmup, **kwargs):
            ck = rec._orig(self_fn, *args, grid=grid, warmup=warmup, **kwargs)
            if warmup or rec.benchmarking:
                return ck
            try:
                names = self_fn.arg_names
                bound = dict(zip(names, args))
                bound.update({k: v for k, v in kwargs.items() if k in names})
                meta = {k: v for k, v in kwargs.items() if k not in names}
                g = grid(dict(bound, **kwargs)) if callable(grid) else grid
                g = tuple(int(x) for x in g) + (1,) * (3 - len(g))
                key = f"{self_fn.__name__}:{ck.hash}" if hasattr(ck, "hash") else f"{self_fn.__name__}:{id(ck)}"
                if key not in rec.kernels:
                    md = ck.metadata
                    md = md._asdict() if hasattr(md, "_asdict") else dict(md.__dict__) if hasattr(md, "__dict__") else {}
                    rec.kernels[key] = {
                        "fn": self_fn.__name__, "module": self_fn.fn.__module__, "kernel_name": ck.name,
                        "num_warps": md.get("num_warps"), "num_stages": md.get("num_stages"),
                        "shared": md.get("shared"), "global_scratch_size": md.get("global_scratch_size"),
                        "profile_scratch_size": md.get("profile_scratch_size"),
                        "constants": {str(k): _describe(v) for k, v in getattr(ck.src, "constants", {}).items()},
                        "signature": {str(k): str(v) for k, v in getattr(ck.src, "signature", {}).items()},
                        "cubin": ck.asm.get("cubin"), "ptx": ck.asm.get("ptx"),
                    }
                rec.launches.append({"key": key, "grid": g, "args": {k: _describe(v) for k, v in bound.items()},
                                     "meta": {k: _describe(v) for k, v in meta.items()}})
            except Exception as e:  # never break the code under test
                rec.launches.append({"error": repr(e), "fn": getattr(self_fn, "__name__", "?")})
            return ck

        JITFunction.run = run
        return self

    def __exit__(self, *exc):
        JITFunction.run = self._orig
        Autotuner._bench = self._orig_bench
        Autotuner.run = self._orig_at_run
        for t in self._tuners:
            name = getattr(getattr(t, "fn", None), "__name__", repr(t))
            self.autotune[name] = {repr(k): str(v) for k, v in t.cache.items()}

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        idx = {}
        for i, (key, k) in enumerate(self.kernels.items()):
            stem = f"{i:02d}_{k['fn']}"
            if k["cubin"]:
                open(os.path.join(self.out_dir, stem + ".cubin"), "wb").write(k["cubin"])
            if k["ptx"]:
                open(os.path.join(self.out_dir, stem + ".ptx"), "w").write(k["ptx"])
            idx[key] = {kk: vv for kk, vv in k.items() if kk not in ("cubin", "ptx")}
            idx[key]["file"] = stem
        json.dump({"triton": triton.__version__, "kernels": idx, "launches": self.launches, "autotune": self.autotune},
                  open(os.path.join(self.out_dir, "record.json"), "w"), indent=1)
        return idx
