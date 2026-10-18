"""AOT build step for the KDA prefill: compile sglv23's own Triton kernels (sglang fla chunk_kda + causal_conv1d_fn)
for every variant class the engine needs, with the autotune configs PINNED (no runtime autotune), and emit
kernels/aot/kda_prefill_aot_sm<cc>.inc: embedded cubins + launch metadata (kernel name, warps, smem, parameter list) per
(T-class, kernel). The engine launches them with the CUDA driver API (kernels/src/kda_prefill.cu).

T-class of a single-sequence prefill chunk of T tokens (Volundr shapes are fixed):
    d16   = T % 16 == 0          (Triton specialises int args divisible by 16)
    fused = ceil(T / 64) * 48 <= 256   (chunk_kda_fwd's _small_grid: fused diagonal + recompute)
    nt    = 0 if NT <= 32 else 1 if NT <= 128 else 2   (fwd_h NT_BUCKET)
plus T == 1 (Triton turns int args equal to 1 into constants).

Pinned configs: tools/aot_pins_sm<cc>.json {kernel fn: config string as printed by triton.Config}, written from the sglv23
capture server's autotune log (TRITON_PRINT_AUTOTUNING=1), or by --pin-from-run on this GPU before the capture.

usage (dev container, GPU 0): python3 tools/triton_aot.py [--pin-from-run]
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402
from triton.runtime.autotuner import Autotuner  # noqa: E402
from triton_record import Recorder  # noqa: E402

# AOT_HEADS=h (tensor parallel rank share: 8 for tp2, 4 for tp4): q / k heads h, v heads 3h; the table and cubin
# names carry the suffix _h<h> (none for the full model's 16)
H = int(os.environ.get("AOT_HEADS", "16"))
HV, K, V = 3 * H, 128, 128
SUF = "" if H == 16 else f"_h{H}"
DIM = 2 * H * K + HV * V
ARCH0 = "sm%d%d" % torch.cuda.get_device_capability(0)  # cubins and autotune pins are per GPU arch
# AOT_TARGET=86: compile for that compute capability instead of this GPU's (Triton GPUTarget), e.g. sm86 cubins
# recorded on an sm89 card (an sm_86 cubin runs on 8.9, same major); the pins default to this GPU's (AOT_PINS=file).
# No tuning: the table only has to load and run on 8.6 (built, not tested on 8.6 hardware).
TARGET = os.environ.get("AOT_TARGET")
PINS0 = os.path.join(os.path.dirname(__file__), f"aot_pins_{ARCH0}{SUF}.json")
if TARGET:
    import triton.runtime.driver as _drv
    from triton.backends.compiler import GPUTarget
    _t = GPUTarget("cuda", int(TARGET), 32)
    _drv.active.get_current_target = lambda: _t
    ARCH0 = f"sm{TARGET}"
ARCH = ARCH0 + SUF
# AOT_PTX=1: emit the kernels' PTX instead of cubins (table kAotKdaPrefill_ptx_<arch>): the driver JIT-compiles it for
# newer GPUs that have no table of their own (e.g. sm120 from the sm89 run: PTX for sm_89 is forward compatible)
PTX = os.environ.get("AOT_PTX") == "1"
PINS = os.environ.get("AOT_PINS") or (PINS0 if TARGET else os.path.join(os.path.dirname(__file__), f"aot_pins_{ARCH}.json"))  # head-share variants: own pins (--pin-from-run)
OUT = os.path.join(os.path.dirname(__file__), "..", "kernels", "aot", f"kda_prefill_aot_{'ptx_' if PTX else ''}{ARCH}.inc")
SLOTS = 64  # engine's KDA state pool for the oracle build (affects only bounds constants, not arithmetic)


def tclass(T):
    nt = math.ceil(T / 64)
    return {"one": T == 1, "d16": T % 16 == 0, "fused": nt * HV <= 256, "nt": 0 if nt <= 32 else 1 if nt <= 128 else 2}


def class_key(c):
    return f"{'one' if c['one'] else ('d16' if c['d16'] else 'nd16')}_{'fused' if c['fused'] else 'split'}_nt{c['nt']}"


REPS = [1, 15, 16, 37, 48, 320, 336, 1000, 1024, 1349, 1360, 2049, 2064, 4095, 4096]  # 1349 / 1360: the head shares' split classes below nt 33


def run_prefill(T, kern, rec_dir=None, initial=True):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    dev = torch.device("cuda:0")
    torch.manual_seed(T)
    qkv = torch.randn(T, DIM, device=dev, dtype=torch.bfloat16)
    w = torch.randn(DIM, 4, device=dev) * 0.3
    conv_state = torch.zeros(SLOTS, 3, DIM, device=dev, dtype=torch.bfloat16).transpose(-1, -2)
    idx = torch.tensor([1], device=dev, dtype=torch.int32)
    qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
    hist = torch.tensor([initial], device=dev)
    a = torch.randn(T, HV * K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(T, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.log(torch.rand(HV, device=dev) * 15 + 1)
    dt = torch.randn(HV * K, device=dev)
    ssm = torch.zeros(SLOTS, HV, V, K, device=dev)
    splits = [H * K, H * K, HV * V]
    q, k, v = qkv.transpose(0, 1).split(splits, dim=0)
    ws_ = w.split(splits, dim=0)
    cs = conv_state.split(splits, dim=-2)
    outs = [causal_conv1d_fn(x_, w_, None, activation="silu", conv_states=c_, has_initial_state=hist,
                             cache_indices=idx, query_start_loc=qsl, seq_lens_cpu=[T]).transpose(0, 1)
            for x_, w_, c_ in zip((q, k, v), ws_, cs)]
    q, k, v = [o.unflatten(-1, (-1, K)).unsqueeze(0) for o in outs]
    g = a.unflatten(-1, (-1, K)).unsqueeze(0)
    beta = b.float().sigmoid().unsqueeze(0)
    return kern.extend(q=q, k=k, v=v, g=g, beta=beta, ssm_states=ssm, cache_indices=idx, query_start_loc=qsl,
                       A_log=A_log, dt_bias=dt, lower_bound=None, extend_seq_lens_cpu=[T])


def pin_autotuners(pins):
    """Seed every Autotuner's cache with the pinned config for its (kernel, key), exactly as Triton 3.6 computes the
    key in Autotuner.run (key args then the dtypes of all tensor args), so nothing is benchmarked. `pins` is either
    {"by_key": [{fn, key, config}]} (from a server log) or {kernel: config} (single pin per kernel, legacy)."""
    orig = Autotuner.run
    by_key = {}
    if "by_key" in pins:
        for p in pins["by_key"]:
            by_key[(p["fn"], tuple(p["key"]))] = p["config"]
    else:
        by_key = {(fn, None): cfg for fn, cfg in pins.items()}

    def run(self, *args, **kwargs):
        name = self.fn.__name__
        nargs = dict(zip(self.arg_names, args))
        all_args = {**nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in self.arg_names}
        key = [_args[k] for k in self.keys if k in _args]
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)
        want = by_key.get((name, key), by_key.get((name, None)))
        if want is not None and key not in self.cache:
            cfg = [c for c in self.configs if str(c) == want]
            if not cfg:
                raise RuntimeError(f"pinned config for {name} {key} not among its configs: {want}")
            self.cache[key] = cfg[0]
        elif want is None and any(n == name for n, _ in by_key):
            raise RuntimeError(f"no pin for {name} key {key}; re-capture the server's autotune log")
        return orig(self, *args, **kwargs)
    Autotuner.run = run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-from-run", action="store_true", help="autotune once here and write tools/aot_pins_<arch>.json")
    a = ap.parse_args()
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    kern = TritonKDAKernel()
    if a.pin_from_run:
        pins = {"by_key": []}
        for T in (37, 4096):
            with Recorder("/tmp/aot_pin") as rec:
                run_prefill(T, kern)
            for name, cache in rec.autotune.items():
                for k, v in cache.items():
                    e = {"fn": name, "key": list(eval(k)), "config": v}
                    if e not in pins.setdefault("by_key", []):
                        pins["by_key"].append(e)
        pins["source"] = "pin-from-run on " + torch.cuda.get_device_name(0)
        json.dump(pins, open(PINS, "w"), indent=1)
    pf = json.load(open(PINS))
    pins = pf if "by_key" in pf else pf["pins"]
    pin_autotuners(pins)
    classes = {}
    for T in REPS:
        key = class_key(tclass(T))
        if key in classes:
            continue
        with Recorder(f"/tmp/aot_{key}") as rec:
            run_prefill(T, kern)
            torch.cuda.synchronize()
        launches = []
        for l in rec.launches:
            k = rec.kernels[l["key"]]
            sig = list(k["signature"].items())
            consts = {int(c.strip("(),").split(",")[0]) for c in k["constants"]}
            params = [n for i, (n, t) in enumerate(sig) if t != "constexpr" and i not in consts]
            launches.append({"fn": k["fn"], "kernel": k["kernel_name"], "warps": k["num_warps"], "smem": k["shared"],
                             "params": params, "grid_T": l["grid"],
                             "cubin": (k["ptx"].encode() + b"\0") if PTX else k["cubin"]})
        classes[key] = {"T": T, "launches": launches}
        print(f"{key:22s} (T={T}): " + " ".join(f"{x['fn']}" for x in launches), flush=True)
    # emit C++ include
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    blobs, lines = {}, []
    TAB = ("ptx_" if PTX else "") + ARCH
    lines.append("// GENERATED by tools/triton_aot.py from sglv23's Triton kernels (triton "
                 f"{__import__('triton').__version__}); pins: {json.dumps(pins)}")
    lines.append("// Do not edit. Re-generate after changing pins or sglang.")
    lines.append("// Embedded binaries compiled from SGLang (Apache-2.0), flash-linear-attention (MIT) and causal-conv1d "
                 "(BSD-3-Clause) / vLLM (Apache-2.0) Triton kernels; see THIRD_PARTY_NOTICES.md.")
    for key, c in classes.items():
        for i, l in enumerate(c["launches"]):
            blob = l["cubin"]
            name = f"cubin_{TAB}_{key}_{i}"
            blobs[name] = blob
            lines.append(f"static const unsigned char {name}[] = {{" + ",".join(str(b) for b in blob) + "};")
    lines.append(f"static const AotLaunch kAotKdaPrefill_{TAB}[] = {{")
    for key, c in classes.items():
        for i, l in enumerate(c["launches"]):
            g = l["grid_T"]
            lines.append(f'  {{"{key}", {i}, "{l["fn"]}", "{l["kernel"]}", {l["warps"]}, {l["smem"]}, cubin_{TAB}_{key}_{i}, '
                         f'sizeof(cubin_{TAB}_{key}_{i}), "{",".join(l["params"])}", {{{g[0]}, {g[1]}, {g[2]}}}}},')
    lines.append("};")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {sum(len(c['launches']) for c in classes.values())} launches in {len(classes)} classes, "
          f"{sum(len(b) for b in blobs.values()) / 1e6:.1f} MB cubin")


if __name__ == "__main__":
    main()
