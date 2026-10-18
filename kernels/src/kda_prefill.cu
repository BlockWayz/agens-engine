// Host glue ported from SGLang (Apache-2.0); launches binaries compiled from SGLang / flash-linear-attention (MIT) /
// causal-conv1d (BSD-3-Clause) Triton kernels (kernels/aot). See THIRD_PARTY_NOTICES.md.
// KDA prefill (one sequence chunk of T tokens), bit-exact with sglv23's extend path by construction: the kernels are
// sglv23's own Triton kernels AOT-compiled with pinned autotune configs (tools/triton_aot.py ->
// kernels/aot/kda_prefill_aot_sm<cc>.inc, one per GPU arch), launched here through the CUDA driver API. This file is the C++ port of the
// Python host glue: kda_backend.forward_extend -> causal_conv1d_fn x3 -> TritonKDAKernel.extend -> chunk_kda.
// Deterministic by construction (no runtime autotune). Rewritten natively in M5.
#include "agens_kernels.h"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct AotLaunch {
  const char* cls;
  int idx;
  const char* fn;
  const char* kernel;
  int warps;
  int smem;
  const unsigned char* cubin;
  unsigned cubin_size;
  const char* params;
  int rep_grid[3];
};
// Triton cubins are arch-specific and the autotune pins differ per arch: one generated table per validated GPU.
// The tables are generated (gitignored); a build carries whichever exist.
#if __has_include("../aot/kda_prefill_aot_sm89.inc")
#include "../aot/kda_prefill_aot_sm89.inc"
#define AGENS_AOT_SM89 1
#endif
#if __has_include("../aot/kda_prefill_aot_sm90.inc")
#include "../aot/kda_prefill_aot_sm90.inc"
#define AGENS_AOT_SM90 1
#endif
// tensor-parallel rank shares (AOT_HEADS=8 / 4 in tools/triton_aot.py): 8 q / k heads + 24 v heads (tp2), 4 + 12 (tp4)
#if __has_include("../aot/kda_prefill_aot_sm89_h8.inc")
#include "../aot/kda_prefill_aot_sm89_h8.inc"
#define AGENS_AOT_SM89_H8 1
#endif
#if __has_include("../aot/kda_prefill_aot_sm89_h4.inc")
#include "../aot/kda_prefill_aot_sm89_h4.inc"
#define AGENS_AOT_SM89_H4 1
#endif
#if __has_include("../aot/kda_prefill_aot_sm90_h8.inc")
#include "../aot/kda_prefill_aot_sm90_h8.inc"
#define AGENS_AOT_SM90_H8 1
#endif
#if __has_include("../aot/kda_prefill_aot_sm90_h4.inc")
#include "../aot/kda_prefill_aot_sm90_h4.inc"
#define AGENS_AOT_SM90_H4 1
#endif
// PTX of the sm89 kernels (AOT_PTX=1): JIT-compiled by the driver on GPUs without a table of their own (sm100 / sm120,
// and sm80-sm88 after load() retargets the PTX: see retarget_ptx)
#if __has_include("../aot/kda_prefill_aot_ptx_sm89.inc")
#include "../aot/kda_prefill_aot_ptx_sm89.inc"
#define AGENS_AOT_PTX_SM89 1
#endif

namespace {
constexpr int K = 128, V = 128, BT = 64;
int H = 16, HV = 48;  // this process's q / k and v heads (tensor parallel: ak_kda_prefill_set_heads)

struct Loaded {
  CUfunction fn;
};
std::map<std::pair<int, const unsigned char*>, Loaded> g_loaded;  // modules are per device (context)

// The sm89 PTX declares `.target sm_89`, which the driver refuses to JIT for an older GPU. On Ampere (sm80-sm88) return
// a copy that targets the device's own arch; the kernels use only sm80 instructions (mma.sync bf16, ldmatrix, cp.async;
// no fp8). Empty string: not PTX, or no rewrite needed. Validated on sm86 (RTX 3090).
std::string retarget_ptx(const unsigned char* image) {
  if (image[0] != '/' && image[0] != '.') return {};  // a cubin (ELF), not PTX text
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (major != 8 || minor >= 9) return {};
  std::string ptx(reinterpret_cast<const char*>(image));
  const std::string from = ".target sm_89";
  if (const size_t at = ptx.find(from); at != std::string::npos) ptx.replace(at, from.size(), ".target sm_8" + std::to_string(minor));
  return ptx;
}

int load(const AotLaunch& l, CUfunction* out) {
  int dev = 0;
  cudaGetDevice(&dev);
  auto it = g_loaded.find({dev, l.cubin});
  if (it != g_loaded.end()) {
    *out = it->second.fn;
    return 0;
  }
  CUmodule m;
  const std::string ptx = retarget_ptx(l.cubin);
  if (const CUresult r = cuModuleLoadData(&m, ptx.empty() ? (const void*)l.cubin : ptx.c_str()); r != CUDA_SUCCESS)  // out of memory is not a bad image
    return r == CUDA_ERROR_OUT_OF_MEMORY ? (int)cudaErrorMemoryAllocation : (int)cudaErrorInvalidKernelImage;
  CUfunction f;
  if (cuModuleGetFunction(&f, m, l.kernel) != CUDA_SUCCESS) return (int)cudaErrorInvalidDeviceFunction;
  if (l.smem > 48 * 1024) cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, l.smem);
  g_loaded[{dev, l.cubin}] = {f};
  *out = f;
  return 0;
}

std::string class_of(int T) {
  const int nt = (T + BT - 1) / BT;
  std::string s = T == 1 ? "one" : (T % 16 == 0 ? "d16" : "nd16");
  s += nt * HV <= 256 ? "_fused" : "_split";
  s += nt <= 32 ? "_nt0" : (nt <= 128 ? "_nt1" : "_nt2");
  return s;
}

// A kernel argument: device pointer, int32, int64 or fp32 scalar (Triton param ABI: pointers 8 bytes, i32 4, f32 4).
struct Arg {
  enum Kind { Ptr, I32, F32 } kind;
  CUdeviceptr p;
  int i;
  float f;
};
Arg P(const void* p) { return {Arg::Ptr, (CUdeviceptr)p, 0, 0.f}; }
Arg I(int i) { return {Arg::I32, 0, i, 0.f}; }
Arg Fl(float f) { return {Arg::F32, 0, 0, f}; }

int launch(const AotLaunch& l, const std::map<std::string, Arg>& args, int gx, int gy, int gz, CUstream s) {
  CUfunction f;
  if (int e = load(l, &f)) return e;
  std::vector<Arg> vals;
  std::string names(l.params);
  size_t pos = 0;
  while (pos <= names.size() && !names.empty()) {
    const size_t c = names.find(',', pos);
    const std::string n = names.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    auto it = args.find(n);
    if (it == args.end()) return (int)cudaErrorInvalidValue;  // glue and AOT manifest disagree
    vals.push_back(it->second);
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  CUdeviceptr scratch0 = 0, scratch1 = 0;  // Triton global/profile scratch (size 0 for these kernels)
  std::vector<void*> kp;
  for (auto& v : vals) kp.push_back(v.kind == Arg::Ptr ? (void*)&v.p : v.kind == Arg::I32 ? (void*)&v.i : (void*)&v.f);
  kp.push_back(&scratch0);
  kp.push_back(&scratch1);
  CUresult r = cuLaunchKernel(f, gx, gy, gz, 32 * l.warps, 1, 1, l.smem, s, kp.data(), nullptr);
  return r == CUDA_SUCCESS ? 0 : (int)cudaErrorLaunchFailure;
}

template <size_t N>
const AotLaunch* find_in(const AotLaunch (&table)[N], const std::string& cls, int idx) {
  for (const auto& l : table)
    if (cls == l.cls && l.idx == idx) return &l;
  return nullptr;
}

// nullptr for an arch without a generated table (callers return an error; regenerate with tools/triton_aot.py)
const AotLaunch* find(const std::string& cls, int idx) {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  switch ((major * 10 + minor) * 100 + H) {
#ifdef AGENS_AOT_SM89
    case 8916: return find_in(kAotKdaPrefill_sm89, cls, idx);
#endif
#ifdef AGENS_AOT_SM90
    case 9016: return find_in(kAotKdaPrefill_sm90, cls, idx);
#endif
#ifdef AGENS_AOT_SM89_H8
    case 8908: return find_in(kAotKdaPrefill_sm89_h8, cls, idx);
#endif
#ifdef AGENS_AOT_SM89_H4
    case 8904: return find_in(kAotKdaPrefill_sm89_h4, cls, idx);
#endif
#ifdef AGENS_AOT_SM90_H8
    case 9008: return find_in(kAotKdaPrefill_sm90_h8, cls, idx);
#endif
#ifdef AGENS_AOT_SM90_H4
    case 9004: return find_in(kAotKdaPrefill_sm90_h4, cls, idx);
#endif
    default:
#ifdef AGENS_AOT_PTX_SM89
      if ((major >= 10 || (major == 8 && minor < 9)) && H == 16) return find_in(kAotKdaPrefill_ptx_sm89, cls, idx);
#endif
      return nullptr;
  }
}

// GVA: [T][16][128] -> [T][48][128], each key head repeated 3x (torch repeat_interleave on dim -2)
__global__ void repeat_heads(const uint4* src, uint4* dst, int T, int H, int HV) {
  const int per_head = K * 2 / 16;  // uint4 per head row
  const long n = (long)T * HV * per_head;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const int c = (int)(i % per_head);
    const long th = i / per_head;
    const int h = (int)(th % HV);
    const long t = th / HV;
    dst[i] = src[(t * H + h / (HV / H)) * per_head + c];
  }
}
// beta = b.float().sigmoid() (fp32 out, torch: 1 / (1 + exp(-x)) with precise expf)
__global__ void sigmoid_f32(const __nv_bfloat16* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = __bfloat162float(b[i]);
    out[i] = 1.0f / (1.0f + expf(-x));
  }
}
__global__ void fill_indices(int32_t* chunk_indices, long long* chunk_offsets, int32_t* qsl, int nt, int T) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nt) {
    chunk_indices[2 * i] = 0;
    chunk_indices[2 * i + 1] = i;
  }
  if (i == 0) {
    chunk_offsets[0] = 0;
    chunk_offsets[1] = nt;
    qsl[0] = 0;
    qsl[1] = T;
  }
}
}  // namespace

// mixed_qkv [T][DIM] bf16 (in_proj_qkv output, pre-conv); a [T][HV*K] bf16 raw gate (f_up output); b [T][HV] bf16 raw
// beta (in_proj_b output); conv_w [DIM][4] fp32; conv_pool [slots][3][DIM] bf16; ssm_pool [slots][HV][V][K] fp32;
// slot = this sequence's state slot; has_init = 1 if this chunk continues a sequence. o [T][HV][V] bf16 out.
extern "C" int ak_kda_prefill(const void* mixed_qkv, const void* a, const void* b, const float* conv_w,
                              const float* A_log, const float* dt_bias, void* conv_pool, float* ssm_pool,
                              const int32_t* slot_dev, const bool* has_init_dev, int T, void* o, void* stream) {
  // head shares (tensor parallel): the small-grid (fused) class is only valid up to 5 chunks of 64 there (the AOT
  // capture's representative); longer fused-size chunks go in pieces of 320 tokens, the state carried between them
  {
    const int nt0 = (T + BT - 1) / BT;
    if (H != 16 && nt0 * HV <= 256 && nt0 > 5) {
      if (getenv("AGENS_KDA_DEBUG")) fprintf(stderr, "kda prefill: T %d in pieces of 320 (heads %d)\n", T, H);
      static std::map<int, bool*> yes;
      int dev = 0;
      cudaGetDevice(&dev);
      if (!yes[dev]) {
        bool t = true;
        cudaMalloc(&yes[dev], 1);
        cudaMemcpy(yes[dev], &t, 1, cudaMemcpyHostToDevice);
      }
      const int DIM = 2 * H * K + HV * V;
      for (int t0 = 0; t0 < T; t0 += 320) {
        const int tn = T - t0 < 320 ? T - t0 : 320;
        const int e = ak_kda_prefill(static_cast<const __nv_bfloat16*>(mixed_qkv) + (size_t)t0 * DIM,
                                     static_cast<const __nv_bfloat16*>(a) + (size_t)t0 * HV * K,
                                     static_cast<const __nv_bfloat16*>(b) + (size_t)t0 * HV, conv_w, A_log, dt_bias,
                                     conv_pool, ssm_pool, slot_dev, t0 ? yes[dev] : has_init_dev, tn,
                                     static_cast<__nv_bfloat16*>(o) + (size_t)t0 * HV * V, stream);
        if (e) return e;
      }
      return 0;
    }
  }
  CUstream s = static_cast<CUstream>(stream);
  cudaStream_t cs = static_cast<cudaStream_t>(stream);
  const std::string cls = class_of(T);
  const bool fused = cls.find("_fused") != std::string::npos;
  const int nt = (T + BT - 1) / BT;
  // ---- temporaries (stream-ordered)
  auto alloc = [&](size_t bytes) {
    void* p = nullptr;
    cudaMallocAsync(&p, bytes > 0 ? bytes : 16, cs);
    return p;
  };
  const size_t tq = (size_t)T * H * K, tv = (size_t)T * HV * V, tk48 = (size_t)T * HV * K;
  auto* cq = (__nv_bfloat16*)alloc(tq * 2);
  auto* ck = (__nv_bfloat16*)alloc(tq * 2);
  auto* cv = (__nv_bfloat16*)alloc(tv * 2);
  auto* q48 = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* k48 = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* qn = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* kn = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* beta = (float*)alloc((size_t)T * HV * 4);
  auto* gcs = (float*)alloc(tk48 * 4);
  auto* Aqk = (__nv_bfloat16*)alloc((size_t)T * HV * BT * 2);
  auto* Akkd = (float*)alloc((size_t)T * HV * 16 * 4);
  auto* Akk = (__nv_bfloat16*)alloc((size_t)T * HV * BT * 2);
  auto* w = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* u = (__nv_bfloat16*)alloc(tv * 2);
  auto* kg = (__nv_bfloat16*)alloc(tk48 * 2);
  auto* h = (__nv_bfloat16*)alloc((size_t)nt * HV * V * K * 2);
  auto* vnew = (__nv_bfloat16*)alloc(tv * 2);
  auto* chunk_indices = (int32_t*)alloc((size_t)nt * 2 * 4);
  auto* chunk_offsets = (long long*)alloc(2 * 8);
  auto* qsl = (int32_t*)alloc(2 * 4);
  fill_indices<<<(nt + 127) / 128, 128, 0, cs>>>(chunk_indices, chunk_offsets, qsl, nt, T);
  if (fused) {  // torch.zeros for Aqk when fuse_diagonal
    cudaMemsetAsync(Aqk, 0, (size_t)T * HV * BT * 2, cs);
  } else {
    cudaMemsetAsync(Akk, 0, (size_t)T * HV * BT * 2, cs);
  }
  int idx = 0, e = 0;
  auto L = [&](const std::map<std::string, Arg>& args, int gx, int gy, int gz) {
    if (e) return;
    const AotLaunch* l = find(cls, idx++);
    if (!l) { e = (int)cudaErrorInvalidValue; return; }
    e = launch(*l, args, gx, gy, gz, s);
  };
  // ---- causal_conv1d_fn x3 (q, k, v parts of the channel-last [T][DIM] input)
  const int offs[3] = {0, H * K, 2 * H * K}, dims[3] = {H * K, H * K, HV * V};
  __nv_bfloat16* conv_out[3] = {cq, ck, cv};
  for (int p = 0; p < 3; ++p) {
    L({{"x_ptr", P((const __nv_bfloat16*)mixed_qkv + offs[p])}, {"w_ptr", P(conv_w + (size_t)offs[p] * 4)},
       {"initial_states_ptr", P((__nv_bfloat16*)conv_pool + offs[p])}, {"cache_indices_ptr", P(slot_dev)},
       {"has_initial_states_ptr", P(has_init_dev)}, {"query_start_loc_ptr", P(qsl)}, {"o_ptr", P(conv_out[p])},
       {"seqlen", I(T)}},
      1, (T + 7) / 8, (dims[p] + 255) / 256);
  }
  repeat_heads<<<1024, 256, 0, cs>>>((const uint4*)cq, (uint4*)q48, T, H, HV);
  repeat_heads<<<1024, 256, 0, cs>>>((const uint4*)ck, (uint4*)k48, T, H, HV);
  // ---- l2norm q, k (rows = T * 48, BT = 16 rows per program)
  L({{"x", P(q48)}, {"y", P(qn)}, {"eps", Fl(1e-6f)}, {"T", I(T * HV)}}, (T * HV + 15) / 16, 1, 1);
  L({{"x", P(k48)}, {"y", P(kn)}, {"eps", Fl(1e-6f)}, {"T", I(T * HV)}}, (T * HV + 15) / 16, 1, 1);
  sigmoid_f32<<<(T * HV + 255) / 256, 256, 0, cs>>>((const __nv_bfloat16*)b, beta, T * HV);
  // ---- gate activation + chunk cumsum: grid (cdiv(K, BS), NT, HV); BS from the pinned config (rep grid x)
  const AotLaunch* lg = find(cls, idx);
  const int gate_gx = lg ? lg->rep_grid[0] : 1;
  L({{"s", P(a)}, {"A_log", P(A_log)}, {"dt_bias", P(dt_bias)}, {"o", P(gcs)}, {"scale", Fl(1.4426950408889634f)},
     {"cu_seqlens", P(qsl)}, {"chunk_indices", P(chunk_indices)}, {"T", I(T)}},
    gate_gx, nt, HV);
  const float scale = 0.08838834764831845f;  // K ** -0.5
  if (fused) {
    L({{"q", P(qn)}, {"k", P(kn)}, {"g", P(gcs)}, {"beta", P(beta)}, {"Aqk", P(Aqk)}, {"Akkd", P(Akkd)}, {"Akk", P(kn)},
       {"scale", Fl(scale)}, {"v_in", P(cv)}, {"w_out", P(w)}, {"u_out", P(u)}, {"kg_out", P(kg)},
       {"cu_seqlens", P(qsl)}, {"chunk_indices", P(chunk_indices)}, {"T", I(T)}},
      nt, HV, 1);
  } else {
    const AotLaunch* lt = find(cls, idx);
    const int bh_gy = lt ? lt->rep_grid[1] : 1;  // cdiv(HV, BH)
    L({{"q", P(qn)}, {"k", P(kn)}, {"g", P(gcs)}, {"beta", P(beta)}, {"Aqk", P(Aqk)}, {"Akk", P(Akkd)},
       {"scale", Fl(scale)}, {"cu_seqlens", P(qsl)}, {"N", I(1)}, {"T", I(T)}},
      T, bh_gy, 1);
    L({{"q", P(qn)}, {"k", P(kn)}, {"g", P(gcs)}, {"beta", P(beta)}, {"Aqk", P(Aqk)}, {"Akkd", P(Akkd)}, {"Akk", P(Akk)},
       {"scale", Fl(scale)}, {"v_in", P(kn)}, {"w_out", P(kn)}, {"u_out", P(kn)}, {"kg_out", P(kn)},
       {"cu_seqlens", P(qsl)}, {"chunk_indices", P(chunk_indices)}, {"T", I(T)}},
      nt, HV, 1);
    L({{"k", P(kn)}, {"kg", P(kg)}, {"v", P(cv)}, {"beta", P(beta)}, {"w", P(w)}, {"u", P(u)}, {"A", P(Akk)},
       {"gk", P(gcs)}, {"cu_seqlens", P(qsl)}, {"chunk_indices", P(chunk_indices)}, {"T", I(T)}},
      nt, HV, 1);
  }
  // ---- recurrent state over chunks (in-place final state into the pool slot)
  const AotLaunch* lh = find(cls, idx);
  L({{"k", P(kg)}, {"v", P(u)}, {"w", P(w)}, {"v_new", P(vnew)}, {"gk", P(gcs)}, {"h", P(h)},
     {"initial_state", P(ssm_pool)}, {"initial_state_indices", P(slot_dev)}, {"stride_init_state", I(HV * V * K)},
     {"cu_seqlens", P(qsl)}, {"chunk_offsets", P(chunk_offsets)}, {"T", I(T)}},
    lh ? lh->rep_grid[0] : 1, HV, 1);
  // ---- output (written into v's buffer in sglv23; here into o)
  const AotLaunch* lo = find(cls, idx);
  L({{"q", P(qn)}, {"v", P(vnew)}, {"g", P(gcs)}, {"h", P(h)}, {"o", P(o)}, {"A", P(Aqk)}, {"cu_seqlens", P(qsl)},
     {"chunk_indices", P(chunk_indices)}, {"scale", Fl(scale)}, {"T", I(T)}},
    lo ? lo->rep_grid[0] : 1, nt, HV);
  for (void* p : {(void*)cq, (void*)ck, (void*)cv, (void*)q48, (void*)k48, (void*)qn, (void*)kn, (void*)beta, (void*)gcs,
                  (void*)Aqk, (void*)Akkd, (void*)Akk, (void*)w, (void*)u, (void*)kg, (void*)h, (void*)vnew,
                  (void*)chunk_indices, (void*)chunk_offsets, (void*)qsl})
    cudaFreeAsync(p, cs);
  if (e) return e;
  return (int)cudaGetLastError();
}
// Load every module of this device's table now (init), so a later prefill of a new T-class never loads one lazily
// with the budget already spent. -> 0, or the error; no table: 0 (the prefill reports it).
template <size_t N>
int preload_all(const AotLaunch (&table)[N]) {
  for (const auto& l : table) {
    CUfunction f;
    if (int e = load(l, &f)) return e;
  }
  return 0;
}
extern "C" int ak_kda_prefill_preload() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  switch ((major * 10 + minor) * 100 + H) {
#ifdef AGENS_AOT_SM89
    case 8916: return preload_all(kAotKdaPrefill_sm89);
#endif
#ifdef AGENS_AOT_SM90
    case 9016: return preload_all(kAotKdaPrefill_sm90);
#endif
#ifdef AGENS_AOT_SM89_H8
    case 8908: return preload_all(kAotKdaPrefill_sm89_h8);
#endif
#ifdef AGENS_AOT_SM89_H4
    case 8904: return preload_all(kAotKdaPrefill_sm89_h4);
#endif
#ifdef AGENS_AOT_SM90_H8
    case 9008: return preload_all(kAotKdaPrefill_sm90_h8);
#endif
#ifdef AGENS_AOT_SM90_H4
    case 9004: return preload_all(kAotKdaPrefill_sm90_h4);
#endif
    default:
#ifdef AGENS_AOT_PTX_SM89
      if ((major >= 10 || (major == 8 && minor < 9)) && H == 16) return preload_all(kAotKdaPrefill_ptx_sm89);
#endif
      return 0;
  }
}
// this process's head share (tensor parallel rank): 16 / 48 (full), 8 / 24, 4 / 12; needs the matching AOT table
extern "C" int ak_kda_prefill_set_heads(int h, int hv) {
  if (hv != 3 * h || (h != 16 && h != 8 && h != 4)) return (int)cudaErrorInvalidValue;
  H = h, HV = hv;
  return 0;
}

