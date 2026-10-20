// W4 g128 decode / verify microbenchmark at the Volundr 32B tp2 per-rank shapes: the engine's KT path
// (ak_wq_gemv_kt_multi, as linear() calls it) vs the sm90 W4H path (ak_w4h_*). Per (shape, m):
//   * time per launch: a CUDA graph of back-to-back launches cycling through enough weight copies that every launch
//     reads its weights from DRAM (copies x bytes >= 8 x L2), replayed after a warm-up; mean us per launch.
//   * GB/s on the logical quantized bytes: n * k / 2 (int4) + n * k / 128 * 2 (bf16 scales), whatever the layout pads.
//   * correctness vs an fp64 GPU reference of the dequantized weights (q - 8) * s: max |err| / max |ref| per launch.
//   * batch invariance: rows of an m-row launch == the same rows of the 128-row launch, and == m = 1 launches of those
//     rows, bitwise.
// Usage: bench_w4h [impl: kt | w4h | both] [shape filter substring] [m list, e.g. 1,8,16]
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                             \
  do {                                                                                    \
    cudaError_t e_ = (cudaError_t)(x);                                                    \
    if (e_ != cudaSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

static uint32_t rng_state = 2463534242u;
static uint32_t rnd() {
  rng_state ^= rng_state << 13, rng_state ^= rng_state >> 17, rng_state ^= rng_state << 5;
  return rng_state;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// fp64 reference: y[j][o] = sum_i x[j][i] * (q[o][i] - 8) * s[o][i / 128]; packed [n][k/8] int32 (nibble i % 8 of word
// i / 8), scales bf16 [n][k/128]
__global__ void ref_kernel(const __nv_bfloat16* x, const uint32_t* packed, const __nv_bfloat16* sc, double* y, int m,
                           int n, int k) {
  const int o = blockIdx.x * blockDim.y + threadIdx.y;
  if (o >= n) return;
  for (int j = 0; j < m; ++j) {
    double acc = 0;
    for (int w = threadIdx.x; w < k / 8; w += 32) {
      const uint32_t q = packed[(size_t)o * (k / 8) + w];
      const double s = __bfloat162float(sc[(size_t)o * (k / 128) + w / 16]);
      for (int t = 0; t < 8; ++t)
        acc += (double)__bfloat162float(x[(size_t)j * k + 8 * w + t]) * (double)((int)((q >> (4 * t)) & 15u) - 8) * s;
    }
    for (int d = 16; d > 0; d >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
    if (threadIdx.x == 0) y[(size_t)j * n + o] = acc;
  }
}

struct Shape {
  const char* name;
  int n, k, per_step;  // calls per verify step per rank
};
// Volundr 32B at tp2 (per rank): 54 KDA layers, 18 attention (BCSA / dense) layers, 72 MLPs. f_up (k = 128) is bf16 at
// load, lm_head / mhc.dyn / in_proj_b / idx_q / idx_k are bf16 in the checkpoint: not W4.
static const Shape kShapes[] = {
    {"kda.in_proj_qkv", 5120, 5120, 54}, {"kda.in_proj_z", 3072, 5120, 54}, {"kda.f_down", 128, 5120, 54},
    {"kda.out_proj", 5120, 3072, 54},    {"attn.q_proj", 6144, 5120, 18},   {"attn.k_proj", 512, 5120, 18},
    {"attn.v_proj", 512, 5120, 18},      {"attn.o_proj", 5120, 3072, 18},   {"mlp.gate_up", 17408, 5120, 72},
    {"mlp.down_proj", 5120, 8704, 72},
};

struct Impl {
  virtual ~Impl() {}
  virtual const char* name() = 0;
  virtual size_t prepare(const void* packed, const void* scales, int n, int k, void** w, void** s) = 0;  // bytes
  virtual int run(const void* x, const void* w, const void* s, void* y, int m, int n, int k, cudaStream_t st) = 0;
  virtual void release(void* w, void* s) { cudaFree(w), cudaFree(s); }
};
struct KtImpl : Impl {
  const char* name() override { return "kt"; }
  size_t prepare(const void* packed, const void* scales, int n, int k, void** w, void** s) override {
    const size_t wb = ak_w4_kt_weight_bytes(n, k), sb = ak_w4_kt_scale_bytes(n, k, 128);
    CK(cudaMalloc(w, wb));
    CK(cudaMalloc(s, sb));
    CK(ak_w4_repack_kt(static_cast<const int32_t*>(packed), scales, *w, *s, n, k, 128, nullptr));
    return wb + sb;
  }
  int run(const void* x, const void* w, const void* s, void* y, int m, int n, int k, cudaStream_t st) override {
    return ak_wq_gemv_kt_multi(4, x, m, 1, &w, &s, &y, &n, &n, k, 128, 0, st);
  }
};
#ifdef AGENS_HAVE_W4H
struct W4hImpl : Impl {
  const char* name() override { return "w4h"; }
  size_t prepare(const void* packed, const void* scales, int n, int k, void** w, void** s) override {
    const size_t wb = ak_w4h_weight_bytes(n, k), sb = ak_w4h_scale_bytes(n, k);
    CK(cudaMalloc(w, wb));
    CK(cudaMalloc(s, sb));
    CK(ak_w4h_repack(static_cast<const int32_t*>(packed), scales, *w, *s, n, k, nullptr));
    return wb + sb;
  }
  int run(const void* x, const void* w, const void* s, void* y, int m, int n, int k, cudaStream_t st) override {
    return ak_w4h_gemm(x, m, 1, &w, &s, &y, &n, &n, k, st);
  }
};
#endif

static std::vector<uint16_t> d2h16(const void* d, size_t n) {
  std::vector<uint16_t> h(n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  return h;
}

int main(int argc, char** argv) {
  const std::string which = argc > 1 ? argv[1] : "kt";
  const std::string filt = argc > 2 ? argv[2] : "";
  std::vector<int> ms = {1, 8, 16, 32, 64, 128};
  if (argc > 3) {
    ms.clear();
    for (const char* p = argv[3]; *p;) {
      ms.push_back(atoi(p));
      while (*p && *p != ',') ++p;
      if (*p) ++p;
    }
  }
  const int iters0 = getenv("BENCH_ITERS") ? atoi(getenv("BENCH_ITERS")) : 200;
  std::vector<Impl*> impls;
  if (which == "kt" || which == "both") impls.push_back(new KtImpl);
#ifdef AGENS_HAVE_W4H
  if (which == "w4h" || which == "both") impls.push_back(new W4hImpl);
#endif
  int dev = 0, l2 = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, dev));
  printf("# %s, %d SMs, L2 %d MB; iters >= %d\n", pr.name, pr.multiProcessorCount, l2 >> 20, iters0);
  printf("%-16s %6s %6s %4s %-5s %9s %8s %10s %6s %6s\n", "shape", "n", "k", "m", "impl", "us", "GB/s", "maxrel", "inv128",
         "inv1");
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const int MMAX = 128;
  double step_us[8][8] = {};
  std::vector<std::vector<uint16_t>> first(16);  // impl 0's output per m (cross-impl bitwise check)
  for (const Shape& sh : kShapes) {
    if (!filt.empty() && std::string(sh.name).find(filt) == std::string::npos) continue;
    const int n = sh.n, k = sh.k;
    const double logical = (double)n * k / 2 + (double)n * (k / 128) * 2;
    // weights: q uniform 0..15, scales ~ 1e-2 * U(0.5, 1.5); x ~ U(-1, 1)
    std::vector<uint32_t> hp((size_t)n * k / 8);
    for (auto& v : hp) v = rnd();
    std::vector<uint16_t> hs((size_t)n * (k / 128));
    for (auto& v : hs) v = f2bf(1e-2f * (0.5f + (rnd() & 0xFFFF) / 65536.f));
    std::vector<uint16_t> hx((size_t)MMAX * k);
    for (auto& v : hx) v = f2bf(((int)(rnd() & 0xFFFF) - 32768) / 32768.f);
    void *dp, *ds, *dx, *dy, *dy2;
    double* dref;
    CK(cudaMalloc(&dp, hp.size() * 4));
    CK(cudaMalloc(&ds, hs.size() * 2));
    CK(cudaMalloc(&dx, hx.size() * 2));
    CK(cudaMalloc(&dy, (size_t)MMAX * n * 2));
    CK(cudaMalloc(&dy2, (size_t)MMAX * n * 2));
    CK(cudaMalloc(&dref, (size_t)MMAX * n * 8));
    CK(cudaMemcpy(dp, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    ref_kernel<<<(n + 7) / 8, dim3(32, 8)>>>(static_cast<const __nv_bfloat16*>(dx), static_cast<const uint32_t*>(dp),
                                            static_cast<const __nv_bfloat16*>(ds), dref, MMAX, n, k);
    CK(cudaDeviceSynchronize());
    std::vector<double> ref((size_t)MMAX * n);
    CK(cudaMemcpy(ref.data(), dref, ref.size() * 8, cudaMemcpyDeviceToHost));
    for (size_t ii = 0; ii < impls.size(); ++ii) {
      Impl* im = impls[ii];
      // weight copies: >= 8 x L2 of weights in the cycle
      const size_t one = ak_w4_kt_weight_bytes(n, k);
      const int copies = std::max(2, (int)((8.0 * l2) / one) + 1);
      std::vector<void*> W(copies), S(copies);
      for (int c = 0; c < copies; ++c) im->prepare(dp, ds, n, k, &W[c], &S[c]);
      const int iters = std::max(iters0, std::min(copies, 4000));
      CK(cudaDeviceSynchronize());
      // full 128-row result (invariance reference)
      CK(im->run(dx, W[0], S[0], dy2, MMAX, n, k, st));
      const std::vector<uint16_t> y128 = d2h16(dy2, (size_t)MMAX * n);
      for (int m : ms) {
        CK(im->run(dx, W[0], S[0], dy, m, n, k, st));
        const std::vector<uint16_t> ym = d2h16(dy, (size_t)m * n);
        double maxerr = 0, maxref = 0;
        for (size_t i = 0; i < (size_t)m * n; ++i) {
          maxerr = std::max(maxerr, std::fabs(bf2f(ym[i]) - ref[i]));
          maxref = std::max(maxref, std::fabs(ref[i]));
        }
        bool inv128 = memcmp(ym.data(), y128.data(), (size_t)m * n * 2) == 0;
        int mi0 = 0;
        while (ms[mi0] != m) ++mi0;
        const char* same0 = "";
        if (ii == 0) first[mi0] = ym;
        else same0 = memcmp(first[mi0].data(), ym.data(), ym.size() * 2) == 0 ? " ==kt" : " !=kt";
        // rows 0, m - 1 (and a middle one) as m = 1 launches
        bool inv1 = true;
        const int rows[3] = {0, m / 2, m - 1};
        for (int r : rows) {
          CK(im->run(static_cast<const __nv_bfloat16*>(dx) + (size_t)r * k, W[0], S[0], dy2, 1, n, k, st));
          const std::vector<uint16_t> y1 = d2h16(dy2, n);
          inv1 = inv1 && memcmp(y1.data(), ym.data() + (size_t)r * n, (size_t)n * 2) == 0;
        }
        // timing: graph of `iters` launches over the copies
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < iters; ++i) CK(im->run(dx, W[i % copies], S[i % copies], dy, m, n, k, st));
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
          CK(cudaEventRecord(e0, st));
          CK(cudaGraphLaunch(ge, st));
          CK(cudaEventRecord(e1, st));
          CK(cudaEventSynchronize(e1));
          float ms_ = 0;
          CK(cudaEventElapsedTime(&ms_, e0, e1));
          best = std::min(best, ms_);
        }
        if (const char* sk = getenv("BENCH_SOAK")) {  // keep the GPU busy (clock / power observation)
          for (int r = 0; r < atoi(sk); ++r) CK(cudaGraphLaunch(ge, st));
          CK(cudaStreamSynchronize(st));
        }
        const double us = best * 1e3 / iters;
        printf("%-16s %6d %6d %4d %-5s %9.2f %8.0f %10.3e %6s %6s%s\n", sh.name, n, k, m, im->name(), us,
               logical / us * 1e-3, maxerr / maxref, inv128 ? "ok" : "DIFF", inv1 ? "ok" : "DIFF", same0);
        fflush(stdout);
        int mi = 0;
        while (ms[mi] != m) ++mi;
        if (mi < 8 && ii < 8) step_us[ii][mi] += us * sh.per_step;
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
      }
      for (int c = 0; c < copies; ++c) im->release(W[c], S[c]);
    }
    cudaFree(dp), cudaFree(ds), cudaFree(dx), cudaFree(dy), cudaFree(dy2), cudaFree(dref);
  }
  if (filt.empty()) {
    double bytes = 0;
    for (const Shape& sh : kShapes) bytes += ((double)sh.n * sh.k / 2 + (double)sh.n * (sh.k / 128) * 2) * sh.per_step;
    printf("# per verify step per rank (all W4 linears, %.2f GB logical):\n", bytes * 1e-9);
    for (size_t ii = 0; ii < impls.size(); ++ii)
      for (size_t mi = 0; mi < ms.size() && mi < 8; ++mi)
        printf("#   %-5s m %4d: %8.1f us  %6.0f GB/s\n", impls[ii]->name(), ms[mi], step_us[ii][mi],
               bytes / step_us[ii][mi] * 1e-3);
  }
  return 0;
}
