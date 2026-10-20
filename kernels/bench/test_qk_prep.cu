// test_qk_prep: ak_qk_prep (one launch) against the chain it replaces, bitwise:
//   split_q_gate -> ak_rmsnorm(q, gemma) -> ak_rmsnorm(k, gemma) -> ak_rope_neox_m (MRoPE) -> store_kv
// T = 1 .. 64 rows, nq / nk = 24 / 4, 12 / 2, 6 / 1 (tp1 / tp2 / tp4), text positions (equal axes) and image-like
// positions (axes differ), with and without a cos|sin cache. Prints QK_PREP_PASS / QK_PREP_FAIL.
//   (perf_s2.sh kt test_qk_prep)
#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                             \
  do {                                                                                    \
    cudaError_t e_ = (cudaError_t)(x);                                                    \
    if (e_ != cudaSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

static uint32_t rs = 88172645u;
static uint32_t rnd() { return rs ^= rs << 13, rs ^= rs >> 17, rs ^= rs << 5, rs; }
static float nrand() {
  float s = 0;
  for (int i = 0; i < 4; ++i) s += (rnd() >> 8) * (1.0f / 16777216.0f);
  return (s - 2.0f) * 1.7320508f;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static uint16_t* dvec(size_t n, float scale, float mean = 0.f) {
  std::vector<uint16_t> h(n);
  for (auto& v : h) v = f2bf(mean + scale * nrand());
  uint16_t* d = nullptr;
  CK(cudaMalloc(&d, n * 2));
  CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
  return d;
}
template <typename T>
static T* dz(size_t n) {
  T* d = nullptr;
  CK(cudaMalloc(&d, n * sizeof(T)));
  CK(cudaMemset(d, 0, n * sizeof(T)));
  return d;
}
__global__ void split_ref(const __nv_bfloat16* qg, __nv_bfloat16* q, __nv_bfloat16* g, int T, int QH) {
  const long n = (long)T * QH * 256;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long t = i / (QH * 256), r = i % (QH * 256), h = r / 256, d = r % 256;
    q[i] = qg[(t * QH + h) * 512 + d];
    g[i] = qg[(t * QH + h) * 512 + 256 + d];
  }
}
__global__ void store_ref(const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* kc, __nv_bfloat16* vc,
                          const int* loc, int T, int KVH) {
  const long n = (long)T * KVH * 256;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long t = i / (KVH * 256), r = i % (KVH * 256);
    kc[(long)loc[t] * KVH * 256 + r] = k[i];
    vc[(long)loc[t] * KVH * 256 + r] = v[i];
  }
}
static bool same(const void* a, const void* b, size_t n, const char* what) {
  std::vector<uint16_t> x(n), y(n);
  CK(cudaMemcpy(x.data(), a, n * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y.data(), b, n * 2, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += x[i] != y[i];
  if (bad) printf("    %s: %zu of %zu differ\n", what, bad, n);
  return !bad;
}

int main() {
  bool ok = true;
  const int rows = 4096;
  float* cache = dz<float>((size_t)rows * 64);
  CK(ak_rope_cache(cache, rows, 64, 10000000.0f, nullptr));
  for (int hs = 0; hs < 3; ++hs) {
    const int nq = 24 >> hs, nk = 4 >> hs;
    uint16_t *wq = dvec(256, 0.2f, 0.0f), *wk = dvec(256, 0.2f, 0.0f);
    for (int T : {1, 3, 8, 16, 32, 64})
      for (int img = 0; img < 2; ++img)
        for (int cc = 0; cc < 2; ++cc) {
          uint16_t *qg = dvec((size_t)T * nq * 512, 2.0f), *k0 = dvec((size_t)T * nk * 256, 2.0f), *v = dvec((size_t)T * nk * 256, 1.0f);
          // positions [3][T]: text = 1000 + t on every axis; image-like = different per axis
          std::vector<long long> ph(3 * T);
          std::vector<int> lh(T);
          for (int t = 0; t < T; ++t) {
            ph[t] = 1000 + t;
            ph[T + t] = img ? 900 + (t % 5) : 1000 + t;
            ph[2 * T + t] = img ? 950 + (t % 7) : 1000 + t;
            lh[t] = (int)(rnd() % rows);
          }
          for (int t = 1; t < T; ++t)  // distinct rows
            for (int u = 0; u < t; ++u)
              if (lh[u] == lh[t]) lh[t] = (lh[t] + 1 + t) % rows, u = -1;
          long long* pos = dz<long long>(3 * T);
          int* loc = dz<int>(T);
          CK(cudaMemcpy(pos, ph.data(), ph.size() * 8, cudaMemcpyHostToDevice));
          CK(cudaMemcpy(loc, lh.data(), lh.size() * 4, cudaMemcpyHostToDevice));
          const size_t QN = (size_t)T * nq * 256, KN = (size_t)T * nk * 256, CN = (size_t)rows * nk * 256;
          uint16_t *uq = dz<uint16_t>(QN), *ug = dz<uint16_t>(QN), *uk = dz<uint16_t>(KN), *ukc = dz<uint16_t>(CN), *uvc = dz<uint16_t>(CN);
          uint16_t *fq = dz<uint16_t>(QN), *fg = dz<uint16_t>(QN), *fk = dz<uint16_t>(KN), *fkc = dz<uint16_t>(CN), *fvc = dz<uint16_t>(CN);
          CK(cudaMemcpy(uk, k0, KN * 2, cudaMemcpyDeviceToDevice));
          CK(cudaMemcpy(fk, k0, KN * 2, cudaMemcpyDeviceToDevice));
          const float* cp = cc ? cache : nullptr;
          split_ref<<<64, 256>>>((const __nv_bfloat16*)qg, (__nv_bfloat16*)uq, (__nv_bfloat16*)ug, T, nq);
          CK(ak_rmsnorm(uq, wq, uq, T * nq, 256, 256, 256, 1e-6f, 1, nullptr));
          CK(ak_rmsnorm(uk, wk, uk, T * nk, 256, 256, 256, 1e-6f, 1, nullptr));
          CK(ak_rope_neox_m(uq, uk, cp, (const int64_t*)pos, T, T, nq, nk, 256, 64, nq * 256, nk * 256, 10000000.0f, 11, 10,
                            nullptr));
          store_ref<<<64, 256>>>((const __nv_bfloat16*)uk, (const __nv_bfloat16*)v, (__nv_bfloat16*)ukc, (__nv_bfloat16*)uvc,
                                 loc, T, nk);
          QkPrepArgs qa{};
          qa.qg = (const __nv_bfloat16*)qg, qa.q = (__nv_bfloat16*)fq, qa.gate = (__nv_bfloat16*)fg, qa.k = (__nv_bfloat16*)fk;
          qa.v = (const __nv_bfloat16*)v, qa.kc = (__nv_bfloat16*)fkc, qa.vc = (__nv_bfloat16*)fvc, qa.loc = loc;
          qa.wq = (const __nv_bfloat16*)wq, qa.wk = (const __nv_bfloat16*)wk, qa.cache = cp, qa.pos = (const int64_t*)pos;
          qa.pstride = T, qa.T = T, qa.nq = nq, qa.nk = nk, qa.sec_h = 11, qa.sec_w = 10, qa.eps = 1e-6f, qa.theta = 10000000.0f;
          CK(ak_qk_prep(&qa, nullptr));
          CK(cudaDeviceSynchronize());
          const bool k = same(uq, fq, QN, "q") & same(ug, fg, QN, "gate") & same(uk, fk, KN, "k") &
                         same(ukc, fkc, CN, "k cache") & same(uvc, fvc, CN, "v cache");
          printf("heads %2d/%d T %2d %s %s: %s\n", nq, nk, T, img ? "image pos" : "text pos ", cc ? "cache" : "fly  ",
                 k ? "identical" : "DIFFERENT");
          ok = ok && k;
          for (void* p : {(void*)qg, (void*)k0, (void*)v, (void*)pos, (void*)loc, (void*)uq, (void*)ug, (void*)uk, (void*)ukc,
                          (void*)uvc, (void*)fq, (void*)fg, (void*)fk, (void*)fkc, (void*)fvc})
            cudaFree(p);
        }
  }
  printf(ok ? "QK_PREP_PASS\n" : "QK_PREP_FAIL\n");
  return ok ? 0 : 1;
}
