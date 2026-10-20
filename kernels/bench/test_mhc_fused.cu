// test_mhc_fused: ak_mhc_fused (one launch per mHC layer boundary) against the unfused chain it replaces, bitwise:
//   ak_bf16_bmm_bi -> ak_mhc_combine(_add) -> ak_rmsnorm (gemma 0) -> ak_bf16_gemv_bi (n 24) -> ak_mhc_mix -> ak_rmsnorm (gemma 1)
// at T = 1, 3, 8, 16, 32, 64 rows, with and without the residual add (yb) and the drafter capture copy; then timing
// (L2 flushed between calls: the weights come from HBM, as in a verify step). Prints MHC_FUSED_PASS / MHC_FUSED_FAIL.
//   nvcc -O3 -arch=sm_90 -I kernels/include kernels/bench/test_mhc_fused.cu -L kernels/build -lagens_kernels_py -o test_mhc_fused
#include "agens_kernels.h"
#include <cuda_runtime.h>
#ifdef AGENS_MHC_PROF
#include "../src/mhc.cu"  // the kernel compiled here with its phase probes (this binary's definitions win over the .so's)
#endif
#include <cmath>
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

static uint32_t rs = 2463534242u;
static uint32_t rnd() { return rs ^= rs << 13, rs ^= rs >> 17, rs ^= rs << 5, rs; }
static float urand() { return (rnd() >> 8) * (1.0f / 16777216.0f); }
static float nrand() {  // ~N(0, 1)
  float s = 0;
  for (int i = 0; i < 4; ++i) s += urand();
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
static uint16_t* dzero(size_t n) {
  uint16_t* d = nullptr;
  CK(cudaMalloc(&d, n * 2));
  CK(cudaMemset(d, 0, n * 2));
  return d;
}
static bool same(const void* a, const void* b, size_t n, const char* what, int T) {
  std::vector<uint16_t> x(n), y(n);
  CK(cudaMemcpy(x.data(), a, n * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y.data(), b, n * 2, cudaMemcpyDeviceToHost));
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < n; ++i)
    if (x[i] != y[i]) {
      if (!bad) first = i;
      ++bad;
    }
  if (bad) printf("  T %d %s: %zu of %zu differ (first %zu: %04x vs %04x)\n", T, what, bad, n, first, x[first], y[first]);
  return bad == 0;
}

int main(int argc, char** argv) {
  const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
  constexpr int D = 5120, NS = 4, MAXT = 64;
  const float eps = 1e-6f;
  // layer i + 1 weights (norm weights around 1, dyn small, statics as trained-ish)
  uint16_t *wn = dvec(D, 0.1f, 1.0f), *wdyn = dvec((size_t)24 * D, 0.02f), *pre_s = dvec(4, 0.3f, 0.5f),
           *post_s = dvec(4, 0.3f, 1.0f), *res_s = dvec(16, 1.0f), *wln = dvec(D, 0.1f, 0.0f);
  uint16_t *pre_c = dvec(4, 0.3f, 0.5f), *post_c = dvec(4, 0.3f, 1.0f), *res_c = dvec(16, 1.0f);  // layer i's statics
  uint16_t* wb = dvec((size_t)48 * D, 0.02f);  // layer i + 1's in_proj_b (n = 12 / 24 / 48: tp4 / tp2 / tp1)
  int tcase = 0;
  bool ok = true;
  for (int pass = 0; pass < 2 && ok; ++pass) {
    for (int T : {1, 3, 8, 16, 32, 64}) {
      for (int add = 0; add < 2; ++add) {
        // layer i state: its dyn rows (res / post come from them, through layer i's mix), H streams, y / yb / x_in rows
        uint16_t *dcur = dvec((size_t)T * 24, 0.7f), *res = dzero((size_t)T * 16), *post = dzero((size_t)T * 4),
                 *xscr = dzero((size_t)T * D);
        uint16_t *H = dvec((size_t)T * NS * D, pass ? 4.0f : 1.0f), *Y = dvec((size_t)T * D, 1.0f),
                 *Yb = dvec((size_t)T * D, 0.5f), *Xi = dvec((size_t)T * D, 1.0f);
        CK(ak_mhc_mix(dcur, 24, pre_c, post_c, res_c, H, NS * D, D, xscr, D, post, res, T, D, 10, nullptr));
        uint16_t* fdc = dzero((size_t)T * 24);  // the fused side's dyn buffer: layer i's rows in, layer i + 1's out
        CK(cudaMemcpy(fdc, dcur, (size_t)T * 24 * 2, cudaMemcpyDeviceToDevice));
        // unfused outputs (u*) and fused (f*); res / post / x_in are overwritten in place, so each side gets copies
        uint16_t *uHn = dzero((size_t)T * NS * D), *uhm = dzero((size_t)T * D), *unrm = dzero((size_t)T * D),
                 *ud = dzero((size_t)T * 24), *uxin = dzero((size_t)T * D), *upost = dzero((size_t)T * 4),
                 *ures = dzero((size_t)T * 16), *uh = dzero((size_t)T * D), *umix = dzero((size_t)T * NS * D),
                 *ucap = dzero((size_t)T * 5 * D);
        uint16_t *fHn = dzero((size_t)T * NS * D), *fhm = dzero((size_t)T * D), *fd = dzero((size_t)T * 24),
                 *fxin = dzero((size_t)T * D), *fpost = dzero((size_t)T * 4), *fres = dzero((size_t)T * 16),
                 *fh = dzero((size_t)T * D), *fcap = dzero((size_t)T * 5 * D);
        CK(cudaMemcpy(fxin, Xi, (size_t)T * D * 2, cudaMemcpyDeviceToDevice));  // fused: x_in in place (Xi == xin)
        CK(cudaMemcpy(fpost, post, (size_t)T * 4 * 2, cudaMemcpyDeviceToDevice));
        // unfused chain
        CK(ak_bf16_bmm_bi(res, H, umix, T, NS, NS, D, nullptr));
        if (add)
          CK(ak_mhc_combine_add(umix, NS * D, D, post, Y, Yb, D, Xi, D, uHn, NS * D, D, uhm, D, T, D, nullptr));
        else
          CK(ak_mhc_combine(umix, NS * D, D, post, Y, D, Xi, D, uHn, NS * D, D, uhm, D, T, D, nullptr));
        CK(cudaMemcpy2D(ucap + 2 * D, 5 * D * 2, uHn, NS * D * 2, D * 2, T, cudaMemcpyDeviceToDevice));
        CK(ak_rmsnorm(uhm, wn, unrm, T, D, D, D, eps, 0, nullptr));
        CK(ak_bf16_gemv_bi(unrm, wdyn, nullptr, ud, T, 24, D, nullptr));
        CK(ak_mhc_mix(ud, 24, pre_s, post_s, res_s, uHn, NS * D, D, uxin, D, upost, ures, T, D, 10, nullptr));
        CK(ak_rmsnorm(uxin, wln, uh, T, D, D, D, eps, 1, nullptr));
        static const int nbs[4] = {0, 24, 48, 12};
        const int nbo = nbs[tcase++ % 4];
        uint16_t *ub = dzero((size_t)T * 48), *fb = dzero((size_t)T * 48);
        if (nbo) CK(ak_bf16_gemv_bi(uh, wb, nullptr, ub, T, nbo, D, nullptr));
        // fused
        MhcFusedArgs fa{};
        fa.dcur = (const __nv_bfloat16*)fdc, fa.res_s_cur = (const __nv_bfloat16*)res_c;
        fa.Hold = (const __nv_bfloat16*)H, fa.post = (const __nv_bfloat16*)fpost;
        fa.Y = (const __nv_bfloat16*)Y, fa.Yb = add ? (const __nv_bfloat16*)Yb : nullptr, fa.Xi = (const __nv_bfloat16*)fxin;
        fa.Hn = (__nv_bfloat16*)fHn, fa.hmn = (__nv_bfloat16*)fhm, fa.cap = (__nv_bfloat16*)(fcap + 2 * D), fa.cap_ld = 5 * D;
        fa.wn = (const __nv_bfloat16*)wn, fa.wdyn = (const __nv_bfloat16*)wdyn, fa.pre_s = (const __nv_bfloat16*)pre_s;
        fa.post_s = (const __nv_bfloat16*)post_s, fa.res_s = (const __nv_bfloat16*)res_s, fa.wln = (const __nv_bfloat16*)wln;
        fa.write_res = add;  // the last-boundary variant on half the cases
        fa.d = (__nv_bfloat16*)fdc, fa.xin = (__nv_bfloat16*)fxin, fa.post_o = (__nv_bfloat16*)fpost, fa.res_o = (__nv_bfloat16*)fres;
        fa.h = (__nv_bfloat16*)fh, fa.eps = eps, fa.iters = 10;
        fa.wb = nbo ? (const __nv_bfloat16*)wb : nullptr, fa.nb_out = nbo, fa.bout = (__nv_bfloat16*)fb;
        CK(ak_mhc_fused(&fa, T, nullptr));
        CK(cudaDeviceSynchronize());
        bool k = same(uHn, fHn, (size_t)T * NS * D, "H'", T) & same(uhm, fhm, (size_t)T * D, "mean", T) &
                 same(ud, fdc, (size_t)T * 24, "dyn", T) & same(uxin, fxin, (size_t)T * D, "x_in", T) &
                 same(upost, fpost, (size_t)T * 4, "post", T) & (!add || same(ures, fres, (size_t)T * 16, "res", T)) &
                 same(uh, fh, (size_t)T * D, "h", T) & same(ucap, fcap, (size_t)T * 5 * D, "cap", T) &
                 same(ub, fb, (size_t)T * 48, "in_proj_b", T);
        printf("T %2d add %d (write_res %d) in_proj_b n %2d scale %s: %s\n", T, add, add, nbo, pass ? "4" : "1",
               k ? "identical" : "DIFFERENT");
        cudaFree(ub), cudaFree(fb);
        ok = ok && k;
        for (void* p : {(void*)dcur, (void*)fdc, (void*)xscr, (void*)res, (void*)H, (void*)post, (void*)Y, (void*)Yb, (void*)Xi, (void*)uHn, (void*)uhm, (void*)unrm,
                        (void*)ud, (void*)uxin, (void*)upost, (void*)ures, (void*)uh, (void*)umix, (void*)ucap, (void*)fHn,
                        (void*)fhm, (void*)fd, (void*)fxin, (void*)fpost, (void*)fres, (void*)fh, (void*)fcap})
          cudaFree(p);
      }
    }
  }
  printf(ok ? "MHC_FUSED_PASS\n" : "MHC_FUSED_FAIL\n");
  if (!bench) return ok ? 0 : 1;
  // timing at T = 8 (single-sequence verify): unfused chain vs fused, L2 flushed (96 MB written) before each call
  const int T = 8;
  uint16_t *dc = dvec((size_t)T * 24, 0.7f), *res = dvec((size_t)T * 16, 0.1f, 0.25f), *H = dvec((size_t)T * NS * D, 1.f), *post = dvec((size_t)T * 4, 0.5f, 1.f),
           *Y = dvec((size_t)T * D, 1.f), *Yb = dvec((size_t)T * D, 0.5f), *Xi = dvec((size_t)T * D, 1.f),
           *Hn = dzero((size_t)T * NS * D), *hm = dzero((size_t)T * D), *nrm = dzero((size_t)T * D), *d = dzero((size_t)T * 24),
           *mix = dzero((size_t)T * NS * D), *h = dzero((size_t)T * D);
  void* flush = nullptr;
  CK(cudaMalloc(&flush, 96 << 20));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  for (int v = 0; v < 3; ++v) {
    double tot = 0;
    const int it = 200;
    for (int r = 0; r < it; ++r) {
      CK(cudaMemsetAsync(flush, r & 255, 96 << 20));
      cudaEventRecord(e0);
      if (v == 0) {
        ak_bf16_bmm_bi(res, H, mix, T, NS, NS, D, nullptr);
        ak_mhc_combine_add(mix, NS * D, D, post, Y, Yb, D, Xi, D, Hn, NS * D, D, hm, D, T, D, nullptr);
        ak_rmsnorm(hm, wn, nrm, T, D, D, D, eps, 0, nullptr);
        ak_bf16_gemv_bi(nrm, wdyn, nullptr, d, T, 24, D, nullptr);
        ak_mhc_mix(d, 24, pre_s, post_s, res_s, Hn, NS * D, D, Xi, D, post, res, T, D, 10, nullptr);
        ak_rmsnorm(Xi, wln, h, T, D, D, D, eps, 1, nullptr);
      } else {
        MhcFusedArgs fa{};
        fa.dcur = (const __nv_bfloat16*)d, fa.res_s_cur = (const __nv_bfloat16*)res_c;
        fa.Hold = (const __nv_bfloat16*)H, fa.post = (const __nv_bfloat16*)post;
        fa.Y = (const __nv_bfloat16*)Y, fa.Yb = (const __nv_bfloat16*)Yb, fa.Xi = (const __nv_bfloat16*)Xi;
        fa.Hn = (__nv_bfloat16*)Hn, fa.hmn = (__nv_bfloat16*)hm, fa.wn = (const __nv_bfloat16*)wn;
        fa.wdyn = (const __nv_bfloat16*)wdyn, fa.pre_s = (const __nv_bfloat16*)pre_s, fa.post_s = (const __nv_bfloat16*)post_s;
        fa.res_s = (const __nv_bfloat16*)res_s, fa.wln = (const __nv_bfloat16*)wln, fa.d = (__nv_bfloat16*)d;
        fa.xin = (__nv_bfloat16*)Xi, fa.post_o = (__nv_bfloat16*)post, fa.res_o = (__nv_bfloat16*)res, fa.h = (__nv_bfloat16*)h;
        fa.eps = eps, fa.iters = 10;
        if (v == 2) fa.wb = (const __nv_bfloat16*)wb, fa.nb_out = 24, fa.bout = (__nv_bfloat16*)d;
        ak_mhc_fused(&fa, T, nullptr);
      }
      cudaEventRecord(e1);
      CK(cudaEventSynchronize(e1));
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      if (r >= 20) tot += ms;
    }
    printf("%s: %.2f us per boundary (T = %d, cold L2)\n", v == 2 ? "fused + in_proj_b" : v ? "fused" : "unfused chain",
           1e3 * tot / (it - 20), T);
#ifdef AGENS_MHC_PROF
    if (v == 1) {
      unsigned long long p[16];
      CK(cudaMemcpyFromSymbol(p, g_mhc_prof, sizeof(p)));
      const char* nm[8] = {"start", "sinkhorn+loads", "combine+push", "norm1", "gemv+push", "dyn sum", "mix+push", "norm2 end"};
      for (int q = 1; q < 8; ++q) printf("  %-12s +%6.2f us\n", nm[q], (p[q] - p[0]) / 1e3);
    }
#endif
  }
  return ok ? 0 : 1;
}
