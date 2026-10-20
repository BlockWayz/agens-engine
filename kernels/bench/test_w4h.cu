// W4H gate: with the W4H layout on (ak_w4h_set_mode(1)), every engine entry point gives the bits of the tensor-core
// KT path (ak_w4h_set_mode(0)) on the same weights. Exit status 0 = all checks pass.
//  For Volundr tp2 decode shapes and ragged ones (k-tile counts 1..9, k not a multiple of 1024, n = 16): repack once
//  as KT and once as W4H (both via ak_w4_repack_kt), then compare bitwise:
//   a. ak_wq_gemv_kt_multi, m = 1..16, 24, 31, 32, 40, 64, 77, 128 (one segment);
//   b. a 3-segment launch (q|k|v-like) and a two-part linear written into one output with row stride ldy;
//   c. ak_w4_gemv_kt / _multi / _batched / ak_w4_mma_gemv;
//   d. ak_wq_dequant_kt_rows / ak_w4_dequant_kt_rows / ak_w4_dequant_kt (rows and the whole weight);
//   e. a mixed launch (one W4H segment, one KT segment);
//   f. n % 16 != 0 stays KT (ak_w4h_is false) and still matches.
// Usage: test_w4h
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
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static int fails = 0, checks = 0;
static void* dmalloc(size_t b) {
  void* p;
  CK(cudaMalloc(&p, b));
  CK(cudaMemset(p, 0, b));
  return p;
}
static bool same(const void* a, const void* b, size_t bytes, const char* what, int n, int k, int m) {
  std::vector<unsigned char> ha(bytes), hb(bytes);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
  ++checks;
  if (memcmp(ha.data(), hb.data(), bytes)) {
    ++fails;
    fprintf(stderr, "FAIL %s n %d k %d m %d\n", what, n, k, m);
    return false;
  }
  return true;
}
struct Lin {
  int n, k;
  void *p, *s, *wk, *sk, *wh, *sh;  // checkpoint packed / scales, KT copy, W4H copy
};
static Lin make(int n, int k) {
  Lin L{n, k};
  std::vector<uint32_t> hp((size_t)n * k / 8);
  for (auto& v : hp) v = rnd();
  std::vector<uint16_t> hs((size_t)n * (k / 128));
  for (auto& v : hs) v = f2bf(1e-2f * (0.25f + (rnd() & 0xFFFF) / 32768.f));
  L.p = dmalloc(hp.size() * 4), L.s = dmalloc(hs.size() * 2);
  CK(cudaMemcpy(L.p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(L.s, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
  L.wk = dmalloc(ak_w4_kt_weight_bytes(n, k)), L.sk = dmalloc(ak_w4_kt_scale_bytes(n, k, 128));
  L.wh = dmalloc(ak_w4_kt_weight_bytes(n, k)), L.sh = dmalloc(ak_w4_kt_scale_bytes(n, k, 128));
  ak_w4h_set_mode(0);
  CK(ak_w4_repack_kt(static_cast<const int32_t*>(L.p), L.s, L.wk, L.sk, n, k, 128, nullptr));
  ak_w4h_set_mode(1);
  CK(ak_w4_repack_kt(static_cast<const int32_t*>(L.p), L.s, L.wh, L.sh, n, k, 128, nullptr));
  CK(cudaDeviceSynchronize());
  return L;
}

int main() {
  int dev = 0, optin = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  if (optin < 200 * 1024) {
    printf("test_w4h: device opt-in shared memory %d KB < 200 KB: W4H unsupported here, skipped\n", optin >> 10);
    return 0;
  }
  const int shapes[][2] = {{5120, 5120}, {3072, 5120}, {128, 5120}, {5120, 3072}, {6144, 5120}, {512, 5120},
                           {17408, 5120}, {5120, 8704}, {48, 640}, {16, 128}, {96, 1152}, {256, 2176}, {80, 7168}};
  const int ms[] = {1, 2, 3, 5, 7, 8, 9, 12, 15, 16, 24, 31, 32, 40, 64, 77, 128};
  const int MMAX = 128;
  for (auto& sh : shapes) {
    const int n = sh[0], k = sh[1];
    Lin L = make(n, k);
    if (!ak_w4h_is(L.wh)) {
      ++fails;
      fprintf(stderr, "FAIL n %d k %d not repacked as W4H\n", n, k);
      continue;
    }
    std::vector<uint16_t> hx((size_t)MMAX * k);
    for (auto& v : hx) v = f2bf(((int)(rnd() & 0xFFFF) - 32768) / 16384.f);
    void* x = dmalloc(hx.size() * 2);
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    void *ya = dmalloc((size_t)MMAX * n * 2), *yb = dmalloc((size_t)MMAX * n * 2);
    // a. single segment, every m
    for (int m : ms) {
      CK(cudaMemset(ya, 0xA5, (size_t)MMAX * n * 2));
      CK(cudaMemset(yb, 0x5A, (size_t)MMAX * n * 2));
      CK(ak_wq_gemv_kt_multi(4, x, m, 1, (const void* const*)&L.wk, (const void* const*)&L.sk, &ya, &n, nullptr, k, 128, 0,
                             nullptr));
      CK(ak_wq_gemv_kt_multi(4, x, m, 1, (const void* const*)&L.wh, (const void* const*)&L.sh, &yb, &n, nullptr, k, 128, 0,
                             nullptr));
      same(ya, yb, (size_t)m * n * 2, "wq_multi", n, k, m);
    }
    // c. the older entry points
    CK(ak_w4_gemv_kt(x, L.wk, L.sk, ya, n, k, 128, 0, nullptr));
    CK(ak_w4_gemv_kt(x, L.wh, L.sh, yb, n, k, 128, 0, nullptr));
    same(ya, yb, (size_t)n * 2, "w4_gemv_kt", n, k, 1);
    for (int m : {4, 16, 48}) {
      CK(ak_w4_gemv_kt_batched(x, L.wk, L.sk, ya, m, n, k, 128, 0, nullptr));
      CK(ak_w4_gemv_kt_batched(x, L.wh, L.sh, yb, m, n, k, 128, 0, nullptr));
      same(ya, yb, (size_t)m * n * 2, "w4_batched", n, k, m);
      CK(ak_w4_mma_gemv(x, L.wk, L.sk, ya, m, n, k, 128, nullptr));
      CK(ak_w4_mma_gemv(x, L.wh, L.sh, yb, m, n, k, 128, nullptr));
      same(ya, yb, (size_t)m * n * 2, "w4_mma_gemv", n, k, m);
    }
    // d. dequant
    {
      const int r0 = n >= 32 ? 16 : 0, nr = n >= 32 ? n - 16 : n;
      void *da = dmalloc((size_t)n * k * 2), *db = dmalloc((size_t)n * k * 2);
      CK(ak_wq_dequant_kt_rows(4, L.wk, L.sk, da, n, k, 128, r0, nr, nullptr));
      CK(ak_wq_dequant_kt_rows(4, L.wh, L.sh, db, n, k, 128, r0, nr, nullptr));
      same(da, db, (size_t)nr * k * 2, "wq_dequant_rows", n, k, 0);
      CK(ak_w4_dequant_kt(L.wk, L.sk, da, n, k, 128, nullptr));
      CK(ak_w4_dequant_kt(L.wh, L.sh, db, n, k, 128, nullptr));
      same(da, db, (size_t)n * k * 2, "w4_dequant", n, k, 0);
      cudaFree(da), cudaFree(db);
    }
    cudaFree(x), cudaFree(ya), cudaFree(yb);
    // keep L for the multi-segment checks below only for one shape per k
    cudaFree(L.p), cudaFree(L.s), cudaFree(L.wk), cudaFree(L.sk);
    ak_w4h_forget(L.wh);
    cudaFree(L.wh), cudaFree(L.sh);
    printf("n %5d k %5d: ok so far (%d checks, %d failures)\n", n, k, checks, fails);
    fflush(stdout);
  }
  // b / e / f: multi-segment and mixed launches at k = 5120 (q|k|v-like: 6144 | 512 | 512; and a 40-row KT linear)
  {
    const int k = 5120;
    Lin q = make(6144, k), kk = make(512, k), v = make(512, k), odd = make(40, k);
    if (ak_w4h_is(odd.wh)) ++fails, fprintf(stderr, "FAIL n 40 repacked as W4H\n");
    std::vector<uint16_t> hx((size_t)MMAX * k);
    for (auto& e : hx) e = f2bf(((int)(rnd() & 0xFFFF) - 32768) / 16384.f);
    void* x = dmalloc(hx.size() * 2);
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    const int nn[3] = {6144, 512, 512};
    for (int m : {1, 8, 16, 32, 128}) {
      void* ya[3];
      void* yb[3];
      for (int i = 0; i < 3; ++i) ya[i] = dmalloc((size_t)m * nn[i] * 2), yb[i] = dmalloc((size_t)m * nn[i] * 2);
      const void* wk[3] = {q.wk, kk.wk, v.wk};
      const void* sk[3] = {q.sk, kk.sk, v.sk};
      const void* wh[3] = {q.wh, kk.wh, v.wh};
      const void* shh[3] = {q.sh, kk.sh, v.sh};
      CK(ak_wq_gemv_kt_multi(4, x, m, 3, wk, sk, ya, nn, nullptr, k, 128, 0, nullptr));
      CK(ak_wq_gemv_kt_multi(4, x, m, 3, wh, shh, yb, nn, nullptr, k, 128, 0, nullptr));
      for (int i = 0; i < 3; ++i) same(ya[i], yb[i], (size_t)m * nn[i] * 2, "3-segment", nn[i], k, m);
      // two parts of one 6656-row output (rows 0..6143 from q, 6144..6655 from k), row stride 6656
      void *za = dmalloc((size_t)m * 6656 * 2), *zb = dmalloc((size_t)m * 6656 * 2);
      void* pa[2] = {za, static_cast<char*>(za) + 6144 * 2};
      void* pb[2] = {zb, static_cast<char*>(zb) + 6144 * 2};
      const int ld[2] = {6656, 6656};
      CK(ak_wq_gemv_kt_multi(4, x, m, 2, wk, sk, pa, nn, ld, k, 128, 0, nullptr));
      CK(ak_wq_gemv_kt_multi(4, x, m, 2, wh, shh, pb, nn, ld, k, 128, 0, nullptr));
      same(za, zb, (size_t)m * 6656 * 2, "two parts ldy", 6656, k, m);
      // mixed: W4H q + KT odd (40 rows)
      const void* wm[2] = {q.wh, odd.wh};
      const void* sm_[2] = {q.sh, odd.sh};
      const void* wr[2] = {q.wk, odd.wk};
      const void* sr[2] = {q.sk, odd.sk};
      const int n2[2] = {6144, 40};
      void* ma[2] = {ya[0], ya[1]};
      void* mb[2] = {yb[0], yb[1]};
      CK(ak_wq_gemv_kt_multi(4, x, m, 2, wr, sr, ma, n2, nullptr, k, 128, 0, nullptr));
      CK(ak_wq_gemv_kt_multi(4, x, m, 2, wm, sm_, mb, n2, nullptr, k, 128, 0, nullptr));
      same(ma[0], mb[0], (size_t)m * 6144 * 2, "mixed W4H", 6144, k, m);
      same(ma[1], mb[1], (size_t)m * 40 * 2, "mixed KT", 40, k, m);
      for (int i = 0; i < 3; ++i) cudaFree(ya[i]), cudaFree(yb[i]);
      cudaFree(za), cudaFree(zb);
    }
  }
  printf("test_w4h: %d checks, %d failures\n", checks, fails);
  return fails ? 1 : 0;
}
