// test_kf: the W4 g128 KF layout (AGENS_W4F) against the KT layout and an fp64 reference.
//   1. prefill dequant: KF rows == KT rows, bitwise (prefill numerics unchanged)
//   2. accuracy: KF GEMV vs fp64 sum of bf16(x) * q * scale, and the KT tensor-core path's error on the same data
//   3. batch invariance: every token column of an m-token call (m = 1 .. 40, segments) == its m = 1 call, bitwise
//   4. --bench: TB/s at m = 1, 8, 16, 32 (KF and KT) on the decode shapes
// prints KF_GATE_PASS / KF_GATE_FAIL
#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
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
template <typename T>
static T* dup(const std::vector<T>& h) {
  T* d = nullptr;
  CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
static std::vector<uint16_t> get(const void* d, size_t n) {
  std::vector<uint16_t> h(n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  return h;
}
struct Lin {
  int n, k;
  std::vector<uint32_t> hp;
  std::vector<uint16_t> hs;
  void *P, *S, *kt, *kts, *kf, *kfs;
};
static Lin make(int n, int k) {
  Lin L{n, k};
  L.hp.resize((size_t)n * k / 8);
  for (auto& w : L.hp) {
    uint32_t v = 0;
    for (int j = 0; j < 8; ++j) v |= (uint32_t)(rnd() % 16) << (4 * j);
    w = v;
  }
  L.hs.resize((size_t)n * (k / 128));
  for (auto& s : L.hs) s = f2bf(0.002f + 0.03f * (rnd() >> 8) / 16777216.f);
  L.P = dup(L.hp), L.S = dup(L.hs);
  CK(cudaMalloc(&L.kt, ak_w4_kt_weight_bytes(n, k)));
  CK(cudaMalloc(&L.kts, ak_w4_kt_scale_bytes(n, k, 128)));
  CK(cudaMalloc(&L.kf, ak_w4_kt_weight_bytes(n, k)));
  CK(cudaMalloc(&L.kfs, ak_w4_kt_scale_bytes(n, k, 128)));
  ak_w4_set_kf(0);
  CK(ak_w4_repack_kt((const int32_t*)L.P, L.S, L.kt, L.kts, n, k, 128, nullptr));
  ak_w4_set_kf(1);
  CK(ak_w4_repack_kt((const int32_t*)L.P, L.S, L.kf, L.kfs, n, k, 128, nullptr));
  ak_w4_set_kf(0);
  if (!ak_w4_is_kf(L.kf) || ak_w4_is_kf(L.kt)) printf("FAIL: KF registry\n"), exit(1);
  return L;
}
static void drop(Lin& L) {
  ak_w4_kf_forget(L.kf);
  for (void* p : {L.P, L.S, L.kt, L.kts, L.kf, L.kfs}) cudaFree(p);
}
static int fails = 0, checks = 0;
static int gemv(const Lin& L, bool kf, const void* x, int m, void* y) {
  const void* w = kf ? L.kf : L.kt;
  const void* s = kf ? L.kfs : L.kts;
  return ak_wq_gemv_kt_multi(4, x, m, 1, &w, &s, &y, &L.n, nullptr, L.k, 128, 0, nullptr);
}

static void test_shape(int n, int k) {
  Lin L = make(n, k);
  // 1. dequant
  void *d1, *d2;
  CK(cudaMalloc(&d1, (size_t)n * k * 2));
  CK(cudaMalloc(&d2, (size_t)n * k * 2));
  const int r0 = n > 48 ? 24 : 0, nr = n - r0 - (n > 48 ? 8 : 0);
  CK(ak_wq_dequant_kt_rows(4, L.kt, L.kts, d1, n, k, 128, r0, nr, nullptr));
  CK(ak_wq_dequant_kt_rows(4, L.kf, L.kfs, d2, n, k, 128, r0, nr, nullptr));
  {
    auto a = get(d1, (size_t)nr * k), b = get(d2, (size_t)nr * k);
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    ++checks;
    if (d) ++fails, printf("FAIL dequant [%d, %d] rows %d..%d: %zu differ\n", n, k, r0, r0 + nr, d);
  }
  cudaFree(d1), cudaFree(d2);
  // 2 / 3. x of 40 tokens
  const int M = 40;
  std::vector<uint16_t> hx((size_t)M * k);
  for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
  void* X = dup(hx);
  void *Y, *Y1;
  CK(cudaMalloc(&Y, (size_t)M * n * 2));
  CK(cudaMalloc(&Y1, (size_t)n * 2));
  // reference for tokens 0..7
  double err_kf = 0, err_kt = 0, ref_rms = 0;
  CK(gemv(L, true, X, 8, Y));
  auto ykf = get(Y, (size_t)8 * n);
  CK(gemv(L, false, X, 8, Y));
  auto ykt = get(Y, (size_t)8 * n);
  for (int j = 0; j < 8; ++j)
    for (int o = 0; o < n; o += std::max(1, n / 512)) {
      double r = 0;
      for (int i = 0; i < k; ++i) {
        const int q = (int)((L.hp[((size_t)o * k + i) / 8] >> (4 * (i % 8))) & 15u) - 8;
        r += (double)bf2f(hx[(size_t)j * k + i]) * q * bf2f(L.hs[(size_t)o * (k / 128) + i / 128]);
      }
      ref_rms += r * r;
      err_kf += std::pow(bf2f(ykf[(size_t)j * n + o]) - r, 2), err_kt += std::pow(bf2f(ykt[(size_t)j * n + o]) - r, 2);
    }
  printf("[%5d, %5d] rel rms error vs fp64: KF %.3e, KT %.3e\n", n, k, std::sqrt(err_kf / ref_rms), std::sqrt(err_kt / ref_rms));
  ++checks;
  if (std::sqrt(err_kf / ref_rms) > 2 * std::sqrt(err_kt / ref_rms) + 1e-4) ++fails, printf("FAIL accuracy\n");
  // 3. every column of m-token calls == its 1-token call
  std::vector<std::vector<uint16_t>> one(M);
  for (int j = 0; j < M; ++j) {
    CK(gemv(L, true, (const uint16_t*)X + (size_t)j * k, 1, Y1));
    one[j] = get(Y1, n);
  }
  for (int m : {2, 5, 8, 9, 16, 24, 31, 32, 40}) {
    CK(gemv(L, true, X, m, Y));
    auto y = get(Y, (size_t)m * n);
    size_t d = 0;
    for (int j = 0; j < m; ++j)
      for (int o = 0; o < n; ++o) d += y[(size_t)j * n + o] != one[j][o];
    ++checks;
    if (d) ++fails, printf("FAIL batch invariance [%d, %d] m %d: %zu differ\n", n, k, m, d);
  }
  // the KT kernels: every column of an m-token call == its 1-token call (decode == verify; tensor-parallel shards
  // have k not a multiple of 1024)
  for (int j = 0; j < M; ++j) {
    CK(gemv(L, false, (const uint16_t*)X + (size_t)j * k, 1, Y1));
    one[j] = get(Y1, n);
  }
  for (int m : {2, 5, 8, 9, 16, 24, 32, 40}) {
    CK(gemv(L, false, X, m, Y));
    auto y = get(Y, (size_t)m * n);
    size_t d = 0;
    for (int j = 0; j < m; ++j)
      for (int o = 0; o < n; ++o) d += y[(size_t)j * n + o] != one[j][o];
    ++checks;
    if (d) ++fails, printf("FAIL KT batch invariance [%d, %d] m %d: %zu differ\n", n, k, m, d);
  }
  cudaFree(X), cudaFree(Y), cudaFree(Y1);
  drop(L);
}

static void bench() {
  const int shapes[][2] = {{34816, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}, {12288, 5120}};
  const char* names[] = {"gate_up", "down", "kda_qkv", "o/out", "attn_q"};
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int si = 0; si < 5; ++si) {
    const int n = shapes[si][0], k = shapes[si][1];
    std::vector<Lin> L;
    for (int c = 0; c < 4; ++c) L.push_back(make(n, k));
    std::vector<uint16_t> hx((size_t)32 * k, f2bf(0.5f));
    void* X = dup(hx);
    void* Y;
    CK(cudaMalloc(&Y, (size_t)32 * n * 2));
    const double bytes = (double)ak_w4_kt_weight_bytes(n, k) + ak_w4_kt_scale_bytes(n, k, 128);
    for (int kf = 1; kf >= 0; --kf) {
      printf("bench %-8s %s", names[si], kf ? "KF" : "KT");
      for (int m : {1, 8, 16, 32}) {
        for (int it = 0; it < 20; ++it) CK(gemv(L[it % 4], kf, X, m, Y));
        cudaEventRecord(a);
        for (int it = 0; it < 200; ++it) gemv(L[it % 4], kf, X, m, Y);
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        const double us = ms * 1e3 / 200;
        printf("  m=%d %6.1f us %5.2f TB/s", m, us, bytes / us / 1e6);
      }
      printf("\n");
    }
    cudaFree(X), cudaFree(Y);
    for (auto& l : L) drop(l);
  }
}

int main(int argc, char** argv) {
  ak_w4_set_mma(1);
  if (argc > 1 && !strcmp(argv[1], "--bench")) return bench(), 0;
  for (auto sh : std::vector<std::pair<int, int>>{{256, 1024}, {1024, 5120}, {512, 17408}, {4096, 6144}, {48, 5120}, {160, 2048},
                                                  {5120, 8704}, {5120, 4352}, {5120, 3072}, {5120, 1536}})
    test_shape(sh.first, sh.second);
  printf("%s: %d checks, %d failed\n", fails ? "KF_GATE_FAIL" : "KF_GATE_PASS", checks, fails);
  return fails != 0;
}
