// Mixed-precision KT gate (track A1e): W2 / W3 KT == W4 KT of the same group on the same values, bitwise, for every
// entry point the engine uses. Exit status 0 = all checks pass.
//  For bits in {2, 3}, group in {64, 128}, Volundr decode shapes + ragged ones (k not a multiple of 1024, odd n):
//  values q uniform in [-2^(bits-1), 2^(bits-1) - 1] written into the int4 container (nibble q + 8), repacked once as
//  W4 KT (ak_w4_repack_kt) and once as W{bits} KT (ak_wq_repack_kt). Then, in both numerics (exact fp32 path and the
//  tensor-core path, ak_w4_set_mma 0 / 1; plus bf16w = 1 on the exact path):
//   a. m = 1 GEMV, b. m = 2..16 (batched / MMA), c. a 3-linear multi launch, d. a linear split in two parts written
//      into one output with row stride ldy (the engine's gate/up parts of different type), e. dequant (rows, tiles):
//   W{bits} output == W4 output, bitwise. f. repack refuses a value outside the range (and accepts the extremes).
//  g. Group equivalence the end-to-end check relies on: on the exact path, a g128 linear == the same linear in the g64
//     container with every scale repeated (per-chunk scaling), bitwise; the tensor-core path differs there (fp32
//     sums per 64 vs per 128 inputs), reported only.
//  i. batched verify (m up to 128 rows in one weight pass): a 72- / 37- / 13-row launch == the same rows as 8-row
//     launches, W2 / W3 / W4 x g64 / g128 and the batch-invariant bf16 GEMV (ak_bf16_gemv_bi), bitwise.
//  h. sm90 only: the bulk-copy W4 g128 path (ak_w4_set_tma 2 / 4 / 8) == the register MMA path (ak_w4_set_tma 0),
//     bitwise: m = 1..16, a 3-segment multi launch, two parts with ldy, ragged shapes.
// Usage: test_wq

#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
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
static int fails = 0, checks = 0;
static void same(const char* what, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int bits, int g,
                 int n, int k, int mode) {
  ++checks;
  size_t d = 0;
  for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
  if (d) {
    ++fails;
    printf("FAIL %s W%d g%d [%d, %d] mode %d: %zu / %zu differ\n", what, bits, g, n, k, mode, d, a.size());
  }
}

struct Lin {  // one linear: int4 container (packed, scales) and its W4 / Wb KT copies
  int n, k, g, bits;
  void *P, *S, *wt4, *st4, *wtb, *stb;
};
static Lin make(int n, int k, int g, int bits) {
  Lin L{n, k, g, bits};
  const int lo = -(1 << (bits - 1)), span = 1 << bits;
  std::vector<uint32_t> hp((size_t)n * k / 8);
  for (size_t w = 0; w < hp.size(); ++w) {
    uint32_t v = 0;
    for (int j = 0; j < 8; ++j) {
      int q = lo + (int)(rnd() % span);
      if (w == 0 && j == 0) q = lo;  // the extremes are always present
      if (w == 0 && j == 1) q = lo + span - 1;
      v |= (uint32_t)(q + 8) << (4 * j);
    }
    hp[w] = v;
  }
  std::vector<uint16_t> hs((size_t)n * (k / g));
  for (auto& s : hs) s = f2bf(0.002f + 0.03f * (rnd() >> 8) / 16777216.f);
  L.P = dup(hp), L.S = dup(hs);
  CK(cudaMalloc(&L.wt4, ak_w4_kt_weight_bytes(n, k)));
  CK(cudaMalloc(&L.st4, ak_w4_kt_scale_bytes(n, k, g)));
  CK(cudaMalloc(&L.wtb, ak_wq_kt_weight_bytes(bits, n, k)));
  CK(cudaMalloc(&L.stb, ak_w4_kt_scale_bytes(n, k, g)));
  CK(ak_w4_repack_kt((const int32_t*)L.P, L.S, L.wt4, L.st4, n, k, g, nullptr));
  CK(ak_wq_repack_kt(bits, (const int32_t*)L.P, L.S, L.wtb, L.stb, n, k, g, nullptr));
  return L;
}
static void drop(Lin& L) {
  for (void* p : {L.P, L.S, L.wt4, L.st4, L.wtb, L.stb}) cudaFree(p);
}

static int bench() {  // test_wq --bench: tensor-core GEMV GB/s per (bits, group) on Volundr decode shapes, m = 1 / 8
  struct Shape { int n, k; const char* name; };
  const Shape shapes[] = {{34816, 5120, "gate_up"}, {5120, 17408, "down"}, {10240, 5120, "kda_qkv"},
                          {5120, 6144, "o/out"}, {12288, 5120, "attn_q"}};
  ak_w4_set_mma(1);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  // WQ_BENCH_M: comma list of row counts (default 1,8,32,64, at most 64); WQ_BENCH_BITS: one width only
  std::vector<int> ms;
  for (const char* c = getenv("WQ_BENCH_M") ? getenv("WQ_BENCH_M") : "1,8,32,64"; *c;) {
    ms.push_back(std::min(64, std::max(1, atoi(c))));
    while (*c && *c != ',') ++c;
    if (*c == ',') ++c;
  }
  const int only_bits = getenv("WQ_BENCH_BITS") ? atoi(getenv("WQ_BENCH_BITS")) : 0;
  for (const Shape& sh : shapes)
    for (int bits : {2, 3, 4})
      for (int g : {64, 128}) {
        if (only_bits && bits != only_bits) continue;
        const int n = sh.n, k = sh.k, copies = 4;
        std::vector<Lin> L;
        for (int c = 0; c < copies; ++c) L.push_back(make(n, k, g, bits));  // > L2 in total: DRAM-bound timing
        std::vector<uint16_t> hx((size_t)64 * k, f2bf(0.5f));
        void* X = dup(hx);
        void* Y;
        CK(cudaMalloc(&Y, (size_t)64 * n * 2));
        const double bytes = (double)ak_wq_kt_weight_bytes(bits, n, k) + ak_w4_kt_scale_bytes(n, k, g);
        printf("bench %-8s W%dg%-3d", sh.name, bits, g);
        for (int m : ms) {
          for (int it = 0; it < 20; ++it)
            ak_wq_gemv_kt_multi(bits, X, m, 1, (const void* const*)&L[it % copies].wtb, (const void* const*)&L[it % copies].stb,
                                &Y, &n, nullptr, k, g, 0, nullptr);
          cudaEventRecord(a);
          const int iters = 200;
          for (int it = 0; it < iters; ++it)
            ak_wq_gemv_kt_multi(bits, X, m, 1, (const void* const*)&L[it % copies].wtb, (const void* const*)&L[it % copies].stb,
                                &Y, &n, nullptr, k, g, 0, nullptr);
          cudaEventRecord(b);
          cudaEventSynchronize(b);
          float ms = 0;
          cudaEventElapsedTime(&ms, a, b);
          const double us = ms * 1e3 / iters;
          printf("  m=%d %7.1f us %6.0f GB/s", m, us, bytes / us / 1e3);
        }
        printf("\n");
        cudaFree(X), cudaFree(Y);
        for (auto& l : L) drop(l);
      }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "--bench")) return bench();
  int dev = 0;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("# device %d %s\n", dev, prop.name);
  struct Shape { int n, k; };
  const Shape shapes[] = {{34816, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}, {1024, 5120}, {301, 1152}, {77, 640}};
  const int M = 16;
  for (int bits : {2, 3})
    for (int g : {64, 128})
      for (const Shape& sh : shapes) {
        const int n = sh.n, k = sh.k;
        if (k % g) continue;
        Lin L = make(n, k, g, bits);
        std::vector<uint16_t> hx((size_t)M * k);
        for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
        void* X = dup(hx);
        void *Y4, *Yb;
        CK(cudaMalloc(&Y4, (size_t)M * 2 * n * 2));
        CK(cudaMalloc(&Yb, (size_t)M * 2 * n * 2));
        // modes: 0 exact, 1 tensor-core, 2 exact with bf16(q * s) weights
        for (int mode = 0; mode < 3; ++mode) {
          ak_w4_set_mma(mode == 1 ? 1 : 0);
          const int bw = mode == 2;
          for (int m : {1, 2, 5, 8, 9, 16}) {
            CK(ak_wq_gemv_kt_multi(4, X, m, 1, (const void* const*)&L.wt4, (const void* const*)&L.st4, &Y4, &n,
                                   nullptr, k, g, bw, nullptr));
            CK(ak_wq_gemv_kt_multi(bits, X, m, 1, (const void* const*)&L.wtb, (const void* const*)&L.stb, &Yb, &n,
                                   nullptr, k, g, bw, nullptr));
            char what[32];
            snprintf(what, sizeof what, "gemv m=%d", m);
            same(what, get(Y4, (size_t)m * n), get(Yb, (size_t)m * n), bits, g, n, k, mode);
          }
          if (mode != 2) {  // W3 / W4 entry points of the existing API agree with ak_wq (m = 1, batched)
            CK(ak_w4_gemv_kt(X, L.wt4, L.st4, Y4, n, k, g, 0, nullptr));
            CK(ak_wq_gemv_kt_multi(bits, X, 1, 1, (const void* const*)&L.wtb, (const void* const*)&L.stb, &Yb, &n,
                                   nullptr, k, g, 0, nullptr));
            same("ak_w4_gemv_kt vs wq", get(Y4, n), get(Yb, n), bits, g, n, k, mode);
            CK(ak_w4_gemv_kt_batched(X, L.wt4, L.st4, Y4, 7, n, k, g, 0, nullptr));
            CK(ak_wq_gemv_kt_multi(bits, X, 7, 1, (const void* const*)&L.wtb, (const void* const*)&L.stb, &Yb, &n,
                                   nullptr, k, g, 0, nullptr));
            same("ak_w4_gemv_kt_batched vs wq", get(Y4, (size_t)7 * n), get(Yb, (size_t)7 * n), bits, g, n, k, mode);
          }
          // c. multi: the linear three times as three segments (row splits) vs W4 multi
          if (n >= 300) {
            const int n0 = n / 3 & ~15, n1 = n / 3, n2 = n - n0 - n1;
            const int ns[3] = {n0, n1, n2};
            std::vector<void*> y4(3), yb(3);
            for (int i = 0; i < 3; ++i) {
              CK(cudaMalloc(&y4[i], (size_t)n * 2));
              CK(cudaMalloc(&yb[i], (size_t)n * 2));
            }
            // separate repacks of the row slices (their own KT records), both widths
            std::vector<Lin> parts;
            const void *w4p[3], *s4p[3], *wbp[3], *sbp[3];
            for (int i = 0; i < 3; ++i) {
              Lin p = make(ns[i], k, g, bits);
              parts.push_back(p);
              w4p[i] = p.wt4, s4p[i] = p.st4, wbp[i] = p.wtb, sbp[i] = p.stb;
            }
            CK(ak_wq_gemv_kt_multi(4, X, 1, 3, w4p, s4p, y4.data(), ns, nullptr, k, g, bw, nullptr));
            CK(ak_wq_gemv_kt_multi(bits, X, 1, 3, wbp, sbp, yb.data(), ns, nullptr, k, g, bw, nullptr));
            for (int i = 0; i < 3; ++i) same("multi", get(y4[i], ns[i]), get(yb[i], ns[i]), bits, g, ns[i], k, mode);
            // d. two parts into one output of stride ldy (m = 1, 3, 8, 13)
            const int ldy[2] = {n0 + n1, n0 + n1};
            for (int m : {1, 3, 8, 13}) {
              void* o4[2] = {Y4, (char*)Y4 + (size_t)n0 * 2};
              void* ob[2] = {Yb, (char*)Yb + (size_t)n0 * 2};
              for (int pi = 0; pi < 2; ++pi) {  // each part on its own launch, as linear() issues them
                CK(ak_wq_gemv_kt_multi(4, X, m, 1, &w4p[pi], &s4p[pi], &o4[pi], &ns[pi], &ldy[pi], k, g, bw, nullptr));
                CK(ak_wq_gemv_kt_multi(bits, X, m, 1, &wbp[pi], &sbp[pi], &ob[pi], &ns[pi], &ldy[pi], k, g, bw,
                                       nullptr));
              }
              same("parts ldy", get(Y4, (size_t)m * ldy[0]), get(Yb, (size_t)m * ldy[0]), bits, g, n0 + n1, k, mode);
              // and the W4 parts == each part alone, row by row (ldy only moves rows)
              void* Z;
              CK(cudaMalloc(&Z, (size_t)m * ns[1] * 2));
              CK(ak_wq_gemv_kt_multi(4, X, m, 1, &w4p[1], &s4p[1], &Z, &ns[1], nullptr, k, g, bw, nullptr));
              const auto z = get(Z, (size_t)m * ns[1]), yy = get(Y4, (size_t)m * ldy[0]);
              std::vector<uint16_t> sl;
              for (int j = 0; j < m; ++j) sl.insert(sl.end(), yy.begin() + (size_t)j * ldy[0] + n0,
                                                    yy.begin() + (size_t)j * ldy[0] + n0 + ns[1]);
              same("ldy rows == plain rows", z, sl, 4, g, ns[1], k, mode);
              cudaFree(Z);
            }
            for (auto& p : parts) drop(p);
            for (int i = 0; i < 3; ++i) cudaFree(y4[i]), cudaFree(yb[i]);
          }
        }
        ak_w4_set_mma(0);
        // e. dequant, whole and row tiles
        {
          void *D4, *Db;
          CK(cudaMalloc(&D4, (size_t)n * k * 2));
          CK(cudaMalloc(&Db, (size_t)n * k * 2));
          CK(ak_w4_dequant_kt(L.wt4, L.st4, D4, n, k, g, nullptr));
          CK(ak_wq_dequant_kt_rows(bits, L.wtb, L.stb, Db, n, k, g, 0, n, nullptr));
          same("dequant", get(D4, (size_t)n * k), get(Db, (size_t)n * k), bits, g, n, k, 0);
          const int r0 = n / 3, nr = n / 2;
          CK(ak_w4_dequant_kt_rows(L.wt4, L.st4, D4, n, k, g, r0, nr, nullptr));
          CK(ak_wq_dequant_kt_rows(bits, L.wtb, L.stb, Db, n, k, g, r0, nr, nullptr));
          same("dequant rows", get(D4, (size_t)nr * k), get(Db, (size_t)nr * k), bits, g, nr, k, 0);
          cudaFree(D4), cudaFree(Db);
        }
        printf("W%d g%-3d [%5d, %5d]: checks so far %d, fails %d\n", bits, g, n, k, checks, fails);
        cudaFree(X), cudaFree(Y4), cudaFree(Yb);
        drop(L);
      }
  // i. batched verify rows == 8-row launches
  {
    ak_w4_set_mma(1);
    const Shape bs[] = {{5120, 6144}, {1280, 5120}, {301, 1152}};
    for (int bits : {2, 3, 4})
      for (int g : {64, 128})
        for (const Shape& sh : bs) {
          const int n = sh.n, k = sh.k;
          if (k % g) continue;
          Lin L = make(n, k, g, bits);
          const int MB = 72;
          std::vector<uint16_t> hx((size_t)MB * k);
          for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
          void* X = dup(hx);
          void *Ya, *Yb;
          CK(cudaMalloc(&Ya, (size_t)MB * n * 2));
          CK(cudaMalloc(&Yb, (size_t)MB * n * 2));
          for (int m : {72, 37, 13}) {
            CK(ak_wq_gemv_kt_multi(bits, X, m, 1, (const void* const*)&L.wtb, (const void* const*)&L.stb, &Ya, &n, nullptr,
                                   k, g, 0, nullptr));
            for (int m0 = 0; m0 < m; m0 += 8) {
              void* yb = (char*)Yb + (size_t)m0 * n * 2;
              CK(ak_wq_gemv_kt_multi(bits, (const char*)X + (size_t)m0 * k * 2, m - m0 < 8 ? m - m0 : 8, 1,
                                     (const void* const*)&L.wtb, (const void* const*)&L.stb, &yb, &n, nullptr, k, g, 0,
                                     nullptr));
            }
            char what[48];
            snprintf(what, sizeof what, "batched m=%d == 8-row launches", m);
            same(what, get(Ya, (size_t)m * n), get(Yb, (size_t)m * n), bits, g, n, k, 1);
          }
          cudaFree(X), cudaFree(Ya), cudaFree(Yb);
          drop(L);
        }
    // bf16 batch-invariant GEMV (+ bias)
    for (const Shape& sh : {Shape{24, 5120}, Shape{1280, 5120}, Shape{4096, 2048}, Shape{8448, 1024}, Shape{9000, 5120}}) {
      const int n = sh.n, k = sh.k, MB = 72;
      std::vector<uint16_t> hw((size_t)n * k), hx((size_t)MB * k), hb(n);
      for (auto& v : hw) v = f2bf(-0.05f + 0.1f * (rnd() >> 8) / 16777216.f);
      for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
      for (auto& v : hb) v = f2bf(-0.5f + (rnd() >> 8) / 16777216.f);
      void *Wd = dup(hw), *X = dup(hx), *Bd = dup(hb), *Ya, *Yb;
      CK(cudaMalloc(&Ya, (size_t)MB * n * 2));
      CK(cudaMalloc(&Yb, (size_t)MB * n * 2));
      for (int m : {72, 37, 13})
        for (void* bias : {(void*)nullptr, Bd}) {
          CK(ak_bf16_gemv_bi(X, Wd, bias, Ya, m, n, k, nullptr));
          for (int m0 = 0; m0 < m; m0 += 8)
            CK(ak_bf16_gemv_bi((const char*)X + (size_t)m0 * k * 2, Wd, bias, (char*)Yb + (size_t)m0 * n * 2,
                               m - m0 < 8 ? m - m0 : 8, n, k, nullptr));
          char what[48];
          snprintf(what, sizeof what, "bf16 bi m=%d%s", m, bias ? " +bias" : "");
          same(what, get(Ya, (size_t)m * n), get(Yb, (size_t)m * n), 16, 0, n, k, 1);
        }
      cudaFree(Wd), cudaFree(X), cudaFree(Bd), cudaFree(Ya), cudaFree(Yb);
    }
    printf("batched verify rows: checks so far %d, fails %d\n", checks, fails);
    ak_w4_set_mma(0);
  }
  // h. sm90 bulk-copy path == register MMA path (W4 g128)
  if (prop.major == 9) {
    ak_w4_set_mma(1);
    const Shape ts[] = {{34816, 5120}, {5120, 17408}, {5120, 6144}, {301, 1152}, {77, 640}, {16, 1024}, {17, 2048}};
    for (const Shape& sh : ts) {
      const int n = sh.n, k = sh.k;
      Lin L = make(n, k, 128, 4);
      std::vector<uint16_t> hx((size_t)M * k);
      for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
      void* X = dup(hx);
      void *Ya, *Yb;
      CK(cudaMalloc(&Ya, (size_t)M * n * 2));
      CK(cudaMalloc(&Yb, (size_t)M * n * 2));
      for (int stg : {2, 4, 8, -1, -2, -4, -8})  // -RB: the lean register path (-8: group split) (shapes it takes: n % 16 == 0, k % 1024 == 0)
        for (int m : {1, 2, 7, 8, 9, 16}) {
          ak_w4_set_tma(0);
          ak_w4_set_lean(0);
          CK(ak_wq_gemv_kt_multi(4, X, m, 1, (const void* const*)&L.wt4, (const void* const*)&L.st4, &Ya, &n, nullptr,
                                 k, 128, 0, nullptr));
          if (stg > 0) ak_w4_set_tma(stg);
          else ak_w4_set_lean(-stg);
          CK(ak_wq_gemv_kt_multi(4, X, m, 1, (const void* const*)&L.wt4, (const void* const*)&L.st4, &Yb, &n, nullptr,
                                 k, 128, 0, nullptr));
          ak_w4_set_lean(0);
          char what[48];
          snprintf(what, sizeof what, "tma%d == reg m=%d", stg, m);
          same(what, get(Ya, (size_t)m * n), get(Yb, (size_t)m * n), 4, 128, n, k, 1);
        }
      if (n >= 300) {  // 3 segments of their own KT records, and two parts into one output of stride ldy
        const int n0 = n / 3 & ~15, n1 = n / 3, n2 = n - n0 - n1, ns[3] = {n0, n1, n2};
        std::vector<Lin> parts;
        const void *wp[3], *sp[3];
        std::vector<void*> ya(3), yb(3);
        for (int i = 0; i < 3; ++i) {
          parts.push_back(make(ns[i], k, 128, 4));
          wp[i] = parts[i].wt4, sp[i] = parts[i].st4;
          CK(cudaMalloc(&ya[i], (size_t)8 * n * 2));
          CK(cudaMalloc(&yb[i], (size_t)8 * n * 2));
        }
        for (int m : {1, 8}) {
          ak_w4_set_tma(0);
          CK(ak_wq_gemv_kt_multi(4, X, m, 3, wp, sp, ya.data(), ns, nullptr, k, 128, 0, nullptr));
          ak_w4_set_tma(4);
          CK(ak_wq_gemv_kt_multi(4, X, m, 3, wp, sp, yb.data(), ns, nullptr, k, 128, 0, nullptr));
          for (int i = 0; i < 3; ++i)
            same("tma multi", get(ya[i], (size_t)m * ns[i]), get(yb[i], (size_t)m * ns[i]), 4, 128, ns[i], k, 1);
          ak_w4_set_tma(0);
          for (int rbl : {1, 2, 4, 8}) {  // segments padded to 16 rows; lean takes the call only when every n % 16 == 0
            ak_w4_set_lean(rbl);
            CK(ak_wq_gemv_kt_multi(4, X, m, 3, wp, sp, yb.data(), ns, nullptr, k, 128, 0, nullptr));
            ak_w4_set_lean(0);
            for (int i = 0; i < 3; ++i)
              same("lean multi", get(ya[i], (size_t)m * ns[i]), get(yb[i], (size_t)m * ns[i]), 4, 128, ns[i], k, 1);
          }

          const int ldy[2] = {n0 + n1, n0 + n1};
          for (int tm : {0, 4}) {
            ak_w4_set_tma(tm);
            void* o[2] = {tm ? Yb : Ya, (char*)(tm ? Yb : Ya) + (size_t)n0 * 2};
            for (int pi = 0; pi < 2; ++pi)
              CK(ak_wq_gemv_kt_multi(4, X, m, 1, &wp[pi], &sp[pi], &o[pi], &ns[pi], &ldy[pi], k, 128, 0, nullptr));
          }
          same("tma parts ldy", get(Ya, (size_t)m * ldy[0]), get(Yb, (size_t)m * ldy[0]), 4, 128, n0 + n1, k, 1);
        }
        for (auto& p : parts) drop(p);
        for (int i = 0; i < 3; ++i) cudaFree(ya[i]), cudaFree(yb[i]);
      }
      printf("tma W4 g128 [%5d, %5d]: checks so far %d, fails %d\n", n, k, checks, fails);
      cudaFree(X), cudaFree(Ya), cudaFree(Yb);
      drop(L);
    }
    ak_w4_set_tma(-1);
    ak_w4_set_lean(-1);
    ak_w4_set_mma(0);
  }
  // f. range: W2 refuses a 3-bit value, W3 a 4-bit value; W4 container values of W2 pass as W3
  {
    Lin L = make(64, 1024, 64, 3);  // values in [-4, 3]
    void *wt, *st;
    CK(cudaMalloc(&wt, ak_wq_kt_weight_bytes(2, 64, 1024)));
    CK(cudaMalloc(&st, ak_w4_kt_scale_bytes(64, 1024, 64)));
    ++checks;
    if (ak_wq_repack_kt(2, (const int32_t*)L.P, L.S, wt, st, 64, 1024, 64, nullptr) != (int)cudaErrorInvalidValue)
      ++fails, printf("FAIL W2 repack accepted 3-bit values\n");
    Lin M4 = make(64, 1024, 64, 4);
    ++checks;
    if (ak_wq_repack_kt(3, (const int32_t*)M4.P, M4.S, wt, st, 64, 1024, 64, nullptr) != (int)cudaErrorInvalidValue)
      ++fails, printf("FAIL W3 repack accepted 4-bit values\n");
    cudaGetLastError();
    Lin M2 = make(64, 1024, 64, 2);
    ++checks;
    if (ak_wq_repack_kt(3, (const int32_t*)M2.P, M2.S, wt, st, 64, 1024, 64, nullptr) != 0)
      ++fails, printf("FAIL W3 repack refused 2-bit values\n");
    cudaFree(wt), cudaFree(st);
    drop(L), drop(M4), drop(M2);
  }
  // g. g128 == g64 container with repeated scales on the exact path
  for (int mode = 0; mode < 2; ++mode) {
    const int n = 5120, k = 6144, M8 = 8;
    std::vector<uint32_t> hp((size_t)n * k / 8);
    for (auto& v : hp) v = rnd();
    std::vector<uint16_t> s128((size_t)n * k / 128), s64((size_t)n * k / 64), hx((size_t)M8 * k);
    for (auto& s : s128) s = f2bf(0.002f + 0.03f * (rnd() >> 8) / 16777216.f);
    for (size_t i = 0; i < s64.size(); ++i) s64[i] = s128[i / 2];
    for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
    void *P = dup(hp), *S1 = dup(s128), *S0 = dup(s64), *X = dup(hx), *w1, *t1, *w0, *t0, *Y1, *Y0;
    CK(cudaMalloc(&w1, ak_w4_kt_weight_bytes(n, k)));
    CK(cudaMalloc(&w0, ak_w4_kt_weight_bytes(n, k)));
    CK(cudaMalloc(&t1, ak_w4_kt_scale_bytes(n, k, 128)));
    CK(cudaMalloc(&t0, ak_w4_kt_scale_bytes(n, k, 64)));
    CK(cudaMalloc(&Y1, (size_t)M8 * n * 2));
    CK(cudaMalloc(&Y0, (size_t)M8 * n * 2));
    CK(ak_w4_repack_kt((const int32_t*)P, S1, w1, t1, n, k, 128, nullptr));
    CK(ak_w4_repack_kt((const int32_t*)P, S0, w0, t0, n, k, 64, nullptr));
    ak_w4_set_mma(mode);
    size_t d = 0;
    for (int m : {1, 4, 8}) {
      CK(ak_wq_gemv_kt_multi(4, X, m, 1, (const void* const*)&w1, (const void* const*)&t1, &Y1, &n, nullptr, k, 128, 0,
                             nullptr));
      CK(ak_wq_gemv_kt_multi(4, X, m, 1, (const void* const*)&w0, (const void* const*)&t0, &Y0, &n, nullptr, k, 64, 0,
                             nullptr));
      const auto a = get(Y1, (size_t)m * n), b = get(Y0, (size_t)m * n);
      for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    }
    if (mode == 0) {
      ++checks;
      if (d) ++fails;
      printf("%s g128 == g64 container (repeated scales), exact path: %zu outputs differ\n", d ? "FAIL" : "ok", d);
    } else {
      printf("info: g128 vs g64 container on the tensor-core path: %zu of %d outputs differ (fp32 sum order)\n", d,
             13 * n);
    }
    ak_w4_set_mma(0);
    for (void* p : {P, S1, S0, X, w1, t1, w0, t0, Y1, Y0}) cudaFree(p);
  }
  // h. RoPE on the fly (ak_rope_neox_fly) == RoPE from the ak_rope_cache table, bitwise: target (rot 64 of 256) and
  //    drafter (rot 128 of 128) shapes, positions up to 262147
  for (int shape = 0; shape < 2; ++shape) {
    const int rot = shape ? 128 : 64, hd = shape ? 128 : 256, nq = shape ? 32 : 24, nk = shape ? 8 : 4, T = 64;
    const int maxp = 262148;
    float* cache;
    CK(cudaMalloc(&cache, (size_t)maxp * rot * 4));
    CK(ak_rope_cache(cache, maxp, rot, 10000000.0f, nullptr));
    std::vector<uint16_t> hq((size_t)T * nq * hd), hk((size_t)T * nk * hd);
    for (auto& v : hq) v = f2bf(-2.f + 4.f * (rnd() >> 8) / 16777216.f);
    for (auto& v : hk) v = f2bf(-2.f + 4.f * (rnd() >> 8) / 16777216.f);
    std::vector<int64_t> hp(T);
    for (int t = 0; t < T; ++t) hp[t] = t < 4 ? maxp - 1 - t : (int64_t)(rnd() % maxp);
    void *q1 = dup(hq), *k1 = dup(hk), *q2 = dup(hq), *k2 = dup(hk), *P = dup(hp);
    CK(ak_rope_neox(q1, k1, cache, (const int64_t*)P, T, nq, nk, hd, rot, nq * hd, nk * hd, nullptr));
    CK(ak_rope_neox_fly(q2, k2, (const int64_t*)P, T, nq, nk, hd, rot, nq * hd, nk * hd, 10000000.0f, nullptr));
    same("rope q", get(q1, hq.size()), get(q2, hq.size()), 0, rot, T, hd, 0);
    same("rope k", get(k1, hk.size()), get(k2, hk.size()), 0, rot, T, hd, 0);
    for (void* p : {(void*)cache, q1, k1, q2, k2, P}) cudaFree(p);
  }
  printf("%s: %d checks, %d failed\n", fails ? "WQ_GATE_FAIL" : "WQ_GATE_PASS", checks, fails);
  return fails ? 1 : 0;
}
