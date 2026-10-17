// W4 kernel gate (KT layout, group 64/128): run on the target GPU, exit status 0 = all checks pass.
//  1. g128: ak_w4_gemv_kt == the CT kernel (ak_w4g128_gemv_bf16, unchanged) == every g128 KT tuning variant, bitwise
//     (the CT kernel is the one the original KT layout was proven bit-identical to, so g128 bits are unchanged).
//  2. GEMV vs an fp64 reference of the fp32-dequantized matmul: |y - ref| <= 1 bf16 ulp of ref + 1e-6 sum|x*w|
//     (fp32 accumulation slack; reports the share of outputs equal to round-to-nearest(ref)).
//  3. Batched GEMV, m = 2..16: every row bit-identical to the m = 1 GEMV of that row.
//  4. Dequant: bitwise equal to bf16(q * scale) on the host; dequant + ak_bf16_linear_f32reduce (m = 17, 64, 300)
//     within the same bound of an fp64 reference over the bf16-dequantized weights, and the error vs the
//     fp32-dequantized reference normalised by sum |x * w| (the weight rounding Marlin also has).
//  5. bf16w = 1 (Marlin numerics, weights bf16(q * scale)): GEMV within the bound of fp64 over those weights, batched
//     rows bit-identical to the GEMV, share of outputs bitwise equal to dequant + cuBLAS. 6. Multi-linear launch ==
//     separate GEMVs, both modes.
//  7. Tensor-core path (g128): same reference bound, rows bit-identical for m = 1..16, entry points route to it.
//  8. Timing on Volundr decode shapes: GEMV g128 vs g64, batched m = 2/4/9/16 (us and effective GB/s), dequant+GEMM.
// Usage: test_w4 [--no-timing]

#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (cudaError_t)(x);                                                         \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));      \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

static uint32_t rng_state = 12345;
static uint32_t rnd() {
  rng_state ^= rng_state << 13, rng_state ^= rng_state >> 17, rng_state ^= rng_state << 5;
  return rng_state;
}
static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t f2bf(float f) {  // round to nearest even (finite inputs)
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static double ulp_bf16(double v) {
  const double a = std::fabs(v);
  if (a < 1e-30) return 1e-30;
  return std::ldexp(1.0, (int)std::floor(std::log2(a)) - 7);
}

struct Case { int n, k, group; };

int main(int argc, char** argv) {
  const bool timing = !(argc > 1 && !strcmp(argv[1], "--no-timing"));
  ak_w4_set_mma(0);  // sections 1-6: the exact path (bit-identity checks); section 7: the tensor-core path
  int dev = 0;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("# device %d %s\n", dev, prop.name);
  const Case cases[] = {
      {34816, 5120, 128}, {5120, 17408, 128}, {10240, 5120, 128}, {12288, 5120, 128}, {5120, 6144, 128},
      {128, 5120, 128},   {1024, 5120, 128},  {301, 1152, 128},   {5120, 17408, 64},  {10240, 5120, 64},
      {5120, 6144, 64},   {128, 5120, 64},    {301, 1152, 64},    {77, 640, 64},
  };
  int fails = 0;
  for (const Case& cs : cases) {
    const int n = cs.n, k = cs.k, g = cs.group, groups = k / g, M = 16;
    std::vector<uint32_t> hp((size_t)n * k / 8);
    std::vector<uint16_t> hs((size_t)n * groups), hx((size_t)300 * k);
    for (auto& v : hp) v = rnd();
    for (auto& v : hs) v = f2bf(0.001f + 0.02f * (rnd() >> 8) / 16777216.f);
    for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
    int32_t* P;
    void *S, *X, *Y, *Y1, *WT, *ST, *WD;
    const size_t wkb = ak_w4_kt_weight_bytes(n, k), skb = ak_w4_kt_scale_bytes(n, k, g);
    CK(cudaMalloc(&P, hp.size() * 4));
    CK(cudaMalloc(&S, hs.size() * 2));
    CK(cudaMalloc(&X, hx.size() * 2));
    CK(cudaMalloc(&Y, (size_t)300 * n * 2));
    CK(cudaMalloc(&Y1, (size_t)n * 2));
    CK(cudaMalloc(&WT, wkb));
    CK(cudaMalloc(&ST, skb));
    CK(cudaMalloc(&WD, (size_t)n * k * 2));
    CK(cudaMemcpy(P, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(ak_w4_repack_kt(P, S, WT, ST, n, k, g, nullptr));
    auto q_at = [&](size_t o, size_t i) { return (int)((hp[(o * k + i) / 8] >> (4 * (i % 8))) & 0xF) - 8; };
    auto w_exact = [&](size_t o, size_t i) { return (double)q_at(o, i) * bf2f(hs[o * groups + i / g]); };
    auto w_bq = [&](size_t o, size_t i) { return (double)bf2f(f2bf((float)q_at(o, i) * bf2f(hs[o * groups + i / g]))); };
    const int step = n > 2000 ? 13 : 1;
    char tag[64];
    snprintf(tag, sizeof tag, "n=%5d k=%5d g%d", n, k, g);

    // 1 + 2: m = 1 GEMV per x row (rows 0..M-1), reference, CT identity
    std::vector<std::vector<uint16_t>> y1(M, std::vector<uint16_t>(n));
    bool ct_same = true, var_same = true;
    double max_ulp = 0;
    size_t nearest = 0, checked = 0;
    for (int i = 0; i < M; ++i) {
      const void* xi = (const uint16_t*)X + (size_t)i * k;
      CK(ak_w4_gemv_kt(xi, WT, ST, Y1, n, k, g, 0, nullptr));
      CK(cudaMemcpy(y1[i].data(), Y1, n * 2, cudaMemcpyDeviceToHost));
      if (g == 128) {
        std::vector<uint16_t> c(n);
        CK(ak_w4g128_gemv_bf16(xi, P, S, Y1, n, k, nullptr));
        CK(cudaMemcpy(c.data(), Y1, n * 2, cudaMemcpyDeviceToHost));
        ct_same &= c == y1[i];
        for (int v : {11, 12, 21, 22, 41}) {
          CK(ak_w4g128_gemv_kt_variant(v, xi, WT, ST, Y1, n, k, nullptr));
          CK(cudaMemcpy(c.data(), Y1, n * 2, cudaMemcpyDeviceToHost));
          var_same &= c == y1[i];
        }
      }
      if (i < 4)
        for (int o = 0; o < n; o += step) {
          double ref = 0, mag = 0;
          for (int j = 0; j < k; ++j) {
            const double t = (double)bf2f(hx[(size_t)i * k + j]) * w_exact(o, j);
            ref += t, mag += std::fabs(t);
          }
          const double e = std::fabs(bf2f(y1[i][o]) - ref) / (ulp_bf16(ref) + 1e-6 * mag);
          max_ulp = std::max(max_ulp, e);
          nearest += y1[i][o] == f2bf((float)ref);
          ++checked;
        }
    }
    const bool ok_ref = max_ulp <= 1.0;
    printf("%s  gemv vs fp64(fp32-dequant): max %.3f ulp, %.2f%% = rn(ref)  %s", tag, max_ulp, 100.0 * nearest / checked,
           ok_ref ? "ok" : "FAIL");
    if (g == 128) printf(" | CT == KT %s, all KT variants %s", ct_same ? "bit-identical" : "DIFFER", var_same ? "bit-identical" : "DIFFER");
    printf("\n");
    fails += !ok_ref + !ct_same + !var_same;

    // 3: batched m = 2..16 rows == m = 1 rows
    bool bat_same = true;
    for (int m = 2; m <= M; ++m) {
      CK(ak_w4_gemv_kt_batched(X, WT, ST, Y, m, n, k, g, 0, nullptr));
      std::vector<uint16_t> yb((size_t)m * n);
      CK(cudaMemcpy(yb.data(), Y, yb.size() * 2, cudaMemcpyDeviceToHost));
      for (int i = 0; i < m; ++i) bat_same &= std::equal(y1[i].begin(), y1[i].end(), yb.begin() + (size_t)i * n);
    }
    {
      CK(ak_w4_gemv_kt_batched(X, WT, ST, Y, 1, n, k, g, 0, nullptr));
      std::vector<uint16_t> yb(n);
      CK(cudaMemcpy(yb.data(), Y, n * 2, cudaMemcpyDeviceToHost));
      bat_same &= yb == y1[0];
    }
    printf("%s  batched m=1..16 rows vs m=1 GEMV: %s\n", tag, bat_same ? "bit-identical" : "DIFFER");
    fails += !bat_same;

    // 4: dequant + cuBLAS
    CK(ak_w4_dequant_kt(WT, ST, WD, n, k, g, nullptr));
    std::vector<uint16_t> hw((size_t)n * k);
    CK(cudaMemcpy(hw.data(), WD, hw.size() * 2, cudaMemcpyDeviceToHost));
    bool dq_same = true;
    for (size_t o = 0; o < (size_t)n; o += step)
      for (size_t i = 0; i < (size_t)k; ++i) dq_same &= hw[o * k + i] == f2bf((float)q_at(o, i) * bf2f(hs[o * groups + i / g]));
    printf("%s  dequant == bf16(q*s): %s\n", tag, dq_same ? "bit-identical" : "DIFFER");
    fails += !dq_same;
    for (int m : {17, 64, 300}) {
      CK(ak_bf16_linear_f32reduce(X, WD, Y, m, n, k, nullptr));
      std::vector<uint16_t> yg((size_t)m * n);
      CK(cudaMemcpy(yg.data(), Y, yg.size() * 2, cudaMemcpyDeviceToHost));
      double mu = 0, mrel = 0;
      for (int i = 0; i < m; i += std::max(1, m / 5))
        for (int o = 0; o < n; o += step * 3) {
          double rb = 0, re = 0, mag = 0;
          for (int j = 0; j < k; ++j) {
            const double xv = bf2f(hx[(size_t)i * k + j]);
            rb += xv * w_bq(o, j), re += xv * w_exact(o, j), mag += std::fabs(xv * w_exact(o, j));
          }
          const double y = bf2f(yg[(size_t)i * n + o]);
          mu = std::max(mu, std::fabs(y - rb) / (ulp_bf16(rb) + 1e-6 * mag));
          mrel = std::max(mrel, std::fabs(y - re) / mag);
        }
      const bool ok = mu <= 1.0;
      printf("%s  dequant+gemm m=%3d: max %.3f ulp vs fp64(bf16-dequant); vs fp32-dequant max |d|/sum|xw| %.2e  %s\n",
             tag, m, mu, mrel, ok ? "ok" : "FAIL");
      fails += !ok;
    }

    // 5: bf16w = 1 (Marlin numerics): GEMV vs fp64 over the bf16-dequantized weights, batched rows == GEMV rows, and
    //    the share of outputs bitwise equal to dequant + cuBLAS (same weights, different fp32 summation order)
    {
      std::vector<std::vector<uint16_t>> yw(M, std::vector<uint16_t>(n));
      double mu = 0;
      for (int i = 0; i < M; ++i) {
        CK(ak_w4_gemv_kt((const uint16_t*)X + (size_t)i * k, WT, ST, Y1, n, k, g, 1, nullptr));
        CK(cudaMemcpy(yw[i].data(), Y1, n * 2, cudaMemcpyDeviceToHost));
        if (i < 2)
          for (int o = 0; o < n; o += step) {
            double rb = 0, mag = 0;
            for (int j = 0; j < k; ++j) {
              const double t = (double)bf2f(hx[(size_t)i * k + j]) * w_bq(o, j);
              rb += t, mag += std::fabs(t);
            }
            mu = std::max(mu, std::fabs(bf2f(yw[i][o]) - rb) / (ulp_bf16(rb) + 1e-6 * mag));
          }
      }
      bool bsame = true;
      for (int m : {2, 9, 16}) {
        CK(ak_w4_gemv_kt_batched(X, WT, ST, Y, m, n, k, g, 1, nullptr));
        std::vector<uint16_t> yb((size_t)m * n);
        CK(cudaMemcpy(yb.data(), Y, yb.size() * 2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < m; ++i) bsame &= std::equal(yw[i].begin(), yw[i].end(), yb.begin() + (size_t)i * n);
      }
      CK(ak_bf16_linear_f32reduce(X, WD, Y, M, n, k, nullptr));
      std::vector<uint16_t> yg((size_t)M * n);
      CK(cudaMemcpy(yg.data(), Y, yg.size() * 2, cudaMemcpyDeviceToHost));
      size_t eq = 0, eq0 = 0;
      for (int i = 0; i < M; ++i)
        for (int o = 0; o < n; ++o) eq += yw[i][o] == yg[(size_t)i * n + o], eq0 += y1[i][o] == yg[(size_t)i * n + o];
      const bool ok = mu <= 1.0 && bsame;
      printf("%s  bf16w gemv: max %.3f ulp vs fp64(bf16-dequant), batched rows %s; == dequant+cuBLAS %.2f%% "
             "(exact-q gemv %.2f%%)  %s\n", tag, mu, bsame ? "bit-identical" : "DIFFER", 100.0 * eq / ((size_t)M * n),
             100.0 * eq0 / ((size_t)M * n), ok ? "ok" : "FAIL");
      fails += !ok;
    }
    cudaFree(P), cudaFree(S), cudaFree(X), cudaFree(Y), cudaFree(Y1), cudaFree(WT), cudaFree(ST), cudaFree(WD);
  }

  // 6: multi-linear launch (KDA in_proj_qkv + in_proj_z + f_down; attention q/k/v): each output == its own GEMV
  for (int bw : {0, 1})
  for (int g : {128, 64}) {
    const int k = 5120;
    const std::vector<std::vector<int>> sets = {{10240, 6144, 128}, {12288, 1024, 1024}, {301, 77}};
    for (const auto& ns : sets) {
      const int nseg = (int)ns.size();
      std::vector<void*> wt(nseg), st(nseg), y(nseg), y1(nseg);
      void* X;
      CK(cudaMalloc(&X, (size_t)k * 2));
      std::vector<uint16_t> hx(k);
      for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
      CK(cudaMemcpy(X, hx.data(), k * 2, cudaMemcpyHostToDevice));
      for (int i = 0; i < nseg; ++i) {
        const int n = ns[i];
        std::vector<uint32_t> hp((size_t)n * k / 8);
        std::vector<uint16_t> hs((size_t)n * (k / g));
        for (auto& v : hp) v = rnd();
        for (auto& v : hs) v = f2bf(0.001f + 0.02f * (rnd() >> 8) / 16777216.f);
        void *P, *S;
        CK(cudaMalloc(&P, hp.size() * 4));
        CK(cudaMalloc(&S, hs.size() * 2));
        CK(cudaMemcpy(P, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&wt[i], ak_w4_kt_weight_bytes(n, k)));
        CK(cudaMalloc(&st[i], ak_w4_kt_scale_bytes(n, k, g)));
        CK(ak_w4_repack_kt((const int32_t*)P, S, wt[i], st[i], n, k, g, nullptr));
        CK(cudaMalloc(&y[i], (size_t)n * 2));
        CK(cudaMalloc(&y1[i], (size_t)n * 2));
        CK(ak_w4_gemv_kt(X, wt[i], st[i], y1[i], n, k, g, bw, nullptr));
        cudaFree(P), cudaFree(S);
      }
      CK(ak_w4_gemv_kt_multi(X, nseg, wt.data(), st.data(), y.data(), ns.data(), k, g, bw, nullptr));
      bool same = true;
      for (int i = 0; i < nseg; ++i) {
        std::vector<uint16_t> a(ns[i]), b(ns[i]);
        CK(cudaMemcpy(a.data(), y[i], ns[i] * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y1[i], ns[i] * 2, cudaMemcpyDeviceToHost));
        same &= a == b;
      }
      printf("multi bf16w=%d g%d %d linears (n0=%d): %s\n", bw, g, nseg, ns[0], same ? "bit-identical to separate GEMVs" : "DIFFER");
      fails += !same;
      if (timing) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        float ms_sep, ms_multi;
        for (int rep = 0; rep < 2; ++rep) {
          cudaEventRecord(e0);
          for (int it = 0; it < 200; ++it)
            for (int i = 0; i < nseg; ++i) ak_w4_gemv_kt(X, wt[i], st[i], y1[i], ns[i], k, g, bw, nullptr);
          cudaEventRecord(e1);
          CK(cudaEventSynchronize(e1));
          cudaEventElapsedTime(&ms_sep, e0, e1);
          cudaEventRecord(e0);
          for (int it = 0; it < 200; ++it) ak_w4_gemv_kt_multi(X, nseg, wt.data(), st.data(), y.data(), ns.data(), k, g, bw, nullptr);
          cudaEventRecord(e1);
          CK(cudaEventSynchronize(e1));
          cudaEventElapsedTime(&ms_multi, e0, e1);
        }
        printf("  timing (L2-warm): separate %.1f us, multi %.1f us\n", ms_sep * 5, ms_multi * 5);
      }
      for (int i = 0; i < nseg; ++i) cudaFree(wt[i]), cudaFree(st[i]), cudaFree(y[i]), cudaFree(y1[i]);
      cudaFree(X);
    }
  }

  // 7: tensor-core path (g128): vs the fp64 reference of the fp32-dequantized matmul (same bound), rows bit-identical
  //    for m = 1..16, multi launch == separate launches
  for (const Case& cs : {Case{34816, 5120, 128}, Case{5120, 17408, 128}, Case{12288, 5120, 128}, Case{1024, 5120, 128},
                         Case{304, 1152, 128}}) {
    const int n = cs.n, k = cs.k, groups = k / 128, M = 16;
    std::vector<uint32_t> hp((size_t)n * k / 8);
    std::vector<uint16_t> hs((size_t)n * groups), hx((size_t)M * k);
    for (auto& v : hp) v = rnd();
    for (auto& v : hs) v = f2bf(0.001f + 0.02f * (rnd() >> 8) / 16777216.f);
    for (auto& v : hx) v = f2bf(-1.f + 2.f * (rnd() >> 8) / 16777216.f);
    void *P, *S, *X, *Y, *Y1, *WT, *ST;
    CK(cudaMalloc(&P, hp.size() * 4));
    CK(cudaMalloc(&S, hs.size() * 2));
    CK(cudaMalloc(&X, hx.size() * 2));
    CK(cudaMalloc(&Y, (size_t)M * n * 2));
    CK(cudaMalloc(&Y1, (size_t)n * 2));
    CK(cudaMalloc(&WT, ak_w4_kt_weight_bytes(n, k)));
    CK(cudaMalloc(&ST, ak_w4_kt_scale_bytes(n, k, 128)));
    CK(cudaMemcpy(P, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(ak_w4_repack_kt((const int32_t*)P, S, WT, ST, n, k, 128, nullptr));
    std::vector<std::vector<uint16_t>> y1(M, std::vector<uint16_t>(n));
    double mu = 0;
    for (int i = 0; i < M; ++i) {
      CK(ak_w4_mma_gemv((const uint16_t*)X + (size_t)i * k, WT, ST, Y1, 1, n, k, 128, nullptr));
      CK(cudaMemcpy(y1[i].data(), Y1, n * 2, cudaMemcpyDeviceToHost));
      if (i < 2)
        for (int o = 0; o < n; o += n > 2000 ? 13 : 1) {
          double ref = 0, mag = 0;
          for (int j = 0; j < k; ++j) {
            const int q = (int)((hp[((size_t)o * k + j) / 8] >> (4 * (j % 8))) & 0xF) - 8;
            const double t = (double)bf2f(hx[(size_t)i * k + j]) * q * bf2f(hs[(size_t)o * groups + j / 128]);
            ref += t, mag += std::fabs(t);
          }
          mu = std::max(mu, std::fabs(bf2f(y1[i][o]) - ref) / (ulp_bf16(ref) + 1e-6 * mag));
        }
    }
    bool bsame = true;
    for (int m = 2; m <= M; ++m) {
      CK(ak_w4_mma_gemv(X, WT, ST, Y, m, n, k, 128, nullptr));
      std::vector<uint16_t> yb((size_t)m * n);
      CK(cudaMemcpy(yb.data(), Y, yb.size() * 2, cudaMemcpyDeviceToHost));
      for (int i = 0; i < m; ++i) bsame &= std::equal(y1[i].begin(), y1[i].end(), yb.begin() + (size_t)i * n);
    }
    ak_w4_set_mma(1);  // the production entry points route to the same kernel
    bool route = true;
    {
      CK(ak_w4_gemv_kt(X, WT, ST, Y1, n, k, 128, 0, nullptr));
      std::vector<uint16_t> a(n);
      CK(cudaMemcpy(a.data(), Y1, n * 2, cudaMemcpyDeviceToHost));
      route &= a == y1[0];
      CK(ak_w4_gemv_kt_batched(X, WT, ST, Y, 9, n, k, 128, 0, nullptr));
      std::vector<uint16_t> yb((size_t)9 * n);
      CK(cudaMemcpy(yb.data(), Y, yb.size() * 2, cudaMemcpyDeviceToHost));
      for (int i = 0; i < 9; ++i) route &= std::equal(y1[i].begin(), y1[i].end(), yb.begin() + (size_t)i * n);
      const int half = n / 32 * 16;  // two segments of the same matrix rows: [0, half) and [half, n)
      if (half > 0) {
        const void* wts[2] = {WT, WT};
        const void* sts[2] = {ST, ST};
        void* ys[2] = {Y, (uint16_t*)Y + n};
        const int ns[2] = {n, n};
        CK(ak_w4_gemv_kt_multi(X, 2, wts, sts, ys, ns, k, 128, 0, nullptr));
        std::vector<uint16_t> b(2 * (size_t)n);
        CK(cudaMemcpy(b.data(), Y, b.size() * 2, cudaMemcpyDeviceToHost));
        route &= std::equal(y1[0].begin(), y1[0].end(), b.begin()) && std::equal(y1[0].begin(), y1[0].end(), b.begin() + n);
      }
    }
    ak_w4_set_mma(0);
    const bool ok = mu <= 1.0 && bsame && route;
    printf("n=%5d k=%5d g128  MMA gemv: max %.3f ulp vs fp64(fp32-dequant); m=1..16 rows %s; entry points %s  %s\n", n, k, mu,
           bsame ? "bit-identical" : "DIFFER", route ? "route to it" : "DIFFER", ok ? "ok" : "FAIL");
    fails += !ok;
    cudaFree(P), cudaFree(S), cudaFree(X), cudaFree(Y), cudaFree(Y1), cudaFree(WT), cudaFree(ST);
  }

  if (timing) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const Case tc[] = {{34816, 5120, 128}, {34816, 5120, 64}, {5120, 17408, 128}, {5120, 17408, 64},
                       {10240, 5120, 128}, {5120, 6144, 128}, {1024, 5120, 128}};
    for (const Case& cs : tc) {
      const int n = cs.n, k = cs.k, g = cs.group;
      const size_t wkb = ak_w4_kt_weight_bytes(n, k), skb = ak_w4_kt_scale_bytes(n, k, g), bytes = wkb + skb;
      const int copies = (int)std::max<size_t>(2, (512ull << 20) / bytes + 1);  // rotate past L2
      std::vector<void*> WT(copies), ST(copies);
      for (int c = 0; c < copies; ++c) {
        CK(cudaMalloc(&WT[c], wkb));
        CK(cudaMalloc(&ST[c], skb));
        CK(cudaMemset(WT[c], 0x37 + c, wkb));
        CK(cudaMemset(ST[c], 0x1c, skb));
      }
      void *X, *Y, *WD;
      CK(cudaMalloc(&X, (size_t)4096 * k * 2));
      CK(cudaMemset(X, 0x3c, (size_t)4096 * k * 2));
      CK(cudaMalloc(&Y, (size_t)4096 * n * 2));
      CK(cudaMalloc(&WD, (size_t)n * k * 2));
      auto time_us = [&](auto&& f, int iters) {
        for (int i = 0; i < 5; ++i) f(i);
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) f(i);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms * 1000.0 / iters;
      };
      printf("timing n=%5d k=%5d g%-3d (%.1f MB):", n, k, g, bytes / 1e6);
      const double t1 = time_us([&](int i) { ak_w4_gemv_kt(X, WT[i % copies], ST[i % copies], Y, n, k, g, 0, nullptr); }, 200);
      printf(" gemv %.1f us %.0f GB/s |", t1, bytes / t1 / 1e3);
      for (int m : {2, 4, 9, 16}) {
        const double t = time_us([&](int i) { ak_w4_gemv_kt_batched(X, WT[i % copies], ST[i % copies], Y, m, n, k, g, 0, nullptr); }, 100);
        printf(" m%d %.1f us (%.2fx) |", m, t, t / t1);
      }
      if (g == 128) {
        ak_w4_set_mma(1);
        const double tm = time_us([&](int i) { ak_w4_gemv_kt(X, WT[i % copies], ST[i % copies], Y, n, k, g, 0, nullptr); }, 200);
        printf(" MMA m1 %.1f us %.0f GB/s", tm, bytes / tm / 1e3);
        for (int m : {4, 9, 16}) {
          const double t = time_us([&](int i) { ak_w4_gemv_kt_batched(X, WT[i % copies], ST[i % copies], Y, m, n, k, g, 0, nullptr); }, 100);
          printf(" m%d %.1f us (%.2fx)", m, t, t / tm);
        }
        printf(" |");
        ak_w4_set_mma(0);
      }
      for (int m : {64, 4096}) {
        const double t = time_us([&](int i) {
          ak_w4_dequant_kt(WT[i % copies], ST[i % copies], WD, n, k, g, nullptr);
          ak_bf16_linear_f32reduce(X, WD, Y, m, n, k, nullptr);
        }, m > 1000 ? 10 : 50);
        printf(" dq+gemm m%d %.0f us", m, t);
      }
      printf("\n");
      for (int c = 0; c < copies; ++c) cudaFree(WT[c]), cudaFree(ST[c]);
      cudaFree(X), cudaFree(Y), cudaFree(WD);
    }
  }
  printf(fails ? "W4 GATE: %d checks FAILED\n" : "W4 GATE: all checks pass\n", fails);
  return fails ? 1 : 0;
}
