// Microbench: W4 g128 GEMV variants vs cuBLAS bf16 GEMV on Volundr decode shapes.
// Weights are rotated through >= 512 MB of copies so nothing is served from L2 (sm89 48 GB card: L2 = 72 MB).
// Usage: bench_gemv [iters]      Output: one line per (shape, kernel): us/call, effective GB/s, max rel err.

#include "agens_kernels.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>


#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));      \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

__device__ __forceinline__ uint32_t hash32(uint32_t a) {
  a ^= a >> 16; a *= 0x7feb352dU; a ^= a >> 15; a *= 0x846ca68bU; a ^= a >> 16;
  return a;
}
__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = hash32((uint32_t)i * 2654435761u + seed);
}
// bf16 values in roughly [-1, 1) (scale = small positive for scales)
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed, float lo, float hi) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(lo + (hi - lo) * (hash32((uint32_t)i * 2246822519u + seed) >> 8) * (1.0f / 16777216.0f));
}

// Read roofline: stream a buffer with 16-byte loads, same access shape as the GEMV (warp-contiguous rows).
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}

static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

struct Shape { const char* name; int n, k; };

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 200;
  const Shape shapes[] = {
      {"mlp.gate_up (fused)", 34816, 5120}, {"mlp.down", 5120, 17408},   {"attn.qkv (fused)", 14336, 5120},
      {"attn.o", 5120, 6144},               {"kda.in_proj_qkv", 10240, 5120}, {"kda.in_proj_z", 6144, 5120},
      {"kda.out_proj", 5120, 6144},         {"engram.value", 5120, 1024},
  };
  const int variants[] = {12, 21, 22};
  const int kt_variants[] = {11, 12, 21, 22, 41};
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("# device %d %s, %d SMs, L2 %d MB\n", dev, prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);
  cublasHandle_t h;
  cublasCreate(&h);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  {  // device read roofline over 1 GiB
    const size_t bytes = 1ull << 30;
    uint4* buf;
    uint32_t* sink;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMalloc(&sink, 4));
    fill_u32<<<1024, 256>>>((uint32_t*)buf, bytes / 4, 1);
    for (int blocks : {prop.multiProcessorCount * 4, prop.multiProcessorCount * 8, prop.multiProcessorCount * 16}) {
      read_kernel<<<blocks, 256>>>(buf, bytes / 16, sink);
      cudaEventRecord(e0);
      for (int i = 0; i < 10; ++i) read_kernel<<<blocks, 256>>>(buf, bytes / 16, sink);
      cudaEventRecord(e1);
      CK(cudaEventSynchronize(e1));
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      printf("# read roofline (%d blocks x 256): %.1f GB/s\n", blocks, bytes * 10 / (ms * 1e6));
    }
    cudaFree(buf);
    cudaFree(sink);
  }
  for (const Shape& s : shapes) {
    const size_t n = s.n, k = s.k;
    const size_t wq = n * k / 2, sc = n * (k / 128) * 2, wb = n * k * 2;
    const size_t w4_bytes = wq + sc;
    const int copies_q = (int)std::max<size_t>(2, (512ull << 20) / w4_bytes + 1);
    const int copies_b = (int)std::max<size_t>(2, (512ull << 20) / wb + 1);
    std::vector<int32_t*> P(copies_q);
    std::vector<__nv_bfloat16*> S(copies_q), B(copies_b);
    for (int c = 0; c < copies_q; ++c) {
      CK(cudaMalloc(&P[c], wq));
      CK(cudaMalloc(&S[c], sc));
      fill_u32<<<1024, 256>>>((uint32_t*)P[c], wq / 4, 17 + c);
      fill_bf16<<<1024, 256>>>(S[c], sc / 2, 99 + c, 0.001f, 0.02f);
    }
    for (int c = 0; c < copies_b; ++c) {
      CK(cudaMalloc(&B[c], wb));
      fill_bf16<<<1024, 256>>>(B[c], wb / 2, 7 + c, -0.05f, 0.05f);
    }
    __nv_bfloat16 *x, *y;
    CK(cudaMalloc(&x, k * 2));
    CK(cudaMalloc(&y, n * 2));
    fill_bf16<<<64, 256>>>(x, k, 5, -1.f, 1.f);
    CK(cudaDeviceSynchronize());

    // reference for copy 0 on a row sample
    std::vector<uint16_t> hx(k), hs(n * (k / 128)), hy(n);
    std::vector<uint32_t> hp(wq / 4);
    CK(cudaMemcpy(hx.data(), x, k * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), S[0], sc, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hp.data(), P[0], wq, cudaMemcpyDeviceToHost));
    std::vector<double> ref(n);
    for (size_t o = 0; o < n; o += 7) {
      double acc = 0;
      for (size_t i = 0; i < k; ++i) {
        const uint32_t w = hp[(o * k + i) / 8];
        const int q = (int)((w >> (4 * (i % 8))) & 0xF) - 8;
        acc += (double)bf2f(hx[i]) * q * bf2f(hs[o * (k / 128) + i / 128]);
      }
      ref[o] = acc;
    }

    for (int v : variants) {
      CK((cudaError_t)ak_w4g128_gemv_bf16_variant(v, x, P[0], S[0], y, n, k, nullptr));
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
      double maxrel = 0;
      for (size_t o = 0; o < n; o += 7)
        maxrel = std::max(maxrel, std::fabs(bf2f(hy[o]) - ref[o]) / (std::fabs(ref[o]) + 1e-2));
      // determinism: second run must be bit-identical
      std::vector<uint16_t> hy2(n);
      CK((cudaError_t)ak_w4g128_gemv_bf16_variant(v, x, P[0], S[0], y, n, k, nullptr));
      CK(cudaMemcpy(hy2.data(), y, n * 2, cudaMemcpyDeviceToHost));
      const bool det = hy2 == hy;
      for (int i = 0; i < 10; ++i)
        ak_w4g128_gemv_bf16_variant(v, x, P[i % copies_q], S[i % copies_q], y, n, k, nullptr);
      cudaEventRecord(e0);
      for (int i = 0; i < iters; ++i)
        ak_w4g128_gemv_bf16_variant(v, x, P[i % copies_q], S[i % copies_q], y, n, k, nullptr);
      cudaEventRecord(e1);
      CK(cudaEventSynchronize(e1));
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      const double us = ms * 1000.0 / iters;
      printf("%-22s n=%6zu k=%6zu  w4 R%dU%d   %8.2f us  %7.1f GB/s  maxrel %.2e  %s\n", s.name, n, k, v / 10, v % 10,
             us, w4_bytes / us / 1e3, maxrel, det ? "det" : "NONDET");
    }
    {  // KT layout: repack every copy, check bit-identity with the CT result, time
      std::vector<uint16_t> ct(n), kt(n);
      CK((cudaError_t)ak_w4g128_gemv_bf16_variant(22, x, P[0], S[0], y, n, k, nullptr));
      CK(cudaMemcpy(ct.data(), y, n * 2, cudaMemcpyDeviceToHost));
      const size_t wkb = ak_w4g128_kt_weight_bytes(n, k), skb = ak_w4g128_kt_scale_bytes(n, k);
      std::vector<void*> WT(copies_q), ST(copies_q);
      for (int c = 0; c < copies_q; ++c) {
        CK(cudaMalloc(&WT[c], wkb));
        CK(cudaMalloc(&ST[c], skb));
        CK((cudaError_t)ak_w4g128_repack_kt(P[c], S[c], WT[c], ST[c], n, k, nullptr));
      }
      for (int v : kt_variants) {
        CK((cudaError_t)ak_w4g128_gemv_kt_variant(v, x, WT[0], ST[0], y, n, k, nullptr));
        CK(cudaMemcpy(kt.data(), y, n * 2, cudaMemcpyDeviceToHost));
        const bool same = kt == ct;
        for (int i = 0; i < 10; ++i) ak_w4g128_gemv_kt_variant(v, x, WT[i % copies_q], ST[i % copies_q], y, n, k, nullptr);
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i)
          ak_w4g128_gemv_kt_variant(v, x, WT[i % copies_q], ST[i % copies_q], y, n, k, nullptr);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = ms * 1000.0 / iters;
        printf("%-22s n=%6zu k=%6zu  kt R%dU%d   %8.2f us  %7.1f GB/s  %s\n", s.name, n, k, v / 10, v % 10, us,
               (wkb + skb) / us / 1e3, same ? "bit-identical to CT" : "DIFFERS FROM CT");
      }
      for (int c = 0; c < copies_q; ++c) { cudaFree(WT[c]); cudaFree(ST[c]); }
    }
    {
      const float alpha = 1.f, beta = 0.f;
      auto run = [&](int c) {
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, 1, (int)k, &alpha, B[c], CUDA_R_16BF, (int)k, x,
                            CUDA_R_16BF, (int)k, &beta, y, CUDA_R_16BF, (int)n, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT);
      };
      for (int i = 0; i < 10; ++i) run(i % copies_b);
      cudaEventRecord(e0);
      for (int i = 0; i < iters; ++i) run(i % copies_b);
      cudaEventRecord(e1);
      CK(cudaEventSynchronize(e1));
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      const double us = ms * 1000.0 / iters;
      printf("%-22s n=%6zu k=%6zu  cublas bf16 %8.2f us  %7.1f GB/s\n", s.name, n, k, us, wb / us / 1e3);
    }
    for (int c = 0; c < copies_q; ++c) { cudaFree(P[c]); cudaFree(S[c]); }
    for (int c = 0; c < copies_b; ++c) cudaFree(B[c]);
    cudaFree(x);
    cudaFree(y);
  }
  return 0;
}
