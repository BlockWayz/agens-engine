// Batch-invariant bf16 GEMV for decode (server mode, AGENS_M1_BATCH_INVARIANT=1): y[m, n] = x[m, k] · W[n, k]^T for
// m <= 16, fp32 accumulation. Each output element is computed by one warp in a fixed order (lane-strided fma chain over
// k in 8-element chunks, then a fixed xor-butterfly), whatever m is, so row r of an m-row call is bit-identical to an
// m = 1 call with x[r]. (cuBLAS is not: different m picks different kernels; tools/check_batch_invariance.py.)
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int kWarps = 8;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

template <int MT>
__global__ void __launch_bounds__(kWarps * 32) bf16_gemv_rows_kernel(const __nv_bfloat16* __restrict__ x, int m,
                                                                     const __nv_bfloat16* __restrict__ w,
                                                                     __nv_bfloat16* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int warps_total = gridDim.x * kWarps;
  for (int row = blockIdx.x * kWarps + (threadIdx.x >> 5); row < n; row += warps_total) {
    float acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = 0.f;
    const __nv_bfloat16* wr = w + (size_t)row * k;
    for (int c = lane * 8; c < k; c += 256) {
      const uint4 wv = ld_stream(reinterpret_cast<const uint4*>(wr + c));
      const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) wf[j] = __bfloat162float(we[j]);
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (i < m) {
          const uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)i * k + c));
          const __nv_bfloat16* xe = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[i] = __fmaf_rn(__bfloat162float(xe[j]), wf[j], acc[i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      if (i < m) {
        float v = acc[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) y[(size_t)i * n + row] = __float2bfloat16_rn(v);
      }
    }
  }
}

template <int MT>
int launch(const void* x, int m, const void* w, void* y, int n, int k, cudaStream_t s) {
  auto kern = bf16_gemv_rows_kernel<MT>;
  int dev = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kWarps * 32, 0);
  if (per_sm < 1) per_sm = 1;
  const int want = (n + kWarps - 1) / kWarps;
  const int blocks = want < sms * per_sm ? want : sms * per_sm;
  kern<<<blocks, kWarps * 32, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, static_cast<const __nv_bfloat16*>(w),
                                       static_cast<__nv_bfloat16*>(y), n, k);
  return (int)cudaGetLastError();
}
}  // namespace

extern "C" int ak_bf16_gemv_rows(const void* x, const void* w, void* y, int m, int n, int k, void* stream) {
  if (m < 1 || m > 16 || k % 8 || ((uintptr_t)x | (uintptr_t)w) % 16) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (m <= 4) return launch<4>(x, m, w, y, n, k, s);
  if (m <= 8) return launch<8>(x, m, w, y, n, k, s);
  return launch<16>(x, m, w, y, n, k, s);
}
