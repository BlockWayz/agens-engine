// W8A16 GEMV for an INT8 lm_head (per-output-row symmetric scale): y[j][o] = scale[o] * sum_i x[j][i] * q[o][i],
// 1 <= m <= 16 tokens. One warp per output row o: lane l reads 16-byte weight chunks l, l+32, ... once and applies
// each to every token, fp32 products accumulated chunk by chunk, then a butterfly. The order of the sum for (o, j)
// does not depend on m, so row j of an m-token call is bit-identical to the 1-token call (speculative verify).
// Quantization (load time, host): q = round(w / s), s = max|w[o]| / 127.

#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kWarps = 8;

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// 4 int8 of word w times 4 x values (2 bf16 pairs)
__device__ __forceinline__ float dot4(uint32_t w, uint32_t x01, uint32_t x23, float s) {
  s = fmaf(__uint_as_float(x01 << 16), (float)(int8_t)(w & 0xFF), s);
  s = fmaf(__uint_as_float(x01 & 0xFFFF0000u), (float)(int8_t)((w >> 8) & 0xFF), s);
  s = fmaf(__uint_as_float(x23 << 16), (float)(int8_t)((w >> 16) & 0xFF), s);
  s = fmaf(__uint_as_float(x23 & 0xFFFF0000u), (float)(int8_t)(w >> 24), s);
  return s;
}

template <int M>
__global__ void __launch_bounds__(kWarps * 32) i8_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                                             const uint4* __restrict__ w,
                                                             const float* __restrict__ scale,
                                                             __nv_bfloat16* __restrict__ y, int m, int n, int k,
                                                             int ldy) {
  const int lane = threadIdx.x & 31, o = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (o >= n) return;
  const int chunks = k >> 4;  // 16 inputs per chunk
  float acc[M];
#pragma unroll
  for (int j = 0; j < M; ++j) acc[j] = 0.f;
  const uint4* wr = w + (size_t)o * chunks;
  for (int c = lane; c < chunks; c += 32) {
    const uint4 q = ld_stream(wr + c);
#pragma unroll
    for (int j = 0; j < M; ++j) {
      if (j >= m) break;
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)j * k) + 2 * c;
      const uint4 a = __ldg(xr), b = __ldg(xr + 1);
      float s = 0.f;
      s = dot4(q.x, a.x, a.y, s);
      s = dot4(q.y, a.z, a.w, s);
      s = dot4(q.z, b.x, b.y, s);
      s = dot4(q.w, b.z, b.w, s);
      acc[j] += s;
    }
  }
  const float sc = scale[o];
#pragma unroll
  for (int j = 0; j < M; ++j) {
    if (j >= m) break;
    float v = acc[j];
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
    if (lane == 0) y[(size_t)j * ldy + o] = __float2bfloat16_rn(v * sc);
  }
}
}  // namespace

extern "C" int ak_i8_gemv(const void* x, const void* w, const void* scale, void* y, int m, int n, int k, int ldy,
                          void* stream) {
  if (m < 1 || m > 16 || k % 16) return (int)cudaErrorInvalidValue;
  const dim3 grid((n + kWarps - 1) / kWarps);
  cudaStream_t s = (cudaStream_t)stream;
  auto X = (const __nv_bfloat16*)x;
  auto W = (const uint4*)w;
  auto S = (const float*)scale;
  auto Y = (__nv_bfloat16*)y;
  if (m == 1)
    i8_gemv_kernel<1><<<grid, kWarps * 32, 0, s>>>(X, W, S, Y, m, n, k, ldy);
  else if (m <= 8)
    i8_gemv_kernel<8><<<grid, kWarps * 32, 0, s>>>(X, W, S, Y, m, n, k, ldy);
  else
    i8_gemv_kernel<16><<<grid, kWarps * 32, 0, s>>>(X, W, S, Y, m, n, k, ldy);
  return (int)cudaGetLastError();
}
