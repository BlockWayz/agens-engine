// Embeds PTX of FlashInfer's CuTe-DSL RMSNorm kernel (Apache-2.0). See THIRD_PARTY_NOTICES.md.
// RMSNorm / GemmaRMSNorm, bit-exact with what sglv23 runs: sglang GemmaRMSNorm/RMSNorm -> sgl_kernel -> flashinfer
// 0.6.15 rmsnorm/gemma_rmsnorm -> the CuTe-DSL RMSNormKernel (flashinfer/norm/kernels/rmsnorm.py; PTX dumped by
// tools/dump_cute_norm.py). Its structure, reproduced here:
//   * threads_per_row TPR = 8 (H<=64) | 16 (<=128) | 32 (<=3072) | 64 (<=6144) | 128 (<=16384); 16-byte vectors
//   * thread lane l of a row owns elements blk*TPR*8 + l*8 + v (blk ascending, v 0..7) and sums x*x as one fma chain
//     starting from 0 in exactly that order
//   * warp reduce: shfl.bfly offsets 1, 2, 4, ... < min(TPR, 32), sum = other + self
//   * TPR > 32: warp sums combined by a second bfly over lanes holding w0..w(k-1) (zero padded)
//   * mean = div.rn(sum, H); rstd = rsqrt.approx.ftz(mean + eps); y = (x * rstd) * (w + bias), bias 1 for Gemma
#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// One row per TPR threads; blockDim.x = max(TPR, 32) (TPR < 32 -> 32 / TPR rows per block).
// ADD: x = bf16(x + xb) elementwise first (torch's bf16 add), written to xsum, then the norm of that (fused residual
// add + RMSNorm: one launch instead of two, the same bits)
// (the body takes the block index bx: rmsnorm_cute_kernel passes blockIdx.x, the pair kernel below 0 per row)
template <int TPR, bool ADD = false>
__device__ __forceinline__ void rmsnorm_cute_body(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                  __nv_bfloat16* __restrict__ y, int rows, int H, int stride_in,
                                                  int stride_out, float bias, float eps,
                                                  const __nv_bfloat16* __restrict__ xb, __nv_bfloat16* __restrict__ xsum,
                                                  int bx) {
  constexpr int kRowsPerBlock = TPR < 32 ? 32 / TPR : 1;
  const int lane = threadIdx.x % TPR;
  const int row = bx * kRowsPerBlock + threadIdx.x / TPR;
  const bool valid = row < rows;
  const int nblk = (H / 8 + TPR - 1) / TPR;
  const __nv_bfloat16* xr = x + (size_t)(valid ? row : 0) * stride_in;
  float sum = 0.f;
  // the row's vectors are loaded together first (kPre per thread; a T = 8 decode launches 8 blocks, so the loop was
  // latency-bound), then summed in the same order; the output pass reuses them
  constexpr int kPre = 12;
  uint4 xv[kPre], wpre[kPre];
  auto addv = [&](uint4 v, int c) -> uint4 {  // bf16(x + xb), stored to xsum (ADD)
    if constexpr (ADD) {
      const uint4 u = *reinterpret_cast<const uint4*>(xb + (size_t)row * stride_in + c);
      __nv_bfloat16* e = reinterpret_cast<__nv_bfloat16*>(&v);
      const __nv_bfloat16* f = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) e[j] = __float2bfloat16(__bfloat162float(e[j]) + __bfloat162float(f[j]));
      *reinterpret_cast<uint4*>(xsum + (size_t)row * stride_in + c) = v;
    }
    return v;
  };
#pragma unroll
  for (int b = 0; b < kPre; ++b) {
    const int c = b * TPR * 8 + lane * 8;
    if (b < nblk && valid && c < H) {
      xv[b] = addv(*reinterpret_cast<const uint4*>(xr + c), c);
      wpre[b] = __ldg(reinterpret_cast<const uint4*>(w + c));
    }
  }
  auto acc = [&](const uint4& v) {
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float f = __bfloat162float(e[j]);
      sum = __fmaf_rn(f, f, sum);
    }
  };
#pragma unroll
  for (int b = 0; b < kPre; ++b)
    if (b < nblk && valid && b * TPR * 8 + lane * 8 < H) acc(xv[b]);
  for (int b = kPre; b < nblk; ++b) {
    const int c = b * TPR * 8 + lane * 8;
    if (valid && c < H) acc(addv(*reinterpret_cast<const uint4*>(xr + c), c));
  }
  constexpr int kWidth = TPR < 32 ? TPR : 32;
#pragma unroll
  for (int o = 1; o < kWidth; o <<= 1) sum = __fadd_rn(__shfl_xor_sync(0xFFFFFFFFu, sum, o, kWidth), sum);
  if constexpr (TPR > 32) {
    __shared__ float ws[TPR / 32 > 0 ? TPR / 32 : 1];
    if ((threadIdx.x & 31) == 0) ws[threadIdx.x >> 5] = sum;
    __syncthreads();
    float v = (threadIdx.x & 31) < TPR / 32 ? ws[threadIdx.x & 31] : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    sum = v;
  }
  if (!valid) return;
  const float rstd = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(sum, (float)H), eps));
  __nv_bfloat16* yr = y + (size_t)row * stride_out;
  auto put = [&](int c, const uint4& v, const uint4& wv) {
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v);
    const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
    uint4 o4;
    __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&o4);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      o[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(e[j]), rstd), __fadd_rn(__bfloat162float(we[j]), bias)));
    *reinterpret_cast<uint4*>(yr + c) = o4;
  };
#pragma unroll
  for (int b = 0; b < kPre; ++b)
    if (b < nblk && b * TPR * 8 + lane * 8 < H) put(b * TPR * 8 + lane * 8, xv[b], wpre[b]);
  for (int b = kPre; b < nblk; ++b) {
    const int c = b * TPR * 8 + lane * 8;
    if (c < H)
      put(c, ADD ? *reinterpret_cast<const uint4*>(xsum + (size_t)row * stride_in + c) : *reinterpret_cast<const uint4*>(xr + c),
          *reinterpret_cast<const uint4*>(w + c));
  }
}

template <int TPR, bool ADD = false>
__global__ void rmsnorm_cute_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ y, int rows, int H, int stride_in, int stride_out,
                                    float bias, float eps, const __nv_bfloat16* __restrict__ xb = nullptr,
                                    __nv_bfloat16* __restrict__ xsum = nullptr) {
  ak_pdl_entry();
  rmsnorm_cute_body<TPR, ADD>(x, w, y, rows, H, stride_in, stride_out, bias, eps, xb, xsum, blockIdx.x);
}
// Two RMSNorms of B rows of H (TPR 64) in one launch, each row exactly rmsnorm_cute_kernel<64>'s: blockIdx.y = 0 the
// embedding rows table[ids[b]] with w0 -> y0 + b * so (the embedding lookup folded in: the same values), 1 the rows
// x1 + b * H with w1 -> y1 + b * so (the MTP chain's pre-fc norms of the token embedding and the hidden state)
__global__ void rmsnorm_pair_embed_kernel(const int32_t* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                          const __nv_bfloat16* __restrict__ w0, __nv_bfloat16* __restrict__ y0,
                                          const __nv_bfloat16* __restrict__ x1, const __nv_bfloat16* __restrict__ w1,
                                          __nv_bfloat16* __restrict__ y1, int H, int so, float bias, float eps) {
  ak_pdl_entry();
  const int b = blockIdx.x;
  if (blockIdx.y == 0)
    rmsnorm_cute_body<64, false>(table + (size_t)ids[b] * H, w0, y0 + (size_t)b * so, 1, H, H, so, bias, eps, nullptr,
                                 nullptr, 0);
  else
    rmsnorm_cute_body<64, false>(x1 + (size_t)b * H, w1, y1 + (size_t)b * so, 1, H, H, so, bias, eps, nullptr, nullptr, 0);
}

template <int TPR>
void launch(const void* x, const void* w, void* y, int rows, int H, int si, int so, float bias, float eps, cudaStream_t s) {
  constexpr int kRowsPerBlock = TPR < 32 ? 32 / TPR : 1;
  constexpr int kThreads = TPR < 32 ? 32 : TPR;
  rmsnorm_cute_kernel<TPR><<<(rows + kRowsPerBlock - 1) / kRowsPerBlock, kThreads, 0, s>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), rows, H,
      si, so, bias, eps);
}
}  // namespace

extern "C" int ak_rmsnorm(const void* x, const void* w, void* y, int rows, int d, int stride_in, int stride_out,
                          float eps, int gemma, void* stream) {
  if (d % 8) return (int)cudaErrorInvalidValue;
  const float bias = gemma ? 1.f : 0.f;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (d <= 64) launch<8>(x, w, y, rows, d, stride_in, stride_out, bias, eps, s);
  else if (d <= 128) launch<16>(x, w, y, rows, d, stride_in, stride_out, bias, eps, s);
  else if (d <= 3072) launch<32>(x, w, y, rows, d, stride_in, stride_out, bias, eps, s);
  else if (d <= 6144) launch<64>(x, w, y, rows, d, stride_in, stride_out, bias, eps, s);
  else if (d <= 16384) launch<128>(x, w, y, rows, d, stride_in, stride_out, bias, eps, s);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}

// y = RMSNorm(s), s = bf16(a + b) also written out (residual add + norm in one launch; rows of d, contiguous)
extern "C" int ak_add_rmsnorm(const void* a, const void* b, void* s_out, const void* w, void* y, int rows, int d, float eps,
                              int gemma, void* stream) {
  if (d % 8 || d <= 3072 || d > 6144) return (int)cudaErrorInvalidValue;  // the TPR 64 shapes (HID)
  rmsnorm_cute_kernel<64, true><<<rows, 64, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), rows, d,
      d, d, gemma ? 1.f : 0.f, eps, static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(s_out));
  return (int)cudaGetLastError();
}


// y0 rows = RMSNorm(table[ids[b]]) * (w0 + bias), y1 rows = RMSNorm(x1 rows) * (w1 + bias), B rows each, row stride so
// out; the same bits as ak_embed + two ak_rmsnorm launches (H in the TPR 64 range)
extern "C" int ak_rmsnorm_pair_embed(const int32_t* ids, const void* table, const void* w0, void* y0, const void* x1,
                                     const void* w1, void* y1, int B, int H, int so, float eps, int gemma, void* stream) {
  if (H % 8 || H <= 3072 || H > 6144 || B < 1) return (int)cudaErrorInvalidValue;
  rmsnorm_pair_embed_kernel<<<dim3(B, 2), 64, 0, static_cast<cudaStream_t>(stream)>>>(
      ids, static_cast<const __nv_bfloat16*>(table), static_cast<const __nv_bfloat16*>(w0), static_cast<__nv_bfloat16*>(y0),
      static_cast<const __nv_bfloat16*>(x1), static_cast<const __nv_bfloat16*>(w1), static_cast<__nv_bfloat16*>(y1), H, so,
      gemma ? 1.f : 0.f, eps);
  return (int)cudaGetLastError();
}
