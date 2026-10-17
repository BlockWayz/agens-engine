// RoPE (NeoX style, partial rotary: the first rot_dim of each head), in place on q and k, bit-exact with sglv23's
// JIT fused_rope_kernel (elementwise/rope.cuh, default flags, no fast-math): per pair (x, y) at offsets i and
// i + rot/2, out_x = x*cos - y*sin, out_y = x*sin + y*cos in fp32 from an fp32 cos|sin cache, cast to bf16,
// with the FMA contraction pinned to what nvcc chose there (a random test missed it: ~1 element in 6K differs).
// Also builds the cos|sin cache the way sglang's RotaryEmbedding does with torch (inv_freq = 1 / base^(2j/rot),
// freqs = pos * inv_freq, cat(cos, sin)).
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
// the cache entry rope_cache_kernel writes for (position p, pair j): same expressions, same compilation flags
__device__ __forceinline__ void rope_cs(int p, int j, int rot, float base, float* c, float* s) {
  const float e = (float)(2 * j) / (float)rot;
  const float inv = 1.0f / powf(base, e);
  const float f = (float)p * inv;
  *c = cosf(f);
  *s = sinf(f);
}

// cache == nullptr: cos / sin computed on the fly (rope_cs, bit-identical to the cache's values)
__global__ void rope_kernel(__nv_bfloat16* q, __nv_bfloat16* k, const float* __restrict__ cache,
                            const int64_t* __restrict__ pos, int tokens, int nq, int nk, int head_dim, int rot,
                            int q_stride, int k_stride, float theta) {
  const int half = rot / 2;
  const long total = (long)tokens * (nq + nk) * half;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int j = (int)(i % half);
    const long th = i / half;
    const int h = (int)(th % (nq + nk));
    const int t = (int)(th / (nq + nk));
    __nv_bfloat16* base = h < nq ? q + (long)t * q_stride + (long)h * head_dim
                                 : k + (long)t * k_stride + (long)(h - nq) * head_dim;
    float cos_, sin_;
    if (cache) {
      const float* c = cache + pos[t] * rot;
      cos_ = c[j], sin_ = c[half + j];
    } else {
      rope_cs((int)pos[t], j, rot, theta, &cos_, &sin_);
    }
    const float x = __bfloat162float(base[j]), y = __bfloat162float(base[half + j]);
    // Contraction exactly as nvcc emitted it for sglang's fused_rope_kernel (SASS, sm_89):
    //   out_x = x*cos - y*sin  ->  FFMA(x, cos, -FMUL(y, sin))   (first product fused)
    //   out_y = x*sin + y*cos  ->  FFMA(y, cos,  FMUL(x, sin))   (SECOND product fused)
    const float out_x = __fmaf_rn(x, cos_, -__fmul_rn(y, sin_));
    const float out_y = __fmaf_rn(y, cos_, __fmul_rn(x, sin_));
    base[j] = __float2bfloat16(out_x);
    base[half + j] = __float2bfloat16(out_y);
  }
}

__global__ void rope_cache_kernel(float* out, int max_pos, int rot, float base) {
  const int half = rot / 2;
  const long total = (long)max_pos * half;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int j = (int)(i % half);
    const int p = (int)(i / half);
    rope_cs(p, j, rot, base, &out[(long)p * rot + j], &out[(long)p * rot + half + j]);
  }
}
}  // namespace

extern "C" int ak_rope_neox(void* q, void* k, const float* cos_sin_cache, const int64_t* positions, int tokens,
                            int nq, int nk, int head_dim, int rot_dim, int q_stride, int k_stride, void* stream) {
  const long total = (long)tokens * (nq + nk) * (rot_dim / 2);
  const int blocks = (int)((total + 255) / 256 < 1024 ? (total + 255) / 256 : 1024);
  rope_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k), cos_sin_cache, positions, tokens, nq, nk,
      head_dim, rot_dim, q_stride, k_stride, 0.f);
  return (int)cudaGetLastError();
}

// RoPE without a cache: cos / sin of each (position, pair) computed in the kernel exactly as ak_rope_cache stores them
// (bit-identical to ak_rope_neox over that cache; no [positions][rot] fp32 table on the device)
extern "C" int ak_rope_neox_fly(void* q, void* k, const int64_t* positions, int tokens, int nq, int nk, int head_dim,
                                int rot_dim, int q_stride, int k_stride, float base, void* stream) {
  const long total = (long)tokens * (nq + nk) * (rot_dim / 2);
  const int blocks = (int)((total + 255) / 256 < 1024 ? (total + 255) / 256 : 1024);
  rope_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k), nullptr, positions, tokens, nq, nk, head_dim,
      rot_dim, q_stride, k_stride, base);
  return (int)cudaGetLastError();
}

extern "C" int ak_rope_cache(float* out, int max_pos, int rot_dim, float base, void* stream) {
  rope_cache_kernel<<<1024, 256, 0, static_cast<cudaStream_t>(stream)>>>(out, max_pos, rot_dim, base);
  return (int)cudaGetLastError();
}
