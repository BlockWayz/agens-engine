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
#include <stdlib.h>
#include <string.h>

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

// Interleaved MRoPE (Volundr image positions): pair j rotates by axis a(j) = H (1) if j % 3 == 1 and
// j < 3 * sec_h, W (2) if j % 3 == 2 and j < 3 * sec_w, else T (0); token t's position on axis a is pos[a * stride + t].
// The math per (position, pair) is rope_kernel's, so a token whose three positions are equal gets its exact output.
// bf16_img (AGENS_MROPE_IMAGE, default off): a token whose positions differ (an image token) gets the reference's
// multimodal rotary arithmetic instead (sglang MRotaryEmbedding.forward_triton: cos|sin cache cast to bf16, bf16 ops).
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__global__ void rope_m_kernel(__nv_bfloat16* q, __nv_bfloat16* k, const float* __restrict__ cache,
                              const int64_t* __restrict__ pos, long stride, int tokens, int nq, int nk, int head_dim,
                              int rot, int q_stride, int k_stride, float theta, int sec_h, int sec_w, int bf16_img) {
  const int half = rot / 2;
  const long total = (long)tokens * (nq + nk) * half;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int j = (int)(i % half);
    const long th = i / half;
    const int h = (int)(th % (nq + nk));
    const int t = (int)(th / (nq + nk));
    const int ax = (j % 3 == 1 && j < 3 * sec_h) ? 1 : ((j % 3 == 2 && j < 3 * sec_w) ? 2 : 0);
    const int64_t p = pos[ax * stride + t];
    __nv_bfloat16* base = h < nq ? q + (long)t * q_stride + (long)h * head_dim
                                 : k + (long)t * k_stride + (long)(h - nq) * head_dim;
    float cos_, sin_;
    if (cache) {
      const float* c = cache + p * rot;
      cos_ = c[j], sin_ = c[half + j];
    } else {
      rope_cs((int)p, j, rot, theta, &cos_, &sin_);
    }
    const float x = __bfloat162float(base[j]), y = __bfloat162float(base[half + j]);
    const bool img = bf16_img && !(pos[t] == pos[stride + t] && pos[t] == pos[2 * stride + t]);
    if (img) {
      const float c = bfr(cos_), s = bfr(sin_);
      if (bf16_img == 2) {  // every product and sum rounded to bf16
        base[j] = __float2bfloat16(__fsub_rn(bfr(__fmul_rn(x, c)), bfr(__fmul_rn(y, s))));
        base[half + j] = __float2bfloat16(__fadd_rn(bfr(__fmul_rn(y, c)), bfr(__fmul_rn(x, s))));
      } else {  // one product contracted into a bf16 FMA (exact product + rounded product, one rounding)
        base[j] = __float2bfloat16(__fmaf_rn(x, c, -bfr(__fmul_rn(y, s))));
        base[half + j] = __float2bfloat16(__fmaf_rn(y, c, bfr(__fmul_rn(x, s))));
      }
      continue;
    }
    const float out_x = __fmaf_rn(x, cos_, -__fmul_rn(y, sin_));
    const float out_y = __fmaf_rn(y, cos_, __fmul_rn(x, sin_));
    base[j] = __float2bfloat16(out_x);
    base[half + j] = __float2bfloat16(out_y);
  }
}

// Attention q / k preparation in one launch (decode / verify rows; head_dim 256, rot 64), bit-identical to the chain
//   split_q_gate (qg [T][nq][2 x 256] -> q, gate) -> ak_rmsnorm(q, q_norm, gemma) -> ak_rmsnorm(k, k_norm, gemma)
//   -> ak_rope_neox_m (MRoPE, q and k) -> store_kv (k, v -> the caches' rows loc[t])
// One warp per (token, head): lane l holds elements 8 l .. 8 l + 7, which is exactly the CuTe RMSNorm's lane layout at
// d = 256 (TPR 32: one 8-element fma chain per lane, butterfly 1 .. 16, other + self); the rotary pairs (j, j + 32)
// sit in lanes l and l ^ 4 (l < 8) and use rope_m_kernel's per-pair arithmetic.
__global__ void __launch_bounds__(256) qk_prep_kernel(const __grid_constant__ QkPrepArgs a, int bf16_img) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const int nh = a.nq + a.nk;
  if (w >= a.T * nh) return;
  const int t = w / nh, h = w % nh, d0 = lane * 8;
  const bool isq = h < a.nq;
  const int kh = h - a.nq;
  uint4 xv;
  if (isq) {
    const __nv_bfloat16* src = a.qg + ((size_t)t * a.nq + h) * 512;
    xv = *reinterpret_cast<const uint4*>(src + d0);
    *reinterpret_cast<uint4*>(a.gate + ((size_t)t * a.nq + h) * 256 + d0) = *reinterpret_cast<const uint4*>(src + 256 + d0);
  } else {
    xv = *reinterpret_cast<const uint4*>(a.k + ((size_t)t * a.nk + kh) * 256 + d0);
  }
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&xv);
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float f = __bfloat162float(e[j]);
    sum = __fmaf_rn(f, f, sum);
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) sum = __fadd_rn(__shfl_xor_sync(0xFFFFFFFFu, sum, o), sum);
  float rstd;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rstd) : "f"(__fadd_rn(__fdiv_rn(sum, 256.f), a.eps)));
  const uint4 wv = __ldg(reinterpret_cast<const uint4*>((isq ? a.wq : a.wk) + d0));
  const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
  uint4 yv;
  __nv_bfloat16* ye = reinterpret_cast<__nv_bfloat16*>(&yv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    ye[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(e[j]), rstd), __fadd_rn(__bfloat162float(we[j]), 1.f)));
  // rotary: lanes 0-3 hold x of pairs 8 l .. 8 l + 7, lanes 4-7 their y
  uint4 pv;
  pv.x = __shfl_xor_sync(0xFFFFFFFFu, yv.x, 4), pv.y = __shfl_xor_sync(0xFFFFFFFFu, yv.y, 4);
  pv.z = __shfl_xor_sync(0xFFFFFFFFu, yv.z, 4), pv.w = __shfl_xor_sync(0xFFFFFFFFu, yv.w, 4);
  uint4 ov = yv;
  if (lane < 8) {
    const __nv_bfloat16* pe = reinterpret_cast<const __nv_bfloat16*>(&pv);
    __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&ov);
    const bool lo = lane < 4;
    const int64_t* pos = a.pos;
    const bool img = bf16_img && !(pos[t] == pos[a.pstride + t] && pos[t] == pos[2 * a.pstride + t]);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int j = (lane & 3) * 8 + v;
      const int ax = (j % 3 == 1 && j < 3 * a.sec_h) ? 1 : ((j % 3 == 2 && j < 3 * a.sec_w) ? 2 : 0);
      const int64_t p = pos[ax * a.pstride + t];
      float cos_, sin_;
      if (a.cache) {
        const float* cc = a.cache + p * 64;
        cos_ = cc[j], sin_ = cc[32 + j];
      } else {
        rope_cs((int)p, j, 64, a.theta, &cos_, &sin_);
      }
      const float x = __bfloat162float(lo ? ye[v] : pe[v]), y = __bfloat162float(lo ? pe[v] : ye[v]);
      float r;
      if (img) {
        const float c = bfr(cos_), s = bfr(sin_);
        if (bf16_img == 2) r = lo ? __fsub_rn(bfr(__fmul_rn(x, c)), bfr(__fmul_rn(y, s))) : __fadd_rn(bfr(__fmul_rn(y, c)), bfr(__fmul_rn(x, s)));
        else r = lo ? __fmaf_rn(x, c, -bfr(__fmul_rn(y, s))) : __fmaf_rn(y, c, bfr(__fmul_rn(x, s)));
      } else {
        r = lo ? __fmaf_rn(x, cos_, -__fmul_rn(y, sin_)) : __fmaf_rn(y, cos_, __fmul_rn(x, sin_));
      }
      oe[v] = __float2bfloat16(r);
    }
  }
  if (isq) {
    *reinterpret_cast<uint4*>(a.q + ((size_t)t * a.nq + h) * 256 + d0) = ov;
  } else {
    const size_t src = ((size_t)t * a.nk + kh) * 256 + d0;
    *reinterpret_cast<uint4*>(a.k + src) = ov;
    if (a.kc) {  // nullable: no cache store (MTP chain)
      const size_t dst = (size_t)a.loc[t] * a.nk * 256 + (size_t)kh * 256 + d0;
      *reinterpret_cast<uint4*>(a.kc + dst) = ov;
      *reinterpret_cast<uint4*>(a.vc + dst) = *reinterpret_cast<const uint4*>(a.v + src);
    }
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

// MRoPE over three position rows (positions[a * stride + t], a = T, H, W; stride 0 = one row for every axis);
// cos_sin_cache == nullptr: on the fly
extern "C" int ak_rope_neox_m(void* q, void* k, const float* cos_sin_cache, const int64_t* positions, long stride,
                              int tokens, int nq, int nk, int head_dim, int rot_dim, int q_stride, int k_stride,
                              float base, int sec_h, int sec_w, void* stream) {
  const long total = (long)tokens * (nq + nk) * (rot_dim / 2);
  if (total <= 0) return 0;
  const int blocks = (int)((total + 255) / 256 < 1024 ? (total + 255) / 256 : 1024);
  // AGENS_MROPE_IMAGE=fp32: image tokens through the fp32 text rotary instead of the reference's bf16 one
  // AGENS_MROPE_IMAGE: fp32 (default; the text rotary for image tokens too) | bf16 (the reference's bf16 cos|sin, one
  // FMA contraction) | bf16ops (bf16 rounding after every operation). Measured vs sglang on the vision gate (1,590
  // teacher-forced image-answer tokens): top-1 99.18 / 98.99 / 99.43%, KL mean 1.94e-2 / 1.95e-2 / 1.92e-2: within noise.
  static const int bf16_img = [] {
    const char* e = getenv("AGENS_MROPE_IMAGE");
    return !e ? 0 : (!strcmp(e, "bf16") ? 1 : (!strcmp(e, "bf16ops") ? 2 : 0));
  }();
  rope_m_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k), cos_sin_cache, positions, stride, tokens, nq, nk,
      head_dim, rot_dim, q_stride, k_stride, base, sec_h, sec_w, bf16_img);
  return (int)cudaGetLastError();
}

// q / k preparation in one launch (qk_prep_kernel); head_dim 256 and rot 64 only
extern "C" int ak_qk_prep(const QkPrepArgs* args, void* stream) {
  if (args->T <= 0) return 0;
  static const int bf16_img = [] {
    const char* e = getenv("AGENS_MROPE_IMAGE");
    return !e ? 0 : (!strcmp(e, "bf16") ? 1 : (!strcmp(e, "bf16ops") ? 2 : 0));
  }();
  const int warps = args->T * (args->nq + args->nk);
  qk_prep_kernel<<<(warps + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(*args, bf16_img);
  return (int)cudaGetLastError();
}
