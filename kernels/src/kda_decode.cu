// Derived from SGLang (Apache-2.0) and flash-linear-attention (MIT, (c) 2023-2025 Songlin Yang, Yu Zhang); the conv
// update from causal-conv1d (BSD-3-Clause, (c) 2024 Tri Dao) via vLLM (Apache-2.0). See THIRD_PARTY_NOTICES.md.
// KDA single-token decode, bit-exact with sglv23's bs < 8 path:
//   sglang.kernels.ops.mamba.causal_conv1d_triton._causal_conv1d_update_kernel   (conv + SiLU, 4 warps)
//   sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_kda_packed_decode_kernel (1 warp, BV=32)
// Each operation below is a transcription of the PTX Triton 3.6 emits for those kernels on sm_89
// (refs: tools/dump_kda_decode_ref.py -> out/kda_ref/*.ptx). Rules that matter for bits:
//   * exp(x) = ex2.approx.f32(x * log2e)            (not expf)
//   * x / y  = div.full.f32                          (not IEEE div, not __fdividef)
//   * sqrt   = sqrt.approx.ftz.f32
//   * log    = libdevice __nv_logf with FTZ reflect = 1 (fma.rn.ftz polynomial), transcribed below
//   * sums over K: per thread a 4-element chain acc = e1*k1; acc = fma(e0,k0,acc); fma(e2..); fma(e3..),
//     then shfl.bfly 16, 8, 4, 2, 1 with acc = acc + other
//   * mul then add that Triton/LLVM contracted are written as fmaf; everything else stays separate (__fmul_rn,
//     __fadd_rn) so nvcc cannot contract on its own.
// State layout [slots][HV][V][K] fp32 (K contiguous), as sglang's mamba pool stores it.

#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <stdlib.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float F(unsigned u) { return __uint_as_float(u); }  // PTX 0fXXXXXXXX literals
#define kLog2e F(0x3FB8AA3Bu)

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tri_exp(float x) { return ex2_approx(__fmul_rn(x, kLog2e)); }
__device__ __forceinline__ float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float sqrt_approx_ftz(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float y;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
  return y;
}
// libdevice __nv_logf, FTZ variant, exactly as inlined by Triton (see PTX: __nv_logf.exit).
__device__ __forceinline__ float tri_log(float a) {
  const bool small = a < 1.17549435e-38f;  // setp.lt.f32 a, 0f00800000
  const float x = small ? __fmul_rn(a, 8388608.0f) : a;
  const float i0 = small ? -23.0f : 0.0f;
  const int xi = __float_as_int(x);
  const int e = (xi - 0x3F2AAAAB) & (int)0xFF800000;
  const float m = __int_as_float(xi - e);
  const float i = fma_ftz((float)e, F(0x34000000), i0);
  const float f = __fadd_rn(m, -1.0f);
  float p = fma_ftz(F(0xBE055027), f, F(0x3E1039F6));
  p = fma_ftz(p, f, F(0xBDF8CDCC));
  p = fma_ftz(p, f, F(0x3E0F2955));
  p = fma_ftz(p, f, F(0xBE2AD8B9));
  p = fma_ftz(p, f, F(0x3E4CED0B));
  p = fma_ftz(p, f, F(0xBE7FFF22));
  p = fma_ftz(p, f, F(0x3EAAAA78));
  p = fma_ftz(p, f, F(0xBF000000));
  const float q = __fmul_rn(f, p);
  const float r = fma_ftz(q, f, f);
  float res = fma_ftz(i, F(0x3F317218), r);
  if (!((unsigned)__float_as_int(x) < 0x7F800000u)) res = fma_ftz(x, __int_as_float(0x7F800000), __int_as_float(0x7F800000));
  return res;
}
__device__ __forceinline__ float bfly_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}
// Triton's in-thread order for a 4-element product-sum (LLVM contracts the first add into an fma).
__device__ __forceinline__ float dot4(const float a[4], const float b[4]) {
  float acc = __fmul_rn(a[1], b[1]);
  acc = __fmaf_rn(a[0], b[0], acc);
  acc = __fmaf_rn(a[2], b[2], acc);
  acc = __fmaf_rn(a[3], b[3], acc);
  return acc;
}

// ------------------------------------------------------------------------------------------- conv1d update
// One thread per channel (Triton: 256 channels per program, 2 per thread; per-channel math is independent).
// state [slots][W-1][dim] bf16; weight [dim][W] fp32 (W = 4); x [B][dim] bf16 (stride x_stride); y [B][dim] bf16.
__global__ void conv_update_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, __nv_bfloat16* __restrict__ state,
                                   const float* __restrict__ w, const int32_t* __restrict__ idx,
                                   __nv_bfloat16* __restrict__ y, int dim) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int bi = blockIdx.y;
  if (c >= dim) return;
  const int slot = idx[bi];
  const __nv_bfloat16 xin = x[(size_t)bi * x_stride + c];
  if (slot < 0) return;  // Triton skips padded entries (output left untouched)
  __nv_bfloat16* s = state + (size_t)slot * 3 * dim;
  const __nv_bfloat16 s0 = s[c], s1 = s[dim + c], s2 = s[2 * dim + c];
  const float* wc = w + (size_t)c * 4;
  float acc = __fmaf_rn(__bfloat162float(s0), wc[0], 0.0f);
  acc = __fmaf_rn(__bfloat162float(s1), wc[1], acc);
  acc = __fmaf_rn(__bfloat162float(s2), wc[2], acc);
  acc = __fmaf_rn(wc[3], __bfloat162float(xin), acc);
  // SiLU: x / (1 + exp(0 - x))
  const float out = div_full(acc, __fadd_rn(tri_exp(__fsub_rn(0.0f, acc)), 1.0f));
  s[c] = s1;
  s[dim + c] = s2;
  s[2 * dim + c] = xin;
  y[(size_t)bi * dim + c] = __float2bfloat16_rn(out);
}

// ------------------------------------------------------------------------------------------- recurrent step
// grid (V / 32, B * HV), 32 threads. Thread l owns k = 4l..4l+3 of 32 state rows (v = 32 * blockIdx.x + r).
template <int K, int V>
__global__ void __launch_bounds__(32)
kda_decode_kernel(const __nv_bfloat16* __restrict__ qkv, int qkv_stride, const __nv_bfloat16* __restrict__ a,
                  int a_stride, const __nv_bfloat16* __restrict__ b, int b_stride, const float* __restrict__ A_log,
                  const float* __restrict__ dt_bias, float* __restrict__ state, const int32_t* __restrict__ idx,
                  __nv_bfloat16* __restrict__ out, float scale, int H, int HV) {
  static_assert(K == 128 && V % 32 == 0, "shapes of Volundr's KDA");
  const int lane = threadIdx.x;
  const int i_v = blockIdx.x;
  const int i_n = blockIdx.y / HV, i_hv = blockIdx.y % HV;
  const int i_h = i_hv / (HV / H);
  __nv_bfloat16* o = out + ((size_t)i_n * HV + i_hv) * V + i_v * 32;
  const int slot = idx[i_n];
  if (slot < 0) {
    o[lane] = __float2bfloat16_rn(0.f);
    return;
  }
  float* h = state + (size_t)slot * HV * V * K + (size_t)i_hv * V * K + (size_t)i_v * 32 * K;
  const __nv_bfloat16* p = qkv + (size_t)i_n * qkv_stride;

  // q, k: this thread's 4 contiguous channels (the norm reduction runs on the same registers)
  float q[4], k[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = __bfloat162float(p[i_h * K + 4 * lane + j]);
    k[j] = __bfloat162float(p[H * K + i_h * K + 4 * lane + j]);
  }
  const float qn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(q, q)), F(0x358637BDu)));
  const float kn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(k, k)), F(0x358637BDu)));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q[j] = __fmul_rn(div_full(q[j], qn), scale);
    k[j] = div_full(k[j], kn);
  }
  // gate: g = -exp(A_log) * softplus(a + dt_bias); decay = exp(g)
  const float negA = __fsub_rn(0.0f, tri_exp(A_log[i_hv]));
  float dec[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int kk = i_hv * K + 4 * lane + j;
    const float xg = __fadd_rn(__bfloat162float(a[(size_t)i_n * a_stride + kk]), dt_bias[kk]);
    const float sp = xg <= 20.0f ? tri_log(__fadd_rn(tri_exp(xg), 1.0f)) : xg;
    dec[j] = tri_exp(__fmul_rn(sp, negA));
  }
  const float bv = __bfloat162float(b[(size_t)i_n * b_stride + i_hv]);
  const float beta = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, bv)), 1.0f));

  // v for the 32 rows of this tile (Triton: [BV] slice layout, every thread holds all 32)
  const __nv_bfloat16* pv = p + 2 * H * K + (size_t)i_hv * V + i_v * 32;
  float res[32];
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    const float4 hv4 = *reinterpret_cast<const float4*>(h + r * K + 4 * lane);
    float hr[4] = {__fmul_rn(hv4.x, dec[0]), __fmul_rn(hv4.y, dec[1]), __fmul_rn(hv4.z, dec[2]),
                   __fmul_rn(hv4.w, dec[3])};
    const float kv = bfly_sum(dot4(k, hr));
    const float vnew = __fmul_rn(__fsub_rn(__bfloat162float(pv[r]), kv), beta);
#pragma unroll
    for (int j = 0; j < 4; ++j) hr[j] = __fmaf_rn(vnew, k[j], hr[j]);
    *reinterpret_cast<float4*>(h + r * K + 4 * lane) = make_float4(hr[0], hr[1], hr[2], hr[3]);
    res[r] = bfly_sum(dot4(q, hr));
  }
  o[lane] = __float2bfloat16_rn(res[lane]);
}

// ------------------------------------------------------------------------------- gated RMSNorm (KDA output)
// fla layer_norm_gated_fwd_kernel (RMS, swish gate), D = 128: Triton tile [32 rows x 128], thread = 8 contiguous
// columns of a row, 16 lanes per row. Here: 16 lanes per row as well, 2 rows per warp (xor 8..1 stays in-half).
// y = ((x * rstd) * w) * g * sigmoid(g),  rstd = 1 / sqrt(sum(x^2) / D + eps)
__device__ __forceinline__ float dot8_sq(const float v[8]) {
  float acc = __fmul_rn(v[1], v[1]);
  acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
  for (int j = 2; j < 8; ++j) acc = __fmaf_rn(v[j], v[j], acc);
  return acc;
}
__global__ void gated_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ g,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows,
                                     float eps) {
  ak_pdl_entry();
  const int row = blockIdx.x * 2 + (threadIdx.x >> 4);
  const int l = threadIdx.x & 15;
  float xv[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) xv[j] = row < rows ? __bfloat162float(x[(size_t)row * 128 + 8 * l + j]) : 0.f;
  float s = dot8_sq(xv);
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, o));
  if (row >= rows) return;
  const float var = div_full(s, 128.0f);
  const float rstd = div_full(1.0f, sqrt_approx_ftz(__fadd_rn(var, eps)));
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const size_t i = (size_t)row * 128 + 8 * l + j;
    const float gv = __bfloat162float(g[i]);
    const float sig = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, gv)), 1.0f));
    float v = __fmul_rn(__fmul_rn(xv[j], rstd), __bfloat162float(w[8 * l + j]));
    v = __fmul_rn(__fmul_rn(v, gv), sig);
    y[i] = __float2bfloat16_rn(v);
  }
}


// ------------------------------------------------------------------------------- multi-token (speculative verify)
// T consecutive tokens of one sequence through the conv update / recurrent step, each token with exactly the
// arithmetic of the single-token kernels above (state carried in registers between tokens instead of memory: same
// values), so token t's output is bit-identical to t + 1 sequential single steps. write_state = 0 leaves the stored
// state untouched (verify); a commit then replays the accepted prefix with write_state = 1. n_dev (optional) caps T
// with a count read on device (the commit length, graph-safe).
// blockIdx.y = sequence z of a batched verify (rows z*T .. z*T+T-1, slot idx[z], count n_dev[z * n_stride])
template <bool PDL = false>
__device__ __forceinline__ void conv_multi_body(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                __nv_bfloat16* __restrict__ state, const float* __restrict__ w,
                                                const int32_t* __restrict__ idx, __nv_bfloat16* __restrict__ y, int dim,
                                                int T, const int* __restrict__ n_dev, int write_state, int n_stride,
                                                int bx, int z) {
  const int c = bx * 256 + threadIdx.x;  // blockDim 256
  if (c >= dim) return;  // (writes nothing)
  const int slot = idx[z];
  if (slot < 0) return;
  x += (size_t)z * T * x_stride;
  if (y) y += (size_t)z * T * dim;
  __nv_bfloat16* s = state + (size_t)slot * 3 * dim;
  // PDL (the verify launch): the conv state (written by the commits only) before the programmatic wait; the inputs and
  // the commit count (n_dev: the accept, maybe the kernel just before) after
  __nv_bfloat16 s0 = s[c], s1 = s[dim + c], s2 = s[2 * dim + c];
  const float* wc = w + (size_t)c * 4;
  if (PDL) ak_pdl_entry();
  const int n = n_dev ? min(n_dev[z * n_stride], T) : T;
  for (int t = 0; t < n; ++t) {
    const __nv_bfloat16 xin = x[(size_t)t * x_stride + c];
    float acc = __fmaf_rn(__bfloat162float(s0), wc[0], 0.0f);
    acc = __fmaf_rn(__bfloat162float(s1), wc[1], acc);
    acc = __fmaf_rn(__bfloat162float(s2), wc[2], acc);
    acc = __fmaf_rn(wc[3], __bfloat162float(xin), acc);
    const float out = div_full(acc, __fadd_rn(tri_exp(__fsub_rn(0.0f, acc)), 1.0f));
    s0 = s1, s1 = s2, s2 = xin;
    if (y) y[(size_t)t * dim + c] = __float2bfloat16_rn(out);
  }
  if (write_state) s[c] = s0, s[dim + c] = s1, s[2 * dim + c] = s2;
}
__global__ void __launch_bounds__(256) conv_multi_kernel(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                         __nv_bfloat16* __restrict__ state, const float* __restrict__ w,
                                                         const int32_t* __restrict__ idx, __nv_bfloat16* __restrict__ y,
                                                         int dim, int T, const int* __restrict__ n_dev, int write_state,
                                                         int n_stride) {
  conv_multi_body<true>(x, x_stride, state, w, idx, y, dim, T, n_dev, write_state, n_stride, blockIdx.x, blockIdx.y);
}

template <int K, int V>
__global__ void __launch_bounds__(32)
kda_multi_kernel(const __nv_bfloat16* __restrict__ qkv, int qkv_stride, const __nv_bfloat16* __restrict__ a,
                 int a_stride, const __nv_bfloat16* __restrict__ b, int b_stride, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, float* __restrict__ state, const int32_t* __restrict__ idx,
                 __nv_bfloat16* __restrict__ out, float scale, int H, int HV, int T, const int* __restrict__ n_dev,
                 int write_state) {
  const int lane = threadIdx.x;
  const int i_v = blockIdx.x, i_hv = blockIdx.y;
  const int i_h = i_hv / (HV / H);
  const int slot = idx[0];
  const int n = n_dev ? min(*n_dev, T) : T;
  if (slot < 0) {
    if (out)
      for (int t = 0; t < n; ++t) out[((size_t)t * HV + i_hv) * V + i_v * 32 + lane] = __float2bfloat16_rn(0.f);
    return;
  }
  float* h = state + (size_t)slot * HV * V * K + (size_t)i_hv * V * K + (size_t)i_v * 32 * K;
  float hs[32][4];
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    const float4 v4 = *reinterpret_cast<const float4*>(h + r * K + 4 * lane);
    hs[r][0] = v4.x, hs[r][1] = v4.y, hs[r][2] = v4.z, hs[r][3] = v4.w;
  }
  const float negA = __fsub_rn(0.0f, tri_exp(A_log[i_hv]));
  for (int t = 0; t < n; ++t) {
    const __nv_bfloat16* p = qkv + (size_t)t * qkv_stride;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] = __bfloat162float(p[i_h * K + 4 * lane + j]);
      k[j] = __bfloat162float(p[H * K + i_h * K + 4 * lane + j]);
    }
    const float qn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(q, q)), F(0x358637BDu)));
    const float kn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(k, k)), F(0x358637BDu)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] = __fmul_rn(div_full(q[j], qn), scale);
      k[j] = div_full(k[j], kn);
    }
    float dec[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = i_hv * K + 4 * lane + j;
      const float xg = __fadd_rn(__bfloat162float(a[(size_t)t * a_stride + kk]), dt_bias[kk]);
      const float sp = xg <= 20.0f ? tri_log(__fadd_rn(tri_exp(xg), 1.0f)) : xg;
      dec[j] = tri_exp(__fmul_rn(sp, negA));
    }
    const float bv = __bfloat162float(b[(size_t)t * b_stride + i_hv]);
    const float beta = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, bv)), 1.0f));
    const __nv_bfloat16* pv = p + 2 * H * K + (size_t)i_hv * V + i_v * 32;
    float res[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      float hr[4] = {__fmul_rn(hs[r][0], dec[0]), __fmul_rn(hs[r][1], dec[1]), __fmul_rn(hs[r][2], dec[2]),
                     __fmul_rn(hs[r][3], dec[3])};
      const float kv = bfly_sum(dot4(k, hr));
      const float vnew = __fmul_rn(__fsub_rn(__bfloat162float(pv[r]), kv), beta);
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[r][j] = __fmaf_rn(vnew, k[j], hr[j]);
      res[r] = bfly_sum(dot4(q, hs[r]));
    }
    if (out) out[((size_t)t * HV + i_hv) * V + i_v * 32 + lane] = __float2bfloat16_rn(res[lane]);
  }
  if (write_state)
#pragma unroll
    for (int r = 0; r < 32; ++r)
      *reinterpret_cast<float4*>(h + r * K + 4 * lane) = make_float4(hs[r][0], hs[r][1], hs[r][2], hs[r][3]);
}

// kda_multi_kernel for T <= 8 with the token-only work hoisted: the T tokens' q / k normalisation, decays, beta and v
// (independent of the state) are computed first, all T at once (their global loads in flight together), into shared
// memory; then the state recurrence runs over the tokens reading them back. Every value is produced by the same
// operations in the same order as in kda_multi_kernel: outputs and state are bit-identical.
template <int K, int V>
__global__ void __launch_bounds__(32)
kda_multi_pre_kernel(const __nv_bfloat16* __restrict__ qkv, int qkv_stride, const __nv_bfloat16* __restrict__ a,
                     int a_stride, const __nv_bfloat16* __restrict__ b, int b_stride, const float* __restrict__ A_log,
                     const float* __restrict__ dt_bias, float* __restrict__ state, const int32_t* __restrict__ idx,
                     __nv_bfloat16* __restrict__ out, float scale, int H, int HV, int T, const int* __restrict__ n_dev,
                     int write_state) {
  constexpr int TM = 8;
  __shared__ float sq[TM][32][4], sk[TM][32][4], sd[TM][32][4], sv[TM][32], sb[TM];
  const int lane = threadIdx.x;
  const int i_v = blockIdx.x, i_hv = blockIdx.y;
  const int i_h = i_hv / (HV / H);
  const int slot = idx[0];
  const int n = n_dev ? min(*n_dev, T) : T;
  if (slot < 0) {
    if (out)
      for (int t = 0; t < n; ++t) out[((size_t)t * HV + i_hv) * V + i_v * 32 + lane] = __float2bfloat16_rn(0.f);
    return;
  }
  const float negA = __fsub_rn(0.0f, tri_exp(A_log[i_hv]));
#pragma unroll
  for (int t = 0; t < TM; ++t) {
    if (t >= n) break;
    const __nv_bfloat16* p = qkv + (size_t)t * qkv_stride;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] = __bfloat162float(p[i_h * K + 4 * lane + j]);
      k[j] = __bfloat162float(p[H * K + i_h * K + 4 * lane + j]);
    }
    const float qn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(q, q)), F(0x358637BDu)));
    const float kn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(k, k)), F(0x358637BDu)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sq[t][lane][j] = __fmul_rn(div_full(q[j], qn), scale);
      sk[t][lane][j] = div_full(k[j], kn);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = i_hv * K + 4 * lane + j;
      const float xg = __fadd_rn(__bfloat162float(a[(size_t)t * a_stride + kk]), dt_bias[kk]);
      const float sp = xg <= 20.0f ? tri_log(__fadd_rn(tri_exp(xg), 1.0f)) : xg;
      sd[t][lane][j] = tri_exp(__fmul_rn(sp, negA));
    }
    if (lane == 0) {
      const float bv = __bfloat162float(b[(size_t)t * b_stride + i_hv]);
      sb[t] = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, bv)), 1.0f));
    }
    sv[t][lane] = __bfloat162float((p + 2 * H * K + (size_t)i_hv * V + i_v * 32)[lane]);
  }
  float* h = state + (size_t)slot * HV * V * K + (size_t)i_hv * V * K + (size_t)i_v * 32 * K;
  float hs[32][4];
#pragma unroll
  for (int r = 0; r < 32; ++r) {
    const float4 v4 = *reinterpret_cast<const float4*>(h + r * K + 4 * lane);
    hs[r][0] = v4.x, hs[r][1] = v4.y, hs[r][2] = v4.z, hs[r][3] = v4.w;
  }
  __syncwarp();
  for (int t = 0; t < n; ++t) {
    float q[4], k[4], dec[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = sq[t][lane][j], k[j] = sk[t][lane][j], dec[j] = sd[t][lane][j];
    const float beta = sb[t];
    float res[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) {
      float hr[4] = {__fmul_rn(hs[r][0], dec[0]), __fmul_rn(hs[r][1], dec[1]), __fmul_rn(hs[r][2], dec[2]),
                     __fmul_rn(hs[r][3], dec[3])};
      const float kv = bfly_sum(dot4(k, hr));
      const float vnew = __fmul_rn(__fsub_rn(sv[t][r], kv), beta);
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[r][j] = __fmaf_rn(vnew, k[j], hr[j]);
      res[r] = bfly_sum(dot4(q, hs[r]));
    }
    if (out) out[((size_t)t * HV + i_hv) * V + i_v * 32 + lane] = __float2bfloat16_rn(res[lane]);
  }
  if (write_state)
#pragma unroll
    for (int r = 0; r < 32; ++r)
      *reinterpret_cast<float4*>(h + r * K + 4 * lane) = make_float4(hs[r][0], hs[r][1], hs[r][2], hs[r][3]);
}


// kda_multi_pre_kernel with the 32 state rows of a block split over NW warps (32 / NW rows each) and the T tokens'
// prologue split over the warps (token t on warp t % NW): every row and every token value is produced by exactly the
// operations of kda_multi_pre_kernel (rows are independent; the prologue is per token), so outputs and state are
// bit-identical, with NW times the warps in flight (the single-warp kernel runs 1.5 warps per sm90 SM: 192 blocks).
template <int K, int V, int NW, bool PDL = false>
__device__ __forceinline__ void
kda_multi_pre_w_body(const __nv_bfloat16* __restrict__ qkv, int qkv_stride, const __nv_bfloat16* __restrict__ a,
                     int a_stride, const __nv_bfloat16* __restrict__ b, int b_stride, const float* __restrict__ A_log,
                     const float* __restrict__ dt_bias, float* __restrict__ state, const int32_t* __restrict__ idx,
                     __nv_bfloat16* __restrict__ out, float scale, int H, int HV, int T, const int* __restrict__ n_dev,
                     int write_state, int n_stride, int i_v, int i_hv, int z) {
  constexpr int TM = 8, RW = 32 / NW;
  __shared__ float sq[TM][32][4], sk[TM][32][4], sd[TM][32][4], sv[TM][32], sb[TM];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;  // z: sequence of a batched verify
  const int i_h = i_hv / (HV / H);
  const int slot = idx[z];
  const int n = n_dev ? min(n_dev[z * n_stride], T) : T;
  qkv += (size_t)z * T * qkv_stride, a += (size_t)z * T * a_stride, b += (size_t)z * T * b_stride;
  if (out) out += (size_t)z * T * HV * V;
  if (slot < 0) {
    if (PDL) ak_pdl_entry();
    if (out && warp == 0)
      for (int t = 0; t < n; ++t) out[((size_t)t * HV + i_hv) * V + i_v * 32 + lane] = __float2bfloat16_rn(0.f);
    return;
  }
  // PDL (the verify launch): this warp's state rows first -- written only by the commits, which are done by now (the
  // kernel before this one, its conv, writes no recurrent state) -- then the programmatic wait, then the token inputs
  float* h = state + (size_t)slot * HV * V * K + (size_t)i_hv * V * K + (size_t)i_v * 32 * K;
  const int rb = warp * RW;  // this warp's rows rb .. rb + RW - 1 of the block's 32
  float hs[RW][4];
  if (PDL) {
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      const float4 v4 = *reinterpret_cast<const float4*>(h + (rb + r) * K + 4 * lane);
      hs[r][0] = v4.x, hs[r][1] = v4.y, hs[r][2] = v4.z, hs[r][3] = v4.w;
    }
    ak_pdl_entry();
  }
  const float negA = __fsub_rn(0.0f, tri_exp(A_log[i_hv]));
  for (int t = warp; t < TM; t += NW) {
    if (t >= n) break;
    const __nv_bfloat16* p = qkv + (size_t)t * qkv_stride;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] = __bfloat162float(p[i_h * K + 4 * lane + j]);
      k[j] = __bfloat162float(p[H * K + i_h * K + 4 * lane + j]);
    }
    const float qn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(q, q)), F(0x358637BDu)));
    const float kn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(k, k)), F(0x358637BDu)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sq[t][lane][j] = __fmul_rn(div_full(q[j], qn), scale);
      sk[t][lane][j] = div_full(k[j], kn);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = i_hv * K + 4 * lane + j;
      const float xg = __fadd_rn(__bfloat162float(a[(size_t)t * a_stride + kk]), dt_bias[kk]);
      const float sp = xg <= 20.0f ? tri_log(__fadd_rn(tri_exp(xg), 1.0f)) : xg;
      sd[t][lane][j] = tri_exp(__fmul_rn(sp, negA));
    }
    if (lane == 0) {
      const float bv = __bfloat162float(b[(size_t)t * b_stride + i_hv]);
      sb[t] = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, bv)), 1.0f));
    }
    sv[t][lane] = __bfloat162float((p + 2 * H * K + (size_t)i_hv * V + i_v * 32)[lane]);
  }
  if (!PDL)
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      const float4 v4 = *reinterpret_cast<const float4*>(h + (rb + r) * K + 4 * lane);
      hs[r][0] = v4.x, hs[r][1] = v4.y, hs[r][2] = v4.z, hs[r][3] = v4.w;
    }
  __syncthreads();
  for (int t = 0; t < n; ++t) {
    float q[4], k[4], dec[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = sq[t][lane][j], k[j] = sk[t][lane][j], dec[j] = sd[t][lane][j];
    const float beta = sb[t];
    float res[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
      float hr[4] = {__fmul_rn(hs[r][0], dec[0]), __fmul_rn(hs[r][1], dec[1]), __fmul_rn(hs[r][2], dec[2]),
                     __fmul_rn(hs[r][3], dec[3])};
      const float kv = bfly_sum(dot4(k, hr));
      const float vnew = __fmul_rn(__fsub_rn(sv[t][rb + r], kv), beta);
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[r][j] = __fmaf_rn(vnew, k[j], hr[j]);
      res[r] = bfly_sum(dot4(q, hs[r]));
    }
    if (out && lane < RW) {
      float v = res[0];
#pragma unroll
      for (int r = 1; r < RW; ++r) v = lane == r ? res[r] : v;
      out[((size_t)t * HV + i_hv) * V + i_v * 32 + rb + lane] = __float2bfloat16_rn(v);
    }
  }
  if (write_state)
#pragma unroll
    for (int r = 0; r < RW; ++r)
      *reinterpret_cast<float4*>(h + (rb + r) * K + 4 * lane) = make_float4(hs[r][0], hs[r][1], hs[r][2], hs[r][3]);
}
template <int K, int V, int NW>
__global__ void __launch_bounds__(32 * NW)
kda_multi_pre_w_kernel(const __nv_bfloat16* __restrict__ qkv, int qkv_stride, const __nv_bfloat16* __restrict__ a,
                       int a_stride, const __nv_bfloat16* __restrict__ b, int b_stride, const float* __restrict__ A_log,
                       const float* __restrict__ dt_bias, float* __restrict__ state, const int32_t* __restrict__ idx,
                       __nv_bfloat16* __restrict__ out, float scale, int H, int HV, int T, const int* __restrict__ n_dev,
                       int write_state, int n_stride) {
  kda_multi_pre_w_body<K, V, NW, true>(qkv, qkv_stride, a, a_stride, b, b_stride, A_log, dt_bias, state, idx, out, scale, H, HV, T,
                                 n_dev, write_state, n_stride, blockIdx.x, blockIdx.y, blockIdx.z);
}
// Speculative commit of every KDA layer in one launch: blockIdx.y = layer, blockIdx.z = sequence; blockIdx.x < nconv:
// that layer's conv-state replay block (conv_multi_body), else its recurrent-state replay block (kda_multi_pre_w_body,
// 8 warps). The layers' updates are independent, and each block runs exactly the per-layer kernels' code: the same bits.
__global__ void __launch_bounds__(256) kda_commit_all_kernel(const __grid_constant__ KdaCommitArgs a) {
  ak_pdl_entry();
  const KdaCommitLayer& L = a.L[blockIdx.y];
  const int nconv = (a.conv_dim + 255) / 256, z = blockIdx.z;
  if ((int)blockIdx.x < nconv) {
    conv_multi_body(L.sq, a.conv_dim, static_cast<__nv_bfloat16*>(L.conv), L.conv_w, a.slot, nullptr, a.conv_dim, a.T,
                    a.n_dev, 1, a.n_stride, blockIdx.x, z);
    return;
  }
  const int bd = blockIdx.x - nconv;
  kda_multi_pre_w_body<128, 128, 8>(L.sc, a.conv_dim, L.sa, a.a_stride, L.sb, a.b_stride, L.A_log, L.dt_bias, L.ssm, a.slot,
                                    nullptr, a.scale, a.H, a.HV, a.T, a.n_dev, 1, a.n_stride, bd % 4, bd / 4, z);
}
// Speculative verify of one KDA layer in one launch (T <= 8 tokens per sequence, K = V = 128): for value head hv (blockIdx.x)
// of sequence z (blockIdx.y), bit-identical to
//   conv_multi_kernel (write_state 0; its y = the stash sc)  ->  kda_multi_pre_w_kernel<128, 128, 8> (write_state 0)
//   ->  gated_rmsnorm_kernel
// 1024 threads = the 4 blocks (i_v = 32-column tiles) of the recurrent kernel: warp w is warp w % 8 of tile w / 8. The
// conv runs here for the head's 384 channels (q / k of head h = hv / (HV / H), v of hv; q / k stash writes repeat across
// the HV / H heads that share them, with the same values); the token prologue (q / k norms, decays, beta), identical
// in the 4 tiles, runs once (token t on warp t); the 4 tiles' outputs meet in shared memory for the 128-column norm.
__global__ void __launch_bounds__(1024) kda_verify_fused_kernel(const __grid_constant__ KdaVerifyArgs a) {
  constexpr int K = 128, V = 128, TM = 8;
  __shared__ __nv_bfloat16 sx[TM][3 * 128];  // conv outputs: q (128), k (128), v (128) channels of the head
  __shared__ float sq[TM][32][4], sk[TM][32][4], sd[TM][32][4], sv[TM][128], sb[TM];
  __shared__ __nv_bfloat16 so[TM][128];      // recurrent outputs (bf16, as the kernel stores them)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int i_hv = blockIdx.x, z = blockIdx.y, H = a.H, HV = a.HV, T = a.T, i_h = i_hv / (HV / H);
  const int slot = a.slot[z];
  const size_t row0 = (size_t)z * T;  // this sequence's first token row
  // programmatic dependent launch: the conv and recurrent states (written by the commits only) and the conv weights
  // before the wait (agens_pdl.cuh), the token inputs after it
  const int c_ch = tid < 384 ? ((tid >> 7) == 0 ? i_h * K + (tid & 127)
                                : (tid >> 7) == 1 ? H * K + i_h * K + (tid & 127) : 2 * H * K + i_hv * V + (tid & 127))
                             : 0;
  __nv_bfloat16 cs0{}, cs1{}, cs2{};
  float4 cw4 = make_float4(0.f, 0.f, 0.f, 0.f);
  float hs[4][4];
  {
    const int i_v = warp >> 3, rb = (warp & 7) * 4;
    if (slot >= 0) {
      const float* h = a.ssm + (size_t)slot * HV * V * K + (size_t)i_hv * V * K + (size_t)i_v * 32 * K;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float4 v4 = *reinterpret_cast<const float4*>(h + (rb + r) * K + 4 * lane);
        hs[r][0] = v4.x, hs[r][1] = v4.y, hs[r][2] = v4.z, hs[r][3] = v4.w;
      }
      if (tid < 384) {
        const __nv_bfloat16* s = a.conv + (size_t)slot * 3 * a.dim;
        cs0 = s[c_ch], cs1 = s[a.dim + c_ch], cs2 = s[2 * a.dim + c_ch];
        cw4 = *reinterpret_cast<const float4*>(a.conv_w + (size_t)c_ch * 4);
      }
    }
  }
  ak_pdl_entry();
  // 1. conv (conv_multi_body's per-channel code; slot < 0: no output, as there)
  if (tid < 384 && slot >= 0) {
    const int c = c_ch;
    __nv_bfloat16 s0 = cs0, s1 = cs1, s2 = cs2;
    const float wc[4] = {cw4.x, cw4.y, cw4.z, cw4.w};
    __nv_bfloat16 xs[TM];  // the T tokens' inputs loaded together (one memory latency, not T)
#pragma unroll
    for (int t = 0; t < TM; ++t) xs[t] = t < T ? a.x[(row0 + t) * a.x_stride + c] : __nv_bfloat16{};
#pragma unroll
    for (int t = 0; t < TM; ++t) {
      if (t >= T) break;
      const __nv_bfloat16 xin = xs[t];
      float acc = __fmaf_rn(__bfloat162float(s0), wc[0], 0.0f);
      acc = __fmaf_rn(__bfloat162float(s1), wc[1], acc);
      acc = __fmaf_rn(__bfloat162float(s2), wc[2], acc);
      acc = __fmaf_rn(wc[3], __bfloat162float(xin), acc);
      const float out = div_full(acc, __fadd_rn(tri_exp(__fsub_rn(0.0f, acc)), 1.0f));
      s0 = s1, s1 = s2, s2 = xin;
      const __nv_bfloat16 ob = __float2bfloat16_rn(out);
      a.y[(row0 + t) * a.dim + c] = ob;
      sx[t][tid] = ob;
    }
  }
  __syncthreads();
  // 2. token prologue (kda_multi_pre_w_body, token t on warp t; the v row: all 128 columns)
  if (slot >= 0 && warp < TM && warp < T) {
    const int t = warp;
    float q[4], k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = __bfloat162float(sx[t][4 * lane + j]), k[j] = __bfloat162float(sx[t][128 + 4 * lane + j]);
    const float qn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(q, q)), F(0x358637BDu)));
    const float kn = sqrt_approx_ftz(__fadd_rn(bfly_sum(dot4(k, k)), F(0x358637BDu)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sq[t][lane][j] = __fmul_rn(div_full(q[j], qn), a.scale);
      sk[t][lane][j] = div_full(k[j], kn);
    }
    const float negA = __fsub_rn(0.0f, tri_exp(a.A_log[i_hv]));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = i_hv * K + 4 * lane + j;
      const float xg = __fadd_rn(__bfloat162float(a.ag[(row0 + t) * a.a_stride + kk]), a.dt_bias[kk]);
      const float sp = xg <= 20.0f ? tri_log(__fadd_rn(tri_exp(xg), 1.0f)) : xg;
      sd[t][lane][j] = tri_exp(__fmul_rn(sp, negA));
    }
    if (lane == 0) {
      const float bv = __bfloat162float(a.b[(row0 + t) * a.b_stride + i_hv]);
      sb[t] = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, bv)), 1.0f));
    }
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) sv[t][q4 * 32 + lane] = __bfloat162float(sx[t][256 + q4 * 32 + lane]);
  }
  __syncthreads();
  // 3. recurrence: tile i_v = warp / 8, rows (warp % 8) * 4 .. + 3 of its 32 (RW = 4)
  {
    const int i_v = warp >> 3, rb = (warp & 7) * 4;
    if (slot < 0) {
      if ((warp & 7) == 0)
        for (int t = 0; t < T; ++t) so[t][i_v * 32 + lane] = __float2bfloat16_rn(0.f);
    } else {  // (hs: loaded at the start)
      for (int t = 0; t < T; ++t) {
        float q[4], k[4], dec[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = sq[t][lane][j], k[j] = sk[t][lane][j], dec[j] = sd[t][lane][j];
        const float beta = sb[t];
        float res[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          float hr[4] = {__fmul_rn(hs[r][0], dec[0]), __fmul_rn(hs[r][1], dec[1]), __fmul_rn(hs[r][2], dec[2]),
                         __fmul_rn(hs[r][3], dec[3])};
          const float kv = bfly_sum(dot4(k, hr));
          const float vnew = __fmul_rn(__fsub_rn(sv[t][i_v * 32 + rb + r], kv), beta);
#pragma unroll
          for (int j = 0; j < 4; ++j) hs[r][j] = __fmaf_rn(vnew, k[j], hr[j]);
          res[r] = bfly_sum(dot4(q, hs[r]));
        }
        if (lane < 4) {
          float v = res[0];
#pragma unroll
          for (int r = 1; r < 4; ++r) v = lane == r ? res[r] : v;
          so[t][i_v * 32 + rb + lane] = __float2bfloat16_rn(v);
        }
      }
    }
  }
  __syncthreads();
  // 4. gated RMSNorm of the T rows (t, hv): row t on half-warp t (16 lanes x 8 columns), gated_rmsnorm_kernel's code
  if (warp < (T + 1) / 2) {  // whole warps (the shuffles); a half-warp past row T - 1 computes and writes nothing
    const int t = tid >> 4, l = tid & 15;
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) xv[j] = t < T ? __bfloat162float(so[t][8 * l + j]) : 0.f;
    float s = dot8_sq(xv);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, o));
    if (t >= T) return;
    const float var = div_full(s, 128.0f);
    const float rstd = div_full(1.0f, sqrt_approx_ftz(__fadd_rn(var, a.eps)));
    const size_t rowi = ((row0 + t) * HV + i_hv) * 128;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const size_t i = rowi + 8 * l + j;
      const float gv = __bfloat162float(a.g[i]);
      const float sig = div_full(1.0f, __fadd_rn(tri_exp(__fsub_rn(0.0f, gv)), 1.0f));
      float v = __fmul_rn(__fmul_rn(xv[j], rstd), __bfloat162float(a.w[8 * l + j]));
      v = __fmul_rn(__fmul_rn(v, gv), sig);
      a.out[i] = __float2bfloat16_rn(v);
    }
  }
}
}  // namespace

// one KDA layer's speculative verify (conv + recurrence + gated norm) in one launch; nb sequences of T <= 8 tokens
// (cudaErrorNotSupported when the per-layer kernels would not be the 8-warp preload ones, or T > 8)
extern "C" int ak_kda_verify_fused(const KdaVerifyArgs* args, int nb, void* stream) {
  static const int pre = [] { const char* e = getenv("AGENS_KDA_MULTI_PRELOAD"); return !(e && e[0] == '0'); }();
  static const int nw = [] { const char* e = getenv("AGENS_KDA_MULTI_WARPS"); return e ? atoi(e) : 8; }();
  if (!pre || nw < 8 || args->T < 1 || args->T > 8 || args->HV % args->H) return (int)cudaErrorNotSupported;
  kda_verify_fused_kernel<<<dim3(args->HV, nb), 1024, 0, static_cast<cudaStream_t>(stream)>>>(*args);
  return (int)cudaGetLastError();
}

// every KDA layer's speculative commit in one launch (nb sequences; nb = 1: the single-sequence commit): the same bits as
// ak_kda_conv_multi(_b) + ak_kda_decode_multi(_b) per layer. cudaErrorNotSupported when the per-layer kernels would not be
// the 8-warp preload kernel (AGENS_KDA_MULTI_PRELOAD=0 / AGENS_KDA_MULTI_WARPS != 8) or the shape is not K = V = 128.
extern "C" int ak_kda_commit_all(const KdaCommitArgs* args, int nb, void* stream) {
  static const int pre = [] { const char* e = getenv("AGENS_KDA_MULTI_PRELOAD"); return !(e && e[0] == '0'); }();
  static const int nw = [] { const char* e = getenv("AGENS_KDA_MULTI_WARPS"); return e ? atoi(e) : 8; }();
  if (!pre || nw < 8 || args->T > 8 || args->nl < 1 || args->nl > kKdaCommitMaxLayers || args->HV % args->H)
    return (int)cudaErrorNotSupported;
  const int nconv = (args->conv_dim + 255) / 256;
  kda_commit_all_kernel<<<dim3(nconv + 4 * args->HV, args->nl, nb), 256, 0, static_cast<cudaStream_t>(stream)>>>(*args);
  return (int)cudaGetLastError();
}

extern "C" int ak_kda_gated_rmsnorm(const void* x, const void* g, const void* w, void* y, int rows, int d, float eps,
                                    void* stream) {
  if (d != 128) return (int)cudaErrorInvalidValue;
  gated_rmsnorm_kernel<<<(rows + 1) / 2, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(g),
      static_cast<const __nv_bfloat16*>(w), static_cast<__nv_bfloat16*>(y), rows, eps);
  return (int)cudaGetLastError();
}

extern "C" int ak_kda_conv_update(const void* x, int x_stride, void* conv_state, const float* weight,
                                  const int32_t* idx, void* y, int batch, int dim, void* stream) {
  dim3 grid((dim + 255) / 256, batch);
  conv_update_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), x_stride, static_cast<__nv_bfloat16*>(conv_state), weight, idx,
      static_cast<__nv_bfloat16*>(y), dim);
  return (int)cudaGetLastError();
}

extern "C" int ak_kda_decode(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b,
                             int b_stride, const float* A_log, const float* dt_bias, float* state,
                             const int32_t* idx, void* out, float scale, int batch, int H, int HV, int K, int V,
                             void* stream) {
  if (K != 128 || V % 32 != 0 || HV % H != 0) return (int)cudaErrorInvalidValue;
  dim3 grid(V / 32, batch * HV);
  if (V == 128)
    kda_decode_kernel<128, 128><<<grid, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(qkv), qkv_stride, static_cast<const __nv_bfloat16*>(a), a_stride,
        static_cast<const __nv_bfloat16*>(b), b_stride, A_log, dt_bias, state, idx,
        static_cast<__nv_bfloat16*>(out), scale, H, HV);
  else
    return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}

extern "C" int ak_kda_conv_multi(const void* x, int x_stride, void* conv_state, const float* weight, const int32_t* idx,
                                 void* y, int dim, int T, const int* n_dev, int write_state, void* stream) {
  conv_multi_kernel<<<(dim + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), x_stride, static_cast<__nv_bfloat16*>(conv_state), weight, idx,
      static_cast<__nv_bfloat16*>(y), dim, T, n_dev, write_state, 0);
  return (int)cudaGetLastError();
}
// nb sequences of T rows each (batched speculative verify / commit): sequence z's rows at z * T, its slot idx[z],
// its commit count n_dev[z * n_stride]; every sequence exactly as its own ak_kda_conv_multi call
extern "C" int ak_kda_conv_multi_b(const void* x, int x_stride, void* conv_state, const float* weight, const int32_t* idx,
                                   void* y, int dim, int T, int nb, const int* n_dev, int n_stride, int write_state,
                                   void* stream) {
  conv_multi_kernel<<<dim3((dim + 255) / 256, nb), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), x_stride, static_cast<__nv_bfloat16*>(conv_state), weight, idx,
      static_cast<__nv_bfloat16*>(y), dim, T, n_dev, write_state, n_stride);
  return (int)cudaGetLastError();
}

extern "C" int ak_kda_decode_multi(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b,
                                   int b_stride, const float* A_log, const float* dt_bias, float* state,
                                   const int32_t* idx, void* out, float scale, int H, int HV, int K, int V, int T,
                                   const int* n_dev, int write_state, void* stream) {
  if (K != 128 || V != 128 || HV % H != 0) return (int)cudaErrorInvalidValue;
  static const int pre = [] { const char* e = getenv("AGENS_KDA_MULTI_PRELOAD"); return !(e && e[0] == '0'); }();
  // AGENS_KDA_MULTI_WARPS: warps per 32-row block of the preload kernel (1, 2, 4 or 8; default 8; bit-identical;
  // sm90 verify: 1 warp 21.8 us, 4 warps 9.2, 8 warps 7.4 per call)
  static const int nw = [] { const char* e = getenv("AGENS_KDA_MULTI_WARPS"); return e ? atoi(e) : 8; }();
  if (pre && T <= 8 && nw > 1) {
    auto kw = nw >= 8 ? kda_multi_pre_w_kernel<128, 128, 8>
                      : nw >= 4 ? kda_multi_pre_w_kernel<128, 128, 4> : kda_multi_pre_w_kernel<128, 128, 2>;
    kw<<<dim3(V / 32, HV), 32 * (nw >= 8 ? 8 : nw >= 4 ? 4 : 2), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(qkv), qkv_stride, static_cast<const __nv_bfloat16*>(a), a_stride,
        static_cast<const __nv_bfloat16*>(b), b_stride, A_log, dt_bias, state, idx, static_cast<__nv_bfloat16*>(out),
        scale, H, HV, T, n_dev, write_state, 0);
    return (int)cudaGetLastError();
  }
  auto kern = pre && T <= 8 ? kda_multi_pre_kernel<128, 128> : kda_multi_kernel<128, 128>;
  kern<<<dim3(V / 32, HV), 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(qkv), qkv_stride, static_cast<const __nv_bfloat16*>(a), a_stride,
      static_cast<const __nv_bfloat16*>(b), b_stride, A_log, dt_bias, state, idx, static_cast<__nv_bfloat16*>(out),
      scale, H, HV, T, n_dev, write_state);
  return (int)cudaGetLastError();
}
// nb sequences of T <= 8 rows (batched speculative verify / commit), each exactly as its own ak_kda_decode_multi call
// with the preload kernel (8 warps): rows of sequence z at z * T (qkv / a / b / out), slot idx[z], count
// n_dev[z * n_stride] (nullable)
extern "C" int ak_kda_decode_multi_b(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b,
                                     int b_stride, const float* A_log, const float* dt_bias, float* state,
                                     const int32_t* idx, void* out, float scale, int H, int HV, int K, int V, int T, int nb,
                                     const int* n_dev, int n_stride, int write_state, void* stream) {
  if (K != 128 || V != 128 || HV % H != 0 || T > 8) return (int)cudaErrorInvalidValue;
  kda_multi_pre_w_kernel<128, 128, 8><<<dim3(V / 32, HV, nb), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(qkv), qkv_stride, static_cast<const __nv_bfloat16*>(a), a_stride,
      static_cast<const __nv_bfloat16*>(b), b_stride, A_log, dt_bias, state, idx, static_cast<__nv_bfloat16*>(out), scale,
      H, HV, T, n_dev, write_state, n_stride);
  return (int)cudaGetLastError();
}
