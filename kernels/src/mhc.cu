// mHC (4-stream manifold hyper-connections) mix + combine, bit-exact with sglv23's fused Triton kernels
// (sglang/srt/layers/volundr_fused.py: _mhc_mix_kernel, _mhc_combine_kernel; enable_fp_fusion=False), which are
// themselves bit-exact with the torch reference. PTX facts (tools/dump_mhc_ref.py -> out/mhc_ref):
//   * libdevice.div_rn  -> div.rn.ftz.f32;  libdevice.exp -> __nv_expf (FTZ) sequence, transcribed in nv_expf_ftz
//   * every add/mul is .rn with no contraction; bf16 roundings exactly where the Triton source rounds
//   * Sinkhorn row sums pairwise (v0+v2)+(v1+v3), column sums serial ((v0+v1)+v2)+v3; clamp = max(s, 1e-9)
// Not here (they stay library calls, issued like torch): the RMSNorm, the `dyn` GEMM and the res·H bmm.

#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float F(unsigned u) { return __uint_as_float(u); }
__device__ __forceinline__ float div_rn_ftz(float a, float b) {
  float y;
  asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fma_rn_ftz(float a, float b, float c) {
  float y;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
  return y;
}
__device__ __forceinline__ float fma_rm_ftz(float a, float b, float c) {
  float y;
  asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
  return y;
}
__device__ __forceinline__ float sat_ftz(float a) {
  float y;
  asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}
__device__ __forceinline__ float ex2_approx_ftz(float a) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}
// libdevice __nv_expf with FTZ reflect = 1, as inlined by Triton.
__device__ __forceinline__ float nv_expf_ftz(float x) {
  const float t = sat_ftz(fma_rn_ftz(x, F(0x3BBB989Du), F(0x3F000000u)));
  const float j = fma_rm_ftz(t, F(0x437C0000u), F(0x4B400001u));
  const float k = __fadd_rn(j, F(0xCB40007Fu));
  float y = fma_rn_ftz(x, F(0x3FB8AA3Bu), -k);
  y = fma_rn_ftz(x, F(0x32A57060u), y);
  return __fmul_rn(ex2_approx_ftz(y), __int_as_float(__float_as_int(j) << 23));
}
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }

constexpr int kBlock = 1024;  // elements per program (Triton BLOCK), 4 per thread

// d [T][24] bf16 (dyn output), statics bf16, H [T][4][DIM] bf16 -> x [T][DIM], post [T][4], res [T][16] bf16
__global__ void __launch_bounds__(256)
mhc_mix_kernel(const __nv_bfloat16* __restrict__ d, int d_stride, const __nv_bfloat16* __restrict__ pre_s,
               const __nv_bfloat16* __restrict__ post_s, const __nv_bfloat16* __restrict__ res_s,
               const __nv_bfloat16* __restrict__ H, int h_st, int h_ss, __nv_bfloat16* __restrict__ X, int x_st,
               __nv_bfloat16* __restrict__ post_o, __nv_bfloat16* __restrict__ res_o, int dim, int iters) {
  const int t = blockIdx.x, pb = blockIdx.y;
  const __nv_bfloat16* drow = d + (size_t)t * d_stride;
  if (pb == gridDim.y - 1) {  // its own block: post + the Sinkhorn of res, one 4x4 element per lane of 16 (below)
    if (threadIdx.x >= 16) return;
    const int e = threadIdx.x, r = e >> 2, c = e & 3;
    if (e < 4) post_o[t * 4 + e] = __float2bfloat16_rn(__fadd_rn(ld(post_s + e), ld(drow + 4 + e)));
    float m = __fadd_rn(ld(res_s + e), ld(drow + 8 + e));
    float mx = m;  // max of the 16 (exact in any order)
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFu, mx, o));
    m = nv_expf_ftz(__fsub_rn(m, mx));
    const float eps = F(0x3089705Fu);  // 1e-9
    for (int it = 0; it < iters; ++it) {
      // row r: s = max((m[4r] + m[4r+2]) + (m[4r+1] + m[4r+3]), eps), the serial kernel's expression
      const float a0 = __shfl_sync(0xFFFFu, m, 4 * r), a1 = __shfl_sync(0xFFFFu, m, 4 * r + 1);
      const float a2 = __shfl_sync(0xFFFFu, m, 4 * r + 2), a3 = __shfl_sync(0xFFFFu, m, 4 * r + 3);
      m = div_rn_ftz(m, fmaxf(__fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3)), eps));
      // column c: s = max(((m[c] + m[4+c]) + m[8+c]) + m[12+c], eps)
      const float b0 = __shfl_sync(0xFFFFu, m, c), b1 = __shfl_sync(0xFFFFu, m, 4 + c);
      const float b2 = __shfl_sync(0xFFFFu, m, 8 + c), b3 = __shfl_sync(0xFFFFu, m, 12 + c);
      m = div_rn_ftz(m, fmaxf(__fadd_rn(__fadd_rn(__fadd_rn(b0, b1), b2), b3), eps));
    }
    res_o[t * 16 + e] = __float2bfloat16_rn(m);
    return;
  }
  float p[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) p[s] = bf(__fadd_rn(ld(pre_s + s), ld(drow + s)));
  const __nv_bfloat16* hb = H + (size_t)t * h_st;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off = pb * kBlock + j * 256 + threadIdx.x;
    if (off < dim) {
      float q[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) q[s] = bf(__fmul_rn(p[s], ld(hb + (size_t)s * h_ss + off)));
      const float x = __fadd_rn(__fadd_rn(__fadd_rn(q[0], q[1]), q[2]), q[3]);
      X[(size_t)t * x_st + off] = __float2bfloat16_rn(x);
    }
  }
}

// mixed [T][4][DIM] (= res·H), post [T][4], y / x_in [T][DIM] -> out [T][4][DIM], mean [T][DIM] (optional)
__global__ void __launch_bounds__(256)
mhc_combine_kernel(const __nv_bfloat16* __restrict__ M, int m_st, int m_ss, const __nv_bfloat16* __restrict__ post,
                   const __nv_bfloat16* __restrict__ Y, int y_st, const __nv_bfloat16* __restrict__ Xi, int x_st,
                   __nv_bfloat16* __restrict__ O, int o_st, int o_ss, __nv_bfloat16* __restrict__ mean, int mean_st,
                   int dim, const __nv_bfloat16* __restrict__ Yb = nullptr) {
  const int t = blockIdx.x, pb = blockIdx.y;
  float w[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) w[s] = ld(post + t * 4 + s);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int off = pb * kBlock + j * 256 + threadIdx.x;
    if (off >= dim) continue;
    // Yb: the layer's last residual add (y = bf16(Y + Yb), torch's bf16 add) folded in here
    const float yv = Yb ? __bfloat162float(__float2bfloat16(ld(Y + (size_t)t * y_st + off) + ld(Yb + (size_t)t * y_st + off)))
                        : ld(Y + (size_t)t * y_st + off);
    const float delta = bf(__fsub_rn(yv, ld(Xi + (size_t)t * x_st + off)));
    float o[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      o[s] = bf(__fadd_rn(ld(M + (size_t)t * m_st + (size_t)s * m_ss + off), bf(__fmul_rn(w[s], delta))));
      O[(size_t)t * o_st + (size_t)s * o_ss + off] = __float2bfloat16_rn(o[s]);
    }
    if (mean) {
      const float sum = __fadd_rn(__fadd_rn(__fadd_rn(o[0], o[1]), o[2]), o[3]);
      mean[(size_t)t * mean_st + off] = __float2bfloat16_rn(__fmul_rn(sum, 0.25f));
    }
  }
}

}  // namespace

extern "C" int ak_mhc_mix(const void* d, int d_stride, const void* pre_static, const void* post_static,
                          const void* res_static, const void* H, int h_st, int h_ss, void* x, int x_st,
                          void* post_out, void* res_out, int T, int dim, int iters, void* stream) {
  if (T == 0) return 0;
  dim3 grid(T, (dim + kBlock - 1) / kBlock + 1);  // + the post / Sinkhorn block
  mhc_mix_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(d), d_stride, static_cast<const __nv_bfloat16*>(pre_static),
      static_cast<const __nv_bfloat16*>(post_static), static_cast<const __nv_bfloat16*>(res_static),
      static_cast<const __nv_bfloat16*>(H), h_st, h_ss, static_cast<__nv_bfloat16*>(x), x_st,
      static_cast<__nv_bfloat16*>(post_out), static_cast<__nv_bfloat16*>(res_out), dim, iters);
  return (int)cudaGetLastError();
}

extern "C" int ak_mhc_combine(const void* mixed, int m_st, int m_ss, const void* post, const void* y, int y_st,
                              const void* x_in, int x_st, void* out, int o_st, int o_ss, void* mean, int mean_st,
                              int T, int dim, void* stream) {
  if (T == 0) return 0;
  dim3 grid(T, (dim + kBlock - 1) / kBlock);
  mhc_combine_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(mixed), m_st, m_ss, static_cast<const __nv_bfloat16*>(post),
      static_cast<const __nv_bfloat16*>(y), y_st, static_cast<const __nv_bfloat16*>(x_in), x_st,
      static_cast<__nv_bfloat16*>(out), o_st, o_ss, static_cast<__nv_bfloat16*>(mean), mean_st, dim);
  return (int)cudaGetLastError();
}
// the same with the layer's last residual add folded in: y = bf16(y + yb) (yb rows as y's)
extern "C" int ak_mhc_combine_add(const void* mixed, int m_st, int m_ss, const void* post, const void* y, const void* yb,
                                  int y_st, const void* x_in, int x_st, void* out, int o_st, int o_ss, void* mean,
                                  int mean_st, int T, int dim, void* stream) {
  if (T == 0) return 0;
  dim3 grid(T, (dim + kBlock - 1) / kBlock);
  mhc_combine_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(mixed), m_st, m_ss, static_cast<const __nv_bfloat16*>(post),
      static_cast<const __nv_bfloat16*>(y), y_st, static_cast<const __nv_bfloat16*>(x_in), x_st,
      static_cast<__nv_bfloat16*>(out), o_st, o_ss, static_cast<__nv_bfloat16*>(mean), mean_st, dim,
      static_cast<const __nv_bfloat16*>(yb));
  return (int)cudaGetLastError();
}

