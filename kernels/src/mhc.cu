// mHC (4-stream manifold hyper-connections) mix + combine, bit-exact with sglv23's fused Triton kernels
// (sglang/srt/layers/volundr_fused.py: _mhc_mix_kernel, _mhc_combine_kernel; enable_fp_fusion=False), which are
// themselves bit-exact with the torch reference. PTX facts (tools/dump_mhc_ref.py -> out/mhc_ref):
//   * libdevice.div_rn  -> div.rn.ftz.f32;  libdevice.exp -> __nv_expf (FTZ) sequence, transcribed in nv_expf_ftz
//   * every add/mul is .rn with no contraction; bf16 roundings exactly where the Triton source rounds
//   * Sinkhorn row sums pairwise (v0+v2)+(v1+v3), column sums serial ((v0+v1)+v2)+v3; clamp = max(s, 1e-9)
// Not here (they stay library calls, issued like torch): the RMSNorm, the `dyn` GEMM and the res·H bmm.

#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include "tp_push.cuh"

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


// ---------------------------------------------------------------- fused layer boundary (decode / verify rows)
// Layer i's end and layer i + 1's start, bit-identical to the unfused chain
//   [mhc_mix's Sinkhorn of layer i] -> ak_bf16_bmm_bi (mixed = res . H) -> mhc_combine(_add) (H', mean)
//   -> ak_rmsnorm(mean, norm.weight, gemma 0) -> ak_bf16_gemv_bi (dyn, n = 24) -> mhc_mix (x_in, post, res)
//   -> ak_rmsnorm(x_in, input_layernorm, gemma 1)
// (7 launches -> 1). Each piece repeats its kernel's operations in its order:
//   * Sinkhorn: mhc_mix_kernel's (16 lanes), from layer i's dyn row and res_static; res enters the bmm as bf16, as the
//     mix kernel stores it. Layer i's res is recomputed here (at the start, under the loads), so the kernel never
//     waits for the Sinkhorn of layer i + 1: that one runs in the next boundary (write_res: also here, for the last).
//   * bmm: fp32 fma chain over the 4 streams from 0, one bf16 rounding
//   * RMSNorm: the CuTe structure at H = 5120 (TPR 64: lane l sums elements 512 b + 8 l + v, b then v ascending, one
//     fma chain; butterfly within each warp, then over the two warp sums), rsqrt.approx.ftz, y = (x * rstd) * (w + bias)
//   * dyn GEMV: bf16_gemv_bi_kernel's MMA chains (chain w of a 16-output block takes k-steps w, w + 8, ..; 2 MMAs per
//     step; chain sums in chain order). The row is token column 0 of the MMAs (columns are independent).
//   * mix / combine: the code of mhc_mix_kernel / mhc_combine_kernel above.
// Layout: a cluster of 8 blocks per token row. Block c owns elements [640 c, 640 c + 640) (threads 0-79, one 16-byte
// vector each) and runs the GEMV chains w = c of both output blocks (warps 3, 4); rows needed whole (the mean and x_in,
// for the norms' exact lane order; the normalized mean for the GEMV) are pushed to every block's shared memory (DSMEM),
// and each block computes the norms' sums itself (same bits everywhere).
constexpr int kFDim = 5120, kFC = 8, kFOwn = kFDim / kFC / 8, kFThreads = 192;  // 80 owner threads per block
#ifdef AGENS_MHC_PROF  // kernels/bench/test_mhc_fused.cu --prof: globaltimer at the phase ends of block 0
__device__ unsigned long long g_mhc_prof[16];
#define MHC_PROF(k, who)                                                       \
  if (blockIdx.x == 0 && threadIdx.x == (who)) {                               \
    unsigned long long tt_;                                                    \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt_));                    \
    g_mhc_prof[k] = tt_;                                                       \
  }
#else
#define MHC_PROF(k, who)
#endif
__device__ __forceinline__ void mma16816f(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float rsqrt_approx_ftz_f(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// the CuTe RMSNorm's sum of squares of row v[kFDim] (shared memory) -> rstd, for every thread of the block (2 barriers)
__device__ __forceinline__ float cute_rstd(const __nv_bfloat16* v, float* ws, float eps) {
  const int tid = threadIdx.x;
  if (tid < 64) {
    float sum = 0.f;
#pragma unroll
    for (int b = 0; b < 10; ++b) {
      const uint4 u = *reinterpret_cast<const uint4*>(v + b * 512 + tid * 8);
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = __bfloat162float(e[j]);
        sum = __fmaf_rn(f, f, sum);
      }
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) sum = __fadd_rn(__shfl_xor_sync(0xFFFFFFFFu, sum, o), sum);
    if ((tid & 31) == 0) ws[tid >> 5] = sum;
  }
  __syncthreads();
  if (tid < 32) {
    float x = tid < 2 ? ws[tid] : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x = __fadd_rn(x, __shfl_xor_sync(0xFFFFFFFFu, x, o));
    if (tid == 0) ws[2] = rsqrt_approx_ftz_f(__fadd_rn(__fdiv_rn(x, (float)kFDim), eps));
  }
  __syncthreads();
  return ws[2];
}
__device__ __forceinline__ uint4 norm_vec(uint4 x, uint4 w, float rstd, float bias) {
  const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&x);
  const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&w);
  uint4 o4;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&o4);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    o[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(e[j]), rstd), __fadd_rn(__bfloat162float(we[j]), bias)));
  return o4;
}
// mhc_mix_kernel's post-free Sinkhorn on 16 consecutive lanes starting at lane `base` (0 or 16) of a warp:
// lane base + e -> element e of res = Sinkhorn(res_s + d[8..23]), returned as the stored bf16 value (float)
__device__ __forceinline__ float sinkhorn16(const __nv_bfloat16* res_s, const __nv_bfloat16* drow, int e, int base,
                                            int iters) {
  const unsigned mask = 0xFFFFu << base;
  const int r = e >> 2, c = e & 3;
  float m = __fadd_rn(ld(res_s + e), ld(drow + 8 + e));
  float mx = m;
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(mask, mx, o));
  m = nv_expf_ftz(__fsub_rn(m, mx));
  const float eps = F(0x3089705Fu);  // 1e-9
  for (int it = 0; it < iters; ++it) {
    const float a0 = __shfl_sync(mask, m, base + 4 * r), a1 = __shfl_sync(mask, m, base + 4 * r + 1);
    const float a2 = __shfl_sync(mask, m, base + 4 * r + 2), a3 = __shfl_sync(mask, m, base + 4 * r + 3);
    m = div_rn_ftz(m, fmaxf(__fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3)), eps));
    const float b0 = __shfl_sync(mask, m, base + c), b1 = __shfl_sync(mask, m, base + 4 + c);
    const float b2 = __shfl_sync(mask, m, base + 8 + c), b3 = __shfl_sync(mask, m, base + 12 + c);
    m = div_rn_ftz(m, fmaxf(__fadd_rn(__fadd_rn(__fadd_rn(b0, b1), b2), b3), eps));
  }
  return bf(m);
}
// ar = 1: layer i's MLP output Yb is this rank's partial, summed here over the tensor-parallel ranks with the push
// all-reduce (tp_push.cuh; the same rank-order sum and rounding as ak_tp_allreduce): the grid is then the push grid
// (clusters past row T only advance their blocks' epochs), each block pushes and sums the Yb units of its slice.
// thread-block clusters: sm90+ only (the host side checks cudaDevAttrClusterLaunch; the other archs get a stub)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#define MHC_CLUSTER __cluster_dims__(kFC, 1, 1)
#else
#define MHC_CLUSTER
#endif
__global__ void MHC_CLUSTER __launch_bounds__(kFThreads)
mhc_fused_kernel(const MhcFusedArgs a, const __grid_constant__ tp_push::Desc pd, int ar, int T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();
#else
  namespace cg = cooperative_groups;
  cg::cluster_group cl = cg::this_cluster();
  const int c = (int)cl.block_rank(), t = blockIdx.x / kFC, tid = threadIdx.x;
  const long punits = (long)T * (kFDim / 8);
  if (t >= T) {  // (ar only) a cluster past the rows: nothing to push or sum
    const unsigned pe = ar ? tp_push::begin(pd) : (ak_pdl_wait(), 0u);
    ak_pdl_launch();
    if (ar) tp_push::end(pd, pe, punits);
    return;
  }
  const int lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const bool own = tid < kFOwn;
  const int off = (c * kFOwn + (own ? tid : 0)) * 8;  // this thread's 8 elements of the row
  __shared__ __align__(16) __nv_bfloat16 smean[kFDim];  // the mean row (pushed by every block), then x_in
  __shared__ __align__(16) __nv_bfloat16 sn[kFDim];     // the normalized mean (GEMV input)
  __shared__ __align__(16) float spart[24][32][4];      // GEMV chain results (chain = output block * 8 + w)
  __shared__ __align__(16) __nv_bfloat16 swn[kFDim];    // norm.weight (cp.async at the start)
  __shared__ float sres[16], sd[24], spost[4], sws[4];
  // in_proj_b weights of this block's k-steps (dynamic, nb_out x kPer x 4 uint4: row q, step u, lane quarter t4)
  extern __shared__ __align__(16) uint4 swb[];
  MHC_PROF(0, 0)
  for (int v = tid; v < kFDim / 8; v += kFThreads)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(swn + v * 8)),
                 "l"(a.wn + v * 8) : "memory");
  const int nbo = a.wb ? a.nb_out : 0, nrbb = (nbo + 15) / 16;  // the in_proj_b GEMV: outputs, 16-output blocks
  // its weights to shared memory now (constant data, before the programmatic wait; read after the combine and norms)
  for (int v = tid; v < nbo * (kFDim / 256) * 4; v += kFThreads) {
    const int q = v / ((kFDim / 256) * 4), u = (v >> 2) % (kFDim / 256), tq = v & 3;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(swb + v)),
                 "l"(a.wb + (size_t)q * kFDim + 32 * (c + 8 * u) + 8 * tq) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  // GEMV chain warps: their weights to registers first (the loads' latency under the combine and the norm)
  constexpr int kSteps = kFDim / 32, kPer = kSteps / 8;  // 160 k-steps, 20 per chain
  uint4 w0[kPer], w1[kPer];
  const int rb = warp - 3, r0 = rb * 16 + g, r1 = r0 + 8;
  if (warp == 3 || warp == 4) {
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int col = 32 * (c + 8 * u) + 8 * t4;
      w0[u] = r0 < 24 ? __ldg(reinterpret_cast<const uint4*>(a.wdyn + (size_t)r0 * kFDim + col)) : make_uint4(0, 0, 0, 0);
      w1[u] = r1 < 24 ? __ldg(reinterpret_cast<const uint4*>(a.wdyn + (size_t)r1 * kFDim + col)) : make_uint4(0, 0, 0, 0);
    }
  }
  // programmatic dependent launch: everything above reads constant weights only (agens_pdl.cuh)
  const unsigned pe = ar ? tp_push::begin(pd) : (ak_pdl_wait(), 0u);  // (begin: its griddepcontrol.wait)
  ak_pdl_launch();
  const size_t ro = (size_t)t * kFDim + off;
  uint4 hv[4], yv4, yb4, xi4, wl4;
  if (ar && own) tp_push::push(pd, pe, reinterpret_cast<const uint4*>(a.Yb), (long)(ro / 8));  // to every peer's inbox
  if (own) {
#pragma unroll
    for (int s = 0; s < 4; ++s) hv[s] = *reinterpret_cast<const uint4*>(a.Hold + (size_t)t * 4 * kFDim + (size_t)s * kFDim + off);
    yv4 = *reinterpret_cast<const uint4*>(a.Y + ro);
    yb4 = a.Yb && !ar ? *reinterpret_cast<const uint4*>(a.Yb + ro) : make_uint4(0, 0, 0, 0);
    xi4 = *reinterpret_cast<const uint4*>(a.Xi + ro);
    wl4 = __ldg(reinterpret_cast<const uint4*>(a.wln + off));
  }
  if (tid < 4) spost[tid] = ld(a.post + t * 4 + tid);
  if (c == 0 && warp == 2 && lane >= 16) {  // layer i's res (Sinkhorn of its dyn row) -> every block
    const int e = lane - 16;
    // res_cur: written by the previous boundary (its write_res: the same sinkhorn16 of the same row, as bf16)
    const float rv = a.res_cur ? ld(a.res_cur + (size_t)t * 16 + e) : sinkhorn16(a.res_s_cur, a.dcur + (size_t)t * 24, e, 16, a.iters);
#pragma unroll
    for (int q = 0; q < kFC; ++q) *cl.map_shared_rank(&sres[e], q) = rv;
  }
  cl.sync();  // 1: sres
  MHC_PROF(1, 0)
  if (ar && own) {  // the peers' Yb units (pushed at their start; the wait overlapped the loads and the Sinkhorn)
    float acc[8];
    tp_push::sum8(pd, pe, reinterpret_cast<const uint4*>(a.Yb), (long)(ro / 8), acc);
    __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(&yb4);
#pragma unroll
    for (int j = 0; j < 8; ++j) yb[j] = __float2bfloat16_rn(acc[j]);
  }
  // bmm + combine (+ the layer's last residual add) on this block's slice
  uint4 hn[4], mn;
  if (own) {
    const __nv_bfloat16 *H[4], *yv = reinterpret_cast<const __nv_bfloat16*>(&yv4), *yb = reinterpret_cast<const __nv_bfloat16*>(&yb4),
                        *xi = reinterpret_cast<const __nv_bfloat16*>(&xi4);
#pragma unroll
    for (int s = 0; s < 4; ++s) H[s] = reinterpret_cast<const __nv_bfloat16*>(&hv[s]);
    __nv_bfloat16* O[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) O[s] = reinterpret_cast<__nv_bfloat16*>(&hn[s]);
    __nv_bfloat16* me = reinterpret_cast<__nv_bfloat16*>(&mn);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float y = a.Yb ? __bfloat162float(__float2bfloat16(__bfloat162float(yv[j]) + __bfloat162float(yb[j]))) : __bfloat162float(yv[j]);
      const float delta = bf(__fsub_rn(y, __bfloat162float(xi[j])));
      float o[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        float m = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) m = __fmaf_rn(sres[s * 4 + q], __bfloat162float(H[q][j]), m);
        o[s] = bf(__fadd_rn(bf(m), bf(__fmul_rn(spost[s], delta))));
        O[s][j] = __float2bfloat16_rn(o[s]);
      }
      me[j] = __float2bfloat16_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(o[0], o[1]), o[2]), o[3]), 0.25f));
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) *reinterpret_cast<uint4*>(a.Hn + (size_t)t * 4 * kFDim + (size_t)s * kFDim + off) = hn[s];
    *reinterpret_cast<uint4*>(a.hmn + ro) = mn;
    if (a.cap) *reinterpret_cast<uint4*>(a.cap + (size_t)t * a.cap_ld + off) = hn[0];
#pragma unroll
    for (int q = 0; q < kFC; ++q) *reinterpret_cast<uint4*>(cl.map_shared_rank(smean + off, q)) = mn;
  }
  cl.sync();  // 2: the whole mean row everywhere
  MHC_PROF(2, 0)
  // RMSNorm of the mean (gemma 0), the whole row (the GEMV chains read all of it)
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  {
    const float rstd = cute_rstd(smean, sws, a.eps);
    for (int v = tid; v < kFDim / 8; v += kFThreads)
      *reinterpret_cast<uint4*>(sn + v * 8) = norm_vec(*reinterpret_cast<const uint4*>(smean + v * 8),
                                                       *reinterpret_cast<const uint4*>(swn + v * 8), rstd, 0.f);
  }
  __syncthreads();
  MHC_PROF(3, 0)
  if (warp == 3 || warp == 4) {  // chain w = c of output block rb: k-steps c, c + 8, ..
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int u = 0; u < kPer; ++u) {
      const int col = 32 * (c + 8 * u) + 8 * t4;
      const uint4 xv = g == 0 ? *reinterpret_cast<const uint4*>(sn + col) : make_uint4(0, 0, 0, 0);
      mma16816f(acc, w0[u].x, w1[u].x, w0[u].y, w1[u].y, xv.x, xv.y);
      mma16816f(acc, w0[u].z, w1[u].z, w0[u].w, w1[u].w, xv.z, xv.w);
    }
    const float4 r4 = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
    for (int q = 0; q < kFC; ++q) *reinterpret_cast<float4*>(cl.map_shared_rank(&spart[rb * 8 + c][lane][0], q)) = r4;
  }
  cl.sync();  // 3: every chain's result everywhere
  MHC_PROF(4, 0)
  if (warp < 2 && t4 == 0) {  // dyn outputs (bf16_gemv_bi_kernel's chain-order sum), every block
    const int b = warp, q0 = b * 16 + g, q1 = q0 + 8;
    float v0 = 0.f, v2 = 0.f;
    for (int ww = 0; ww < 8; ++ww) v0 += spart[b * 8 + ww][lane][0], v2 += spart[b * 8 + ww][lane][2];
    if (q0 < 24) {
      const __nv_bfloat16 bv = __float2bfloat16_rn(v0);
      sd[q0] = __bfloat162float(bv);
      if (c == 0) a.d[t * 24 + q0] = bv;
    }
    if (q1 < 24) {
      const __nv_bfloat16 bv = __float2bfloat16_rn(v2);
      sd[q1] = __bfloat162float(bv);
      if (c == 0) a.d[t * 24 + q1] = bv;
    }
  }
  __syncthreads();
  MHC_PROF(5, 0)
  if (c == 0 && tid >= 96 && tid < 100) {
    const int e = tid - 96;
    a.post_o[t * 4 + e] = __float2bfloat16_rn(__fadd_rn(ld(a.post_s + e), sd[4 + e]));
  }
  if (a.write_res && c == 0 && warp == 2 && lane >= 16) {  // layer i + 1's res (the last boundary: no next one)
    const int e = lane - 16;
    a.res_o[t * 16 + e] = __float2bfloat16_rn(sinkhorn16(a.res_s, a.d + (size_t)t * 24, e, 16, a.iters));
  }
  uint4 xo = make_uint4(0, 0, 0, 0);
  if (own) {
    float p[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) p[s] = bf(__fadd_rn(ld(a.pre_s + s), sd[s]));
    __nv_bfloat16* xe = reinterpret_cast<__nv_bfloat16*>(&xo);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float q[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) q[s] = bf(__fmul_rn(p[s], __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&hn[s])[j])));
      xe[j] = __float2bfloat16_rn(__fadd_rn(__fadd_rn(__fadd_rn(q[0], q[1]), q[2]), q[3]));
    }
    *reinterpret_cast<uint4*>(a.xin + ro) = xo;
#pragma unroll
    for (int q = 0; q < kFC; ++q) *reinterpret_cast<uint4*>(cl.map_shared_rank(smean + off, q)) = xo;  // smean: x_in now
  }
  cl.sync();  // 4: the whole x_in row everywhere (no DSMEM access after this)
  MHC_PROF(6, 0)
  const float rstd = cute_rstd(smean, sws, a.eps);
  uint4 hv4 = make_uint4(0, 0, 0, 0);
  if (own) {
    hv4 = norm_vec(xo, wl4, rstd, 1.f);
    *reinterpret_cast<uint4*>(a.h + ro) = hv4;
  }
  MHC_PROF(7, 0)
  if (!nbo) {
    if (ar) tp_push::end(pd, pe, punits);
    return;
  }
  // the next (KDA) layer's in_proj_b = h . Wb^T (ak_bf16_gemv_bi, n <= 48): the whole h row to every block, then the
  // chains w = c of each 16-output block on warps 3 .. 3 + nrbb - 1, as the dyn GEMV
  if (own)
#pragma unroll
    for (int q = 0; q < kFC; ++q) *reinterpret_cast<uint4*>(cl.map_shared_rank(sn + off, q)) = hv4;
  cl.sync();  // 5: h everywhere (sn's normalized mean was last read before barrier 3)
  if (warp >= 3 && warp - 3 < nrbb) {
    const int b = warp - 3, q0 = b * 16 + g, q1 = q0 + 8;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    constexpr int kU = 5;
#pragma unroll 1
    for (int u0 = 0; u0 < kPer; u0 += kU) {
      uint4 x0[kU], x1[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {  // (staged in swb at the start: the same values as the global rows)
        x0[u] = q0 < nbo ? swb[(q0 * kPer + u0 + u) * 4 + t4] : make_uint4(0, 0, 0, 0);
        x1[u] = q1 < nbo ? swb[(q1 * kPer + u0 + u) * 4 + t4] : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int col = 32 * (c + 8 * (u0 + u)) + 8 * t4;
        const uint4 xv = g == 0 ? *reinterpret_cast<const uint4*>(sn + col) : make_uint4(0, 0, 0, 0);
        mma16816f(acc, x0[u].x, x1[u].x, x0[u].y, x1[u].y, xv.x, xv.y);
        mma16816f(acc, x0[u].z, x1[u].z, x0[u].w, x1[u].w, xv.z, xv.w);
      }
    }
    const float4 r4 = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
    for (int q = 0; q < kFC; ++q) *reinterpret_cast<float4*>(cl.map_shared_rank(&spart[b * 8 + c][lane][0], q)) = r4;
  }
  cl.sync();  // 6: every in_proj_b chain everywhere (no DSMEM access after this)
  if (c == 0 && warp < nrbb && t4 == 0) {
    const int b = warp, q0 = b * 16 + g, q1 = q0 + 8;
    float v0 = 0.f, v2 = 0.f;
    for (int ww = 0; ww < 8; ++ww) v0 += spart[b * 8 + ww][lane][0], v2 += spart[b * 8 + ww][lane][2];
    if (q0 < nbo) a.bout[(size_t)t * nbo + q0] = __float2bfloat16_rn(v0);
    if (q1 < nbo) a.bout[(size_t)t * nbo + q1] = __float2bfloat16_rn(v2);
  }
  if (ar) tp_push::end(pd, pe, punits);
#endif
}
// AGENS_MHC_FUSE_AR=0: the MLP all-reduce stays its own launch
bool args_ok_env() {
  static const bool on = !(getenv("AGENS_MHC_FUSE_AR") && getenv("AGENS_MHC_FUSE_AR")[0] == '0') && !getenv("AGENS_TP_TRACE");
  return on;
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


// Fused layer boundary (decode / verify rows, H = 5120 only; see mhc_fused_kernel): T blocks of 640 threads.
// whether this device can launch the fused boundary (thread-block clusters: sm90+)
extern "C" int ak_mhc_fused_ok() {
  static int ok = -1;
  if (ok < 0) {
    int dev = 0, v = 0;
    cudaGetDevice(&dev);
    ok = cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev) == cudaSuccess && v != 0;
    cudaGetLastError();
  }
  return ok;
}
// dynamic shared memory of a launch: the staged in_proj_b weights (nb_out <= 48 rows x 1280 B per block)
int fused_smem(const MhcFusedArgs* args) {
  static const bool attr = cudaFuncSetAttribute(mhc_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                48 * (kFDim / 256) * 64) == cudaSuccess;
  const int nbo = args->wb ? args->nb_out : 0;
  return attr || nbo * (kFDim / 256) * 64 <= 8192 ? nbo * (kFDim / 256) * 64 : -1;
}
extern "C" int ak_mhc_fused(const MhcFusedArgs* args, int T, void* stream) {
  if (T == 0) return 0;
  if (!ak_mhc_fused_ok()) return (int)cudaErrorNotSupported;
  const int sm = fused_smem(args);
  if (sm < 0) return (int)cudaErrorInvalidValue;
  mhc_fused_kernel<<<T * kFC, kFThreads, sm, static_cast<cudaStream_t>(stream)>>>(*args, tp_push::Desc{}, 0, T);
  return (int)cudaGetLastError();
}
// with layer i's MLP all-reduce folded in (args->Yb = this rank's partial MLP output): cudaErrorNotSupported when the push
// path cannot take it (no peer regions, push grid not 8-block clusters for T rows, too many elements); the caller then
// all-reduces Yb first and calls ak_mhc_fused
extern "C" int ak_mhc_fused_ar_ok(int T) {
  const tp_push::Desc* d = ak_tp_push_desc();
  return d && ak_mhc_fused_ok() && args_ok_env() && d->grid % kFC == 0 && T >= 1 && T <= d->grid / kFC && (long)T * kFDim <= d->elems;
}
extern "C" int ak_mhc_fused_ar(const MhcFusedArgs* args, int T, void* stream) {
  if (!ak_mhc_fused_ar_ok(T) || !args->Yb) return (int)cudaErrorNotSupported;
  const tp_push::Desc* d = ak_tp_push_desc();
  const int sm = fused_smem(args);
  if (sm < 0) return (int)cudaErrorInvalidValue;
  mhc_fused_kernel<<<d->grid, kFThreads, sm, static_cast<cudaStream_t>(stream)>>>(*args, *d, 1, T);
  return (int)cudaGetLastError();
}
