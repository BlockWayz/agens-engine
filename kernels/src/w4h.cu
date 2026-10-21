// W4H: sm90 W4 g128 decode / verify GEMM on a fragment-major layout built at load time.
//
// Numerics: exactly the tensor-core KT path (w4_mma_kernel<8, 1> in w4_gemv.cu and its lean / TMA / GEMM twins), so
// every output is bit-identical to it: per 16-row block, warp-slot i (0..7) runs group G = 8t + i of every k-tile t in
// increasing t; per group the 8-MMA m16n8k16 chain over its 128 inputs (A = exact bf16 q - 8, B = x, fp32 accumulate,
// the KT pair mapping), then acc = fmaf(group sum, scale, acc); the 8 slot sums are added in slot order from 0 and
// rounded to bf16. An output column depends only on its own token, so rows are bit-identical for any m (decode m = 1 ==
// verify m = 8 == batched verify) and to the KT kernels.
//
// LAYOUT (frozen 2026-10-20; shared with the prefill W4 GEMM -- changes go through main). Two arrays, n % 16 == 0,
// k % 128 == 0, no k padding; rb = row block (16 rows), G = group (128 inputs), both row-major in that order:
//   weights wt: [rb][G][h = 0, 1][lane = 0..31][16 B]  -- one 1024 B group record per (rb, G), at (rb * ng + G) * 1024,
//            ng = k / 128. The 16 B of (h, lane) are the 4 compressed-tensors int32 words (unchanged: nibble j of word w
//            = input 8 w + j, value q + 8, q in [-8, 7]) of inputs 128 G + 32 t4 .. 128 G + 32 t4 + 31 of row
//            16 rb + g + 8 h, where g = lane / 4, t4 = lane % 4 (the m16n8k16 A-fragment owner of those weights).
//   scales st: [rb][G][8 x uint32]                      -- one 32 B record per (rb, G), at (rb * ng + G) * 32; word g
//            (0..7) = bf16x2 (low: scale of row 16 rb + g, high: scale of row 16 rb + g + 8) of group G.
//   Sizes: n * k / 2 and n * k / 64 bytes (exactly the logical int4 + bf16-scale bytes). Both fit in the KT buffers of
//   the same linear (ak_w4_kt_weight_bytes / ak_w4_kt_scale_bytes), so the engine repacks W4H into those allocations
//   and keeps no other copy. Repack: ak_w4h_repack (from the checkpoint's packed int32 [n][k/8] + bf16 scales [n][k/128]);
//   dense bf16 rows (value bf16(q * scale)): ak_w4h_dequant_rows.
//
// Kernels (all: persistent, one block per SM, a producer warp streams weight stages with cp.async.bulk into a
// shared-memory ring, mbarrier full / empty per stage; consumer warp i = slot i; mma.sync + bulk copies only, so the code
// builds for sm90 / sm100 / sm120 -- the launches need >= 200 KB of shared memory per block):
//   * w4h_xr_kernel  m <= 8, 3 / 5 / 6 / 9 k-tiles: x B fragments of the warp's groups in registers, stage =
//                    (row block, k-tile). The decode / verify kernel.
//   * w4h_kernel     m <= 16 otherwise: x fragments of the whole k in shared memory (generic k-tile count).
//   * w4h_bt_kernel  8 < m <= 32 per launch (larger m: 32-token chunks): stage = (pass of up to RB row blocks, k-tile)
//                    with the raw x tile of the tokens; x converted once per stage, reused across the pass.
// ak_w4h_gemm picks among them; up to 4 linears sharing x per launch (q|k|v, qkv|z|f_down), each with a row stride.
#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace {

int sms();
int smem_optin();
constexpr int kSegs = 4;
struct W4hSegs {
  const uint4* wt[kSegs];
  const uint32_t* st[kSegs];
  __nv_bfloat16* y[kSegs];
  int rb_end[kSegs];  // cumulative row-block ends
  int ldy[kSegs];
  int nseg;
};
constexpr int kStageW = 8 * 1024, kStageS = 8 * 32, kStage = kStageW + kStageS;
constexpr int kMaxStages = 32;

// The bulk-copy / transaction-count mbarrier ops need sm90+; the library also builds for sm86 / sm89, where these
// kernels are never launched (ak_w4h_* refuse devices without the shared memory, the engine never repacks W4H there).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define W4H_SM90(...)
#else
#define W4H_SM90(...) __VA_ARGS__
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) {
  W4H_SM90(uint32_t done = 0; do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done)
                 : "r"(a), "r"(par)
                 : "memory");
  } while (!done);)
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  W4H_SM90(asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");)
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  W4H_SM90(asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                        "l"(src), "r"(bytes), "r"(bar)
                        : "memory");)
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// bf16x2 (q_j - 8, q_{j+4} - 8), exact (w4_gemv.cu deq_pair)
__device__ __forceinline__ uint32_t deq_pair(uint32_t w, int j) {
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t x_pair(const uint32_t* xw, int wd, int j) {
  const uint32_t a = xw[4 * wd + (j >> 1)], b = xw[4 * wd + 2 + (j >> 1)];
  return __byte_perm(a, b, (j & 1) ? 0x7632 : 0x5410);
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm(  // not volatile: a pure register op the compiler may schedule around the (volatile) ring loads
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int seg_of(const W4hSegs& s, int rb) {
  int sg = 0;
#pragma unroll
  for (int i = 1; i < kSegs; ++i) sg += i < s.nseg && rb >= s.rb_end[i - 1];
  return sg;
}

// x [m][k] bf16 -> per block, shared xs[G][tt][q4 0..3][lane][4 words]: the B registers (b0, b1 of MMAs 2 q4, 2 q4 + 1)
// of lane (g, t4) for token 8 tt + g, group G (zeros for tokens >= m). NT token tiles of 8 (m <= 8 NT).
// A stage is TPS consecutive k-tiles of one row block: [TPS * 8 group records of 1 KB][TPS * 8 scale records of 32 B];
// a row block is nst = ceil(ntile / TPS) stages. CS consumer sets of 8 warps (warp i of a set = slot i: groups
// 8 t + i): set c takes the block's row blocks j = c, c + CS, ...; the producer fills stages group by group of CS row
// blocks, stage-interleaved across the group's sets, so the sets advance together and no consumer ever waits more than
// CS stages past the producer (the ring's parity stays unambiguous with stages >= 2 CS).
constexpr int kTPS = 2;
constexpr int kStageB = kTPS * kStage;
template <int NT, int CS>
__global__ void __launch_bounds__((8 * CS + 1) * 32, 1)
w4h_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ W4hSegs segs, int k, int stages) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages];
  // dynamic shared memory: [red: CS x 2 x 8 x 32 x 4 NT fp32][x fragments: ng x NT x 2 KB][ring: stages x kStageB]
  float (*red)[2][8][32][4 * NT] = reinterpret_cast<float (*)[2][8][32][4 * NT]>(sm);
  const int ng = k >> 7, ntile = (ng + 7) >> 3, nst = (ntile + kTPS - 1) / kTPS;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nrb = segs.rb_end[segs.nseg - 1];
  const int lgs = 31 - __clz(stages);  // stages: a power of two
  // this block's row blocks: rb = blockIdx.x + j * gridDim.x, j < nj
  const int nj = blockIdx.x < nrb ? (nrb - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  const uint32_t xs = smem_u32(sm) + CS * NT * 8192, ring = xs + (uint32_t)ng * NT * 2048;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) mbar_init(smem_u32(&full[i]), 1), mbar_init(smem_u32(&empty[i]), 8);
    W4H_SM90(asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");)
  }
  __syncthreads();
  if (warp == 8 * CS) {  // producer
    if (lane == 0) {
      int seq = 0;
      for (int j0 = 0; j0 < nj; j0 += CS) {
        const int cnt = nj - j0 < CS ? nj - j0 : CS;
        for (int st = 0; st < nst; ++st)
          for (int cc = 0; cc < cnt; ++cc, ++seq) {
            const int rb = blockIdx.x + (j0 + cc) * gridDim.x;
            const int sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
            const size_t g0 = (size_t)lrb * ng + 8 * kTPS * st;
            const int gn = ng - 8 * kTPS * st < 8 * kTPS ? ng - 8 * kTPS * st : 8 * kTPS;
            const int slot = seq & (stages - 1);
            mbar_wait(smem_u32(&empty[slot]), ((seq >> lgs) & 1) ^ 1);
            const uint32_t fb = smem_u32(&full[slot]), dst = ring + (uint32_t)slot * kStageB;
            mbar_expect_tx(fb, gn * (1024 + 32));
            bulk_g2s(dst, reinterpret_cast<const unsigned char*>(segs.wt[sg]) + g0 * 1024, gn * 1024, fb);
            bulk_g2s(dst + kTPS * kStageW, reinterpret_cast<const unsigned char*>(segs.st[sg]) + g0 * 32, gn * 32, fb);
          }
      }
    }
    return;
  }
  const int c = warp >> 3, i = warp & 7, g = lane >> 2, t4 = lane & 3;
  ak_pdl_entry();  // the producer above streams weights only (constant): it starts before the previous kernel ends
  // x -> B fragments of slot i's groups (G = i + 8 r; set c converts r = c, c + CS, ...): the loads of XB groups are
  // issued together (one L2 round trip per batch instead of per group)
  {
    constexpr int XB = NT == 1 ? 4 : 2;
    for (int G0 = i + 8 * c; G0 < ng; G0 += 8 * CS * XB) {
      uint4 xv[XB][NT][4];
#pragma unroll
      for (int b = 0; b < XB; ++b)
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
          const int G = G0 + 8 * CS * b, tok = tt * 8 + g;
          const bool in = G < ng && tok < m;
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(in ? tok : 0) * k + (size_t)(in ? G : 0) * 128 + t4 * 32);
#pragma unroll
          for (int q = 0; q < 4; ++q) xv[b][tt][q] = in ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
      for (int b = 0; b < XB; ++b) {
        const int G = G0 + 8 * CS * b;
        if (G >= ng) break;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
          const uint32_t xw[16] = {xv[b][tt][0].x, xv[b][tt][0].y, xv[b][tt][0].z, xv[b][tt][0].w,
                                   xv[b][tt][1].x, xv[b][tt][1].y, xv[b][tt][1].z, xv[b][tt][1].w,
                                   xv[b][tt][2].x, xv[b][tt][2].y, xv[b][tt][2].z, xv[b][tt][2].w,
                                   xv[b][tt][3].x, xv[b][tt][3].y, xv[b][tt][3].z, xv[b][tt][3].w};
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            uint32_t bb[4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int q = 2 * q4 + h, wd = q >> 1, j0 = 2 * (q & 1);
              bb[2 * h] = x_pair(xw, wd, j0), bb[2 * h + 1] = x_pair(xw, wd, j0 + 1);
            }
            sts128(xs + (uint32_t)(((G * NT + tt) * 4 + q4) * 32 + lane) * 16, make_uint4(bb[0], bb[1], bb[2], bb[3]));
          }
        }
      }
    }
  }
  if (CS > 1) asm volatile("bar.sync 1, %0;" ::"r"(CS * 256) : "memory");
  else __syncwarp();
  // this set's stages, flat over (row block jl, stage st), software-pipelined: stage e + 1 is fetched (and its slot
  // released) before stage e's MMAs, so a warp never holds a slot while it computes
  const int nmine = nj > c ? ((nj - 1 - c) / CS + 1) * nst : 0;
  int fj = 0, fs = 0, fcnt = nj < CS ? nj : CS, fseq = c;  // fetch cursor: (row block, stage), ring sequence number
  const uint32_t wofs = i * 1024 + lane * 16, sofs = kTPS * kStageW + i * 32 + g * 4;
  struct Ops {
    uint4 w0[kTPS], w1[kTPS];
    uint32_t sc[kTPS];
  };
  auto fetch = [&](Ops& o) {
    const int slot = fseq & (stages - 1);
    mbar_wait(smem_u32(&full[slot]), (fseq >> lgs) & 1);
    const uint32_t base = ring + (uint32_t)slot * kStageB;
    // unconditional: a k-tile past ng reads stale ring bytes, whose sums compute() discards
#pragma unroll
    for (int u = 0; u < kTPS; ++u) {
      o.w0[u] = lds128(base + u * 8192 + wofs);
      o.w1[u] = lds128(base + u * 8192 + 512 + wofs);
      o.sc[u] = lds32(base + u * 256 + sofs);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(&empty[slot]));
    if (++fs < nst) {
      fseq += fcnt;
    } else {
      fs = 0, ++fj;
      fseq = fj * CS * nst + c;
      fcnt = nj - fj * CS < CS ? nj - fj * CS : CS;
    }
  };
  float acc[NT][4];
#pragma unroll
  for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  // the stage's groups: cf chains of all its k-tiles first (independent), then the scaled adds in k-tile order
  auto compute = [&](const Ops& o, int st) {
    float cf[kTPS][NT][4];
#pragma unroll
    for (int u = 0; u < kTPS; ++u) {
      const int G = 8 * (kTPS * st + u) + i;
      const uint32_t ww0[4] = {o.w0[u].x, o.w0[u].y, o.w0[u].z, o.w0[u].w};
      const uint32_t ww1[4] = {o.w1[u].x, o.w1[u].y, o.w1[u].z, o.w1[u].w};
      uint32_t a[8][4];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
        a[q][0] = deq_pair(ww0[wd], j0), a[q][1] = deq_pair(ww1[wd], j0);
        a[q][2] = deq_pair(ww0[wd], j1), a[q][3] = deq_pair(ww1[wd], j1);
      }
      const int Gx = G < ng ? G : ng - 1;  // out-of-range group: its sums are discarded below
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        cf[u][tt][0] = cf[u][tt][1] = cf[u][tt][2] = cf[u][tt][3] = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const uint4 b = lds128(xs + (uint32_t)(((Gx * NT + tt) * 4 + q4) * 32 + lane) * 16);
          mma16816(cf[u][tt], a[2 * q4][0], a[2 * q4][1], a[2 * q4][2], a[2 * q4][3], b.x, b.y);
          mma16816(cf[u][tt], a[2 * q4 + 1][0], a[2 * q4 + 1][1], a[2 * q4 + 1][2], a[2 * q4 + 1][3], b.z, b.w);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kTPS; ++u) {
      if (8 * (kTPS * st + u) + i >= ng) continue;  // k padding: no group (the KT path adds an exact 0 there)
      const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&o.sc[u]);
      const float s0 = __low2float(s2), s1 = __high2float(s2);
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        acc[tt][0] = fmaf(cf[u][tt][0], s0, acc[tt][0]);
        acc[tt][1] = fmaf(cf[u][tt][1], s0, acc[tt][1]);
        acc[tt][2] = fmaf(cf[u][tt][2], s1, acc[tt][2]);
        acc[tt][3] = fmaf(cf[u][tt][3], s1, acc[tt][3]);
      }
    }
  };
  int jl = 0, st = 0;
  // after the last stage of a row block: slot-order sum (double-buffered: one set barrier per row block)
  auto finish_rb = [&]() {
    const int p = jl & 1, rb = blockIdx.x + (c + jl * CS) * gridDim.x;
#pragma unroll
    for (int tt = 0; tt < NT; ++tt)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) red[c][p][i][lane][4 * tt + jj] = acc[tt][jj], acc[tt][jj] = 0.f;
    asm volatile("bar.sync %0, 256;" ::"r"(2 + c) : "memory");
    const bool reducer = i == (jl & 7);
    ++jl;
    if (reducer) {
      const int sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
      __nv_bfloat16* __restrict__ y = segs.y[sg];
      const int ldy = segs.ldy[sg], r0 = lrb * 16 + g;
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int w = 0; w < 8; ++w)
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) v[jj] += red[c][p][w][lane][4 * tt + jj];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int col = tt * 8 + 2 * t4 + cc;
          if (col < m) {
            y[(size_t)col * ldy + r0] = __float2bfloat16_rn(v[cc]);
            y[(size_t)col * ldy + r0 + 8] = __float2bfloat16_rn(v[2 + cc]);
          }
        }
      }
    }
  };
  // ping-pong operand buffers (no register copies): compute A while B is fetched, then the reverse
  Ops A, B;
  if (nmine > 0) fetch(A);
  for (int e = 0; e < nmine; e += 2) {
    if (e + 1 < nmine) fetch(B);
    compute(A, st);
    if (++st == nst) st = 0, finish_rb();
    if (e + 1 >= nmine) break;
    if (e + 2 < nmine) fetch(A);
    compute(B, st);
    if (++st == nst) st = 0, finish_rb();
  }
}

// m <= 8, k-tiles per row block NTILE (compile time): x B fragments held in registers. CS consumer sets of 8 warps
// (warp i of a set = slot i); warp i keeps the B fragments of its groups 8 t + i, t < NTILE (16 registers each), loaded
// once, so a group costs it only the weight / scale shared loads, the dequant and 8 MMAs. The block's row blocks are a
// contiguous range; set c takes its row blocks c, c + CS, ... Stage = (row block, one k-tile): [8 group records][8 scale
// records]; the producer fills them in groups of CS row blocks, k-tile-interleaved across the group's sets (no consumer
// waits more than CS stages past the producer). Per row block: the KT path's MMA chain and sum order (bit-identical).
// registers: one block per SM (shared memory); 65536 / (threads rounded to 4 warps) per thread
template <int NTILE, int CS, bool PAIR>
__global__ void __launch_bounds__((8 * CS + 1) * 32, 1)
w4h_xr_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ W4hSegs segs, int k, int stages) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages];
  __shared__ float red[CS][2][8][32][4];
  // reduction hand-off per (set, parity): rfull (8 slot warps arrive after storing), rempty (the reducer arrives after
  // reading) -- point-to-point, so the slot warps never wait for each other or for the reducer's sum
  __shared__ __align__(8) uint64_t rfull[CS][2], rempty[CS][2];
  const int ng = k >> 7;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nrb = segs.rb_end[segs.nseg - 1];
  const int lgs = 31 - __clz(stages);
  const int rb_lo = (int)((long)nrb * blockIdx.x / gridDim.x), rb_hi = (int)((long)nrb * (blockIdx.x + 1) / gridDim.x);
  const int nj = rb_hi - rb_lo;
  const uint32_t ring = smem_u32(sm), full0 = smem_u32(&full[0]), empty0 = smem_u32(&empty[0]);
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) mbar_init(full0 + 8 * i, 1), mbar_init(empty0 + 8 * i, 8);
    for (int cc = 0; cc < CS; ++cc)
      for (int pp = 0; pp < 2; ++pp) mbar_init(smem_u32(&rfull[cc][pp]), 8), mbar_init(smem_u32(&rempty[cc][pp]), 1);
    W4H_SM90(asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");)
  }
  __syncthreads();
  if (warp == 8 * CS) {  // producer: stage order (row-block group j0 of CS, k-tile t, set cc)
    if (lane == 0) {
      int seq = 0;
      for (int j0 = 0; j0 < nj; j0 += CS) {
        const int cnt = nj - j0 < CS ? nj - j0 : CS;
#pragma unroll 1
        for (int t = 0; t < NTILE; ++t)
#pragma unroll 1
          for (int cc = 0; cc < cnt; ++cc, ++seq) {
            const int rb = rb_lo + j0 + cc, sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
            const size_t g0 = (size_t)lrb * ng + 8 * t;
            const int slot = seq & (stages - 1);
            mbar_wait(empty0 + 8 * slot, ((seq >> lgs) & 1) ^ 1);
            const int gn = ng - 8 * t < 8 ? ng - 8 * t : 8;
            const uint32_t fb = full0 + 8 * slot, dst = ring + (uint32_t)slot * kStage;
            mbar_expect_tx(fb, gn * (1024 + 32));
            bulk_g2s(dst, reinterpret_cast<const unsigned char*>(segs.wt[sg]) + g0 * 1024, gn * 1024, fb);
            bulk_g2s(dst + kStageW, reinterpret_cast<const unsigned char*>(segs.st[sg]) + g0 * 32, gn * 32, fb);
          }
      }
    }
    return;
  }
  const int c = warp >> 3, i = warp & 7, g = lane >> 2, t4 = lane & 3;
  ak_pdl_entry();  // the producer above streams weights only (constant): it starts before the previous kernel ends
  // B fragments of this warp's groups, in registers: xb[t][q4] = (b0, b1 of MMA 2 q4, b0, b1 of MMA 2 q4 + 1)
  uint4 xb[NTILE][4];
  {
    uint4 raw[NTILE][4];
#pragma unroll
    for (int t = 0; t < NTILE; ++t) {
      const int G = 8 * t + i;
      const bool in = G < ng && g < m;
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(in ? g : 0) * k + (size_t)(in ? G : 0) * 128 + t4 * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) raw[t][q] = in ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int t = 0; t < NTILE; ++t) {
      const uint32_t xw[16] = {raw[t][0].x, raw[t][0].y, raw[t][0].z, raw[t][0].w, raw[t][1].x, raw[t][1].y,
                               raw[t][1].z, raw[t][1].w, raw[t][2].x, raw[t][2].y, raw[t][2].z, raw[t][2].w,
                               raw[t][3].x, raw[t][3].y, raw[t][3].z, raw[t][3].w};
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int qa = 2 * q4, qb = qa + 1;
        xb[t][q4] = make_uint4(x_pair(xw, qa >> 1, 2 * (qa & 1)), x_pair(xw, qa >> 1, 2 * (qa & 1) + 1),
                               x_pair(xw, qb >> 1, 2 * (qb & 1)), x_pair(xw, qb >> 1, 2 * (qb & 1) + 1));
      }
    }
  }
  const uint32_t wofs = i * 1024 + lane * 16, sofs = kStageW + i * 32 + g * 4;
  int pass = 0;
  for (int j0 = 0; j0 < nj; j0 += CS, ++pass) {
    const int cnt = nj - j0 < CS ? nj - j0 : CS;
    if (c >= cnt) break;
    int seq = j0 * NTILE + c;  // stage (j0 + c, t) = j0 * NTILE + t * cnt + c
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // PAIR: k-tiles two at a time -- both stages fetched, their two MMA chains interleaved (independent), then the
    // scaled adds in k-tile order; the next pair is fetched before. The same per-group chains and add order: same bits.
    constexpr int NB = PAIR ? 4 : 2;
    uint4 w0[NB], w1[NB];
    uint32_t sc[NB];
    auto fetch = [&](int b) {
      const int slot = seq & (stages - 1);
      mbar_wait(full0 + 8 * slot, (seq >> lgs) & 1);
      const uint32_t base = ring + (uint32_t)slot * kStage;
      w0[b] = lds128(base + wofs), w1[b] = lds128(base + 512 + wofs), sc[b] = lds32(base + sofs);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      seq += cnt;
    };
    auto chain = [&](float* cf, int t, int b) {
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int wd = q >> 1, j0q = 2 * (q & 1);
        const uint32_t u0 = wd == 0 ? w0[b].x : wd == 1 ? w0[b].y : wd == 2 ? w0[b].z : w0[b].w;
        const uint32_t u1 = wd == 0 ? w1[b].x : wd == 1 ? w1[b].y : wd == 2 ? w1[b].z : w1[b].w;
        const uint32_t b0 = (q & 1) ? xb[t][q >> 1].z : xb[t][q >> 1].x, b1 = (q & 1) ? xb[t][q >> 1].w : xb[t][q >> 1].y;
        mma16816(cf, deq_pair(u0, j0q), deq_pair(u1, j0q), deq_pair(u0, j0q + 1), deq_pair(u1, j0q + 1), b0, b1);
      }
    };
    auto scaled_add = [&](const float* cf, int b) {
      const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sc[b]);
      const float s0 = __low2float(s2), s1 = __high2float(s2);
      acc[0] = fmaf(cf[0], s0, acc[0]);
      acc[1] = fmaf(cf[1], s0, acc[1]);
      acc[2] = fmaf(cf[2], s1, acc[2]);
      acc[3] = fmaf(cf[3], s1, acc[3]);
    };
    if (PAIR) {
      fetch(0);
      if (NTILE > 1) fetch(1);
#pragma unroll
      for (int t = 0; t < NTILE; t += 2) {
        const int b = t & 3;
        if (t + 2 < NTILE) fetch((t + 2) & 3);
        if (t + 3 < NTILE) fetch((t + 3) & 3);
        const bool va = 8 * t + i < ng, vb = t + 1 < NTILE && 8 * (t + 1) + i < ng;
        float ca[4] = {0.f, 0.f, 0.f, 0.f}, cb[4] = {0.f, 0.f, 0.f, 0.f};
        if (vb) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {  // the two chains interleaved, MMA by MMA
            const int wd = q >> 1, j0q = 2 * (q & 1);
            const int ba = b, bb = (t + 1) & 3;
            const uint32_t ua0 = wd == 0 ? w0[ba].x : wd == 1 ? w0[ba].y : wd == 2 ? w0[ba].z : w0[ba].w;
            const uint32_t ua1 = wd == 0 ? w1[ba].x : wd == 1 ? w1[ba].y : wd == 2 ? w1[ba].z : w1[ba].w;
            const uint32_t ub0 = wd == 0 ? w0[bb].x : wd == 1 ? w0[bb].y : wd == 2 ? w0[bb].z : w0[bb].w;
            const uint32_t ub1 = wd == 0 ? w1[bb].x : wd == 1 ? w1[bb].y : wd == 2 ? w1[bb].z : w1[bb].w;
            const int tb = t + 1 < NTILE ? t + 1 : t;
            const uint32_t xa0 = (q & 1) ? xb[t][q >> 1].z : xb[t][q >> 1].x, xa1 = (q & 1) ? xb[t][q >> 1].w : xb[t][q >> 1].y;
            const uint32_t xb0 = (q & 1) ? xb[tb][q >> 1].z : xb[tb][q >> 1].x, xb1 = (q & 1) ? xb[tb][q >> 1].w : xb[tb][q >> 1].y;
            mma16816(ca, deq_pair(ua0, j0q), deq_pair(ua1, j0q), deq_pair(ua0, j0q + 1), deq_pair(ua1, j0q + 1), xa0, xa1);
            mma16816(cb, deq_pair(ub0, j0q), deq_pair(ub1, j0q), deq_pair(ub0, j0q + 1), deq_pair(ub1, j0q + 1), xb0, xb1);
          }
          scaled_add(ca, b);  // vb implies va (groups ascend with t)
          scaled_add(cb, (t + 1) & 3);
        } else if (va) {
          chain(ca, t, b);
          scaled_add(ca, b);
        }
      }
    } else {
      fetch(0);
#pragma unroll
      for (int t = 0; t < NTILE; ++t) {
        const int b = t & 1;
        if (t + 1 < NTILE) fetch(b ^ 1);
        if (8 * t + i < ng) {
          float cf[4] = {0.f, 0.f, 0.f, 0.f};
          chain(cf, t, b);
          scaled_add(cf, b);
        }
      }
    }
    // slot-order sum, handed to warp pass % 8 through red[c][p] (p = pass parity; use u = pass / 2 of that buffer)
    const int p = pass & 1, use = pass >> 1, rb = rb_lo + j0 + c;
    const uint32_t rf = smem_u32(&rfull[c][p]), re = smem_u32(&rempty[c][p]);
    mbar_wait(re, (use & 1) ^ 1);  // the reducer of pass - 2 has read the buffer
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) red[c][p][i][lane][jj] = acc[jj];
    __syncwarp();
    if (lane == 0) mbar_arrive(rf);
    if (i == (pass & 7)) {
      mbar_wait(rf, use & 1);
      const int sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
      __nv_bfloat16* __restrict__ y = segs.y[sg];
      const int ldy = segs.ldy[sg], q0 = lrb * 16 + g;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) v[jj] += red[c][p][w][lane][jj];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = 2 * t4 + cc;
        if (col < m) {
          y[(size_t)col * ldy + q0] = __float2bfloat16_rn(v[cc]);
          y[(size_t)col * ldy + q0 + 8] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(re);
    }
  }
}

// Batched kernel ("bt"), 8 < m <= 8 NTT tokens per launch: a block owns a contiguous range of row blocks and runs it in
// passes of up to RB row blocks; a ring stage is (pass, k-tile): the raw x tile of the TOK tokens (TOK rows of 2 KB,
// padded to kXRow bytes against bank conflicts) + the pass's row blocks' group / scale records of that k-tile. Warp i
// (slot i) converts its group's x once per stage into B fragments and reuses them for every row block of the pass
// (x traffic / weight traffic = 2 TOK / (8 RB) per stage). Per row block: the KT path's chain and sum order, so rows are
// bit-identical to the m <= 8 kernels (and to KT) for any m.
constexpr int kXRow = 2048 + 16;
template <int NTT, int RB>
struct BtCfg {
  static constexpr int TOK = 8 * NTT, XB = TOK * kXRow, WB = RB * 8192, SB = RB * 256, STAGE = XB + WB + SB;
};
template <int NTT, int RB>
__global__ void __launch_bounds__(9 * 32, 1)
w4h_bt_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ W4hSegs segs, int k, int stages) {
  using C = BtCfg<NTT, RB>;
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) uint64_t full[kMaxStages], empty[kMaxStages];
  __shared__ float red[8][32][4 * NTT];
  const int ng = k >> 7, ntile = (ng + 7) >> 3;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nrb = segs.rb_end[segs.nseg - 1];
  const int rb_lo = (int)((long)nrb * blockIdx.x / gridDim.x), rb_hi = (int)((long)nrb * (blockIdx.x + 1) / gridDim.x);
  const uint32_t ring = smem_u32(sm), full0 = smem_u32(&full[0]), empty0 = smem_u32(&empty[0]);
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) mbar_init(full0 + 8 * s, 1), mbar_init(empty0 + 8 * s, 8);
    W4H_SM90(asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");)
  }
  __syncthreads();
  ak_pdl_entry();  // the producer copies x too
  if (warp == 8) {
    if (lane == 0) {
      int slot = 0, ph = 0;
      for (int r0 = rb_lo; r0 < rb_hi; r0 += RB) {
        const int cnt = rb_hi - r0 < RB ? rb_hi - r0 : RB;
        for (int t = 0; t < ntile; ++t) {
          mbar_wait(empty0 + 8 * slot, ph ^ 1);
          const int gn = ng - 8 * t < 8 ? ng - 8 * t : 8, kin = gn * 128;  // inputs of this k-tile
          const int tok = m < C::TOK ? m : C::TOK;
          const uint32_t fb = full0 + 8 * slot, dst = ring + (uint32_t)slot * C::STAGE;
          mbar_expect_tx(fb, tok * kin * 2 + cnt * gn * (1024 + 32));
          for (int r = 0; r < tok; ++r) bulk_g2s(dst + r * kXRow, x + (size_t)r * k + (size_t)t * 1024, kin * 2, fb);
          for (int r = 0; r < cnt; ++r) {
            const int rb = r0 + r, sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
            const size_t g0 = (size_t)lrb * ng + 8 * t;
            bulk_g2s(dst + C::XB + r * 8192, reinterpret_cast<const unsigned char*>(segs.wt[sg]) + g0 * 1024, gn * 1024, fb);
            bulk_g2s(dst + C::XB + C::WB + r * 256, reinterpret_cast<const unsigned char*>(segs.st[sg]) + g0 * 32, gn * 32, fb);
          }
          if (++slot == stages) slot = 0, ph ^= 1;
        }
      }
    }
    return;
  }
  const int i = warp, g = lane >> 2, t4 = lane & 3;
  const uint32_t wofs = i * 1024 + lane * 16, sofs = i * 32 + g * 4;
  int slot = 0, ph = 0;
  for (int r0 = rb_lo; r0 < rb_hi; r0 += RB) {
    const int cnt = rb_hi - r0 < RB ? rb_hi - r0 : RB;
    float acc[RB][NTT][4];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt) acc[r][tt][0] = acc[r][tt][1] = acc[r][tt][2] = acc[r][tt][3] = 0.f;
    for (int t = 0; t < ntile; ++t) {
      mbar_wait(full0 + 8 * slot, ph);
      const uint32_t base = ring + (uint32_t)slot * C::STAGE;
      const bool on = 8 * t + i < ng;
      // B fragments of group 8 t + i for the NTT token tiles (tokens >= m: stale rows, never stored)
      uint4 xb[NTT][4];
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt) {
        const uint32_t xa = base + (uint32_t)(tt * 8 + g) * kXRow + i * 256 + t4 * 64;
        const uint4 v0 = lds128(xa), v1 = lds128(xa + 16), v2 = lds128(xa + 32), v3 = lds128(xa + 48);
        const uint32_t xw[16] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w};
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const int qa = 2 * q4, qb = qa + 1;
          xb[tt][q4] = make_uint4(x_pair(xw, qa >> 1, 2 * (qa & 1)), x_pair(xw, qa >> 1, 2 * (qa & 1) + 1),
                                  x_pair(xw, qb >> 1, 2 * (qb & 1)), x_pair(xw, qb >> 1, 2 * (qb & 1) + 1));
        }
      }
      uint4 w0[RB], w1[RB];
      uint32_t sc[RB];
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        w0[r] = lds128(base + C::XB + r * 8192 + wofs), w1[r] = lds128(base + C::XB + r * 8192 + 512 + wofs);
        sc[r] = lds32(base + C::XB + C::WB + r * 256 + sofs);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      if (++slot == stages) slot = 0, ph ^= 1;
      if (!on) continue;
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        if (r >= cnt) break;
        float cf[NTT][4];
#pragma unroll
        for (int tt = 0; tt < NTT; ++tt) cf[tt][0] = cf[tt][1] = cf[tt][2] = cf[tt][3] = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1);
          const uint32_t u0 = wd == 0 ? w0[r].x : wd == 1 ? w0[r].y : wd == 2 ? w0[r].z : w0[r].w;
          const uint32_t u1 = wd == 0 ? w1[r].x : wd == 1 ? w1[r].y : wd == 2 ? w1[r].z : w1[r].w;
          const uint32_t a0 = deq_pair(u0, j0), a1 = deq_pair(u1, j0), a2 = deq_pair(u0, j0 + 1), a3 = deq_pair(u1, j0 + 1);
#pragma unroll
          for (int tt = 0; tt < NTT; ++tt)
            mma16816(cf[tt], a0, a1, a2, a3, (q & 1) ? xb[tt][q >> 1].z : xb[tt][q >> 1].x,
                     (q & 1) ? xb[tt][q >> 1].w : xb[tt][q >> 1].y);
        }
        const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sc[r]);
        const float s0 = __low2float(s2), s1 = __high2float(s2);
#pragma unroll
        for (int tt = 0; tt < NTT; ++tt) {
          acc[r][tt][0] = fmaf(cf[tt][0], s0, acc[r][tt][0]);
          acc[r][tt][1] = fmaf(cf[tt][1], s0, acc[r][tt][1]);
          acc[r][tt][2] = fmaf(cf[tt][2], s1, acc[r][tt][2]);
          acc[r][tt][3] = fmaf(cf[tt][3], s1, acc[r][tt][3]);
        }
      }
    }
    // slot-order sums, one row block at a time (reducer warp r % 8)
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      if (r >= cnt) break;
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) red[i][lane][4 * tt + jj] = acc[r][tt][jj];
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (i == (r & 7)) {
        const int rb = r0 + r, sg = seg_of(segs, rb), lrb = rb - (sg ? segs.rb_end[sg - 1] : 0);
        __nv_bfloat16* __restrict__ y = segs.y[sg];
        const int ldy = segs.ldy[sg], q0 = lrb * 16 + g;
#pragma unroll
        for (int tt = 0; tt < NTT; ++tt) {
          float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int w = 0; w < 8; ++w)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) v[jj] += red[w][lane][4 * tt + jj];
#pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            const int col = tt * 8 + 2 * t4 + cc;
            if (col < m) {
              y[(size_t)col * ldy + q0] = __float2bfloat16_rn(v[cc]);
              y[(size_t)col * ldy + q0 + 8] = __float2bfloat16_rn(v[2 + cc]);
            }
          }
        }
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
    }
  }
}
template <int NTT, int RB>
int launch_bt(const void* x, int m, const W4hSegs& s, int k, int max_stages, cudaStream_t st) {
  using C = BtCfg<NTT, RB>;
  auto kern = w4h_bt_kernel<NTT, RB>;
  static int static_smem = -1;
  if (static_smem < 0) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return (int)cudaErrorInvalidValue;
    static_smem = (int)fa.sharedSizeBytes;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_optin() - static_smem) != cudaSuccess)
      return (int)cudaErrorInvalidValue;
  }
  const int stages = std::min(std::min(max_stages, kMaxStages), (smem_optin() - static_smem) / C::STAGE);
  if (stages < 2) return -1;
  const int nrb = s.rb_end[s.nseg - 1];
  const int grid = std::min(nrb, sms());
  kern<<<grid, 9 * 32, stages * C::STAGE, st>>>(static_cast<const __nv_bfloat16*>(x), m, s, k, stages);
  return (int)cudaGetLastError();
}

// CT packed [n][k/8] int32 + bf16 scales [n][k/128] -> W4H
__global__ void w4h_repack_kernel(const uint4* __restrict__ src, const uint16_t* __restrict__ ssrc, uint4* __restrict__ dst,
                                  uint32_t* __restrict__ sdst, int n, int k) {
  const int ng = k >> 7, nrb = n >> 4;
  const size_t total = (size_t)nrb * ng * 64;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    const int lane = (int)(idx & 31), h = (int)((idx >> 5) & 1);
    const size_t rg = idx >> 6;
    const int G = (int)(rg % ng), rb = (int)(rg / ng);
    const int row = rb * 16 + (lane >> 2) + 8 * h;
    dst[idx] = src[(size_t)row * (k >> 5) + G * 4 + (lane & 3)];
    if (h == 0 && lane < 8) {
      const int r0 = rb * 16 + lane;
      sdst[rg * 8 + lane] = (uint32_t)ssrc[(size_t)r0 * ng + G] | ((uint32_t)ssrc[(size_t)(r0 + 8) * ng + G] << 16);
    }
  }
}
// W4H rows [r0, r0 + nr) -> dense bf16 w [nr][k], value bf16(q * scale) (dequant_kt_kernel's rounding)
__global__ void w4h_dequant_kernel(const uint4* __restrict__ wt, const uint32_t* __restrict__ st, __nv_bfloat16* __restrict__ w,
                                   int k, int r0, int nr) {
  const int ng = k >> 7;
  const size_t total = (size_t)nr * ng * 4;  // (row, group, quarter t4): 32 inputs
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    const int t4 = (int)(idx & 3);
    const size_t rg = idx >> 2;
    const int G = (int)(rg % ng), o = (int)(rg / ng), row = r0 + o;
    const int rb = row >> 4, rr = row & 15, gg = rr & 7, h = rr >> 3;
    const size_t rec = (size_t)rb * ng + G;
    const uint4 q = wt[rec * 64 + h * 32 + gg * 4 + t4];
    const uint32_t s2 = st[rec * 8 + gg];
    const float s = __uint_as_float(h ? (s2 & 0xFFFF0000u) : (s2 << 16));
    const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
    uint4* out = reinterpret_cast<uint4*>(w + (size_t)o * k + (size_t)G * 128 + t4 * 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t v[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float lo = (float)((int)((qq[j] >> (8 * p)) & 15u) - 8), hi = (float)((int)((qq[j] >> (8 * p + 4)) & 15u) - 8);
        v[p] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(lo * s)) |
               ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hi * s)) << 16);
      }
      out[j] = make_uint4(v[0], v[1], v[2], v[3]);
    }
  }
}

int sms() {
  static int v = 0;
  if (!v) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
  }
  return v;
}
int smem_optin() {
  static int v = 0;
  if (!v) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}
// Launch configuration (one table entry per GPU class; a future autotune can fill it per device). Read once;
// AGENS_W4H_CS / AGENS_W4H_STAGES / AGENS_W4H_XR / AGENS_W4H_BT_RB override (bench / tuning).
struct W4hCfg {
  int cs;          // consumer sets of 8 warps of the generic (x in shared memory) kernel
  int max_stages;  // ring depth cap (shared memory permitting)
  int xr;          // m <= 8: x fragments in registers (w4h_xr_kernel) for k-tile counts with an instance
  int pair;        // w4h_xr_kernel: k-tiles two at a time, chains interleaved (AGENS_W4H_XR_PAIR=0: one at a time)
};
const W4hCfg& w4h_cfg() {
  static const W4hCfg c = [] {
    W4hCfg v{2, kMaxStages, 1, 1};
    if (const char* e = getenv("AGENS_W4H_XR_PAIR")) v.pair = atoi(e);
    if (const char* e = getenv("AGENS_W4H_XR")) v.xr = atoi(e);
    if (const char* e = getenv("AGENS_W4H_CS")) v.cs = atoi(e);
    if (const char* e = getenv("AGENS_W4H_STAGES")) v.max_stages = std::min(kMaxStages, atoi(e));
    return v;
  }();
  return c;
}
template <int NT, int CS>
int launch_cs(const void* x, int m, const W4hSegs& s, int k, int max_stages, cudaStream_t st) {
  auto kern = w4h_kernel<NT, CS>;
  static int static_smem = -1;
  if (static_smem < 0) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return (int)cudaErrorInvalidValue;
    static_smem = (int)fa.sharedSizeBytes;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_optin() - static_smem) != cudaSuccess)
      return (int)cudaErrorInvalidValue;
  }
  const int ng = k >> 7, xbytes = CS * NT * 8192 + ng * NT * 2048;
  const int avail = smem_optin() - static_smem - xbytes;
  int stages = std::min(max_stages, avail / kStageB);
  while (stages & (stages - 1)) stages &= stages - 1;  // power of two
  if (stages < 2 * CS) return -1;  // x does not fit next to the ring: caller splits the tokens
  const int nrb = s.rb_end[s.nseg - 1];
  const int grid = std::min(nrb, sms());
  kern<<<grid, (8 * CS + 1) * 32, xbytes + stages * kStageB, st>>>(static_cast<const __nv_bfloat16*>(x), m, s, k, stages);
  return (int)cudaGetLastError();
}
template <int NTILE, int CS, bool PAIR>
int launch_xr(const void* x, int m, const W4hSegs& s, int k, int max_stages, cudaStream_t st) {
  auto kern = w4h_xr_kernel<NTILE, CS, PAIR>;
  static int static_smem = -1;
  if (static_smem < 0) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return (int)cudaErrorInvalidValue;
    static_smem = (int)fa.sharedSizeBytes;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_optin() - static_smem) != cudaSuccess)
      return (int)cudaErrorInvalidValue;
  }
  int stages = std::min(max_stages, (smem_optin() - static_smem) / kStage);
  while (stages & (stages - 1)) stages &= stages - 1;
  const int nrb = s.rb_end[s.nseg - 1];
  const int grid = std::min(nrb, sms());
  kern<<<grid, (8 * CS + 1) * 32, stages * kStage, st>>>(static_cast<const __nv_bfloat16*>(x), m, s, k, stages);
  const int e = (int)cudaGetLastError();
  if (e && getenv("AGENS_W4H_DEBUG")) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    fprintf(stderr, "w4h_xr<%d,%d>: %s (grid %d, threads %d, dyn smem %d, static %zu, regs %d, max threads %d)\n", NTILE,
            CS, cudaGetErrorString((cudaError_t)e), grid, (8 * CS + 1) * 32, stages * kStage, fa.sharedSizeBytes,
            fa.numRegs, fa.maxThreadsPerBlock);
  }
  return e;
}
template <int NT>
int launch(const void* x, int m, const W4hSegs& s, int k, cudaStream_t st) {
  const W4hCfg& c = w4h_cfg();
  if (NT == 1 && c.xr) {
    switch ((k / 128 + 7) / 8) {  // k-tiles
      case 3: return c.pair ? launch_xr<3, 1, true>(x, m, s, k, c.max_stages, st) : launch_xr<3, 1, false>(x, m, s, k, c.max_stages, st);
      case 5: return c.pair ? launch_xr<5, 1, true>(x, m, s, k, c.max_stages, st) : launch_xr<5, 1, false>(x, m, s, k, c.max_stages, st);
      case 6: return c.pair ? launch_xr<6, 1, true>(x, m, s, k, c.max_stages, st) : launch_xr<6, 1, false>(x, m, s, k, c.max_stages, st);
      case 9: return c.pair ? launch_xr<9, 1, true>(x, m, s, k, c.max_stages, st) : launch_xr<9, 1, false>(x, m, s, k, c.max_stages, st);
      default: break;
    }
  }
  // fewer consumer sets when x + the reduction buffers leave too small a ring
  int e = -1;
  if (c.cs >= 3) e = launch_cs<NT, 3>(x, m, s, k, c.max_stages, st);
  if (e == -1 && c.cs >= 2) e = launch_cs<NT, 2>(x, m, s, k, c.max_stages, st);
  if (e == -1) e = launch_cs<NT, 1>(x, m, s, k, c.max_stages, st);
  return e;
}

}  // namespace

extern "C" size_t ak_w4h_weight_bytes(int n, int k) { return (size_t)n * k / 2; }
extern "C" size_t ak_w4h_scale_bytes(int n, int k) { return (size_t)n * k / 64; }
// n % 16, k % 128; k <= 9216 (9 k-tiles: the m <= 8 kernels keep x of the whole k on chip)
extern "C" int ak_w4h_ok(int n, int k) { return n > 0 && n % 16 == 0 && k > 0 && k % 128 == 0 && k <= 9216; }
extern "C" int ak_w4h_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, void* stream) {
  if (!ak_w4h_ok(n, k)) return (int)cudaErrorInvalidValue;
  w4h_repack_kernel<<<1024, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint4*>(packed), static_cast<const uint16_t*>(scales), static_cast<uint4*>(wt),
      static_cast<uint32_t*>(st), n, k);
  return (int)cudaGetLastError();
}
extern "C" int ak_w4h_dequant_rows(const void* wt, const void* st, void* w, int n, int k, int r0, int nr, void* stream) {
  if (!ak_w4h_ok(n, k) || r0 < 0 || nr < 1 || r0 + nr > n) return (int)cudaErrorInvalidValue;
  w4h_dequant_kernel<<<sms() * 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(wt), static_cast<const uint32_t*>(st), static_cast<__nv_bfloat16*>(w), k, r0, nr);
  return (int)cudaGetLastError();
}
// y[i] (token j at y[i] + j * ldy[i]) = x [m][k] times linear i, nseg <= 4 linears sharing x (n[i] % 16 == 0)
extern "C" int ak_w4h_gemm(const void* x, int m, int nseg, const void* const* wt, const void* const* st, void* const* y,
                           const int* n, const int* ldy, int k, void* stream) {
  if (nseg < 1 || nseg > kSegs || m < 1 || m > 128 || k % 128) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  W4hSegs g{};
  g.nseg = nseg;
  for (int i = 0, e = 0; i < nseg; ++i) {
    if (!ak_w4h_ok(n[i], k)) return (int)cudaErrorInvalidValue;
    g.wt[i] = static_cast<const uint4*>(wt[i]), g.st[i] = static_cast<const uint32_t*>(st[i]);
    g.y[i] = static_cast<__nv_bfloat16*>(y[i]), g.ldy[i] = ldy ? ldy[i] : n[i];
    e += n[i] / 16, g.rb_end[i] = e;
  }
  static const int bt = getenv("AGENS_W4H_BT") ? atoi(getenv("AGENS_W4H_BT")) : 1;
  if (m > 8 && bt) {  // batched: chunks of up to 32 tokens, one weight pass each
    for (int m0 = 0; m0 < m; m0 += 32) {
      W4hSegs gg = g;
      for (int i = 0; i < nseg; ++i) gg.y[i] = g.y[i] + (size_t)m0 * g.ldy[i];
      const int mm = std::min(32, m - m0);
      const __nv_bfloat16* xx = static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k;
      int e;
      if (mm <= 8) e = launch<1>(xx, mm, gg, k, s);
      else if (mm <= 16) e = launch_bt<2, 4>(xx, mm, gg, k, kMaxStages, s);
      else if (mm <= 24) e = launch_bt<3, 4>(xx, mm, gg, k, kMaxStages, s);
      else e = launch_bt<4, 4>(xx, mm, gg, k, kMaxStages, s);
      if (e) return e < 0 ? (int)cudaErrorInvalidValue : e;
    }
    return 0;
  }
  // token chunks of 16 (8 if x of 16 tokens does not fit next to a 2-stage ring); each chunk one weight pass
  for (int m0 = 0; m0 < m; m0 += 16) {
    W4hSegs gg = g;
    for (int i = 0; i < nseg; ++i) gg.y[i] = g.y[i] + (size_t)m0 * g.ldy[i];
    const int mm = std::min(16, m - m0);
    const __nv_bfloat16* xx = static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k;
    int e = mm <= 8 ? launch<1>(xx, mm, gg, k, s) : launch<2>(xx, mm, gg, k, s);
    if (e == -1 && mm > 8) {
      e = launch<1>(xx, 8, gg, k, s);
      if (!e) {
        for (int i = 0; i < nseg; ++i) gg.y[i] += (size_t)8 * g.ldy[i];
        e = launch<1>(xx + (size_t)8 * k, mm - 8, gg, k, s);
      }
    }
    if (e) return e < 0 ? (int)cudaErrorInvalidValue : e;
  }
  return 0;
}

// ---- engine hooks (w4_gemv.cu): which W4 g128 linears are repacked as W4H, and the registry of W4H weights.
// AGENS_W4H: 0 = off (default), 1 = every eligible W4 g128 linear (ak_w4h_ok) on a device with the shared memory these
// kernels need (>= 200 KB per block opt-in: sm90 / sm100), 2 = "auto": only the shape classes where W4H measured faster
// than KT at m = 1 / 8 on compute capability 9.0 (n >= 12288: MLP gate_up; n <= 1024: attention k / v, KDA f_down), and
// (compute capability 9.0) k 4096 .. 6144: the hidden-state projections that share x, so the verify runs them as one
// multi-segment W4H launch
// (KDA qkv | z | f_down, attention q | k | v: one launch instead of three; linear_shared). ak_w4h_set_mode
// overrides (tests).
namespace {
int g_w4h_mode = -1;
std::unordered_set<const void*> g_w4h_set;
std::mutex g_w4h_mu;
bool w4h_device_ok() {
  static const bool ok = smem_optin() >= 200 * 1024;
  return ok;
}
int w4h_mode() {
  if (g_w4h_mode < 0) {
    const char* e = getenv("AGENS_W4H");
    g_w4h_mode = !e ? 0 : e[0] == '1' ? 1 : (e[0] == '2' || !strcmp(e, "auto")) ? 2 : 0;
  }
  return g_w4h_mode;
}

// Dual layout (AGENS_W4H_MMAX = N > 0, with AGENS_W4H on; 1.3.4): an eligible linear keeps its KT layout in the engine's
// buffers and gets a W4H copy of its own; launches of m <= N tokens take the copy, larger m the KT GEMM (W4H's batched
// kernel streams the weights once per 32 tokens: on sm90 tp2 at m = 96 the KT path's step was 42 vs 50 ms). Same bits
// either way (W4H == KT); the copy costs the W4H bytes again (memory: ak_w4h_dual_bytes).
struct W4hDual {
  void *wt, *st;
  int n, k;
};
std::unordered_map<const void*, W4hDual> g_w4h_dual;
size_t g_w4h_dual_bytes = 0;
int w4h_mmax() {
  static const int v = [] { const char* e = getenv("AGENS_W4H_MMAX"); return e ? atoi(e) : 0; }();
  return v;
}
void w4h_dual_drop(const void* wt) {  // under g_w4h_mu
  auto it = g_w4h_dual.find(wt);
  if (it == g_w4h_dual.end()) return;
  g_w4h_set.erase(it->second.wt);
  cudaFree(it->second.wt), cudaFree(it->second.st);
  g_w4h_dual_bytes -= ak_w4h_weight_bytes(it->second.n, it->second.k) + ak_w4h_scale_bytes(it->second.n, it->second.k);
  g_w4h_dual.erase(it);
}
}  // namespace
// 1 = not handled (W4H off, shape or device unsupported): the caller repacks KT; else a CUDA error code (0 = ok)
// (dual layout: the W4H copy is made here and 1 returned, so the caller writes KT into wt / st as without W4H)
int agens_w4h_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, cudaStream_t s) {
  {
    std::lock_guard<std::mutex> l(g_w4h_mu);
    w4h_dual_drop(wt);  // a weight at an address an earlier one (with a copy) had
  }
  const int mode = w4h_mode();
  if (!mode || !ak_w4h_ok(n, k) || !w4h_device_ok()) return 1;
  if (mode == 2 && !(n >= 12288 || n <= 1024 || (k >= 4096 && k <= 6144 && ak_cc90()))) return 1;
  if (w4h_mmax() > 0) {
    void *w2 = nullptr, *s2 = nullptr;
    const size_t wb = ak_w4h_weight_bytes(n, k), sb = ak_w4h_scale_bytes(n, k);
    if (cudaMalloc(&w2, wb) != cudaSuccess || cudaMalloc(&s2, sb) != cudaSuccess) {
      cudaFree(w2);
      return (int)cudaErrorMemoryAllocation;
    }
    const int e = ak_w4h_repack(packed, scales, w2, s2, n, k, s);
    if (e) {
      cudaFree(w2), cudaFree(s2);
      return e;
    }
    std::lock_guard<std::mutex> l(g_w4h_mu);
    g_w4h_set.insert(w2);
    g_w4h_dual[wt] = W4hDual{w2, s2, n, k};
    g_w4h_dual_bytes += wb + sb;
    return 1;
  }
  const int e = ak_w4h_repack(packed, scales, wt, st, n, k, s);
  if (e == 0) {
    std::lock_guard<std::mutex> l(g_w4h_mu);
    g_w4h_set.insert(wt);
  }
  return e;
}
bool agens_w4h_is(const void* wt) {
  std::lock_guard<std::mutex> l(g_w4h_mu);
  return !g_w4h_set.empty() && g_w4h_set.count(wt);
}
// dual layout: the W4H copy of KT weight wt for an m-token launch (m <= AGENS_W4H_MMAX), else false
bool agens_w4h_dual(const void* wt, int m, const void** w2, const void** s2) {
  if (m > w4h_mmax()) return false;
  std::lock_guard<std::mutex> l(g_w4h_mu);
  auto it = g_w4h_dual.find(wt);
  if (it == g_w4h_dual.end()) return false;
  *w2 = it->second.wt, *s2 = it->second.st;
  return true;
}
extern "C" size_t ak_w4h_dual_bytes() { return g_w4h_dual_bytes; }
// the W4H layout to use for an m-token launch of weight wt: wt itself (W4H) or its dual-layout copy; 0 = none (KT)
extern "C" int ak_w4h_ptrs(const void* wt, const void* st, int m, const void** w, const void** s) {
  if (agens_w4h_is(wt)) return *w = wt, *s = st, 1;
  return agens_w4h_dual(wt, m, w, s) ? 1 : 0;
}
extern "C" int ak_w4h_is_m(const void* wt, int m) {
  const void *a, *b;
  return agens_w4h_is(wt) || agens_w4h_dual(wt, m, &a, &b);
}
void agens_w4h_forget(const void* wt) {
  std::lock_guard<std::mutex> l(g_w4h_mu);
  g_w4h_set.erase(wt);
}
extern "C" int ak_w4h_set_mode(int mode) {
  const int prev = w4h_mode();
  g_w4h_mode = mode < 0 ? -1 : mode;
  return prev;
}
extern "C" int ak_w4h_is(const void* wt) { return agens_w4h_is(wt); }
extern "C" int ak_w4h_forget(const void* wt) { return agens_w4h_forget(wt), 0; }
