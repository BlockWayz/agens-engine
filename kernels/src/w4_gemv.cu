// W4A16 GEMV for the compressed-tensors pack-quantized format (symmetric, group 128).
//
// Weight row o is k/8 int32 words, i.e. k/32 16-byte chunks; chunk c covers inputs [32c, 32c+32) and lies in
// group c/4. One warp owns R rows; lane l walks chunks l, l+32, ... and streams R chunks (one per row) per step,
// so each x value it reads from shared memory is reused R times. x is staged in shared memory as bf16 pairs in a
// chunk-transposed layout (pair p of chunk c at [p * chunks + c]) so the 32 lanes of a warp read 32 consecutive
// words: no bank conflicts.
//
// Dequant: nibble n (0..15) -> float via the exponent trick: as_float(0x4B000000 | n) = 2^23 + n exactly, minus
// (2^23 + 8) gives n - 8 exactly. Per chunk: fp32 dot over 32 inputs, times the group scale, added to the row
// accumulator; warp butterfly reduction. Fixed order everywhere -> deterministic run to run.
//
// Two weight layouts:
//  * CT  — the on-disk compressed-tensors layout [n][k/8] (row-major). Used directly by ak_w4g128_gemv_bf16.
//  * KT  — "k-tiled", built once at load time by ak_w4g128_repack_kt: k is padded to a multiple of 1024 and split
//          into tiles of 1024 inputs; tile t of row o is one contiguous 512-byte record at [(t * n + o) * 512], its 8
//          group scales at [(t * n + o) * 8]. In one k-step all warps of all blocks read one contiguous slab instead
//          of n scattered row streams — DRAM row-buffer friendly (measured: contiguous segments read ~880 GB/s on a
//          sm89 vs ~860 for strided streams). Same per-lane chunk order as CT, so results are bit-identical.
//          The KT path takes the group size as a parameter (64 or 128: 16 or 8 scales per tile record; the 3-bit probe
//          writes g64 into the same int4 container). Entry points (ak_w4_*): m = 1 GEMV, m <= 16 batched GEMV (rows
//          bit-identical to m = 1), and dequant-to-bf16 for the prefill path (scratch + cuBLAS).
//  * W3 KT — the same tiling with 3-bit values (q in [-4, 3], e.g. the 3-bit probe's int4 container re-packed at
//          load, ak_w3_repack_kt): a tile record is 384 bytes, 32 lanes x 8 bytes of 2-bit planes (inputs 0..15 of the
//          lane's chunk in word 0, 16..31 in word 1) then 32 lanes x 4 bytes of the third bit; scales as W4 KT. The
//          load expands a lane's 12 bytes back into the 4 nibble words of the W4 chunk (nibble = q + 8), after which
//          every kernel is the W4 one: W3 results are bit-identical to W4 KT on the same values (ak_w3_*).
//  * W2 KT — the same with 2-bit values (q in [-2, 1]): a tile record is 256 bytes, 32 lanes x 8 bytes of 2-bit fields
//          (n = q + 2; inputs 0..15 of the lane's chunk in word 0, 16..31 in word 1), expanded on load to the W4 nibble
//          words (nibble = n + 6). Mixed precision (GSQ-RCO: per linear W2/W3/W4 x g64/g128) uses the ak_wq_* entry
//          points with the linear's bit width; every (bits, group) is bit-identical to W4 KT of that group.

#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <unordered_set>

// KF layout (end of file): repack (1 = not handled: KF off or shape unsupported), membership, dequant rows
int agens_kf_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, cudaStream_t s);
bool agens_kf_is(const void* wt);
void agens_kf_forget(const void* wt);
int agens_kf_dequant_rows(const void* wt, const void* st, void* w, int n, int k, int r0, int nr, cudaStream_t s);
// W4H layout (w4h.cu, sm90 / sm100; AGENS_W4H=1): repack (1 = not handled), membership, forget
int agens_w4h_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, cudaStream_t s);
bool agens_w4h_is(const void* wt);
void agens_w4h_forget(const void* wt);
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr int kWarps = 8;  // 256 threads per block

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  // Weights are read exactly once per step: bypass L1.
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ld_stream2(const uint2* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_stream1(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
// 8 2-bit fields (16 bits) -> the low 2 bits of 8 nibbles; 8 bits -> bit 0 of 8 nibbles
__device__ __forceinline__ uint32_t spread2(uint32_t x) {
  x &= 0xFFFFu;
  x = (x | (x << 8)) & 0x00FF00FFu;
  x = (x | (x << 4)) & 0x0F0F0F0Fu;
  return (x | (x << 2)) & 0x33333333u;
}
__device__ __forceinline__ uint32_t spread1(uint32_t y) {
  y &= 0xFFu;
  y = (y | (y << 12)) & 0x000F000Fu;
  y = (y | (y << 6)) & 0x03030303u;
  return (y | (y << 3)) & 0x11111111u;
}
// W3 values n = q + 4 in 0..7 -> W4 nibbles q + 8 = n + 4 (no carry: n + 4 <= 11)
__device__ __forceinline__ uint32_t w3_word(uint32_t lo16, uint32_t hi8) {
  return (spread2(lo16) | (spread1(hi8) << 2)) + 0x44444444u;
}
// W2 values n = q + 2 in 0..3 -> W4 nibbles q + 8 = n + 6 (no carry: n + 6 <= 9)
__device__ __forceinline__ uint32_t w2_word(uint32_t lo16) { return spread2(lo16) + 0x66666666u; }
// 32-bit words per KT tile record (1024 inputs) at BITS bits per weight
template <int BITS>
constexpr int rec_words() { return 32 * BITS; }
// lane's 16-byte chunk of tile record rec: W4 KT directly, W3 / W2 KT expanded to the same nibble words
template <int BITS>
__device__ __forceinline__ uint4 ldw(const uint4* __restrict__ wt, size_t rec, int lane) {
  static_assert(BITS == 2 || BITS == 3 || BITS == 4, "bits");
  if (BITS == 4) return ld_stream(wt + rec * 32 + lane);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(wt) + rec * rec_words<BITS>();
  const uint2 lo = ld_stream2(reinterpret_cast<const uint2*>(b) + lane);
  if (BITS == 2) return make_uint4(w2_word(lo.x), w2_word(lo.x >> 16), w2_word(lo.y), w2_word(lo.y >> 16));
  const uint32_t hi = ld_stream1(b + 64 + lane);
  return make_uint4(w3_word(lo.x, hi), w3_word(lo.x >> 16, hi >> 8), w3_word(lo.y, hi >> 16), w3_word(lo.y >> 16, hi >> 24));
}

__device__ __forceinline__ float nib(uint32_t w, int i) {
  return __int_as_float(0x4B000000u | ((w >> (4 * i)) & 0xFu)) - 8388616.0f;
}

// Dot of 8 nibbles of one word with 8 x values held as 4 bf16 pairs.
__device__ __forceinline__ float dot8(uint32_t w, const uint32_t* xp) {
  float s = 0.f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float xl = __uint_as_float(xp[p] << 16);
    const float xh = __uint_as_float(xp[p] & 0xFFFF0000u);
    s = fmaf(xl, nib(w, 2 * p), s);
    s = fmaf(xh, nib(w, 2 * p + 1), s);
  }
  return s;
}

template <int R, int U>
__global__ void __launch_bounds__(kWarps * 32)
w4g128_gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint4* __restrict__ packed,
                   const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ y, int n, int k) {
  extern __shared__ uint32_t xs[];  // k/2 words, chunk-transposed
  const int chunks = k >> 5;
  const int groups = k >> 7;
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(x);
  for (int i = threadIdx.x; i < (k >> 1); i += blockDim.x) {
    const int c = i >> 4, p = i & 15;  // word i = pair p of chunk c
    xs[p * chunks + c] = __ldg(xw + i);
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int warps_total = gridDim.x * kWarps;

  for (int row0 = warp_global * R; row0 < n; row0 += warps_total * R) {
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    for (int c0 = lane; c0 < chunks; c0 += 32 * U) {
      // Issue every load of this step (weights and their group scales) before any math.
      uint4 w[U][R];
      __nv_bfloat16 sc[U][R];
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int c = c0 + 32 * u;
          const bool ok = c < chunks && row0 + r < n;
          w[u][r] = ok ? ld_stream(packed + (size_t)(row0 + r) * chunks + c)
                       : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
          sc[u][r] = ok ? scales[(size_t)(row0 + r) * groups + (c >> 2)] : __float2bfloat16_rn(0.f);
        }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int c = c0 + 32 * u;
        if (c >= chunks) break;
        uint32_t xp[16];
#pragma unroll
        for (int p = 0; p < 16; ++p) xp[p] = xs[p * chunks + c];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (row0 + r >= n) break;
          const float s = __bfloat162float(sc[u][r]);
          const float d = dot8(w[u][r].x, xp) + dot8(w[u][r].y, xp + 4) + dot8(w[u][r].z, xp + 8) +
                          dot8(w[u][r].w, xp + 12);
          acc[r] = fmaf(d, s, acc[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], o);
      if (lane == 0 && row0 + r < n) y[row0 + r] = __float2bfloat16_rn(acc[r]);
    }
  }
}

template <int R, int U>
int launch(const void* x, const int32_t* packed, const void* scales, void* y, int n, int k, cudaStream_t st) {
  static int sms = 0;
  if (sms == 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }
  const size_t smem = (size_t)k * 2;
  auto kern = w4g128_gemv_kernel<R, U>;
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
  }
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kWarps * 32, smem);
  if (per_sm < 1) per_sm = 1;
  const int row_groups = (n + R - 1) / R;
  const int want = (row_groups + kWarps - 1) / kWarps;
  const int blocks = want < sms * per_sm ? want : sms * per_sm;
  kern<<<blocks, kWarps * 32, smem, st>>>(static_cast<const __nv_bfloat16*>(x), reinterpret_cast<const uint4*>(packed),
                                           static_cast<const __nv_bfloat16*>(scales),
                                           static_cast<__nv_bfloat16*>(y), n, k);
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------- KT layout
// Group size G (64 or 128) is a template parameter: a tile record carries 1024/G scales and lane l's chunk lies in
// group slot l / (G/32). For G = 128 every expression is the one of the original g128 kernel (bit-identical).
template <int G>
__device__ __forceinline__ int kt_scale_slot(int lane) {
  return lane / (G / 32);
}

// BW ("bf16 weights", Marlin numerics): each weight is first rounded to bf16(q * scale) -- the value Marlin's bf16
// dequant feeds its MMA and ak_w4_dequant_kt writes -- and the chunk dot of x with those weights is added to the row
// accumulator (no per-chunk scale fma). Default (BW = 0): exact fp32 q, scale applied per chunk (the g128 bits).
__device__ __forceinline__ float bf16w(float q, float s) { return __bfloat162float(__float2bfloat16_rn(q * s)); }

// dot8 with the weights rounded as above: same fma chain shape as dot8
__device__ __forceinline__ float dot8w(uint32_t w, const uint32_t* xp, float sc) {
  float s = 0.f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    s = fmaf(__uint_as_float(xp[p] << 16), bf16w(nib(w, 2 * p), sc), s);
    s = fmaf(__uint_as_float(xp[p] & 0xFFFF0000u), bf16w(nib(w, 2 * p + 1), sc), s);
  }
  return s;
}

// Up to kMaxSegs W4 linears that share the input x (and k, group) in one launch: the warps walk the concatenated row
// space (each segment padded to a multiple of R rows, so no row unit straddles two segments); per row the math is the
// single-linear kernel's, so every output is bit-identical to a separate launch.
constexpr int kMaxSegs = 4;
struct KtSegs {
  const uint4* wt[kMaxSegs];
  const __nv_bfloat16* st[kMaxSegs];
  __nv_bfloat16* y[kMaxSegs];
  int n[kMaxSegs];
  int end[kMaxSegs];  // cumulative padded row ends
  int ldy[kMaxSegs];  // output row stride of token j: y[j * ldy + row] (= n unless a linear is split in parts)
  int nseg;
};
// tensor-core path (end of file)
bool mma_on(int group, int bf16w);
int mma_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s, int group, int bits);
KtSegs make_segs(int nseg, const void* const* wt, const void* const* st, void* const* y, const int* n, int R);

template <int G, int R, int U, bool BW = false, int WARPS = kWarps, int BITS = 4>
__global__ void __launch_bounds__(WARPS * 32)
w4_gemv_kt_kernel(const __nv_bfloat16* __restrict__ x, const __grid_constant__ KtSegs segs, int k) {
  extern __shared__ uint32_t xs[];
  constexpr int SPT = 1024 / G;  // scales per tile record
  const int kp = (k + 1023) & ~1023;
  const int chunks = kp >> 5, tiles = kp >> 10;
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(x);
  for (int i = threadIdx.x; i < (kp >> 1); i += blockDim.x) {
    const int c = i >> 4, p = i & 15;
    xs[p * chunks + c] = i < (k >> 1) ? __ldg(xw + i) : 0u;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * WARPS + (threadIdx.x >> 5);
  const int warps_total = gridDim.x * WARPS;
  for (int g0 = warp_global * R; g0 < segs.end[segs.nseg - 1]; g0 += warps_total * R) {
    int sg = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
    const int row0 = g0 - (sg ? segs.end[sg - 1] : 0), n = segs.n[sg];
    const uint4* __restrict__ wt = segs.wt[sg];
    const __nv_bfloat16* __restrict__ st = segs.st[sg];
    __nv_bfloat16* __restrict__ y = segs.y[sg];
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int t0 = 0; t0 < tiles; t0 += U) {
      uint4 w[U][R];
      __nv_bfloat16 sc[U][R];
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const int t = t0 + u;
          const bool ok = t < tiles && row0 + r < n;
          const size_t rec = (size_t)t * n + row0 + r;
          w[u][r] = ok ? ldw<BITS>(wt, rec, lane) : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
          sc[u][r] = ok ? st[rec * SPT + kt_scale_slot<G>(lane)] : __float2bfloat16_rn(0.f);
        }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int t = t0 + u;
        if (t >= tiles) break;
        const int c = t * 32 + lane;
        uint32_t xp[16];
#pragma unroll
        for (int p = 0; p < 16; ++p) xp[p] = xs[p * chunks + c];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (row0 + r >= n) break;
          if (BW) {
            const float sf = __bfloat162float(sc[u][r]);
            acc[r] += dot8w(w[u][r].x, xp, sf) + dot8w(w[u][r].y, xp + 4, sf) + dot8w(w[u][r].z, xp + 8, sf) +
                      dot8w(w[u][r].w, xp + 12, sf);
          } else {
            const float d = dot8(w[u][r].x, xp) + dot8(w[u][r].y, xp + 4) + dot8(w[u][r].z, xp + 8) +
                            dot8(w[u][r].w, xp + 12);
            acc[r] = fmaf(d, __bfloat162float(sc[u][r]), acc[r]);
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], o);
      if (lane == 0 && row0 + r < n) y[row0 + r] = __float2bfloat16_rn(acc[r]);
    }
  }
}

// Dot of 8 inputs (4 bf16 pairs) with 8 pre-dequantized nibbles: the same fma chain as dot8, so the same bits.
__device__ __forceinline__ float dot8f(const float* wf, const uint32_t* xp) {
  float s = 0.f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    s = fmaf(__uint_as_float(xp[p] << 16), wf[2 * p], s);
    s = fmaf(__uint_as_float(xp[p] & 0xFFFF0000u), wf[2 * p + 1], s);
  }
  return s;
}

// Small-batch GEMV (m <= MT <= 16 rows of x; DFlash verify, short prefill tails): one pass over the weights, every
// weight chunk is dequantized once into registers and applied to all m inputs. x rows are read straight from global
// memory (they stay in L1/L2: all warps of a block walk the same tile at about the same time). Each output element
// follows exactly the per-lane order of w4_gemv_kt_kernel (same tiles in the same order, same dot8 chain, same scale
// fma, same butterfly), so row i of the batched result is bit-identical to the m = 1 GEMV of row i.
template <int G, int R, int MT, bool BW = false, int BITS = 4>
__global__ void __launch_bounds__(kWarps * 32)
w4_gemv_kt_batched_kernel(const __nv_bfloat16* __restrict__ x, int m, const uint4* __restrict__ wt,
                          const __nv_bfloat16* __restrict__ st, __nv_bfloat16* __restrict__ y, int n, int k, int ldy) {
  constexpr int SPT = 1024 / G;
  const int kp = (k + 1023) & ~1023;
  const int tiles = kp >> 10, kc = k >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int warps_total = gridDim.x * kWarps;
  for (int row0 = warp_global * R; row0 < n; row0 += warps_total * R) {
    float acc[MT][R];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
    for (int t = 0; t < tiles; ++t) {
      uint4 w[R];
      float s[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const bool ok = row0 + r < n;
        const size_t rec = (size_t)t * n + row0 + r;
        w[r] = ok ? ldw<BITS>(wt, rec, lane) : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
        s[r] = __bfloat162float(ok ? st[rec * SPT + kt_scale_slot<G>(lane)] : __float2bfloat16_rn(0.f));
      }
      float wf[R][32];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t ww[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int i = 0; i < 8; ++i) wf[r][8 * j + i] = BW ? bf16w(nib(ww[j], i), s[r]) : nib(ww[j], i);
      }
      const int c = t * 32 + lane;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (i >= m) break;
        uint32_t xp[16];
        if (c < kc) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)i * k + (size_t)c * 32);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint4 v = __ldg(xr + q);
            xp[4 * q] = v.x, xp[4 * q + 1] = v.y, xp[4 * q + 2] = v.z, xp[4 * q + 3] = v.w;
          }
        } else {
#pragma unroll
          for (int p = 0; p < 16; ++p) xp[p] = 0u;  // k padding: x = 0 like the GEMV's staged zeros
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const float d = dot8f(wf[r], xp) + dot8f(wf[r] + 8, xp + 4) + dot8f(wf[r] + 16, xp + 8) +
                          dot8f(wf[r] + 24, xp + 12);
          acc[i][r] = BW ? acc[i][r] + d : fmaf(d, s[r], acc[i][r]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      if (i >= m) break;
#pragma unroll
      for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[i][r] += __shfl_xor_sync(0xFFFFFFFFu, acc[i][r], o);
        if (lane == 0 && row0 + r < n) y[(size_t)i * ldy + row0 + r] = __float2bfloat16_rn(acc[i][r]);
      }
    }
  }
}

// One thread per 16-byte chunk of the KT output (padding chunks get nibble 8 = 0 and scale 0).
template <int G>
__global__ void repack_kt_kernel(const uint4* __restrict__ src, const __nv_bfloat16* __restrict__ ssrc, uint4* __restrict__ dst,
                                 __nv_bfloat16* __restrict__ sdst, int n, int k) {
  constexpr int SPT = 1024 / G, CPG = G / 32;  // scales per tile, chunks per group
  const int kp = (k + 1023) & ~1023;
  const int chunks = k >> 5, tiles = kp >> 10, groups = k / G;
  const size_t total = (size_t)tiles * n * 32;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const size_t rec = i >> 5;
    const int o = (int)(rec % n), t = (int)(rec / n);
    const int c = t * 32 + lane;
    dst[i] = c < chunks ? src[(size_t)o * chunks + c] : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
    if (lane % CPG == 0) {
      const int g = c / CPG;
      sdst[rec * SPT + lane / CPG] = g < groups ? ssrc[(size_t)o * groups + g] : __float2bfloat16_rn(0.f);
    }
  }
}

// KT -> dense bf16 [n][k] (prefill path: dequantize into a scratch buffer, then cuBLAS). Value = bf16(q * scale),
// the rounding Marlin's bf16 dequant applies before its MMA. One thread per 16-byte chunk (32 outputs).
template <int G, int BITS = 4>
// rows [r0, r0 + nr) of the n-row weight -> w [nr][k] (r0 = 0, nr = n: the whole weight)
__global__ void dequant_kt_kernel(const uint4* __restrict__ wt, const __nv_bfloat16* __restrict__ st,
                                  __nv_bfloat16* __restrict__ w, int n, int k, int r0, int nr) {
  constexpr int SPT = 1024 / G;
  const int kp = (k + 1023) & ~1023;
  const int kc = k >> 5, tiles = kp >> 10;
  const size_t total = (size_t)tiles * nr * 32;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const size_t rl = i >> 5;
    const int o = (int)(rl % nr), t = (int)(rl / nr);
    const size_t rec = (size_t)t * n + r0 + o;
    const int c = t * 32 + lane;
    if (c >= kc) continue;
    const uint4 q = ldw<BITS>(wt, rec, lane);
    const float s = __bfloat162float(st[rec * SPT + kt_scale_slot<G>(lane)]);
    const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
    uint4* out = reinterpret_cast<uint4*>(w + (size_t)o * k + (size_t)c * 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t v[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const __nv_bfloat16 lo = __float2bfloat16_rn(nib(qq[j], 2 * p) * s);
        const __nv_bfloat16 hi = __float2bfloat16_rn(nib(qq[j], 2 * p + 1) * s);
        v[p] = (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
      }
      out[j] = make_uint4(v[0], v[1], v[2], v[3]);
    }
  }
}

int num_sms() {
  static int sms = 0;
  if (sms == 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }
  return sms;
}

// bps_cap > 0 caps resident blocks per SM below the occupancy limit (fewer, longer-lived blocks: x is staged once per
// block, which matters on high-bandwidth parts where a GEMV lasts only a few microseconds).
template <int G, int R, int U, bool BW = false, int WARPS = kWarps, int BITS = 4>
int launch_kt(const void* x, const KtSegs& segs, int k, cudaStream_t s, int bps_cap = 0) {
  const size_t smem = (size_t)((k + 1023) & ~1023) * 2;
  auto kern = w4_gemv_kt_kernel<G, R, U, BW, WARPS, BITS>;
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
  }
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, WARPS * 32, smem);
  if (per_sm < 1) per_sm = 1;
  if (bps_cap > 0 && per_sm > bps_cap) per_sm = bps_cap;
  const int want = (segs.end[segs.nseg - 1] / R + WARPS - 1) / WARPS;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, WARPS * 32, smem, s>>>(static_cast<const __nv_bfloat16*>(x), segs, k);
  return (int)cudaGetLastError();
}

KtSegs make_segs(int nseg, const void* const* wt, const void* const* st, void* const* y, const int* n, int R) {
  KtSegs g{};
  g.nseg = nseg;
  for (int i = 0, e = 0; i < nseg; ++i) {
    g.wt[i] = static_cast<const uint4*>(wt[i]), g.st[i] = static_cast<const __nv_bfloat16*>(st[i]);
    g.y[i] = static_cast<__nv_bfloat16*>(y[i]), g.n[i] = n[i], g.ldy[i] = n[i];
    e += (n[i] + R - 1) / R * R;
    g.end[i] = e;
  }
  return g;
}

template <int G, int R, int U>
int launch_kt(const void* x, const void* wt, const void* st, void* y, int n, int k, cudaStream_t s) {
  return launch_kt<G, R, U>(x, make_segs(1, &wt, &st, &y, &n, R), k, s);
}

template <int G, int R, int MT, bool BW = false, int BITS = 4>
int launch_kt_batched(const void* x, int m, const void* wt, const void* st, void* y, int n, int k, cudaStream_t s,
                      int ldy) {
  auto kern = w4_gemv_kt_batched_kernel<G, R, MT, BW, BITS>;
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kWarps * 32, 0);
  if (per_sm < 1) per_sm = 1;
  const int want = ((n + R - 1) / R + kWarps - 1) / kWarps;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, kWarps * 32, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, static_cast<const uint4*>(wt),
                                       static_cast<const __nv_bfloat16*>(st), static_cast<__nv_bfloat16*>(y), n, k, ldy);
  return (int)cudaGetLastError();
}

bool group_ok(int k, int group) { return (group == 64 || group == 128) && k % group == 0; }

}  // namespace

extern "C" size_t ak_w4_kt_weight_bytes(int n, int k) { return (size_t)n * ((k + 1023) & ~1023) / 2; }
extern "C" size_t ak_w4_kt_scale_bytes(int n, int k, int group) { return (size_t)n * ((k + 1023) & ~1023) / group * 2; }
extern "C" size_t ak_w4g128_kt_weight_bytes(int n, int k) { return ak_w4_kt_weight_bytes(n, k); }
extern "C" size_t ak_w4g128_kt_scale_bytes(int n, int k) { return ak_w4_kt_scale_bytes(n, k, 128); }

extern "C" int ak_w4_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, int group,
                               void* stream) {
  if (!group_ok(k, group) || k % 32 != 0) return (int)cudaErrorInvalidValue;
  if (group == 128) {
    // W4H only carries the tensor-core numerics: not with the exact fp32 path (AGENS_W4_EXACT=1)
    int e = mma_on(128, 0) ? agens_w4h_repack(packed, scales, wt, st, n, k, static_cast<cudaStream_t>(stream)) : 1;
    if (e != 1) return e;
    agens_w4h_forget(wt);  // a KT / KF weight at an address a freed W4H weight had
    e = agens_kf_repack(packed, scales, wt, st, n, k, static_cast<cudaStream_t>(stream));
    if (e != 1) return e;
  }
  agens_w4h_forget(wt);
  agens_kf_forget(wt);  // a KT weight at an address a freed KF weight had
  auto kern = group == 64 ? repack_kt_kernel<64> : repack_kt_kernel<128>;
  kern<<<1024, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint4*>(packed), static_cast<const __nv_bfloat16*>(scales), static_cast<uint4*>(wt),
      static_cast<__nv_bfloat16*>(st), n, k);
  return (int)cudaGetLastError();
}

extern "C" int ak_w4g128_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k,
                                   void* stream) {
  return ak_w4_repack_kt(packed, scales, wt, st, n, k, 128, stream);
}

extern "C" int ak_w4g128_gemv_kt_variant(int variant, const void* x, const void* wt, const void* st, void* y, int n,
                                         int k, void* stream) {
  if (k % 128 != 0) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (variant) {
    case 11: return launch_kt<128, 1, 1>(x, wt, st, y, n, k, s);
    case 12: return launch_kt<128, 1, 2>(x, wt, st, y, n, k, s);
    case 21: return launch_kt<128, 2, 1>(x, wt, st, y, n, k, s);
    case 22: return launch_kt<128, 2, 2>(x, wt, st, y, n, k, s);
    case 41: return launch_kt<128, 4, 1>(x, wt, st, y, n, k, s);
    default: return (int)cudaErrorInvalidValue;
  }
}

extern "C" int ak_w4g128_gemv_kt_bf16(const void* x, const void* wt, const void* st, void* y, int n, int k,
                                      void* stream) {
  return ak_w4g128_gemv_kt_variant(21, x, wt, st, y, n, k, stream);
}

namespace {
template <int R, int U, int WARPS>
int tune_one(const void* x, const KtSegs& g, int k, int bps, cudaStream_t s) {
  return launch_kt<128, R, U, false, WARPS>(x, g, k, s, bps);
}
template <int R, int U>
int tune_w(int warps, const void* x, const KtSegs& g, int k, int bps, cudaStream_t s) {
  switch (warps) {
    case 4: return tune_one<R, U, 4>(x, g, k, bps, s);
    case 8: return tune_one<R, U, 8>(x, g, k, bps, s);
    case 16: return tune_one<R, U, 16>(x, g, k, bps, s);
    default: return (int)cudaErrorInvalidValue;
  }
}
template <int R>
int tune_u(int u, int warps, const void* x, const KtSegs& g, int k, int bps, cudaStream_t s) {
  switch (u) {
    case 1: return tune_w<R, 1>(warps, x, g, k, bps, s);
    case 2: return tune_w<R, 2>(warps, x, g, k, bps, s);
    case 4: return tune_w<R, 4>(warps, x, g, k, bps, s);
    default: return (int)cudaErrorInvalidValue;
  }
}
}  // namespace

// Tuning entry (bench only, g128, exact): rows per warp r in {1,2,4}, unroll u in {1,2,4}, warps per block in
// {4,8,16}, blocks per SM cap (0 = occupancy). Every combination gives the same bits.
extern "C" int ak_w4g128_gemv_kt_tune(int r, int u, int warps, int bps, const void* x, const void* wt, const void* st,
                                      void* y, int n, int k, void* stream) {
  if (k % 128 != 0) return (int)cudaErrorInvalidValue;
  const KtSegs g = make_segs(1, &wt, &st, &y, &n, 4);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (r) {
    case 1: return tune_u<1>(u, warps, x, g, k, bps, s);
    case 2: return tune_u<2>(u, warps, x, g, k, bps, s);
    case 4: return tune_u<4>(u, warps, x, g, k, bps, s);
    default: return (int)cudaErrorInvalidValue;
  }
}

namespace {
// production m = 1 / multi / batched dispatch: group 64 or 128, exact or bf16-rounded weights (bf16w).
// Rows per warp / unroll by shape (every variant gives the same bits: per-row order is fixed), from bench_gemv on a
// sm89: k >= 16K (down_proj, 35 KB of x in smem per block) R1U2 897 vs 787 GB/s for R2U1; >= 10K rows R2U2
// (gate_up 992, KDA qkv 892 GB/s); else R2U1 (5120/6144-row shapes 807-817 GB/s).
template <int G, bool BW, int BITS = 4>
int gemv_shape(const void* x, const KtSegs& g, int k, cudaStream_t s) {
  if (k >= 16384) return launch_kt<G, 1, 2, BW, kWarps, BITS>(x, g, k, s);
  if (g.end[g.nseg - 1] >= 10240) return launch_kt<G, 2, 2, BW, kWarps, BITS>(x, g, k, s);
  return launch_kt<G, 2, 1, BW, kWarps, BITS>(x, g, k, s);
}

template <bool BW>
int gemv_segs(const void* x, const KtSegs& g, int k, int group, cudaStream_t s) {
  return group == 64 ? gemv_shape<64, BW>(x, g, k, s) : gemv_shape<128, BW>(x, g, k, s);
}

template <int G, bool BW, int BITS = 4>
int batched(const void* x, const void* wt, const void* st, void* y, int m, int n, int k, cudaStream_t s, int ldy) {
  if (m <= 4) return launch_kt_batched<G, 2, 4, BW, BITS>(x, m, wt, st, y, n, k, s, ldy);
  if (m <= 8) return launch_kt_batched<G, 2, 8, BW, BITS>(x, m, wt, st, y, n, k, s, ldy);
  return launch_kt_batched<G, 2, 16, BW, BITS>(x, m, wt, st, y, n, k, s, ldy);
}

}  // namespace

namespace {
// W4H weights among the segments: all W4H -> one W4H launch (bit-identical to the tensor-core KT path); a mix -> per
// segment; none -> -1 (not handled here). W4H has only the tensor-core numerics (group 128, bf16w = 0).
int w4h_route(const void* x, int m, int nseg, const void* const* wt, const void* const* st, void* const* y,
              const int* n, const int* ldy, int k, int group, int bf16w, void* stream) {
  int nh = 0;
  for (int i = 0; i < nseg; ++i) nh += agens_w4h_is(wt[i]);
  if (!nh) return -1;
  if (group != 128 || bf16w) return (int)cudaErrorInvalidValue;
  if (nh == nseg) return ak_w4h_gemm(x, m, nseg, wt, st, y, n, ldy, k, stream);
  for (int i = 0; i < nseg; ++i) {
    const int e = agens_w4h_is(wt[i]) ? ak_w4h_gemm(x, m, 1, wt + i, st + i, y + i, n + i, ldy ? ldy + i : nullptr, k, stream)
                                      : ak_wq_gemv_kt_multi(4, x, m, 1, wt + i, st + i, y + i, n + i, ldy ? ldy + i : nullptr,
                                                            k, group, bf16w, stream);
    if (e) return e;
  }
  return 0;
}
}  // namespace

extern "C" int ak_w4_gemv_kt(const void* x, const void* wt, const void* st, void* y, int n, int k, int group,
                             int bf16w, void* stream) {
  if (!group_ok(k, group)) return (int)cudaErrorInvalidValue;
  if (const int e = w4h_route(x, 1, 1, &wt, &st, &y, &n, nullptr, k, group, bf16w, stream); e >= 0) return e;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (mma_on(group, bf16w)) return mma_segs(x, 1, make_segs(1, &wt, &st, &y, &n, 16), k, s, group, 4);
  const KtSegs g = make_segs(1, &wt, &st, &y, &n, 4);
  return bf16w ? gemv_segs<true>(x, g, k, group, s) : gemv_segs<false>(x, g, k, group, s);
}

extern "C" int ak_w4_gemv_kt_multi(const void* x, int nseg, const void* const* wt, const void* const* st,
                                   void* const* y, const int* n, int k, int group, int bf16w, void* stream) {
  if (!group_ok(k, group) || nseg < 1 || nseg > kMaxSegs) return (int)cudaErrorInvalidValue;
  if (const int e = w4h_route(x, 1, nseg, wt, st, y, n, nullptr, k, group, bf16w, stream); e >= 0) return e;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (mma_on(group, bf16w)) return mma_segs(x, 1, make_segs(nseg, wt, st, y, n, 16), k, s, group, 4);
  const KtSegs g = make_segs(nseg, wt, st, y, n, 4);
  return bf16w ? gemv_segs<true>(x, g, k, group, s) : gemv_segs<false>(x, g, k, group, s);
}

extern "C" int ak_w4_gemv_kt_batched(const void* x, const void* wt, const void* st, void* y, int m, int n, int k,
                                     int group, int bf16w, void* stream) {
  if (!group_ok(k, group) || m < 1 || m > 128) return (int)cudaErrorInvalidValue;
  if (const int e = w4h_route(x, m, 1, &wt, &st, &y, &n, nullptr, k, group, bf16w, stream); e >= 0) return e;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (mma_on(group, bf16w)) return mma_segs(x, m, make_segs(1, &wt, &st, &y, &n, 16), k, s, group, 4);
  if (m > 16) return (int)cudaErrorInvalidValue;  // exact path: <= 16 rows (ak_wq_gemv_kt_multi chunks larger m)
  if (group == 64) return bf16w ? batched<64, true>(x, wt, st, y, m, n, k, s, n) : batched<64, false>(x, wt, st, y, m, n, k, s, n);
  return bf16w ? batched<128, true>(x, wt, st, y, m, n, k, s, n) : batched<128, false>(x, wt, st, y, m, n, k, s, n);
}

extern "C" int ak_w4_dequant_kt(const void* wt, const void* st, void* w, int n, int k, int group, void* stream) {
  if (!group_ok(k, group) || k % 32 != 0) return (int)cudaErrorInvalidValue;
  if (agens_w4h_is(wt)) return ak_w4h_dequant_rows(wt, st, w, n, k, 0, n, stream);
  if (agens_kf_is(wt)) return agens_kf_dequant_rows(wt, st, w, n, k, 0, n, static_cast<cudaStream_t>(stream));
  auto kern = group == 64 ? dequant_kt_kernel<64> : dequant_kt_kernel<128>;
  kern<<<num_sms() * 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(wt), static_cast<const __nv_bfloat16*>(st), static_cast<__nv_bfloat16*>(w), n, k, 0, n);
  return (int)cudaGetLastError();
}

extern "C" int ak_w4_dequant_kt_rows(const void* wt, const void* st, void* w, int n, int k, int group, int r0, int nr,
                                     void* stream) {
  if (!group_ok(k, group) || k % 32 != 0 || r0 < 0 || nr < 1 || r0 + nr > n) return (int)cudaErrorInvalidValue;
  if (agens_w4h_is(wt)) return ak_w4h_dequant_rows(wt, st, w, n, k, r0, nr, stream);
  if (agens_kf_is(wt)) return agens_kf_dequant_rows(wt, st, w, n, k, r0, nr, static_cast<cudaStream_t>(stream));
  auto kern = group == 64 ? dequant_kt_kernel<64> : dequant_kt_kernel<128>;
  kern<<<num_sms() * 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(wt), static_cast<const __nv_bfloat16*>(st), static_cast<__nv_bfloat16*>(w), n, k, r0, nr);
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------- W3 / W2 KT
namespace {
// One thread per lane chunk of the W3 / W2 KT output: the int4 container's 32 nibbles (q + 8) of chunk c -> 2-bit fields
// (+ the bit plane for W3) of n = q + 2^(BITS-1); a nibble whose q needs more than BITS bits sets *bad. Padding: q = 0.
template <int G, int BITS>
__global__ void repack_wq_kernel(const uint4* __restrict__ src, const __nv_bfloat16* __restrict__ ssrc,
                                 uint32_t* __restrict__ dst, __nv_bfloat16* __restrict__ sdst, int n, int k, int* bad) {
  constexpr int SPT = 1024 / G, CPG = G / 32;
  constexpr int OFF = 8 - (1 << (BITS - 1)), MAXN = (1 << BITS) - 1;
  const int kp = (k + 1023) & ~1023;
  const int chunks = k >> 5, tiles = kp >> 10, groups = k / G;
  const size_t total = (size_t)tiles * n * 32;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const size_t rec = i >> 5;
    const int o = (int)(rec % n), t = (int)(rec / n);
    const int c = t * 32 + lane;
    const uint4 q = c < chunks ? src[(size_t)o * chunks + c] : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
    const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
    uint32_t lo[2] = {0u, 0u}, hi = 0u;
#pragma unroll
    for (int v = 0; v < 32; ++v) {
      const int nv = (int)((qq[v >> 3] >> (4 * (v & 7))) & 0xFu) - OFF;
      if (nv < 0 || nv > MAXN) *bad = 1;
      const uint32_t mv = (uint32_t)nv & (uint32_t)MAXN;
      lo[v >> 4] |= (mv & 3u) << (2 * (v & 15));
      if (BITS == 3) hi |= (mv >> 2) << v;
    }
    uint32_t* b = dst + rec * rec_words<BITS>();
    b[2 * lane] = lo[0], b[2 * lane + 1] = lo[1];
    if (BITS == 3) b[64 + lane] = hi;
    if (lane % CPG == 0) {
      const int g = c / CPG;
      sdst[rec * SPT + lane / CPG] = g < groups ? ssrc[(size_t)o * groups + g] : __float2bfloat16_rn(0.f);
    }
  }
}

// GEMV of nseg linears sharing x (same k, group, bits), m tokens: tensor-core path (default), else exact (m = 1: the
// multi-segment GEMV; m > 1: the batched GEMV per segment). Output of token j, segment i at y[i] + j * ldy[i].
template <int BITS>
int wq_run(const void* x, int m, int nseg, const void* const* wt, const void* const* st, void* const* y, const int* n,
           const int* ldy, int k, int group, int bf16w, cudaStream_t s) {
  if (mma_on(group, bf16w)) {
    KtSegs g = make_segs(nseg, wt, st, y, n, 16);
    if (ldy)
      for (int i = 0; i < nseg; ++i) g.ldy[i] = ldy[i];
    return mma_segs(x, m, g, k, s, group, BITS);
  }
  if (m == 1) {
    const KtSegs g = make_segs(nseg, wt, st, y, n, 4);
    if (group == 64) return bf16w ? gemv_shape<64, true, BITS>(x, g, k, s) : gemv_shape<64, false, BITS>(x, g, k, s);
    return bf16w ? gemv_shape<128, true, BITS>(x, g, k, s) : gemv_shape<128, false, BITS>(x, g, k, s);
  }
  for (int i = 0; i < nseg; ++i) {
    const int ld = ldy ? ldy[i] : n[i];
    const int e = group == 64 ? (bf16w ? batched<64, true, BITS>(x, wt[i], st[i], y[i], m, n[i], k, s, ld)
                                       : batched<64, false, BITS>(x, wt[i], st[i], y[i], m, n[i], k, s, ld))
                              : (bf16w ? batched<128, true, BITS>(x, wt[i], st[i], y[i], m, n[i], k, s, ld)
                                       : batched<128, false, BITS>(x, wt[i], st[i], y[i], m, n[i], k, s, ld));
    if (e) return e;
  }
  return 0;
}
}  // namespace

// Mixed precision (per linear W2 / W3 / W4, group 64 or 128): bytes, repack from the int4 container, GEMV, dequant.
extern "C" size_t ak_wq_kt_weight_bytes(int bits, int n, int k) { return (size_t)n * ((k + 1023) & ~1023) * bits / 8; }
extern "C" size_t ak_w3_kt_weight_bytes(int n, int k) { return ak_wq_kt_weight_bytes(3, n, k); }

// int4-container weights whose values all lie in [-2^(bits-1), 2^(bits-1) - 1] -> W{bits} KT (scales as W4 KT,
// ak_w4_kt_scale_bytes). cudaErrorInvalidValue if any value needs more bits (the caller keeps that linear wider).
extern "C" int ak_wq_repack_kt(int bits, const int32_t* packed, const void* scales, void* wt, void* st, int n, int k,
                               int group, void* stream) {
  if (bits == 4) return ak_w4_repack_kt(packed, scales, wt, st, n, k, group, stream);
  if ((bits != 2 && bits != 3) || !group_ok(k, group) || k % 32 != 0) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  int* bad = nullptr;
  if (cudaMalloc(&bad, 4) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
  cudaMemsetAsync(bad, 0, 4, s);
  auto kern = bits == 3 ? (group == 64 ? repack_wq_kernel<64, 3> : repack_wq_kernel<128, 3>)
                        : (group == 64 ? repack_wq_kernel<64, 2> : repack_wq_kernel<128, 2>);
  kern<<<1024, 256, 0, s>>>(reinterpret_cast<const uint4*>(packed), static_cast<const __nv_bfloat16*>(scales),
                            static_cast<uint32_t*>(wt), static_cast<__nv_bfloat16*>(st), n, k, bad);
  int h = 0;
  cudaError_t e = cudaMemcpyAsync(&h, bad, 4, cudaMemcpyDeviceToHost, s);
  if (e == cudaSuccess) e = cudaStreamSynchronize(s);
  cudaFree(bad);
  if (e != cudaSuccess) return (int)e;
  return h ? (int)cudaErrorInvalidValue : 0;
}
extern "C" int ak_w3_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, int group,
                               void* stream) {
  return ak_wq_repack_kt(3, packed, scales, wt, st, n, k, group, stream);
}

// 1 <= m <= 16 tokens, up to 4 linears sharing x (same bits, k, group); ldy = nullptr: rows of n[i]. bits = 4 is exactly
// ak_w4_gemv_kt (m = 1) / _multi / _batched; bits 2, 3 the same kernels on expanded chunks (bit-identical to W4 KT).
extern "C" int ak_wq_gemv_kt_multi(int bits, const void* x, int m, int nseg, const void* const* wt,
                                   const void* const* st, void* const* y, const int* n, const int* ldy, int k,
                                   int group, int bf16w, void* stream) {
  if (!group_ok(k, group) || nseg < 1 || nseg > kMaxSegs || m < 1 || m > 128) return (int)cudaErrorInvalidValue;
  if (bits == 4)
    if (const int e = w4h_route(x, m, nseg, wt, st, y, n, ldy, k, group, bf16w, stream); e >= 0) return e;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (m > 16 && !mma_on(group, bf16w)) {  // exact path: 16-row chunks (rows bit-identical for any m)
    for (int m0 = 0; m0 < m; m0 += 16) {
      void* yy[kMaxSegs];
      for (int i = 0; i < nseg; ++i)
        yy[i] = static_cast<__nv_bfloat16*>(y[i]) + (size_t)m0 * (ldy ? ldy[i] : n[i]);
      const int e = ak_wq_gemv_kt_multi(bits, static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k, m - m0 < 16 ? m - m0 : 16,
                                        nseg, wt, st, yy, n, ldy, k, group, bf16w, stream);
      if (e) return e;
    }
    return 0;
  }
  switch (bits) {
    case 4: return wq_run<4>(x, m, nseg, wt, st, y, n, ldy, k, group, bf16w, s);
    case 3: return wq_run<3>(x, m, nseg, wt, st, y, n, ldy, k, group, bf16w, s);
    case 2: return wq_run<2>(x, m, nseg, wt, st, y, n, ldy, k, group, bf16w, s);
    default: return (int)cudaErrorInvalidValue;
  }
}
extern "C" int ak_wq_dequant_kt_rows(int bits, const void* wt, const void* st, void* w, int n, int k, int group, int r0,
                                     int nr, void* stream) {
  if (!group_ok(k, group) || k % 32 != 0 || r0 < 0 || nr < 1 || r0 + nr > n) return (int)cudaErrorInvalidValue;
  if (bits == 4 && agens_w4h_is(wt)) return ak_w4h_dequant_rows(wt, st, w, n, k, r0, nr, stream);
  if (bits == 4 && agens_kf_is(wt)) return agens_kf_dequant_rows(wt, st, w, n, k, r0, nr, static_cast<cudaStream_t>(stream));
  decltype(&dequant_kt_kernel<64, 4>) kern;
  switch (bits * 1000 + group) {
    case 4064: kern = dequant_kt_kernel<64, 4>; break;
    case 4128: kern = dequant_kt_kernel<128, 4>; break;
    case 3064: kern = dequant_kt_kernel<64, 3>; break;
    case 3128: kern = dequant_kt_kernel<128, 3>; break;
    case 2064: kern = dequant_kt_kernel<64, 2>; break;
    case 2128: kern = dequant_kt_kernel<128, 2>; break;
    default: return (int)cudaErrorInvalidValue;
  }
  kern<<<num_sms() * 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(wt), static_cast<const __nv_bfloat16*>(st), static_cast<__nv_bfloat16*>(w), n, k, r0, nr);
  return (int)cudaGetLastError();
}

// W3 KT twins of the W4 KT entry points (kept for callers of the 3-bit probe path)
extern "C" int ak_w3_gemv_kt(const void* x, const void* wt, const void* st, void* y, int n, int k, int group,
                             void* stream) {
  return ak_wq_gemv_kt_multi(3, x, 1, 1, &wt, &st, &y, &n, nullptr, k, group, 0, stream);
}
extern "C" int ak_w3_gemv_kt_multi(const void* x, int nseg, const void* const* wt, const void* const* st,
                                   void* const* y, const int* n, int k, int group, void* stream) {
  return ak_wq_gemv_kt_multi(3, x, 1, nseg, wt, st, y, n, nullptr, k, group, 0, stream);
}
extern "C" int ak_w3_gemv_kt_batched(const void* x, const void* wt, const void* st, void* y, int m, int n, int k,
                                     int group, void* stream) {
  return ak_wq_gemv_kt_multi(3, x, m, 1, &wt, &st, &y, &n, nullptr, k, group, 0, stream);
}
extern "C" int ak_w3_dequant_kt_rows(const void* wt, const void* st, void* w, int n, int k, int group, int r0, int nr,
                                     void* stream) {
  return ak_wq_dequant_kt_rows(3, wt, st, w, n, k, group, r0, nr, stream);
}

// Variant selection is exposed for the bench; production uses ak_w4g128_gemv_bf16.
extern "C" int ak_w4g128_gemv_bf16_variant(int variant, const void* x, const int32_t* packed, const void* scales,
                                           void* y, int n, int k, void* stream) {
  if (k % 128 != 0) return (int)cudaErrorInvalidValue;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  switch (variant) {
    case 11: return launch<1, 1>(x, packed, scales, y, n, k, st);
    case 12: return launch<1, 2>(x, packed, scales, y, n, k, st);
    case 14: return launch<1, 4>(x, packed, scales, y, n, k, st);
    case 21: return launch<2, 1>(x, packed, scales, y, n, k, st);
    case 22: return launch<2, 2>(x, packed, scales, y, n, k, st);
    case 41: return launch<4, 1>(x, packed, scales, y, n, k, st);
    case 42: return launch<4, 2>(x, packed, scales, y, n, k, st);
    default: return (int)cudaErrorInvalidValue;
  }
}

extern "C" int ak_w4g128_gemv_bf16(const void* x, const int32_t* packed, const void* scales, void* y, int n, int k,
                                   void* stream) {
  return ak_w4g128_gemv_bf16_variant(22, x, packed, scales, y, n, k, stream);
}

// ---------------------------------------------------------------------------------------------- tensor-core path
// The exact GEMV spends ~5 ALU instructions per weight (nibble -> fp32, fma), which caps it near 1.5 TB/s on an
// sm90 (it is bandwidth-bound on sm89 / sm120 parts). This path reads the same k-tiled layout and does the
// dot products on tensor cores (mma.m16n8k16 bf16 -> fp32): a lop3 + one bf16x2 subtract turn two nibbles into the
// exact bf16 pair (q - 8) (~1.4 instructions per weight), the products are exact, sums are fp32 in the tensor core,
// and each group's partial sum is scaled in fp32 (y = sum_g s_g * sum_{k in g} x_k q_k, as the exact path).
// Numerics differ from the exact path only in fp32 summation order. Columns of an MMA are independent, so up to 8
// tokens per launch give rows bit-identical to the 1-token launch (decode == DFlash verify).
// Mapping: per 16-row block, lane (g = lane/4, t4 = lane%4) loads, per k-tile and step i < 8, chunk 4i + t4 (32
// inputs) of rows g and g+8, so one step's 4 chunks are one g128 group. Input order inside an MMA is free as long as
// A and B agree: pair p of a chunk = inputs (8(p/4) + p%4, +4) = nibbles (p%4, p%4 + 4) of word p/4; MMA q of a step
// takes pairs 2q (k slots 2*t4, +1) and 2q + 1 (slots 2*t4 + 8, +9).
namespace {
__device__ __forceinline__ uint32_t deq_pair(uint32_t w, int j) {  // bf16x2 (q_j - 8, q_{j+4} - 8), exact
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  // 0x4300 = 128.0 bf16: (128 + q) - 136 = q - 8
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
// W2 KT on the tensor-core path without the nibble expansion: a 16-bit word of 2-bit fields f_0..f_7 (n = q + 2,
// inputs 8wd .. 8wd+7) prepared once as w2x = (fields 0..3 at bits 0..7, fields 4..7 at bits 16..23); pair j is then
// bf16x2 (128 + f_j, 128 + f_{j+4}) - 130 = (q_j, q_{j+4}): the same bf16 operands deq_pair builds from the nibble
// word, so the MMA results are bit-identical.
__device__ __forceinline__ uint32_t w2x(uint32_t f16) { return (f16 & 0xFFu) | ((f16 & 0xFF00u) << 8); }
__device__ __forceinline__ uint32_t deq2_pair(uint32_t x, int j) {
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(x >> (2 * j)), "r"(0x00030003u), "r"(0x43004300u));
  // 0x4300 = 128.0 bf16: (128 + n) - 130 = n - 2 = q
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(130.f, 130.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
template <int BITS>
__device__ __forceinline__ uint32_t deqb(uint32_t w, int j) { return BITS == 2 ? deq2_pair(w, j) : deq_pair(w, j); }
// lane's chunk for the tensor-core g128 kernel: W2 as 4 prepared w2x words, else the nibble words of ldw
template <int BITS>
__device__ __forceinline__ uint4 ldw_mma(const uint4* __restrict__ wt, size_t rec, int c) {
  if (BITS != 2) return ldw<BITS>(wt, rec, c);
  const uint2 lo = ld_stream2(reinterpret_cast<const uint2*>(reinterpret_cast<const uint32_t*>(wt) + rec * rec_words<2>()) + c);
  return make_uint4(w2x(lo.x & 0xFFFFu), w2x(lo.x >> 16), w2x(lo.y & 0xFFFFu), w2x(lo.y >> 16));
}
__device__ __forceinline__ uint32_t x_pair(const uint32_t* xw, int wd, int j) {  // bf16x2 (x[8wd + j], x[8wd + j + 4])
  const uint32_t a = xw[4 * wd + (j >> 1)], b = xw[4 * wd + 2 + (j >> 1)];
  return __byte_perm(a, b, (j & 1) ? 0x7632 : 0x5410);
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// x [m][k] bf16 (m <= 8 tokens), y [m][n] per segment. Segments padded to 16 rows (KtSegs.end).
// A block of WARPS warps owns one 16-row block at a time and splits its k-steps (step = k-tile t, group i: chunks
// 4i..4i+3 of rows g, g+8) across the warps (warp w takes steps w, w + WARPS, ...; U steps' loads issued together);
// the warps' scaled partial sums are added in warp order through shared memory: fixed order, so deterministic and the
// same for every column count m.
template <int WARPS, int U, int BITS = 4>
__global__ void __launch_bounds__(WARPS * 32)
w4_mma_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  __shared__ float red[WARPS][32][4];
  const int kp = (k + 1023) & ~1023, steps = (kp >> 10) * 8, kc = k >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int g0 = rb << 4;
    int sg = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
    const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    const uint4* __restrict__ wt = segs.wt[sg];
    const __nv_bfloat16* __restrict__ st = segs.st[sg];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s0 = warp; s0 < steps; s0 += WARPS * U) {
      uint4 w0[U], w1[U], xv[U][4];
      float sc0[U], sc1[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int s = s0 + u * WARPS, t = s >> 3, i = s & 7;
        const bool in = s < steps;
        const size_t rec0 = (size_t)t * n + r0, rec1 = (size_t)t * n + r1;
        const int c = 4 * i + t4, cg = t * 32 + c;
        // padding: q = 0 (nibble 8 / 2-bit field 2 -> w2x word 0x00AA00AA)
        const uint4 pad = BITS == 2 ? make_uint4(0x00AA00AAu, 0x00AA00AAu, 0x00AA00AAu, 0x00AA00AAu)
                                    : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
        w0[u] = in && ok0 ? ldw_mma<BITS>(wt, rec0, c) : pad;
        w1[u] = in && ok1 ? ldw_mma<BITS>(wt, rec1, c) : pad;
        sc0[u] = in && ok0 ? __bfloat162float(st[rec0 * 8 + i]) : 0.f;
        sc1[u] = in && ok1 ? __bfloat162float(st[rec1 * 8 + i]) : 0.f;
        const bool xin = in && g < m && cg < kc;
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)g * k + (size_t)cg * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[u][q] = xin ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t xw[16] = {xv[u][0].x, xv[u][0].y, xv[u][0].z, xv[u][0].w, xv[u][1].x, xv[u][1].y,
                                 xv[u][1].z, xv[u][1].w, xv[u][2].x, xv[u][2].y, xv[u][2].z, xv[u][2].w,
                                 xv[u][3].x, xv[u][3].y, xv[u][3].z, xv[u][3].w};
        const uint32_t ww0[4] = {w0[u].x, w0[u].y, w0[u].z, w0[u].w}, ww1[4] = {w1[u].x, w1[u].y, w1[u].z, w1[u].w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, deqb<BITS>(ww0[wd], j0), deqb<BITS>(ww1[wd], j0), deqb<BITS>(ww0[wd], j1), deqb<BITS>(ww1[wd], j1),
                   x_pair(xw, wd, j0), x_pair(xw, wd, j1));
        }
        acc[0] = fmaf(cf[0], sc0[u], acc[0]);
        acc[1] = fmaf(cf[1], sc0[u], acc[1]);
        acc[2] = fmaf(cf[2], sc1[u], acc[2]);
        acc[3] = fmaf(cf[3], sc1[u], acc[3]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp][lane][j] = acc[j];
    __syncthreads();
    if (warp == 0) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int w = 0; w < WARPS; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[w][lane][j];
      __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
          if (ok1) y[(size_t)col * segs.ldy[sg] + r1] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}

// Batched speculative verify (B sequences x 8 tokens per launch): NT tiles of 8 tokens share one pass over the
// weights. Per step the warp dequantizes its A operands once and runs w4_mma_kernel<8, 1>'s 8-MMA chain for every
// tile (an MMA column depends only on its own token's x: each token's chain, its scaled accumulation and the warp-order
// sum are exactly the 8-token launch's), so every row is bit-identical to its m <= 8 launch. Portable (mma.sync).
template <int BITS, int NT>
__global__ void __launch_bounds__(256)
w4_mma_nt_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  __shared__ float red[8][32][4 * NT];
  const int kp = (k + 1023) & ~1023, steps = (kp >> 10) * 8, kc = k >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int g0 = rb << 4;
    int sg = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
    const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    const uint4* __restrict__ wt = segs.wt[sg];
    const __nv_bfloat16* __restrict__ st = segs.st[sg];
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int s = warp; s < steps; s += 8) {
      const int t = s >> 3, i = s & 7;
      const size_t rec0 = (size_t)t * n + r0, rec1 = (size_t)t * n + r1;
      const int c = 4 * i + t4, cg = t * 32 + c;
      const uint4 pad = BITS == 2 ? make_uint4(0x00AA00AAu, 0x00AA00AAu, 0x00AA00AAu, 0x00AA00AAu)
                                  : make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
      const uint4 w0 = ok0 ? ldw_mma<BITS>(wt, rec0, c) : pad, w1 = ok1 ? ldw_mma<BITS>(wt, rec1, c) : pad;
      const float sc0 = ok0 ? __bfloat162float(st[rec0 * 8 + i]) : 0.f, sc1 = ok1 ? __bfloat162float(st[rec1 * 8 + i]) : 0.f;
      const uint32_t ww0[4] = {w0.x, w0.y, w0.z, w0.w}, ww1[4] = {w1.x, w1.y, w1.z, w1.w};
      uint32_t a[8][4];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
        a[q][0] = deqb<BITS>(ww0[wd], j0), a[q][1] = deqb<BITS>(ww1[wd], j0);
        a[q][2] = deqb<BITS>(ww0[wd], j1), a[q][3] = deqb<BITS>(ww1[wd], j1);
      }
      uint4 xn[4];
      auto ldx = [&](int tt, uint4* xv) {
        const int tok = tt * 8 + g;
        const bool xin = tok < m && cg < kc;
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)tok * k + (size_t)cg * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = xin ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
      };
      ldx(0, xn);
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        const uint32_t xw[16] = {xn[0].x, xn[0].y, xn[0].z, xn[0].w, xn[1].x, xn[1].y, xn[1].z, xn[1].w,
                                 xn[2].x, xn[2].y, xn[2].z, xn[2].w, xn[3].x, xn[3].y, xn[3].z, xn[3].w};
        if (tt + 1 < NT) ldx(tt + 1, xn);
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, a[q][0], a[q][1], a[q][2], a[q][3], x_pair(xw, wd, j0), x_pair(xw, wd, j1));
        }
        acc[tt][0] = fmaf(cf[0], sc0, acc[tt][0]);
        acc[tt][1] = fmaf(cf[1], sc0, acc[tt][1]);
        acc[tt][2] = fmaf(cf[2], sc1, acc[tt][2]);
        acc[tt][3] = fmaf(cf[3], sc1, acc[tt][3]);
      }
    }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][lane][4 * tt + j] = acc[tt][j];
    __syncthreads();
    // the NT tiles' warp-order sums, tile tt by warp tt % 8 (each sum exactly w4_mma_kernel's warp 0 loop)
    for (int tt = warp; tt < NT; tt += 8) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[w][lane][4 * tt + j];
      __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = tt * 8 + 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
          if (ok1) y[(size_t)col * segs.ldy[sg] + r1] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}
// The same for group 64 (w4_mma64_kernel<8, BITS, U>'s per-group 4-MMA chain and per-step scaling)
template <int BITS, int NT>
__global__ void __launch_bounds__(256)
w4_mma64_nt_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  __shared__ float red[8][32][4 * NT];
  const int kp = (k + 1023) & ~1023, steps = (kp >> 10) * 16, kg = k >> 6;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  const int h = t4 & 1;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int g0 = rb << 4;
    int sg = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
    const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    const uint4* __restrict__ wt = segs.wt[sg];
    const __nv_bfloat16* __restrict__ st = segs.st[sg];
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    for (int s = warp; s < steps; s += 8) {
      const int t = s >> 4, i = s & 15, c = 2 * i + (t4 >> 1);
      const size_t rec0 = (size_t)t * n + r0, rec1 = (size_t)t * n + r1;
      uint32_t w0[2], w1[2];
      w0[0] = w0[1] = w1[0] = w1[1] = BITS == 2 ? 0x00AA00AAu : 0x88888888u;
      if (BITS == 2) {
        if (ok0) {
          const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec0 * rec_words<2>() + 2 * c + h);
          w0[0] = w2x(lo & 0xFFFFu), w0[1] = w2x(lo >> 16);
        }
        if (ok1) {
          const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec1 * rec_words<2>() + 2 * c + h);
          w1[0] = w2x(lo & 0xFFFFu), w1[1] = w2x(lo >> 16);
        }
      } else if (BITS == 3) {
        if (ok0) {
          const uint32_t* b = reinterpret_cast<const uint32_t*>(wt) + rec0 * 96;
          const uint32_t lo = ld_stream1(b + 2 * c + h), hi = ld_stream1(b + 64 + c) >> (16 * h);
          w0[0] = w3_word(lo, hi), w0[1] = w3_word(lo >> 16, hi >> 8);
        }
        if (ok1) {
          const uint32_t* b = reinterpret_cast<const uint32_t*>(wt) + rec1 * 96;
          const uint32_t lo = ld_stream1(b + 2 * c + h), hi = ld_stream1(b + 64 + c) >> (16 * h);
          w1[0] = w3_word(lo, hi), w1[1] = w3_word(lo >> 16, hi >> 8);
        }
      } else {
        if (ok0) {
          const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec0 * 32 + c) + h);
          w0[0] = v.x, w0[1] = v.y;
        }
        if (ok1) {
          const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec1 * 32 + c) + h);
          w1[0] = v.x, w1[1] = v.y;
        }
      }
      const float sc0 = ok0 ? __bfloat162float(st[rec0 * 16 + i]) : 0.f;
      const float sc1 = ok1 ? __bfloat162float(st[rec1 * 16 + i]) : 0.f;
      uint32_t a[4][4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
        a[q][0] = deqb<BITS>(w0[wd], j0), a[q][1] = deqb<BITS>(w1[wd], j0);
        a[q][2] = deqb<BITS>(w0[wd], j1), a[q][3] = deqb<BITS>(w1[wd], j1);
      }
      const int gg = t * 16 + i;
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        const int tok = tt * 8 + g;
        uint4 xv0 = make_uint4(0, 0, 0, 0), xv1 = make_uint4(0, 0, 0, 0);
        if (tok < m && gg < kg) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)tok * k + (size_t)gg * 64 + 16 * t4);
          xv0 = __ldg(xr), xv1 = __ldg(xr + 1);
        }
        const uint32_t xw[8] = {xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, a[q][0], a[q][1], a[q][2], a[q][3], x_pair(xw, wd, j0), x_pair(xw, wd, j1));
        }
        acc[tt][0] = fmaf(cf[0], sc0, acc[tt][0]);
        acc[tt][1] = fmaf(cf[1], sc0, acc[tt][1]);
        acc[tt][2] = fmaf(cf[2], sc1, acc[tt][2]);
        acc[tt][3] = fmaf(cf[3], sc1, acc[tt][3]);
      }
    }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][lane][4 * tt + j] = acc[tt][j];
    __syncthreads();
    for (int tt = warp; tt < NT; tt += 8) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[w][lane][4 * tt + j];
      __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = tt * 8 + 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
          if (ok1) y[(size_t)col * segs.ldy[sg] + r1] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}
// Batched verify GEMM (m > 8 tokens): block = 2 row blocks (32 rows) x one tile of 8 * NTT tokens, 8 warps = the
// 8 k-groups of w4_mma_kernel<8, 1> / w4_mma64_kernel<8, BITS, U> (warp w: group w of every 1024-input tile, g64: groups
// w and w + 8 in that order), each warp accumulating its own chain, then the warp-order sum: every output is
// bit-identical to the 8-token launches. x of the block's tokens is staged per k-tile in shared memory with cp.async
// (double-buffered, XOR-swizzled 16-byte pieces: conflict-free fragment reads), so it is read from L2 once per block
// instead of once per lane per tile; A operands are dequantized once per step for all token tiles. sm80+ (portable:
// NTT = 2 fits sm89 / sm120 shared memory, 80 KB; sm90 takes NTT = 4, 160 KB).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src),
               "r"(src_bytes) : "memory");
}
template <bool G64>
__device__ __forceinline__ int xs_swz(int tok, int p) {  // 16-byte piece p (0..127) of a token's 1024-input row
  const int f = G64 ? (tok & 1) : (((p >> 3) & 1) | ((tok & 1) << 1));
  return (p & ~7) | ((p & 7) ^ f);
}
// v4: STAGES-deep cp.async pipeline of whole k-tiles -- the block's weight records (RB row blocks x 16 rows, KT
// records are contiguous per tile), their scales, and x of the block's 8 * NTT tokens -- so no warp waits on DRAM.
// 16 warps: warp w = k-group w & 7, row-block half w >> 3 (RB / 2 row blocks each).
// weight records in shared memory: 16-byte piece p of a row stored at p ^ ((row & 1) << 2) (rows g and g + 1 of a
// fragment read then hit different banks); word w of the row is at sw_word(w, sw)
__device__ __forceinline__ int sw_word(int w, int sw) { return (((w >> 2) ^ sw) << 2) | (w & 3); }
template <int BITS>
__device__ __forceinline__ void smem_w128(const uint32_t* row, int sw, int c, uint32_t* w) {  // == ldw_mma<BITS>(.., c)
  if (BITS == 4) {
    const uint4 v = *reinterpret_cast<const uint4*>(row + sw_word(4 * c, sw));
    w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
  } else if (BITS == 2) {
    const uint2 lo = *reinterpret_cast<const uint2*>(row + sw_word(2 * c, sw));
    w[0] = w2x(lo.x & 0xFFFFu), w[1] = w2x(lo.x >> 16), w[2] = w2x(lo.y & 0xFFFFu), w[3] = w2x(lo.y >> 16);
  } else {
    const uint2 lo = *reinterpret_cast<const uint2*>(row + sw_word(2 * c, sw));
    const uint32_t hi = row[sw_word(64 + c, sw)];
    w[0] = w3_word(lo.x, hi), w[1] = w3_word(lo.x >> 16, hi >> 8), w[2] = w3_word(lo.y, hi >> 16),
    w[3] = w3_word(lo.y >> 16, hi >> 24);
  }
}
template <int BITS>
__device__ __forceinline__ void smem_w64(const uint32_t* row, int sw, int c, int h, uint32_t* w) {  // w4_mma64's words
  if (BITS == 2) {
    const uint32_t lo = row[sw_word(2 * c + h, sw)];
    w[0] = w2x(lo & 0xFFFFu), w[1] = w2x(lo >> 16);
  } else if (BITS == 3) {
    const uint32_t lo = row[sw_word(2 * c + h, sw)], hi = row[sw_word(64 + c, sw)] >> (16 * h);
    w[0] = w3_word(lo, hi), w[1] = w3_word(lo >> 16, hi >> 8);
  } else {
    const uint2 v = *reinterpret_cast<const uint2*>(row + sw_word(4 * c + 2 * h, sw));
    w[0] = v.x, w[1] = v.y;
  }
}
template <int BITS, bool G64, int RB, int NTT, int STAGES, int NW = 16>
struct GemmCfg {
  static constexpr int TOK = 8 * NTT, RBYTES = 128 * BITS, SCN = G64 ? 16 : 8;
  static constexpr int WB = RB * 16 * RBYTES, SB = RB * 16 * SCN * 2, XB = TOK * 128 * 16;
  static constexpr int STAGE = WB + SB + XB, ACC = (RB / (NW / 8)) * NTT * 4;
  static constexpr int RED = NW * 32 * ACC * 4;  // the final k-group sums reuse the stage buffers
  static constexpr int SMEM = STAGES * STAGE > RED ? STAGES * STAGE : RED;
};
template <int BITS, bool G64, int RB, int NTT, int STAGES, int NW = 16>
__global__ void __launch_bounds__(NW * 32, 1)
w4_gemm_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  using C = GemmCfg<BITS, G64, RB, NTT, STAGES, NW>;
  constexpr int RBW = RB / (NW / 8), TOK = C::TOK, NT_ = NW * 32;
  extern __shared__ __align__(16) unsigned char gsm[];
  float* red = reinterpret_cast<float*>(gsm);  // [16][32][ACC], after the k loop (aliases the stages)
  const int kp = (k + 1023) & ~1023, ntile = kp >> 10;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, grp = warp & 7, hlf = warp >> 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4, ntt = (m + TOK - 1) / TOK;
  const int item = blockIdx.x / ntt, tok0 = (blockIdx.x % ntt) * TOK;
  // per row block of the block: segment, rows in it, base row
  auto rb_info = [&](int r, int& sg, int& n, int& base) {
    const int rb = item * RB + r, g0 = (rb < nblk ? rb : nblk - 1) << 4;
    int q = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) q += i < segs.nseg && g0 >= segs.end[i - 1];
    sg = q, n = segs.n[q], base = g0 - (q ? segs.end[q - 1] : 0);
  };
  // this thread's 16-byte pieces of a stage, resolved once: a source pointer advanced by its tile stride per stage
  // (out-of-range pieces copy 0 bytes from a valid address: zero fill), so a stage is straight-line cp.async
  constexpr int WP = C::WB / 16, SP = C::SB / 16, XP = C::XB / 16;
  static_assert(XP % NT_ == 0, "stage pieces");
  constexpr int NWP = (WP + NT_ - 1) / NT_, NXP = XP / NT_;
  const unsigned char* wsrc[NWP];
  int wstr[NWP], wbytes[NWP], wdst[NWP];
#pragma unroll
  for (int j = 0; j < NWP; ++j) {
    const int P0 = threadIdx.x + NT_ * j, P = P0 < WP ? P0 : 0, r = P / C::RBYTES, rem = P % C::RBYTES;
    int sg, n, base;
    rb_info(r, sg, n, base);
    const int row = base + rem / (C::RBYTES / 16);
    const bool in = P0 < WP && item * RB + r < nblk && row < n;
    wsrc[j] = reinterpret_cast<const unsigned char*>(segs.wt[sg]) + (size_t)(in ? row : 0) * C::RBYTES + (rem % (C::RBYTES / 16)) * 16;
    wstr[j] = in ? n * C::RBYTES : 0, wbytes[j] = in ? 16 : 0;
    const int rowl = rem / (C::RBYTES / 16), pc = rem % (C::RBYTES / 16);
    wdst[j] = ((r * 16 + rowl) * (C::RBYTES / 16) + (pc ^ ((rowl & 1) << 2))) * 16;
  }
  const unsigned char* ssrc = nullptr;
  int sstr = 0, sbytes = 0;
  if (threadIdx.x < SP) {
    const int Q = threadIdx.x, r = Q / (C::SCN * 2), rem = Q % (C::SCN * 2);
    int sg, n, base;
    rb_info(r, sg, n, base);
    const int row = base + rem / (C::SCN * 2 / 16);
    const bool in = item * RB + r < nblk && row < n;
    ssrc = reinterpret_cast<const unsigned char*>(segs.st[sg]) + (size_t)(in ? row : 0) * C::SCN * 2 + (rem % (C::SCN * 2 / 16)) * 16;
    sstr = in ? n * C::SCN * 2 : 0, sbytes = in ? 16 : 0;
  }
  const unsigned char* xsrc[NXP];
  int xdst[NXP], xcol[NXP], xbytes[NXP];
#pragma unroll
  for (int j = 0; j < NXP; ++j) {
    const int Q = threadIdx.x + NT_ * j, tok = Q >> 7, p = Q & 127, xr = tok0 + tok;
    xsrc[j] = reinterpret_cast<const unsigned char*>(x + (size_t)(xr < m ? xr : 0) * k + p * 8);
    xdst[j] = (tok * 128 + xs_swz<G64>(tok, p)) * 16, xcol[j] = p * 8, xbytes[j] = xr < m ? 16 : 0;
  }
  auto stage = [&](int t, int sl) {
    unsigned char* S0 = gsm + (size_t)sl * C::STAGE;
    if (t < ntile) {
#pragma unroll
      for (int j = 0; j < NWP; ++j)
        if (WP % NT_ == 0 || threadIdx.x + NT_ * j < WP)
          cp_async16(S0 + wdst[j], wsrc[j] + (size_t)t * wstr[j], wbytes[j]);
      if (threadIdx.x < SP) cp_async16(S0 + C::WB + (size_t)threadIdx.x * 16, ssrc + (size_t)t * sstr, sbytes);
      const int tc = t * 1024;
#pragma unroll
      for (int j = 0; j < NXP; ++j) {
        const bool in = tc + xcol[j] < k;
        cp_async16(S0 + C::WB + C::SB + xdst[j], xsrc[j] + (in ? (size_t)tc * 2 : 0), in ? xbytes[j] : 0);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  float acc[RBW][NTT][4];
#pragma unroll
  for (int r = 0; r < RBW; ++r)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt) acc[r][tt][0] = acc[r][tt][1] = acc[r][tt][2] = acc[r][tt][3] = 0.f;
#pragma unroll
  for (int sl = 0; sl < STAGES - 1; ++sl) stage(sl, sl);
  for (int t = 0; t < ntile; ++t) {
    stage(t + STAGES - 1, (t + STAGES - 1) % STAGES);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1) : "memory");
    __syncthreads();
    const unsigned char* S0 = gsm + (size_t)(t % STAGES) * C::STAGE;
    const uint32_t* Wsm = reinterpret_cast<const uint32_t*>(S0);
    const __nv_bfloat16* Ssm = reinterpret_cast<const __nv_bfloat16*>(S0 + C::WB);
    const uint4* xb = reinterpret_cast<const uint4*>(S0 + C::WB + C::SB);
#pragma unroll
    for (int half = 0; half < (G64 ? 2 : 1); ++half) {  // g64: groups w, then w + 8 (w4_mma64_kernel's order)
      const int i = grp + 8 * half;
      constexpr int NQ = G64 ? 4 : 8;
      uint32_t bq[NTT][NQ][2];
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt) {
        const int tok = tt * 8 + g;
        if (!G64) {
          uint32_t xw[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            const uint4 v = xb[tok * 128 + xs_swz<false>(tok, i * 16 + t4 * 4 + q4)];
            xw[4 * q4] = v.x, xw[4 * q4 + 1] = v.y, xw[4 * q4 + 2] = v.z, xw[4 * q4 + 3] = v.w;
          }
#pragma unroll
          for (int q = 0; q < NQ; ++q) {
            const int wd = q >> 1, j0 = 2 * (q & 1);
            bq[tt][q][0] = x_pair(xw, wd, j0), bq[tt][q][1] = x_pair(xw, wd, j0 + 1);
          }
        } else {
          const int p = 8 * i + 2 * t4;
          const uint4 v0 = xb[tok * 128 + xs_swz<true>(tok, p)], v1 = xb[tok * 128 + xs_swz<true>(tok, p + 1)];
          const uint32_t xw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
          for (int q = 0; q < NQ; ++q) {
            const int wd = q >> 1, j0 = 2 * (q & 1);
            bq[tt][q][0] = x_pair(xw, wd, j0), bq[tt][q][1] = x_pair(xw, wd, j0 + 1);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < RBW; ++r) {
        const int rl = hlf * RBW + r;  // row block within the block
        const uint32_t* row0 = Wsm + (size_t)(rl * 16 + g) * (C::RBYTES / 4);
        const uint32_t* row1 = row0 + 8 * (C::RBYTES / 4);
        uint32_t w0[4], w1[4];
        float sa, sb;
        if (!G64) {
          const int c = 4 * i + t4;
          smem_w128<BITS>(row0, (g & 1) << 2, c, w0), smem_w128<BITS>(row1, (g & 1) << 2, c, w1);
          sa = __bfloat162float(Ssm[(rl * 16 + g) * 8 + i]), sb = __bfloat162float(Ssm[(rl * 16 + g + 8) * 8 + i]);
        } else {
          const int c = 2 * i + (t4 >> 1), h = t4 & 1;
          smem_w64<BITS>(row0, (g & 1) << 2, c, h, w0), smem_w64<BITS>(row1, (g & 1) << 2, c, h, w1);
          sa = __bfloat162float(Ssm[(rl * 16 + g) * 16 + i]), sb = __bfloat162float(Ssm[(rl * 16 + g + 8) * 16 + i]);
        }
        uint32_t a[NQ][4];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          a[q][0] = deqb<BITS>(w0[wd], j0), a[q][1] = deqb<BITS>(w1[wd], j0);
          a[q][2] = deqb<BITS>(w0[wd], j1), a[q][3] = deqb<BITS>(w1[wd], j1);
        }
#pragma unroll
        for (int tt = 0; tt < NTT; ++tt) {
          float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int q = 0; q < NQ; ++q) mma16816(cf, a[q][0], a[q][1], a[q][2], a[q][3], bq[tt][q][0], bq[tt][q][1]);
          acc[r][tt][0] = fmaf(cf[0], sa, acc[r][tt][0]);
          acc[r][tt][1] = fmaf(cf[1], sa, acc[r][tt][1]);
          acc[r][tt][2] = fmaf(cf[2], sb, acc[r][tt][2]);
          acc[r][tt][3] = fmaf(cf[3], sb, acc[r][tt][3]);
        }
      }
    }
    __syncthreads();  // slot t % STAGES is restaged by the next iteration
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  // padding rows (beyond n) were zero-filled: their sums are never stored
#pragma unroll
  for (int r = 0; r < RBW; ++r)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[((size_t)warp * 32 + lane) * C::ACC + (r * NTT + tt) * 4 + j] = acc[r][tt][j];
  __syncthreads();
  for (int pr = warp; pr < RB * NTT; pr += NW) {  // (row block, token tile): the k-group-order sum
    const int rl = pr / NTT, tt = pr % NTT, h2 = rl / RBW, r = rl % RBW;
    if (item * RB + rl >= nblk) continue;
    int sg, n, base;
    rb_info(rl, sg, n, base);
    const int q0 = base + g;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < 8; ++w)
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] += red[((size_t)(h2 * 8 + w) * 32 + lane) * C::ACC + (r * NTT + tt) * 4 + j];
    __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int col = tok0 + tt * 8 + 2 * t4 + cc;
      if (col < m) {
        if (q0 < n) y[(size_t)col * segs.ldy[sg] + q0] = __float2bfloat16_rn(v[cc]);
        if (q0 + 8 < n) y[(size_t)col * segs.ldy[sg] + q0 + 8] = __float2bfloat16_rn(v[2 + cc]);
      }
    }
  }
}
template <int BITS, bool G64, int RB, int NTT, int STAGES, int NW = 16>
int launch_gemm(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  using C = GemmCfg<BITS, G64, RB, NTT, STAGES, NW>;
  auto kern = w4_gemm_kernel<BITS, G64, RB, NTT, STAGES, NW>;
  static int ok = -1;
  if (ok < 0) ok = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess;
  if (!ok) return -1;
  const int nblk = segs.end[segs.nseg - 1] >> 4, ntt = (m + C::TOK - 1) / C::TOK;
  kern<<<((nblk + RB - 1) / RB) * ntt, NW * 32, C::SMEM, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k);
  return (int)cudaGetLastError();
}
// AGENS_W4_GEMM=0: the NT register kernel instead (x from L1 per token tile). Default on sm90: 16-token tiles
// (RB 2, 8 warps) up to 16 rows (sm90 W4 g128 m = 16: gate_up 58 us vs 100 with 32-token tiles), 32-token tiles above
int gemm_ntt(int m) {
  static int v = -2, major = 0;
  if (v == -2) {
    const char* e = getenv("AGENS_W4_GEMM");
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    v = e && e[0] == '0' ? 0 : e ? atoi(e) : -1;
  }
  if (v >= 0) return v;
  return major == 9 ? (m <= 16 ? 8222 : 8442) : 222;
}
template <bool G64, int BITS, int NT>
int launch_nt(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  auto kern = G64 ? w4_mma64_nt_kernel<BITS, NT> : w4_mma_nt_kernel<BITS, NT>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, 0);
    if (per_sm < 1) per_sm = 1;
  }
  const int want = segs.end[segs.nseg - 1] >> 4;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k);
  return (int)cudaGetLastError();
}
// Small n, m > 8 (batched verify of the narrow projections: KDA gate / beta, indexer, f_down; n <= ~4K rows): few
// row blocks leave the GEMM with a handful of blocks that each stage all of x. Group split instead: block = (8 row
// blocks, k-group slot w, 8 * NTT tokens); x of the block's group slot for every k-tile is staged once; warp r runs
// row block 8 item + r of slot w (w4_mma_kernel<8, 1> / w4_mma64_kernel's chain: groups w, w + 8 for g64) and stores
// its sums; the item's last block to finish adds them in group order (from 0): bit-identical to the 8-token launches.
template <int BITS, bool G64, int NTT>
__global__ void __launch_bounds__(256)
w4_gsnt_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k,
               float* __restrict__ part, unsigned* __restrict__ cnt) {
  constexpr int TOK = 8 * NTT;
  extern __shared__ __align__(16) uint4 gxs[];  // [TOK][ntile][16] pieces of this slot's inputs
  __shared__ int last_s;
  const int kp = (k + 1023) & ~1023, ntile = kp >> 10;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4, ntt = (m + TOK - 1) / TOK;
  const int w = blockIdx.x & 7, it_tt = blockIdx.x >> 3, tt0 = it_tt % ntt, item = it_tt / ntt, tok0 = tt0 * TOK;
  const int rb = item * 8 + warp;
  const bool live = rb < nblk;
  // stage this slot's x: token tok, tile t, piece p (16 B = 8 inputs): g128 inputs t*1024 + w*128 + 8p; g64: groups
  // w (p < 8) and w + 8 (p >= 8), inputs t*1024 + (w + 8 (p >> 3)) * 64 + 8 (p & 7)
  for (int P = threadIdx.x; P < TOK * ntile * 16; P += 256) {
    const int tok = P / (ntile * 16), rem = P % (ntile * 16), t = rem >> 4, p = rem & 15;
    const int col = G64 ? t * 1024 + (w + 8 * (p >> 3)) * 64 + 8 * (p & 7) : t * 1024 + w * 128 + 8 * p;
    const int row = tok0 + tok;
    const bool in = row < m && col < k;
    cp_async16(gxs + ((size_t)tok * ntile + t) * 16 + xs_swz<G64>(tok, p), x + (size_t)(in ? row : 0) * k + (in ? col : 0),
               in ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  int sg = 0;
  const int g0 = (live ? rb : nblk - 1) << 4;
#pragma unroll
  for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
  const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g;
  const bool ok0 = live && r0 < n, ok1 = live && r0 + 8 < n;
  const uint4* __restrict__ wt = segs.wt[sg];
  const __nv_bfloat16* __restrict__ st = segs.st[sg];
  const uint32_t padw = BITS == 2 ? 0x00AA00AAu : 0x88888888u;
  float acc[NTT][4];
#pragma unroll
  for (int tt = 0; tt < NTT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  for (int t = 0; t < ntile; ++t) {
#pragma unroll
    for (int half = 0; half < (G64 ? 2 : 1); ++half) {
      const int i = w + 8 * half;
      constexpr int NQ = G64 ? 4 : 8;
      uint32_t w0[4], w1[4];
      float sa, sb;
      const size_t rec0 = (size_t)t * n + r0, rec1 = rec0 + 8;
      if (!G64) {
        const int c = 4 * i + t4;
        const uint4 pad = make_uint4(padw, padw, padw, padw);
        const uint4 a0 = ok0 ? ldw_mma<BITS>(wt, rec0, c) : pad, a1 = ok1 ? ldw_mma<BITS>(wt, rec1, c) : pad;
        w0[0] = a0.x, w0[1] = a0.y, w0[2] = a0.z, w0[3] = a0.w, w1[0] = a1.x, w1[1] = a1.y, w1[2] = a1.z, w1[3] = a1.w;
        sa = ok0 ? __bfloat162float(st[rec0 * 8 + i]) : 0.f, sb = ok1 ? __bfloat162float(st[rec1 * 8 + i]) : 0.f;
      } else {
        const int c = 2 * i + (t4 >> 1), h = t4 & 1;
        w0[0] = w0[1] = w1[0] = w1[1] = padw;
        if (BITS == 2) {
          if (ok0) { const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec0 * rec_words<2>() + 2 * c + h); w0[0] = w2x(lo & 0xFFFFu), w0[1] = w2x(lo >> 16); }
          if (ok1) { const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec1 * rec_words<2>() + 2 * c + h); w1[0] = w2x(lo & 0xFFFFu), w1[1] = w2x(lo >> 16); }
        } else if (BITS == 3) {
          if (ok0) { const uint32_t* bp = reinterpret_cast<const uint32_t*>(wt) + rec0 * 96; const uint32_t lo = ld_stream1(bp + 2 * c + h), hi = ld_stream1(bp + 64 + c) >> (16 * h); w0[0] = w3_word(lo, hi), w0[1] = w3_word(lo >> 16, hi >> 8); }
          if (ok1) { const uint32_t* bp = reinterpret_cast<const uint32_t*>(wt) + rec1 * 96; const uint32_t lo = ld_stream1(bp + 2 * c + h), hi = ld_stream1(bp + 64 + c) >> (16 * h); w1[0] = w3_word(lo, hi), w1[1] = w3_word(lo >> 16, hi >> 8); }
        } else {
          if (ok0) { const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec0 * 32 + c) + h); w0[0] = v.x, w0[1] = v.y; }
          if (ok1) { const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec1 * 32 + c) + h); w1[0] = v.x, w1[1] = v.y; }
        }
        sa = ok0 ? __bfloat162float(st[rec0 * 16 + i]) : 0.f, sb = ok1 ? __bfloat162float(st[rec1 * 16 + i]) : 0.f;
      }
      uint32_t a[NQ][4];
#pragma unroll
      for (int q = 0; q < NQ; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
        a[q][0] = deqb<BITS>(w0[wd], j0), a[q][1] = deqb<BITS>(w1[wd], j0);
        a[q][2] = deqb<BITS>(w0[wd], j1), a[q][3] = deqb<BITS>(w1[wd], j1);
      }
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt) {
        const int tok = tt * 8 + g;
        const uint4* xr = gxs + ((size_t)tok * ntile + t) * 16;
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
        if (!G64) {
          uint32_t xw[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            const uint4 v = xr[xs_swz<false>(tok, t4 * 4 + q4)];
            xw[4 * q4] = v.x, xw[4 * q4 + 1] = v.y, xw[4 * q4 + 2] = v.z, xw[4 * q4 + 3] = v.w;
          }
#pragma unroll
          for (int q = 0; q < NQ; ++q) {
            const int wd = q >> 1, j0 = 2 * (q & 1);
            mma16816(cf, a[q][0], a[q][1], a[q][2], a[q][3], x_pair(xw, wd, j0), x_pair(xw, wd, j0 + 1));
          }
        } else {
          const int p = 8 * half + 2 * t4;
          const uint4 v0 = xr[xs_swz<true>(tok, p)], v1 = xr[xs_swz<true>(tok, p + 1)];
          const uint32_t xw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
          for (int q = 0; q < NQ; ++q) {
            const int wd = q >> 1, j0 = 2 * (q & 1);
            mma16816(cf, a[q][0], a[q][1], a[q][2], a[q][3], x_pair(xw, wd, j0), x_pair(xw, wd, j0 + 1));
          }
        }
        acc[tt][0] = fmaf(cf[0], sa, acc[tt][0]);
        acc[tt][1] = fmaf(cf[1], sa, acc[tt][1]);
        acc[tt][2] = fmaf(cf[2], sb, acc[tt][2]);
        acc[tt][3] = fmaf(cf[3], sb, acc[tt][3]);
      }
    }
  }
  // this (row block, slot, token tile)'s sums: part[((rb * 8 + w) * ntt + tt0)][lane][NTT][4]
  if (live)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt)
      reinterpret_cast<float4*>(part)[(((size_t)rb * 8 + w) * ntt + tt0) * 32 * NTT + lane * NTT + tt] =
          make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_s = (atomicAdd(&cnt[it_tt], 1u) & 7u) == 7u;
  __syncthreads();
  if (!last_s || !live) return;
  __threadfence();
  float v[NTT][4];
#pragma unroll
  for (int tt = 0; tt < NTT; ++tt) v[tt][0] = v[tt][1] = v[tt][2] = v[tt][3] = 0.f;
  for (int gi = 0; gi < 8; ++gi)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt) {
      const float4 q = __ldcg(reinterpret_cast<const float4*>(part) + (((size_t)rb * 8 + gi) * ntt + tt0) * 32 * NTT +
                              lane * NTT + tt);
      v[tt][0] += q.x, v[tt][1] += q.y, v[tt][2] += q.z, v[tt][3] += q.w;
    }
  __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
  for (int tt = 0; tt < NTT; ++tt)
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int col = tok0 + tt * 8 + 2 * t4 + cc;
      if (col < m) {
        if (ok0) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[tt][cc]);
        if (ok1) y[(size_t)col * segs.ldy[sg] + r0 + 8] = __float2bfloat16_rn(v[tt][2 + cc]);
      }
    }
}
// scratch per (device, stream): sums for up to kGsBlk row blocks x 8 slots x 128 tokens, counters
constexpr int kGsBlk = 800;
struct GsnScratch {
  float* part = nullptr;
  unsigned* cnt = nullptr;
  int max_blk = 0;
};
int launch_gsnt_any(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s, int bits, bool g64) {
  static std::map<std::pair<int, cudaStream_t>, GsnScratch> scr;
  int dev = 0;
  cudaGetDevice(&dev);
  GsnScratch& z = scr[{dev, s}];
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  if (nblk > kGsBlk) return -1;
  if (!z.part) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &cs);
    if (cs != cudaStreamCaptureStatusNone) return -1;
    // sums [rb][8 slots][tokens][16 rows] fp32 for up to 8 x AGENS_SPEC_BATCH_MAX tokens (default 128)
    const char* eb = getenv("AGENS_SPEC_BATCH_MAX");
    const size_t mmax = eb ? (size_t)std::max(2, std::min(16, atoi(eb))) * 8 : 128;
    if (cudaMalloc(&z.part, (size_t)kGsBlk * 8 * mmax * 16 * 4) != cudaSuccess) return -1;
    z.max_blk = (int)mmax;  // tokens the sums hold
    if (cudaMalloc(&z.cnt, (size_t)(kGsBlk / 8 + 1) * 16 * 4) != cudaSuccess) return -1;
    cudaMemset(z.cnt, 0, (size_t)(kGsBlk / 8 + 1) * 16 * 4);
    cudaDeviceSynchronize();
  }
  if (m > z.max_blk) return -1;
  const int ntile = ((k + 1023) & ~1023) >> 10;
  const int NTT = m > 16 && (size_t)32 * ntile * 256 <= 96 * 1024 ? 4 : 2;  // tokens per block: x of a slot <= 96 KB
  const int tok = 8 * NTT, ntt = (m + tok - 1) / tok, items = (nblk + 7) / 8;
  const size_t smem = (size_t)tok * ntile * 256;
  // the device's opt-in shared memory per block (sm90 227 KB; sm89 / sm120 99 KB): larger x stages fall back
  static std::map<int, int> optin;
  if (!optin.count(dev)) {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    optin[dev] = std::min(v, 200 * 1024);
  }
  const int smax = optin[dev];
  if (smem > (size_t)smax) return -1;
  auto go = [&](auto kern) {
    // dynamic shared memory up to the opt-in limit less the kernel's static shared memory
    static std::map<std::pair<const void*, int>, int> dmax;
    auto it = dmax.find({(const void*)kern, dev});
    if (it == dmax.end()) {
      cudaFuncAttributes fa;
      int lim = 48 * 1024;
      if (cudaFuncGetAttributes(&fa, kern) == cudaSuccess) {
        const int want = smax - (int)fa.sharedSizeBytes;
        if (want > lim && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, want) == cudaSuccess)
          lim = want;
        else if (want > lim && getenv("AGENS_W4_DEBUG"))
          fprintf(stderr, "gsnt: attribute %d: %s\n", want, cudaGetErrorString(cudaGetLastError()));
      }
      it = dmax.emplace(std::make_pair((const void*)kern, dev), lim).first;
    }
    if (smem > (size_t)it->second) return -1;
    kern<<<items * ntt * 8, 256, smem, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k, z.part, z.cnt);
    const int e = (int)cudaGetLastError();
    if (e && getenv("AGENS_W4_DEBUG")) {
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      fprintf(stderr, "gsnt: launch %d blocks smem %zu m %d k %d nblk %d: %s (max dyn %d, regs %d, max threads %d, lim %d)\n",
              items * ntt * 8, smem, m, k, nblk, cudaGetErrorString((cudaError_t)e), fa.maxDynamicSharedSizeBytes,
              fa.numRegs, fa.maxThreadsPerBlock, it->second);
    }
    return e;
  };
#define GS_CASE(B, G)                                                     \
  if (bits == B && g64 == G) return NTT == 4 ? go(w4_gsnt_kernel<B, G, 4>) : go(w4_gsnt_kernel<B, G, 2>);
  GS_CASE(4, false) GS_CASE(3, false) GS_CASE(2, false) GS_CASE(4, true) GS_CASE(3, true) GS_CASE(2, true)
#undef GS_CASE
  return -1;
}
// AGENS_W4_GEMM8 = RB * 100 + 10 + STAGES: the pipelined GEMM for m <= 8 too (experiment; 0 = off)
int gemm8_cfg() {
  static const int v = [] { const char* e = getenv("AGENS_W4_GEMM8"); return e ? atoi(e) : 0; }();
  return v;
}
template <bool G64, int BITS>
int gemm8(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  switch (gemm8_cfg()) {
    case 413: return launch_gemm<BITS, G64, 4, 1, 3>(x, m, g, k, s);
    case 414: return launch_gemm<BITS, G64, 4, 1, 4>(x, m, g, k, s);
    case 813: return launch_gemm<BITS, G64, 8, 1, 3>(x, m, g, k, s);
    case 213: return launch_gemm<BITS, G64, 2, 1, 3>(x, m, g, k, s);
    default: return -1;
  }
}
// m > 8 tokens: launches of up to 64 (8 tiles), each one pass over the weights (AGENS_W4_NT=0: 8-token launches)
template <bool G64, int BITS>
int nt_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  // few row blocks (n < 2 x SMs x 16 rows): the group-split kernel (AGENS_W4_GSNT=0: off)
  static const bool gs_on = [] { const char* e = getenv("AGENS_W4_GSNT"); return !(e && e[0] == '0'); }();
  // (also long k with < 6 x SMs row blocks: the MLP down projection, 320 row blocks x 17 k-tiles; sm90 m = 32:
  // 48 us vs 52 for the GEMM)
  const int nb_ = g.end[g.nseg - 1] >> 4;
  if (gs_on && (nb_ < 2 * num_sms() || (k >= 12288 && nb_ < 6 * num_sms()))) {
    const int e = launch_gsnt_any(x, m, g, k, s, BITS, G64);
    if (e >= 0) return e;
  }
  // AGENS_W4_GEMM = RB * 100 + NTT * 10 + STAGES (0: off); -1 from a launch = shared memory too small -> fall back
  int e = -1;
  switch (gemm_ntt(m)) {
    case 0: break;
    case 443: e = launch_gemm<BITS, G64, 4, 4, 3>(x, m, g, k, s); break;
    case 8442: e = launch_gemm<BITS, G64, 4, 4, 2, 8>(x, m, g, k, s); break;
    case 8423: e = launch_gemm<BITS, G64, 4, 2, 3, 8>(x, m, g, k, s); break;
    case 8822: e = launch_gemm<BITS, G64, 8, 2, 2, 8>(x, m, g, k, s); break;
    case 8222: e = launch_gemm<BITS, G64, 2, 2, 2, 8>(x, m, g, k, s); break;
    case 442: e = launch_gemm<BITS, G64, 4, 4, 2>(x, m, g, k, s); break;
    case 423: e = launch_gemm<BITS, G64, 4, 2, 3>(x, m, g, k, s); break;
    case 422: e = launch_gemm<BITS, G64, 4, 2, 2>(x, m, g, k, s); break;
    case 843: e = launch_gemm<BITS, G64, 8, 4, 3>(x, m, g, k, s); break;
    case 822: e = launch_gemm<BITS, G64, 8, 2, 2>(x, m, g, k, s); break;
    case 242: e = launch_gemm<BITS, G64, 2, 4, 2>(x, m, g, k, s); break;
    default: e = launch_gemm<BITS, G64, 2, 2, 2>(x, m, g, k, s); break;
  }
  if (e >= 0) return e;
  for (int m0 = 0; m0 < m; m0 += 64) {
    KtSegs gg = g;
    if (m0)
      for (int i = 0; i < g.nseg; ++i) gg.y[i] = g.y[i] + (size_t)m0 * g.ldy[i];
    const int mm = m - m0 < 64 ? m - m0 : 64, nt = (mm + 7) / 8;
    const void* xx = static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k;
    const int e = nt <= 2 ? launch_nt<G64, BITS, 2>(xx, mm, gg, k, s)
                : nt <= 4 ? launch_nt<G64, BITS, 4>(xx, mm, gg, k, s)
                          : launch_nt<G64, BITS, 8>(xx, mm, gg, k, s);
    if (e) return e;
  }
  return 0;
}
bool nt_on() {
  static const int on = [] { const char* e = getenv("AGENS_W4_NT"); return !(e && e[0] == '0'); }();
  return on;
}
template <int WARPS, int U, int BITS = 4>
int launch_mma(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  auto kern = w4_mma_kernel<WARPS, U, BITS>;
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, WARPS * 32, 0);
  if (per_sm < 1) per_sm = 1;
  const int want = segs.end[segs.nseg - 1] >> 4;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, WARPS * 32, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k);
  return (int)cudaGetLastError();
}
// Lean register path (W4 g128, every segment's n a multiple of 16, k a multiple of 1024): w4_mma_kernel<8, 1>'s MMA
// chain and sum order (warp w takes group w of every k-tile in tile order; warp-order sum), so bit-identical to it,
// without its per-step predicates and padding moves (all loads in bounds; the x row of a column >= m is row 0, whose
// MMA column is never stored) and with RB row blocks per work item sharing the warp's x fragment of a tile (x is 2 KB
// of L1 / L2 reads per 1 KB of weights at 8 tokens; ncu on sm90: L1 82% busy, long-scoreboard stalls).
template <int RB>
__global__ void __launch_bounds__(256)
w4_lean_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  __shared__ float red[8][32][4 * RB];
  const int ntile = k >> 10;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3, i = warp;
  const int c = 4 * i + t4;
  const int nblk = segs.end[segs.nseg - 1] >> 4, nitem = (nblk + RB - 1) / RB;
  const __nv_bfloat162 c136 = __floats2bfloat162_rn(136.f, 136.f);
  auto deq = [&](uint32_t w, int j) -> uint32_t {
    uint32_t v;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
    const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), c136);
    return *reinterpret_cast<const uint32_t*>(&r);
  };
  const uint4* __restrict__ xr0 = reinterpret_cast<const uint4*>(x + (size_t)(g < m ? g : 0) * k + (size_t)c * 32);
  for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
    const uint4* wp[RB];
    const __nv_bfloat16* sp[RB];
    int nn[RB];
    bool live[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int b = it * RB + r < nblk ? it * RB + r : nblk - 1, g0 = b << 4;
      int sg = 0;
#pragma unroll
      for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && g0 >= segs.end[j - 1];
      const int r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g;
      nn[r] = segs.n[sg], live[r] = it * RB + r < nblk;
      wp[r] = segs.wt[sg] + (size_t)r0 * 32 + c;  // tile t, row r0: + t * n * 32; row r0 + 8: + 8 * 32
      sp[r] = segs.st[sg] + (size_t)r0 * 8 + i;
    }
    float acc[RB][4];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;
    for (int t = 0; t < ntile; ++t) {
      uint4 xv[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) xv[q] = __ldg(xr0 + (size_t)t * 128 + q);
      uint4 w0[RB], w1[RB];
      float s0[RB], s1[RB];
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const size_t o = (size_t)t * nn[r];
        w0[r] = ld_stream(wp[r] + o * 32);
        w1[r] = ld_stream(wp[r] + o * 32 + 8 * 32);
        s0[r] = __bfloat162float(sp[r][o * 8]);
        s1[r] = __bfloat162float(sp[r][o * 8 + 64]);
      }
      const uint32_t xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                               xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
      uint32_t xb[16];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1);
        xb[2 * q] = x_pair(xw, wd, j0), xb[2 * q + 1] = x_pair(xw, wd, j0 + 1);
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint32_t ww0[4] = {w0[r].x, w0[r].y, w0[r].z, w0[r].w}, ww1[4] = {w1[r].x, w1[r].y, w1[r].z, w1[r].w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, deq(ww0[wd], j0), deq(ww1[wd], j0), deq(ww0[wd], j1), deq(ww1[wd], j1), xb[2 * q], xb[2 * q + 1]);
        }
        acc[r][0] = fmaf(cf[0], s0[r], acc[r][0]);
        acc[r][1] = fmaf(cf[1], s0[r], acc[r][1]);
        acc[r][2] = fmaf(cf[2], s1[r], acc[r][2]);
        acc[r][3] = fmaf(cf[3], s1[r], acc[r][3]);
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][lane][4 * r + j] = acc[r][j];
    __syncthreads();
    if (warp < RB && live[warp]) {  // warp r writes row block r: w4_mma_kernel's warp-order sum
      const int r = warp, g0 = (it * RB + r) << 4;
      int sg = 0;
#pragma unroll
      for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && g0 >= segs.end[j - 1];
      const int q0 = g0 - (sg ? segs.end[sg - 1] : 0) + g;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int w = 0; w < 8; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[w][lane][4 * r + j];
      __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = 2 * t4 + cc;
        if (col < m) {
          y[(size_t)col * segs.ldy[sg] + q0] = __float2bfloat16_rn(v[cc]);
          y[(size_t)col * segs.ldy[sg] + q0 + 8] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}
// Group-split lean path (sm90 default for >= SMs row blocks): block (item of 8 row blocks, group i) -- warp w takes row
// block 8 item + w, group i of every k-tile in tile order, exactly warp i's chain in w4_mma_kernel<8, 1> -- and the
// group sums of a row block are added in group order (0 .. 7, from 0) by the item's last block to finish (atomic
// count), as w4_mma_kernel's warp-order sum: bit-identical. The block's 8 warps share one x fragment per tile (the same
// group), staged once per tile in shared memory already in MMA B order (x_pair applied), so x costs 1 / 8 of the L1
// traffic of the register path, where x reads were ~90% of its L1 wavefronts (L1 85% busy on sm90).
// Scratch per (device, stream): group sums [rb][8][32][4] fp32 and a counter per item.
struct GsScratch {
  float* part = nullptr;
  unsigned* cnt = nullptr;
  int max_blk = 0;
};
__global__ void __launch_bounds__(256)
w4_gsplit_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k,
                 float* __restrict__ part, unsigned* __restrict__ cnt) {
  __shared__ uint4 xs[2][4][8][4];  // [buf][q4][token g][t4]: lane (g, t4)'s B words 4 q4 .. 4 q4 + 3 of the tile
  __shared__ int last_s;
  const int ntile = k >> 10;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int item = blockIdx.x >> 3, i = blockIdx.x & 7, c = 4 * i + t4;
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  const int rb = item * 8 + warp;
  const bool live = rb < nblk;
  const __nv_bfloat162 c136 = __floats2bfloat162_rn(136.f, 136.f);
  auto deq = [&](uint32_t w, int j) -> uint32_t {
    uint32_t v;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
    const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), c136);
    return *reinterpret_cast<const uint32_t*>(&r);
  };
  // staging: thread s < 128 loads 16-byte piece p = s % 16 (q4 = p / 4 ... see below) of token s / 16's 256 bytes of
  // group i; piece p of token tg = inputs 8p .. 8p + 7 of the group = lane t4 = p / 4's words 4 (p % 4) .. + 3 (xw)
  const int st_tok = threadIdx.x >> 4, st_p = threadIdx.x & 15, st_t4 = st_p >> 2, st_wd = st_p & 3;
  const uint4* xsrc = reinterpret_cast<const uint4*>(x + (size_t)(st_tok < m ? st_tok : 0) * k + i * 128 + st_p * 8);
  auto stage = [&](int t, int buf, uint4 v) {
    // x_pair(xw, wd, j) for j = 0..3 from piece wd's 4 words: B words 2q, 2q + 1 of q = 2 wd, 2 wd + 1
    const uint32_t a0 = v.x, a1 = v.y, b0 = v.z, b1 = v.w;  // xw[4wd + 0..3]
    uint4 o;
    o.x = __byte_perm(a0, b0, 0x5410), o.y = __byte_perm(a0, b0, 0x7632);
    o.z = __byte_perm(a1, b1, 0x5410), o.w = __byte_perm(a1, b1, 0x7632);
    xs[buf][st_wd][st_tok][st_t4] = o;
  };
  uint4 xn = make_uint4(0, 0, 0, 0);
  if (threadIdx.x < 128) xn = __ldg(xsrc);
  int sg = 0, n = 0, r0 = 0;
  {
    const int g0 = (live ? rb : nblk - 1) << 4;
#pragma unroll
    for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && g0 >= segs.end[j - 1];
    n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g;
  }
  const uint4* wp = segs.wt[sg] + (size_t)r0 * 32 + c;
  const __nv_bfloat16* sp = segs.st[sg] + (size_t)r0 * 8 + i;
  uint4 w0 = ld_stream(wp), w1 = ld_stream(wp + 8 * 32);
  float s0 = __bfloat162float(sp[0]), s1 = __bfloat162float(sp[64]);
  if (threadIdx.x < 128) stage(0, 0, xn);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int t = 0; t < ntile; ++t) {
    const int buf = t & 1;
    __syncthreads();  // tile t staged; the other buffer's readers (tile t - 1) are done
    uint4 nw0 = w0, nw1 = w1;
    float ns0 = s0, ns1 = s1;
    if (t + 1 < ntile) {
      const size_t o = (size_t)(t + 1) * n;
      nw0 = ld_stream(wp + o * 32), nw1 = ld_stream(wp + o * 32 + 8 * 32);
      ns0 = __bfloat162float(sp[o * 8]), ns1 = __bfloat162float(sp[o * 8 + 64]);
      if (threadIdx.x < 128) xn = __ldg(xsrc + (size_t)(t + 1) * 128);
    }
    uint32_t xb[16];
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const uint4 v = xs[buf][q4][g][t4];
      xb[4 * q4] = v.x, xb[4 * q4 + 1] = v.y, xb[4 * q4 + 2] = v.z, xb[4 * q4 + 3] = v.w;
    }
    const uint32_t ww0[4] = {w0.x, w0.y, w0.z, w0.w}, ww1[4] = {w1.x, w1.y, w1.z, w1.w};
    float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
      mma16816(cf, deq(ww0[wd], j0), deq(ww1[wd], j0), deq(ww0[wd], j1), deq(ww1[wd], j1), xb[2 * q], xb[2 * q + 1]);
    }
    acc[0] = fmaf(cf[0], s0, acc[0]);
    acc[1] = fmaf(cf[1], s0, acc[1]);
    acc[2] = fmaf(cf[2], s1, acc[2]);
    acc[3] = fmaf(cf[3], s1, acc[3]);
    if (t + 1 < ntile && threadIdx.x < 128) stage(t + 1, buf ^ 1, xn);
    w0 = nw0, w1 = nw1, s0 = ns0, s1 = ns1;
  }
  if (live) reinterpret_cast<float4*>(part)[((size_t)rb * 8 + i) * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_s = (atomicAdd(&cnt[item], 1u) & 7u) == 7u;
  __syncthreads();
  if (!last_s || !live) return;
  __threadfence();
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  for (int gi = 0; gi < 8; ++gi) {
    const float4 p = __ldcg(reinterpret_cast<const float4*>(part) + ((size_t)rb * 8 + gi) * 32 + lane);
    v[0] += p.x, v[1] += p.y, v[2] += p.z, v[3] += p.w;
  }
  __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
  for (int cc = 0; cc < 2; ++cc) {
    const int col = 2 * t4 + cc;
    if (col < m) {
      y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
      y[(size_t)col * segs.ldy[sg] + r0 + 8] = __float2bfloat16_rn(v[2 + cc]);
    }
  }
}
int launch_gsplit(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  static std::map<std::pair<int, cudaStream_t>, GsScratch> scr;
  int dev = 0;
  cudaGetDevice(&dev);
  GsScratch& z = scr[{dev, s}];
  const int nblk = segs.end[segs.nseg - 1] >> 4, nitem = (nblk + 7) / 8;
  if (nblk > z.max_blk) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &cs);
    if (cs != cudaStreamCaptureStatusNone) return -1;  // never allocate inside a capture: the caller falls back
    const int mb = nblk < 4096 ? 4096 : nblk;          // 4096 row blocks = 65536 rows (8 MB of sums)
    float* p = nullptr;
    unsigned* q = nullptr;
    if (cudaMalloc(&p, (size_t)mb * 8 * 128 * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&q, (size_t)(mb / 8 + 1) * 4) != cudaSuccess) return -1;
    cudaMemset(q, 0, (size_t)(mb / 8 + 1) * 4);
    cudaDeviceSynchronize();
    if (z.part) cudaFree(z.part), cudaFree(z.cnt);
    z.part = p, z.cnt = q, z.max_blk = mb;
  }
  w4_gsplit_kernel<<<nitem * 8, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k, z.part, z.cnt);
  return (int)cudaGetLastError();
}
template <int RB>
int launch_lean(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  auto kern = w4_lean_kernel<RB>;
  static int per_sm = -1;
  if (per_sm < 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, 0);
    if (per_sm < 1) per_sm = 1;
  }
  const int want = ((segs.end[segs.nseg - 1] >> 4) + RB - 1) / RB;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k);
  return (int)cudaGetLastError();
}
bool lean_ok(const KtSegs& g, int k) {
  if (k & 1023) return false;
  for (int i = 0; i < g.nseg; ++i)
    if (g.n[i] & 15) return false;
  return true;
}
// AGENS_W4_LEAN=RB (1 / 2 / 4, 8 = group split; 0 = off); default on sm90: group split for k >= 12288, RB 2 for
// >= 5 x SMs row blocks (gate_up, attention q), else 1 (sm90 m = 8: gate_up 40.8 -> 37.3 us, down 26.1 -> 23.5, KDA qkv 17.2 -> 13.2); ak_w4_set_lean overrides
int g_lean = -1;
int lean_rb(int nblk, int k) {
  static int v = -2;
  if (g_lean >= 0) return g_lean;
  if (v == -2) {
    const char* e = getenv("AGENS_W4_LEAN");
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    v = e ? atoi(e) : major == 9 ? -1 : 0;
  }
  // auto: the group split for long k (MLP down, k = 17408: 23.5 -> 21.1 us), else RB 2 / 1 by row blocks
  return v == -1 ? (k >= 12288 ? 8 : nblk >= 5 * num_sms() ? 2 : 1) : v;
}
// sm90 g128 W4 path: the same MMA chain and accumulation order as w4_mma_kernel<8, 1> -- warp w takes group
// i = w of every k-tile in increasing tile order, then the 8 warps' sums in warp order -- so every output is bit-identical
// to it; only the weight movement differs. A producer warp streams each (16-row block, k-tile) record pair (8 KB of
// nibbles + 256 B of scales, contiguous in the KT layout) into a STAGES-deep shared-memory ring with cp.async.bulk
// (mbarrier complete_tx), so the bytes in flight per SM no longer depend on registers (the register path saturates at
// ~2.2 TB/s on sm90: more steps in flight per warp cost occupancy). x stays on __ldg (L1 / L2 hits).
// Opt-in (AGENS_W4_TMA=1; not faster than the register path at 8 tokens on sm90 so far), AGENS_W4_TMA_STAGES=2|4|8.
constexpr int kTmaW = 8;                       // consumer warps (= w4_mma_kernel's WARPS: the step-to-warp mapping)
constexpr int kTmaRec = 16 * 512, kTmaSc = 16 * 16;  // bytes per stage: 16 rows x one k-tile record / its 8 scales
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t done = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(a), "r"(par) : "memory");
  } while (!done);
#endif
}
// RB 16-row blocks per work item: each consumer warp reuses its x fragment of a tile for all RB of them (the register
// path re-reads x from L1 / L2 per row block: 2 KB of x per 1 KB of weights at 8 tokens, which bounds it near
// 2.2 TB/s on sm90). Per row block the MMA chain and the order of every sum are those of w4_mma_kernel<8, 1>.
template <int STAGES, int RB>
__global__ void __launch_bounds__((kTmaW + 1) * 32)
w4_tma_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int STG = RB * (kTmaRec + kTmaSc);
  extern __shared__ __align__(128) unsigned char tsm[];
  __shared__ float red[kTmaW][32][4 * RB];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
  const int kp = (k + 1023) & ~1023, ntile = kp >> 10, kc = k >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4, nitem = (nblk + RB - 1) / RB;
  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(&empty[i])), "r"(kTmaW * 32));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == kTmaW) {  // producer
    if (lane == 0) {
      int seq = 0;
      for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
        int nn[RB], rbase[RB], rows[RB];
        const unsigned char *wt[RB], *st[RB];
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          const int g0 = (it * RB + r) << 4;
          int sg = 0;
#pragma unroll
          for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
          nn[r] = segs.n[sg], rbase[r] = g0 - (sg ? segs.end[sg - 1] : 0);
          rows[r] = it * RB + r >= nblk ? 0 : (nn[r] - rbase[r] < 16 ? nn[r] - rbase[r] : 16);
          wt[r] = reinterpret_cast<const unsigned char*>(segs.wt[sg]);
          st[r] = reinterpret_cast<const unsigned char*>(segs.st[sg]);
        }
        uint32_t tx = 0;
#pragma unroll
        for (int r = 0; r < RB; ++r) tx += rows[r] * (512 + 16);
        for (int t = 0; t < ntile; ++t, ++seq) {
          const int sl = seq % STAGES;
          if (seq >= STAGES) {  // every consumer thread has read the stage; order those reads before the async write
            mbar_wait(smem_u32(&empty[sl]), ((seq / STAGES) - 1) & 1);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          const uint32_t fb = smem_u32(&full[sl]);
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fb), "r"(tx) : "memory");
          unsigned char* dst = tsm + (size_t)sl * STG;
#pragma unroll
          for (int r = 0; r < RB; ++r) {
            if (!rows[r]) continue;
            const size_t rec = (size_t)t * nn[r] + rbase[r];
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                             smem_u32(dst + r * kTmaRec)), "l"(wt[r] + rec * 512), "r"(rows[r] * 512), "r"(fb) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                             smem_u32(dst + RB * kTmaRec + r * kTmaSc)), "l"(st[r] + rec * 16), "r"(rows[r] * 16), "r"(fb)
                         : "memory");
          }
        }
      }
    }
    return;
  }
  const int i = warp;  // this warp's group in every tile (step s = 8 t + i, as w4_mma_kernel<8, 1>)
  const uint4 pad = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
  int seq = 0;
  for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
    bool ok0[RB], ok1[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int g0 = (it * RB + r) << 4;
      int sg = 0;
#pragma unroll
      for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && g0 >= segs.end[j - 1];
      const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g;
      const bool live = it * RB + r < nblk;
      ok0[r] = live && r0 < n, ok1[r] = live && r0 + 8 < n;
    }
    float acc[RB][4];
#pragma unroll
    for (int r = 0; r < RB; ++r) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;
    // x of the next tile is loaded while this tile's MMAs run (one tile of x in flight per warp)
    const int c = 4 * i + t4;
    uint4 xn[4];
    {
      const bool xin = g < m && c < kc;
      const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)g * k + (size_t)c * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) xn[q] = xin ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
    }
    for (int t = 0; t < ntile; ++t, ++seq) {
      const int sl = seq % STAGES;
      const uint32_t xw[16] = {xn[0].x, xn[0].y, xn[0].z, xn[0].w, xn[1].x, xn[1].y, xn[1].z, xn[1].w,
                               xn[2].x, xn[2].y, xn[2].z, xn[2].w, xn[3].x, xn[3].y, xn[3].z, xn[3].w};
      {
        const int cg = (t + 1) * 32 + c;
        const bool xin = t + 1 < ntile && g < m && cg < kc;
        const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)g * k + (size_t)cg * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) xn[q] = xin ? __ldg(xr + q) : make_uint4(0, 0, 0, 0);
      }
      const unsigned char* sbase = tsm + (size_t)sl * STG;
      mbar_wait(smem_u32(&full[sl]), (seq / STAGES) & 1);
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint4* sw = reinterpret_cast<const uint4*>(sbase + r * kTmaRec);
        const __nv_bfloat16* ss = reinterpret_cast<const __nv_bfloat16*>(sbase + RB * kTmaRec + r * kTmaSc);
        const uint4 w0 = ok0[r] ? sw[g * 32 + c] : pad, w1 = ok1[r] ? sw[(g + 8) * 32 + c] : pad;
        const float sc0 = ok0[r] ? __bfloat162float(ss[g * 8 + i]) : 0.f;
        const float sc1 = ok1[r] ? __bfloat162float(ss[(g + 8) * 8 + i]) : 0.f;
        if (r == RB - 1) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&empty[sl])) : "memory");
        const uint32_t ww0[4] = {w0.x, w0.y, w0.z, w0.w}, ww1[4] = {w1.x, w1.y, w1.z, w1.w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, deq_pair(ww0[wd], j0), deq_pair(ww1[wd], j0), deq_pair(ww0[wd], j1), deq_pair(ww1[wd], j1),
                   x_pair(xw, wd, j0), x_pair(xw, wd, j1));
        }
        acc[r][0] = fmaf(cf[0], sc0, acc[r][0]);
        acc[r][1] = fmaf(cf[1], sc0, acc[r][1]);
        acc[r][2] = fmaf(cf[2], sc1, acc[r][2]);
        acc[r][3] = fmaf(cf[3], sc1, acc[r][3]);
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][lane][4 * r + j] = acc[r][j];
    asm volatile("bar.sync 1, %0;" ::"r"(kTmaW * 32) : "memory");
    if (warp < RB) {  // warp r writes row block r: the same warp-order sum as w4_mma_kernel's warp 0
      const int r = warp, g0 = (it * RB + r) << 4;
      if (it * RB + r < nblk) {
        int sg = 0;
#pragma unroll
        for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && g0 >= segs.end[j - 1];
        const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g, r1 = r0 + 8;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int w = 0; w < kTmaW; ++w)
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] += red[w][lane][4 * r + j];
        __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int col = 2 * t4 + cc;
          if (col < m) {
            if (r0 < n) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
            if (r1 < n) y[(size_t)col * segs.ldy[sg] + r1] = __float2bfloat16_rn(v[2 + cc]);
          }
        }
      }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(kTmaW * 32) : "memory");
  }
#endif
}
template <int STAGES, int RB>
int launch_tma_rb(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  auto kern = w4_tma_kernel<STAGES, RB>;
  const int smem = STAGES * RB * (kTmaRec + kTmaSc);
  static int per_sm = -1;
  if (per_sm < 0) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (kTmaW + 1) * 32, smem);
    if (per_sm < 1) per_sm = 1;
  }
  const int want = ((segs.end[segs.nseg - 1] >> 4) + RB - 1) / RB;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  kern<<<blocks, (kTmaW + 1) * 32, smem, s>>>(static_cast<const __nv_bfloat16*>(x), m, segs, k);
  return (int)cudaGetLastError();
}
// RB: the largest of 4 / 2 / 1 that still gives every SM a work item (AGENS_W4_TMA_RB forces one)
template <int STAGES>
int launch_tma(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s) {
  static int force = -2;
  if (force == -2) {
    const char* e = getenv("AGENS_W4_TMA_RB");
    force = e ? atoi(e) : 0;
  }
  const int nblk = segs.end[segs.nseg - 1] >> 4, sms = num_sms();
  int rb = force > 0 ? force : nblk >= 4 * sms ? 4 : nblk >= 2 * sms ? 2 : 1;
  while (rb > 1 && STAGES * rb * (kTmaRec + kTmaSc) > 200 * 1024) rb >>= 1;  // shared memory per block
  switch (rb) {
    case 4: return launch_tma_rb<STAGES, 4>(x, m, segs, k, s);
    case 2: return launch_tma_rb<STAGES, 2>(x, m, segs, k, s);
    default: return launch_tma_rb<STAGES, 1>(x, m, segs, k, s);
  }
}
// sm90 with AGENS_W4_TMA=1 (other archs: the register path; sm120 keeps it); ak_w4_set_tma overrides (tests)
int g_tma = -1;
int tma_stages() {
  static int v = -1;
  if (g_tma >= 0) return g_tma;
  if (v < 0) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    const char* e = getenv("AGENS_W4_TMA");
    const char* st = getenv("AGENS_W4_TMA_STAGES");
    v = (major == 9 && e && e[0] == '1') ? (st ? atoi(st) : 2) : 0;
  }
  return v;
}
// Group 64 (the 3-bit probe; W4 container or W3 KT): one step = one 64-input group, so every MMA of a step lies in
// one group and its partial is scaled once. Lane (g, t4) takes inputs 16 t4 .. 16 t4 + 15 of the group: half t4 & 1
// of chunk 2i + (t4 >> 1) of rows g and g + 8 (8 bytes of W4 KT, or its 2-bit word + 16 bits of the bit plane of W3
// KT expanded to the same 2 nibble words; W2 KT: its 2-bit word); 4 MMAs per step with the pair mapping of w4_mma_kernel
// restricted to 2 words.
template <int WARPS, int BITS, int U = 1>
__global__ void __launch_bounds__(WARPS * 32)
w4_mma64_kernel(const __nv_bfloat16* __restrict__ x, int m, const __grid_constant__ KtSegs segs, int k) {
  // U: steps whose loads a warp issues together (memory parallelism); each warp still takes steps w, w + WARPS, ...
  // in increasing order and accumulates them in that order, so the result is the U = 1 result bit for bit
  __shared__ float red[WARPS][32][4];
  const int kp = (k + 1023) & ~1023, steps = (kp >> 10) * 16, kg = k >> 6;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4;
  const int h = t4 & 1;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int g0 = rb << 4;
    int sg = 0;
#pragma unroll
    for (int i = 1; i < kMaxSegs; ++i) sg += i < segs.nseg && g0 >= segs.end[i - 1];
    const int n = segs.n[sg], r0 = g0 - (sg ? segs.end[sg - 1] : 0) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    const uint4* __restrict__ wt = segs.wt[sg];
    const __nv_bfloat16* __restrict__ st = segs.st[sg];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s0 = warp; s0 < steps; s0 += WARPS * U) {
      uint32_t w0[U][2], w1[U][2];
      float sc0[U], sc1[U];
      uint4 xv[U][2];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int s = s0 + u * WARPS;
        const bool in = s < steps;
        const int t = s >> 4, i = s & 15, c = 2 * i + (t4 >> 1);
        const size_t rec0 = (size_t)t * n + r0, rec1 = (size_t)t * n + r1;
        w0[u][0] = w0[u][1] = w1[u][0] = w1[u][1] = BITS == 2 ? 0x00AA00AAu : 0x88888888u;
        if (BITS == 2) {
          if (in && ok0) {
            const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec0 * rec_words<2>() + 2 * c + h);
            w0[u][0] = w2x(lo & 0xFFFFu), w0[u][1] = w2x(lo >> 16);
          }
          if (in && ok1) {
            const uint32_t lo = ld_stream1(reinterpret_cast<const uint32_t*>(wt) + rec1 * rec_words<2>() + 2 * c + h);
            w1[u][0] = w2x(lo & 0xFFFFu), w1[u][1] = w2x(lo >> 16);
          }
        } else if (BITS == 3) {
          if (in && ok0) {
            const uint32_t* b = reinterpret_cast<const uint32_t*>(wt) + rec0 * 96;
            const uint32_t lo = ld_stream1(b + 2 * c + h), hi = ld_stream1(b + 64 + c) >> (16 * h);
            w0[u][0] = w3_word(lo, hi), w0[u][1] = w3_word(lo >> 16, hi >> 8);
          }
          if (in && ok1) {
            const uint32_t* b = reinterpret_cast<const uint32_t*>(wt) + rec1 * 96;
            const uint32_t lo = ld_stream1(b + 2 * c + h), hi = ld_stream1(b + 64 + c) >> (16 * h);
            w1[u][0] = w3_word(lo, hi), w1[u][1] = w3_word(lo >> 16, hi >> 8);
          }
        } else {
          if (in && ok0) {
            const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec0 * 32 + c) + h);
            w0[u][0] = v.x, w0[u][1] = v.y;
          }
          if (in && ok1) {
            const uint2 v = ld_stream2(reinterpret_cast<const uint2*>(wt + rec1 * 32 + c) + h);
            w1[u][0] = v.x, w1[u][1] = v.y;
          }
        }
        sc0[u] = in && ok0 ? __bfloat162float(st[rec0 * 16 + i]) : 0.f;
        sc1[u] = in && ok1 ? __bfloat162float(st[rec1 * 16 + i]) : 0.f;
        const int gg = t * 16 + i;  // global group
        const bool xin = in && g < m && gg < kg;
        xv[u][0] = xv[u][1] = make_uint4(0, 0, 0, 0);
        if (xin) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)g * k + (size_t)gg * 64 + 16 * t4);
          xv[u][0] = __ldg(xr), xv[u][1] = __ldg(xr + 1);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (s0 + u * WARPS >= steps) break;
        const uint32_t xw[8] = {xv[u][0].x, xv[u][0].y, xv[u][0].z, xv[u][0].w,
                                xv[u][1].x, xv[u][1].y, xv[u][1].z, xv[u][1].w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          mma16816(cf, deqb<BITS>(w0[u][wd], j0), deqb<BITS>(w1[u][wd], j0), deqb<BITS>(w0[u][wd], j1),
                   deqb<BITS>(w1[u][wd], j1), x_pair(xw, wd, j0), x_pair(xw, wd, j1));
        }
        acc[0] = fmaf(cf[0], sc0[u], acc[0]);
        acc[1] = fmaf(cf[1], sc0[u], acc[1]);
        acc[2] = fmaf(cf[2], sc1[u], acc[2]);
        acc[3] = fmaf(cf[3], sc1[u], acc[3]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp][lane][j] = acc[j];
    __syncthreads();
    if (warp == 0) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int w = 0; w < WARPS; ++w)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[w][lane][j];
      __nv_bfloat16* __restrict__ y = segs.y[sg];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * segs.ldy[sg] + r0] = __float2bfloat16_rn(v[cc]);
          if (ok1) y[(size_t)col * segs.ldy[sg] + r1] = __float2bfloat16_rn(v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}
template <int BITS, int U>
int mma64_segs_u(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  constexpr int WARPS = 8;
  auto kern = w4_mma64_kernel<WARPS, BITS, U>;
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, WARPS * 32, 0);
  if (per_sm < 1) per_sm = 1;
  const int want = g.end[g.nseg - 1] >> 4;
  const int blocks = want < num_sms() * per_sm ? want : num_sms() * per_sm;
  for (int m0 = 0; m0 < m; m0 += 8) {
    KtSegs gg = g;
    if (m0)
      for (int i = 0; i < g.nseg; ++i) gg.y[i] = g.y[i] + (size_t)m0 * g.ldy[i];
    kern<<<blocks, WARPS * 32, 0, s>>>(static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k, m - m0 < 8 ? m - m0 : 8,
                                       gg, k);
    const int e = (int)cudaGetLastError();
    if (e) return e;
  }
  return 0;
}
int launch_gsnt_any(const void* x, int m, const KtSegs& segs, int k, cudaStream_t s, int bits, bool g64);
// AGENS_W4_GS8=1: small-n W4 at <= 8 tokens on the group-split kernel (bit-identical; off by default: slower on sm90
// at 8 tokens, 8.2 vs 7.6 us per call)
bool gs8_on() {
  static const int on = [] { const char* e = getenv("AGENS_W4_GS8"); return e && e[0] == '1'; }();
  return on;
}
// steps in flight per warp (bit-identical for any U): AGENS_W4_MMA64_U (default 1)
template <int BITS>
int mma64_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  static int u = -1;
  static bool fixed = false;
  if (u < 0) {
    const char* e = getenv("AGENS_W4_MMA64_U");
    u = e ? atoi(e) : 1;
    fixed = e != nullptr;
  }
  if (m > 8 && nt_on()) return nt_segs<true, BITS>(x, m, g, k, s);
  if (gs8_on() && (g.end[g.nseg - 1] >> 4) < 2 * num_sms()) {  // small n, <= 8 tokens: the group split (same bits)
    const int e = launch_gsnt_any(x, m, g, k, s, BITS, true);
    if (e >= 0) return e;
  }
  if (!fixed && (g.end[g.nseg - 1] >> 4) < num_sms()) return mma64_segs_u<BITS, 4>(x, m, g, k, s);  // as mma_launch
  switch (u) {
    case 2: return mma64_segs_u<BITS, 2>(x, m, g, k, s);
    case 4: return mma64_segs_u<BITS, 4>(x, m, g, k, s);
    default: return mma64_segs_u<BITS, 1>(x, m, g, k, s);
  }
}

int g_mma_mode = -1;  // -1 unresolved (on unless AGENS_W4_EXACT=1), 0 off, 1 on
bool kf_on();
bool mma_on(int group, int bf16w) {
  if (group == 128 && !bf16w && kf_on()) return true;  // KF weights only have the tensor-core path
  if ((group != 128 && group != 64) || bf16w) return false;
  if (g_mma_mode < 0) {
    // default on every arch (2026-10-14: agreement with sglang as the exact path, flat cost for 2..8-token verify);
    // AGENS_W4_EXACT=1 selects the exact fp32 path (bit-identical to the original g128 GEMV)
    const char* e = getenv("AGENS_W4_EXACT");
    g_mma_mode = e && e[0] == '1' ? 0 : 1;
  }
  return g_mma_mode == 1;
}
// launch shape (warps per 16-row block, steps per load batch): AGENS_W4_MMA_CFG="W,U" (bench), default 8,1 (sm90 sweep:
// 8,1 best on every decode shape; gate_up 2.84 TB/s, down 2.17, KDA qkv 1.93, out 1.70 vs 1.62 / 1.07 / 1.16 / 0.90 exact)
template <int BITS>
int mma_launch(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  if (BITS == 4) {
    // (fewer row blocks than SMs: the register path below, with its loads batched, is the faster one there)
    if (const int rb = lean_rb(g.end[g.nseg - 1] >> 4, k); rb && lean_ok(g, k) && (g.end[g.nseg - 1] >> 4) >= num_sms()) {
      if (rb == 8) {
        const int e = launch_gsplit(x, m, g, k, s);
        if (e >= 0) return e;
      }
    }
    if (const int rb = lean_rb(g.end[g.nseg - 1] >> 4, k); rb && rb != 8 && lean_ok(g, k) && (g.end[g.nseg - 1] >> 4) >= num_sms())
      return rb >= 4 ? launch_lean<4>(x, m, g, k, s) : rb == 2 ? launch_lean<2>(x, m, g, k, s) : launch_lean<1>(x, m, g, k, s);
    if (const int st = tma_stages()) {
      switch (st) {
        case 2: return launch_tma<2>(x, m, g, k, s);
        case 8: return launch_tma<8>(x, m, g, k, s);
        default: return launch_tma<4>(x, m, g, k, s);
      }
    }
  }
  static int cfg = -1;
  static bool fixed = false;
  if (cfg < 0) {
    int w = 8, u = 1;
    if (const char* e = getenv("AGENS_W4_MMA_CFG")) sscanf(e, "%d,%d", &w, &u), fixed = true;
    cfg = w * 10 + u;
  }
  // fewer 16-row blocks than SMs (n <= ~2K: KDA gate / beta projections, indexer, drafter): the launch is latency-bound,
  // so each warp issues 4 steps' loads together (same steps, same order per warp: bit-identical to 8,1)
  if (!fixed && (g.end[g.nseg - 1] >> 4) < num_sms()) return launch_mma<8, 4, BITS>(x, m, g, k, s);
  switch (cfg) {
    case 41: return launch_mma<4, 1, BITS>(x, m, g, k, s);
    case 42: return launch_mma<4, 2, BITS>(x, m, g, k, s);
    case 44: return launch_mma<4, 4, BITS>(x, m, g, k, s);
    case 81: return launch_mma<8, 1, BITS>(x, m, g, k, s);
    case 84: return launch_mma<8, 4, BITS>(x, m, g, k, s);
    case 161: return launch_mma<16, 1, BITS>(x, m, g, k, s);
    case 162: return launch_mma<16, 2, BITS>(x, m, g, k, s);
    case 164: return launch_mma<16, 4, BITS>(x, m, g, k, s);
    case 82: return launch_mma<8, 2, BITS>(x, m, g, k, s);
    default: return launch_mma<8, 1, BITS>(x, m, g, k, s);
  }
}
// up to 16 tokens: launches of <= 8 (the MMA width), same rows each
template <int BITS>
int mma128_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  if (m > 8 && nt_on()) return nt_segs<false, BITS>(x, m, g, k, s);
  if (gs8_on() && (g.end[g.nseg - 1] >> 4) < 2 * num_sms()) {  // small n, <= 8 tokens: the group split (same bits)
    const int e = launch_gsnt_any(x, m, g, k, s, BITS, false);
    if (e >= 0) return e;
  }
  if (gemm8_cfg() && lean_ok(g, k)) {
    const int e = gemm8<false, BITS>(x, m, g, k, s);
    if (e >= 0) return e;
  }
  for (int m0 = 0; m0 < m; m0 += 8) {
    KtSegs gg = g;
    if (m0)
      for (int i = 0; i < g.nseg; ++i) gg.y[i] = g.y[i] + (size_t)m0 * g.ldy[i];
    const int e = mma_launch<BITS>(static_cast<const __nv_bfloat16*>(x) + (size_t)m0 * k, m - m0 < 8 ? m - m0 : 8, gg, k, s);
    if (e) return e;
  }
  return 0;
}
// ---------------------------------------------------------------------------------------------- W4 g128 KF layout
// KF ("fragment-major") layout of a W4 g128 weight, n % 16 == 0 (AGENS_W4F=1 at repack time): the bytes a warp needs
// for one 16-row block x 64 inputs are one contiguous 512 B record, lane-major, so a warp streams its row block with
// whole-sector loads (the KT layout's 64 B per row per load runs at ~3 TB/s on sm90 even without any math; contiguous
// records stream at ~4.3). The k order inside an MMA step is KT's (the same B fragments, x_pair), so a lane's word for
// step q of group G holds rows g and g + 8: nibbles {0, 4} = row g (KT nibbles j0, j0 + 4 of word q / 2 of chunk
// 4 (G % 8) + t4), {1, 5} = row g + 8 (same), {2, 6} = row g (j1, j1 + 4), {3, 7} = row g + 8, j0 = 2 (q & 1).
//   weights  [G][rb][h (steps 4h .. 4h + 3)][lane][4 words]          (= n * kp / 2 bytes, as KT)
//   scales   [G][rb][g (0..7)] bf16x2 (row 16 rb + g, row 16 rb + g + 8) (= n * kp / 128 bf16, as KT)
// Group-major: the warps of the GPU sweep a group's row blocks together, so concurrent requests hit contiguous DRAM.
// Numerics (the product numerics with AGENS_W4F; not bit-identical to the KT kernels): k is cut into chunks of 2048
// (S = kp / 2048 rounded up, a function of k only); per (row, token, chunk) a warp runs the group steps in k order, each
// group's 8 MMAs into a zeroed fp32 fragment, then acc = fmaf(group sum, scale, acc); the chunk sums are added in chunk
// order (0, 1, ..) from 0. A token column's arithmetic never depends on m, the batch or the launch shape, so decode
// (m = 1), verify (m = 8) and batched verify (8 x NT rows) give the same bits per row.
constexpr int kKfChunkMax = 2048;
// k per chunk (AGENS_W4F_CHUNK = 1024 | 2048, default 1024): part of the product numerics, fixed per process
int kf_chunk() {
  static const int v = [] {
    const char* e = getenv("AGENS_W4F_CHUNK");
    return e && (atoi(e) == 2048 || atoi(e) == 4096 || atoi(e) == 8192) ? atoi(e) : 1024;
  }();
  return v;
}
std::unordered_set<const void*> g_kf_set;  // weights stored in KF
int g_kf = -1;
bool kf_on() {
  if (g_kf < 0) {
    const char* e = getenv("AGENS_W4F");
    g_kf = e && e[0] == '1' ? 1 : 0;
  }
  return g_kf == 1;
}
// KT -> KF, one thread per output word (weights) / scale word
__global__ void kt_to_kf_kernel(const uint32_t* __restrict__ kt, const __nv_bfloat16* __restrict__ kst,
                                uint32_t* __restrict__ kf, uint32_t* __restrict__ kfs, int n, int kp) {
  const int ng = kp >> 7, nrb = n >> 4;
  const size_t words = (size_t)nrb * ng * 2 * 32 * 4, swords = (size_t)nrb * ng * 8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words + swords; i += (size_t)gridDim.x * blockDim.x) {
    if (i < words) {
      const int w = (int)(i & 3), lane = (int)((i >> 2) & 31), h = (int)((i >> 7) & 1);
      const size_t r = i >> 8;  // G * nrb + rb
      const int rb = (int)(r % nrb), G = (int)(r / nrb);
      const int g = lane >> 2, t4 = lane & 3, q = 4 * h + w, wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
      const int tile = G >> 3, c = 4 * (G & 7) + t4;  // KT chunk (uint4) within the tile record
      auto ktw = [&](int row) { return kt[(((size_t)tile * n + row) * 32 + c) * 4 + wd]; };
      const uint32_t a = ktw(16 * rb + g), b = ktw(16 * rb + g + 8);
      auto nib = [](uint32_t v, int j) { return (v >> (4 * j)) & 15u; };
      const uint32_t o = nib(a, j0) | nib(b, j0) << 4 | nib(a, j1) << 8 | nib(b, j1) << 12 | nib(a, j0 + 4) << 16 |
                         nib(b, j0 + 4) << 20 | nib(a, j1 + 4) << 24 | nib(b, j1 + 4) << 28;
      kf[i] = o;
    } else {
      const size_t j = i - words;
      const int g = (int)(j & 7);
      const size_t r = j >> 3;
      const int rb = (int)(r % nrb), G = (int)(r / nrb), tile = G >> 3, gi = G & 7;
      const __nv_bfloat16 lo = kst[((size_t)tile * n + 16 * rb + g) * 8 + gi];
      const __nv_bfloat16 hi = kst[((size_t)tile * n + 16 * rb + g + 8) * 8 + gi];
      kfs[j] = (uint32_t)__bfloat16_as_ushort(lo) | (uint32_t)__bfloat16_as_ushort(hi) << 16;
    }
  }
}
// KF -> dense bf16 rows [r0, r0 + nr) (prefill): value = bf16(q * scale) as the KT dequant, so prefill is unchanged
__global__ void dequant_kf_kernel(const uint32_t* __restrict__ kf, const uint32_t* __restrict__ kfs,
                                  __nv_bfloat16* __restrict__ w, int n, int k, int r0, int nr) {
  const int kp = (k + 1023) & ~1023, ng = kp >> 7;
  const int rb0 = r0 >> 4, rb1 = (r0 + nr + 15) >> 4;
  const size_t words = (size_t)(rb1 - rb0) * ng * 256;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
    const int wq = (int)(i & 3), lane = (int)((i >> 2) & 31), h = (int)((i >> 7) & 1);
    const size_t r = i >> 8;
    const int G = (int)(r % ng), rb = rb0 + (int)(r / ng);
    const uint32_t v = kf[((size_t)G * (n >> 4) + rb) * 256 + (i & 255)];
    const int g = lane >> 2, t4 = lane & 3, q = 4 * h + wq, wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
    const uint32_t sw = kfs[((size_t)G * (n >> 4) + rb) * 8 + g];
    const float s[2] = {__bfloat162float(__ushort_as_bfloat16((unsigned short)(sw & 0xffff))),
                        __bfloat162float(__ushort_as_bfloat16((unsigned short)(sw >> 16)))};
    const int kb = G * 128 + t4 * 32 + 8 * wd;
    // nibble p of v: row (p & 1 ? g + 8 : g), k offset {j0, j1, j0 + 4, j1 + 4}[...]
    const int koff[8] = {j0, j0, j1, j1, j0 + 4, j0 + 4, j1 + 4, j1 + 4};
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int row = 16 * rb + g + (p & 1 ? 8 : 0), kk = kb + koff[p];
      if (row < r0 || row >= r0 + nr || kk >= k) continue;
      const float q4 = (float)((int)((v >> (4 * p)) & 15u) - 8);
      w[(size_t)(row - r0) * k + kk] = __float2bfloat16_rn(q4 * s[p & 1]);
    }
  }
}
__device__ __forceinline__ uint32_t kf_deq(uint32_t w, int j) {  // nibbles j, j + 4 -> bf16x2 (q - 8)
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t kf_raw(uint32_t w) {  // nibbles 0, 4 -> bf16x2 (128 + nibble) = (q + 136)
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w), "r"(0x000F000Fu), "r"(0x43004300u));
  return v;
}
__device__ __forceinline__ uint4 kf_ld(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
// One CTA = (chunk s, CTA c of the chunk): x of the chunk staged once in B-fragment order [step][tile][lane][2], then
// warp w walks row blocks c * 8 + w, + C * 8, ..; per row block the chunk's groups in order. The weight stream of a
// warp is continuous across its row blocks: P groups (P KB) in flight, the first ones issued before x is staged.
// S > 1: per-chunk sums to scratch, the row block's last chunk to finish adds them in chunk order.
template <int NT, int P, int MINB>
__global__ void __launch_bounds__(256, MINB) w4f_kernel(const __nv_bfloat16* __restrict__ x, int m, int m0,
                                                  const __grid_constant__ KtSegs segs, int k, int S, int C,
                                                  float* __restrict__ part, unsigned* __restrict__ cnt, int kch,
                                                  int probe) {
  extern __shared__ uint4 kf_smem[];
  uint4* xs = kf_smem;  // [step pairs][NT][32]: (b0, b1) of steps 2 wd and 2 wd + 1
  // per (group, token) sum of x over the group's 128 inputs, fixed order (the weights enter the MMA as q + 136; the
  // 136 x sum comes back out once per group)
  float* sxs = reinterpret_cast<float*>(kf_smem + (size_t)(kch / 32) * NT * 32);  // [groups][8 NT]
  float* sxp = sxs + (kch / 128) * 8 * NT;  // [groups][8 NT][16]: sums of the 8-input pieces (t4, wd)
  // per-lane weight ring: a lane streams (cp.async) exactly the 2 x 16 B per group it consumes, P groups deep
  uint4* wring = reinterpret_cast<uint4*>(sxp + (kch / 128) * 8 * NT * 16);  // [P][2][256]
  uint32_t* sring = reinterpret_cast<uint32_t*>(wring + P * 2 * 256);       // [P][256]
  const int kp = (k + 1023) & ~1023, ng = kp >> 7;
  const int s = blockIdx.x % S, c = blockIdx.x / S;
  const int G0 = s * (kch / 128), G1 = min(ng, G0 + kch / 128), ngc = G1 - G0, steps = 8 * ngc;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4, rb0 = c * 8 + warp, rstride = C * 8;
  const int nmine = rb0 < nblk && probe != 3 ? (nblk - rb0 + rstride - 1) / rstride : 0, T = nmine * ngc;
  int fi = 0, fit = 0, fg = 0;  // next group to fetch: index, (item, group)
  auto fetch = [&]() {
    if (fi < T) {
      const int rb = rb0 + fit * rstride, G = G0 + fg;
      int sg = 0;
#pragma unroll
      for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && 16 * rb >= segs.end[j - 1];
      const int lrb = rb - (sg ? segs.end[sg - 1] >> 4 : 0);
      const size_t rec = (size_t)G * (segs.n[sg] >> 4) + lrb;  // group-major: a group's row blocks contiguous
      const uint4* wp = segs.wt[sg] + rec * 64 + lane;
      const int slot = fi % P;
      cp_async16(wring + (slot * 2) * 256 + tid, wp, 16);
      cp_async16(wring + (slot * 2 + 1) * 256 + tid, wp + 32, 16);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)__cvta_generic_to_shared(sring + slot * 256 + tid)),
                   "l"(reinterpret_cast<const uint32_t*>(segs.st[sg]) + rec * 8 + g) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    ++fi;
    if (++fg == ngc) fg = 0, ++fit;
  };
#pragma unroll 1
  for (int p = 0; p < P; ++p) fetch();
  // stage x: thread item (step pair qp = (G, wd), tile tt, lane) reads x[tok][kb .. kb + 8) once (16 B) and writes the
  // entries of steps 2 wd (j0 = 0) and 2 wd + 1 (j0 = 2): (x0 x4, x1 x5) and (x2 x6, x3 x7)
  for (int e = tid; e < (steps / 2) * NT * 32; e += blockDim.x) {
    const int ln = e & 31, tt = (e >> 5) % NT, qp = (e >> 5) / NT;
    const int G = G0 + (qp >> 2), wd = qp & 3;
    const int tok = m0 + 8 * tt + (ln >> 2), kb = G * 128 + (ln & 3) * 32 + 8 * wd;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (tok < m) {
      const __nv_bfloat16* xr = x + (size_t)tok * k + kb;
      if (kb + 8 <= k && ((reinterpret_cast<uintptr_t>(xr) & 15) == 0)) {
        v = __ldg(reinterpret_cast<const uint4*>(xr));
      } else {
        uint32_t h[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) h[j] = kb + j < k ? (uint32_t)__bfloat16_as_ushort(xr[j]) : 0u;
        v = make_uint4(h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16, h[6] | h[7] << 16);
      }
    }
    // v = (x0 x1)(x2 x3)(x4 x5)(x6 x7): step 2 wd: (x0 x4), (x1 x5); step 2 wd + 1: (x2 x6), (x3 x7)
    xs[(qp * NT + tt) * 32 + ln] = make_uint4(__byte_perm(v.x, v.z, 0x5410), __byte_perm(v.x, v.z, 0x7632),
                                              __byte_perm(v.y, v.w, 0x5410), __byte_perm(v.y, v.w, 0x7632));
    {  // this piece's sum, in input order
      const uint32_t u[4] = {v.x, v.y, v.z, v.w};
      float ps = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) ps += __uint_as_float(u[j] << 16), ps += __uint_as_float(u[j] & 0xffff0000u);
      sxp[(((qp >> 2) * 8 * NT) + 8 * tt + (ln >> 2)) * 16 + (ln & 3) * 4 + (qp & 3)] = ps;
    }
  }
  __syncthreads();
  for (int e = tid; e < ngc * 8 * NT; e += blockDim.x) {  // group sums: the 16 pieces in input order
    float sx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) sx += sxp[e * 16 + j];
    sxs[e] = sx;
  }
  __syncthreads();
  float acc[NT][4];
#pragma unroll
  for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  // groups two at a time (two independent MMA chains; each group's chain and the acc order are unchanged); T is even
  // (chunks are whole multiples of 8 groups)
#pragma unroll 1
  for (int i = 0; i < T; i += 2) {
    asm volatile("cp.async.wait_group %0;" ::"n"(P - 2) : "memory");
    uint32_t w8[2][8], sw[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int slot = (i + h) % P;
      const uint4 u0 = wring[(slot * 2) * 256 + tid], u1 = wring[(slot * 2 + 1) * 256 + tid];
      w8[h][0] = u0.x, w8[h][1] = u0.y, w8[h][2] = u0.z, w8[h][3] = u0.w;
      w8[h][4] = u1.x, w8[h][5] = u1.y, w8[h][6] = u1.z, w8[h][7] = u1.w;
      sw[h] = sring[slot * 256 + tid];
    }
    fetch();
    fetch();
    const int Gl = i % ngc;
    if (probe) {  // bench probe: the stream without the math
      acc[0][0] += (float)((w8[0][0] ^ w8[1][7] ^ sw[0]) & 1);
      if (Gl + 1 != ngc - 1) continue;
    } else {
    float cf[2][NT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) cf[h][tt][0] = cf[h][tt][1] = cf[h][tt][2] = cf[h][tt][3] = 0.f;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint4* xq = xs + ((size_t)(Gl + h) * 4 + wd) * NT * 32 + lane;
        uint4 b[NT];
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) b[tt] = xq[tt * 32];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const uint32_t w = w8[h][2 * wd + e];
          const uint32_t a0 = kf_raw(w), a1 = kf_raw(w >> 4), a2 = kf_raw(w >> 8), a3 = kf_raw(w >> 12);
#pragma unroll
          for (int tt = 0; tt < NT; ++tt)
            mma16816(cf[h][tt], a0, a1, a2, a3, e ? b[tt].z : b[tt].x, e ? b[tt].w : b[tt].y);
        }
      }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float s0 = __uint_as_float(sw[h] << 16), s1 = __uint_as_float(sw[h] & 0xffff0000u);
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        const float2 sx = *reinterpret_cast<const float2*>(sxs + (Gl + h) * 8 * NT + 8 * tt + 2 * t4);
        cf[h][tt][0] = fmaf(-136.f, sx.x, cf[h][tt][0]);
        cf[h][tt][1] = fmaf(-136.f, sx.y, cf[h][tt][1]);
        cf[h][tt][2] = fmaf(-136.f, sx.x, cf[h][tt][2]);
        cf[h][tt][3] = fmaf(-136.f, sx.y, cf[h][tt][3]);
        acc[tt][0] = fmaf(cf[h][tt][0], s0, acc[tt][0]);
        acc[tt][1] = fmaf(cf[h][tt][1], s0, acc[tt][1]);
        acc[tt][2] = fmaf(cf[h][tt][2], s1, acc[tt][2]);
        acc[tt][3] = fmaf(cf[h][tt][3], s1, acc[tt][3]);
      }
    }
    if (Gl + 1 != ngc - 1) continue;
    }
    // the row block's chunk is done
    const int rb = rb0 + (i / ngc) * rstride;
    int sg = 0;
#pragma unroll
    for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && 16 * rb >= segs.end[j - 1];
    const int lrb = rb - (sg ? segs.end[sg - 1] >> 4 : 0);
    __nv_bfloat16* __restrict__ y = segs.y[sg];
    const int ldy = segs.ldy[sg], row = 16 * lrb + g;
    bool out = S == 1;
    if (!out) {
      float4* pp = reinterpret_cast<float4*>(part) + ((size_t)(s * nblk + rb) * 32 + lane) * NT;
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) __stcg(pp + tt, make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]));
      if (probe != 2) __threadfence();
      __syncwarp();
      unsigned old = 0;
      if (lane == 0) old = atomicAdd(cnt + rb, 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == (unsigned)(S - 1)) {
        __threadfence();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
        for (int ss = 0; ss < S; ++ss) {
          const float4* qq = reinterpret_cast<const float4*>(part) + ((size_t)(ss * nblk + rb) * 32 + lane) * NT;
#pragma unroll
          for (int tt = 0; tt < NT; ++tt) {
            const float4 u = __ldcg(qq + tt);
            acc[tt][0] += u.x, acc[tt][1] += u.y, acc[tt][2] += u.z, acc[tt][3] += u.w;
          }
        }
        if (lane == 0) cnt[rb] = 0;  // ready for the next launch on this stream
        out = true;
      }
    }
    if (out)
#pragma unroll
      for (int tt = 0; tt < NT; ++tt)
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int tok = m0 + 8 * tt + 2 * t4 + cc;
          if (tok < m) {
            y[(size_t)tok * ldy + row] = __float2bfloat16_rn(acc[tt][cc]);
            y[(size_t)tok * ldy + row + 8] = __float2bfloat16_rn(acc[tt][2 + cc]);
          }
        }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
// Bulk-copy variant (sm90+: cp.async.bulk + mbarrier; sm89 runs w4f_kernel): one producer warp per CTA streams, per
// group, the octet's 8 contiguous 1 KB records (group-major KF: one copy of up to 8 KB) and their scales into a
// STAGES-deep ring; consumer warp w takes row block 8 o + w. Same arithmetic per (row, token, chunk) as w4f_kernel.
template <int NT, int STAGES>
__global__ void __launch_bounds__(9 * 32) w4fb_kernel(const __nv_bfloat16* __restrict__ x, int m, int m0,
                                                      const __grid_constant__ KtSegs segs, int k, int S, int C,
                                                      float* __restrict__ part, unsigned* __restrict__ cnt, int kch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int STG = 8 * 1024 + 8 * 32;  // 8 row blocks x (1 KB records + 32 B scales)
  extern __shared__ __align__(128) uint4 kfb_smem[];
  __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
  unsigned char* ring = reinterpret_cast<unsigned char*>(kfb_smem);
  uint4* xs = reinterpret_cast<uint4*>(ring + (size_t)STAGES * STG);       // [step pairs][NT][32]
  float* sxs = reinterpret_cast<float*>(xs + (size_t)(kch / 32) * NT * 32);  // [groups][8 NT]
  float* sxp = sxs + (kch / 128) * 8 * NT;                                    // [groups][8 NT][16]
  const int kp = (k + 1023) & ~1023, ng = kp >> 7;
  const int s = blockIdx.x % S, c = blockIdx.x / S;
  const int G0 = s * (kch / 128), G1 = min(ng, G0 + kch / 128), ngc = G1 - G0, steps = 8 * ngc;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = segs.end[segs.nseg - 1] >> 4, noct = (nblk + 7) >> 3;
  const int nmine = c < noct ? (noct - c + C - 1) / C : 0, T = nmine * ngc;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(&empty[i])), "r"(8));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 8) {  // producer: stream index i -> (octet c + (i / ngc) C, group G0 + i % ngc)
    if (lane == 0) {
      for (int i = 0; i < T; ++i) {
        const int sl = i % STAGES, o = c + (i / ngc) * C, G = G0 + i % ngc;
        if (i >= STAGES) {
          mbar_wait(smem_u32(&empty[sl]), ((i / STAGES) - 1) & 1);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        const uint32_t fb = smem_u32(&full[sl]);
        unsigned char* dst = ring + (size_t)sl * STG;
        // runs of consecutive row blocks in one segment: one weight copy and one scale copy each
        int rb = 8 * o;
        const int rbe = min(8 * o + 8, nblk);
        uint32_t tx = (uint32_t)(rbe - rb) * (1024 + 32);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fb), "r"(tx) : "memory");
        while (rb < rbe) {
          int sg = 0;
#pragma unroll
          for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && 16 * rb >= segs.end[j - 1];
          const int send = segs.end[sg] >> 4, r1 = min(rbe, send), lrb = rb - (sg ? segs.end[sg - 1] >> 4 : 0);
          const size_t rec = (size_t)G * (segs.n[sg] >> 4) + lrb;
          const int nr = r1 - rb, w = rb - 8 * o;
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                           smem_u32(dst + w * 1024)), "l"(reinterpret_cast<const unsigned char*>(segs.wt[sg]) + rec * 1024),
                       "r"(nr * 1024), "r"(fb) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                           smem_u32(dst + 8 * 1024 + w * 32)),
                       "l"(reinterpret_cast<const unsigned char*>(segs.st[sg]) + rec * 32), "r"(nr * 32), "r"(fb)
                       : "memory");
          rb = r1;
        }
      }
    }
    return;
  }
  // consumers (256 threads): stage x of the chunk + the group sums; 8 items per thread per round, loads issued first
  const int nst = (steps / 2) * NT * 32;
  for (int e0 = tid; e0 < nst; e0 += 256 * 8) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = e0 + j * 256;
      v[j] = make_uint4(0u, 0u, 0u, 0u);
      if (e >= nst) continue;
      const int ln = e & 31, tt = (e >> 5) % NT, qp = (e >> 5) / NT;
      const int G = G0 + (qp >> 2), wd = qp & 3;
      const int tok = m0 + 8 * tt + (ln >> 2), kb = G * 128 + (ln & 3) * 32 + 8 * wd;
      if (tok < m) {
        const __nv_bfloat16* xr = x + (size_t)tok * k + kb;
        if (kb + 8 <= k && ((reinterpret_cast<uintptr_t>(xr) & 15) == 0)) {
          v[j] = __ldg(reinterpret_cast<const uint4*>(xr));
        } else {
          uint32_t h[8];
#pragma unroll
          for (int q = 0; q < 8; ++q) h[q] = kb + q < k ? (uint32_t)__bfloat16_as_ushort(xr[q]) : 0u;
          v[j] = make_uint4(h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16, h[6] | h[7] << 16);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = e0 + j * 256;
      if (e >= nst) continue;
      const int ln = e & 31, tt = (e >> 5) % NT, qp = (e >> 5) / NT;
      xs[(qp * NT + tt) * 32 + ln] = make_uint4(__byte_perm(v[j].x, v[j].z, 0x5410), __byte_perm(v[j].x, v[j].z, 0x7632),
                                                __byte_perm(v[j].y, v[j].w, 0x5410), __byte_perm(v[j].y, v[j].w, 0x7632));
      const uint32_t u[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
      float ps = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) ps += __uint_as_float(u[q] << 16), ps += __uint_as_float(u[q] & 0xffff0000u);
      sxp[(((qp >> 2) * 8 * NT) + 8 * tt + (ln >> 2)) * 16 + (ln & 3) * 4 + (qp & 3)] = ps;
    }
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  for (int e = tid; e < ngc * 8 * NT; e += 256) {
    float sx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) sx += sxp[e * 16 + j];
    sxs[e] = sx;
  }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  float acc[NT][4];
#pragma unroll
  for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  for (int i = 0; i < T; i += 2) {
    uint32_t w8[2][8], sw[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int sl = (i + h) % STAGES;
      mbar_wait(smem_u32(&full[sl]), ((i + h) / STAGES) & 1);
      const unsigned char* src = ring + (size_t)sl * STG;
      const uint4 u0 = reinterpret_cast<const uint4*>(src + warp * 1024)[lane];
      const uint4 u1 = reinterpret_cast<const uint4*>(src + warp * 1024 + 512)[lane];
      sw[h] = reinterpret_cast<const uint32_t*>(src + 8 * 1024 + warp * 32)[g];
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&empty[sl])) : "memory");
      w8[h][0] = u0.x, w8[h][1] = u0.y, w8[h][2] = u0.z, w8[h][3] = u0.w;
      w8[h][4] = u1.x, w8[h][5] = u1.y, w8[h][6] = u1.z, w8[h][7] = u1.w;
    }
    const int Gl = i % ngc, rb = 8 * (c + (i / ngc) * C) + warp;
    if (rb >= nblk) {  // past the last row block: the copies were consumed, nothing to compute
      continue;
    }
    float cf[2][NT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) cf[h][tt][0] = cf[h][tt][1] = cf[h][tt][2] = cf[h][tt][3] = 0.f;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint4* xq = xs + ((size_t)(Gl + h) * 4 + wd) * NT * 32 + lane;
        uint4 b[NT];
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) b[tt] = xq[tt * 32];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const uint32_t w = w8[h][2 * wd + e];
          const uint32_t a0 = kf_raw(w), a1 = kf_raw(w >> 4), a2 = kf_raw(w >> 8), a3 = kf_raw(w >> 12);
#pragma unroll
          for (int tt = 0; tt < NT; ++tt)
            mma16816(cf[h][tt], a0, a1, a2, a3, e ? b[tt].z : b[tt].x, e ? b[tt].w : b[tt].y);
        }
      }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float s0 = __uint_as_float(sw[h] << 16), s1 = __uint_as_float(sw[h] & 0xffff0000u);
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) {
        const float2 sx = *reinterpret_cast<const float2*>(sxs + (Gl + h) * 8 * NT + 8 * tt + 2 * t4);
        cf[h][tt][0] = fmaf(-136.f, sx.x, cf[h][tt][0]);
        cf[h][tt][1] = fmaf(-136.f, sx.y, cf[h][tt][1]);
        cf[h][tt][2] = fmaf(-136.f, sx.x, cf[h][tt][2]);
        cf[h][tt][3] = fmaf(-136.f, sx.y, cf[h][tt][3]);
        acc[tt][0] = fmaf(cf[h][tt][0], s0, acc[tt][0]);
        acc[tt][1] = fmaf(cf[h][tt][1], s0, acc[tt][1]);
        acc[tt][2] = fmaf(cf[h][tt][2], s1, acc[tt][2]);
        acc[tt][3] = fmaf(cf[h][tt][3], s1, acc[tt][3]);
      }
    }
    if (Gl + 1 != ngc - 1) continue;
    int sg = 0;
#pragma unroll
    for (int j = 1; j < kMaxSegs; ++j) sg += j < segs.nseg && 16 * rb >= segs.end[j - 1];
    const int lrb = rb - (sg ? segs.end[sg - 1] >> 4 : 0);
    __nv_bfloat16* __restrict__ y = segs.y[sg];
    const int ldy = segs.ldy[sg], row = 16 * lrb + g;
    bool out = S == 1;
    if (!out) {
      float4* pp = reinterpret_cast<float4*>(part) + ((size_t)(s * nblk + rb) * 32 + lane) * NT;
#pragma unroll
      for (int tt = 0; tt < NT; ++tt) __stcg(pp + tt, make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]));
      __threadfence();
      __syncwarp();
      unsigned old = 0;
      if (lane == 0) old = atomicAdd(cnt + rb, 1u);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old == (unsigned)(S - 1)) {
        __threadfence();
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
        for (int ss = 0; ss < S; ++ss) {
          const float4* qq = reinterpret_cast<const float4*>(part) + ((size_t)(ss * nblk + rb) * 32 + lane) * NT;
#pragma unroll
          for (int tt = 0; tt < NT; ++tt) {
            const float4 u = __ldcg(qq + tt);
            acc[tt][0] += u.x, acc[tt][1] += u.y, acc[tt][2] += u.z, acc[tt][3] += u.w;
          }
        }
        if (lane == 0) cnt[rb] = 0;
        out = true;
      }
    }
    if (out)
#pragma unroll
      for (int tt = 0; tt < NT; ++tt)
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int tok = m0 + 8 * tt + 2 * t4 + cc;
          if (tok < m) {
            y[(size_t)tok * ldy + row] = __float2bfloat16_rn(acc[tt][cc]);
            y[(size_t)tok * ldy + row + 8] = __float2bfloat16_rn(acc[tt][2 + cc]);
          }
        }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
  }
#endif
}
struct KfScratch {
  float* part = nullptr;
  unsigned* cnt = nullptr;
  size_t part_floats = 0;
  int cnt_n = 0;
};
template <int NT>
int kf_launch(const void* x, int m, int m0, const KtSegs& g, int k, cudaStream_t s, KfScratch& z) {
  static const int Pe = getenv("AGENS_W4F_P") ? atoi(getenv("AGENS_W4F_P")) : 8;
  const int P = Pe == 4 ? 4 : Pe == 16 ? 16 : 8;
  static const int minb = getenv("AGENS_W4F_MINB") ? atoi(getenv("AGENS_W4F_MINB")) : 2;
  auto kern = P == 4 ? (minb >= 4 ? w4f_kernel<NT, 4, 4> : minb == 3 ? w4f_kernel<NT, 4, 3> : w4f_kernel<NT, 4, 2>)
            : P == 16 ? w4f_kernel<NT, 16, 1>
            : (minb >= 3 ? w4f_kernel<NT, 8, 3> : w4f_kernel<NT, 8, 2>);
  const int kKfChunk = kf_chunk();
  const int kp = (k + 1023) & ~1023, S = (kp + kKfChunk - 1) / kKfChunk, nblk = g.end[g.nseg - 1] >> 4;
  const int smem = (kKfChunk / 16) * NT * 32 * 8 + (kKfChunk / 128) * 8 * NT * 4 * 17  // x chunk + group sums
                 + P * 256 * (32 + 4);                                                // weight / scale ring
  static int per_sm = -1;
  if (per_sm < 0) {
    if (smem > 48 * 1024 && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 256, smem);
    if (per_sm < 1) return -1;
  }
  if (S > 1) {
    const size_t need = (size_t)S * nblk * 32 * NT * 4;
    if (need > z.part_floats || nblk > z.cnt_n) {
      cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(s, &cs);
      if (cs != cudaStreamCaptureStatusNone) return -2;  // never allocate inside a capture
      cudaStreamSynchronize(s);
      if (z.part) cudaFree(z.part), cudaFree(z.cnt);
      z.part_floats = std::max(need, (size_t)8 << 20), z.cnt_n = std::max(nblk, 8192);
      if (cudaMalloc(&z.part, z.part_floats * 4) || cudaMalloc(&z.cnt, (size_t)z.cnt_n * 4)) return -2;
      cudaMemset(z.cnt, 0, (size_t)z.cnt_n * 4);
      cudaDeviceSynchronize();
    }
  }
  // CTAs per chunk: enough warps for the row blocks, at most the resident CTAs spread over the S chunks
  const int resident = num_sms() * per_sm;
  int C = std::max(1, resident / S);
  C = std::min(C, (nblk + 7) / 8);
  static const int bulk = [] {
    const char* e = getenv("AGENS_W4F_BULK");
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 9 && !(e && e[0] == '0') ? (e ? atoi(e) : 8) : 0;
  }();
  if (bulk) {
    auto kb = bulk == 4 ? w4fb_kernel<NT, 4> : bulk == 12 ? w4fb_kernel<NT, 12> : w4fb_kernel<NT, 8>;
    const int stages = bulk == 4 ? 4 : bulk == 12 ? 12 : 8;
    const int smb = stages * (8 * 1024 + 8 * 32) + (kKfChunk / 16) * NT * 32 * 8 + (kKfChunk / 128) * 8 * NT * 4 * 17;
    static int per_b = -1;
    if (per_b < 0) {
      if (cudaFuncSetAttribute(kb, cudaFuncAttributeMaxDynamicSharedMemorySize, smb)) return -1;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_b, kb, 9 * 32, smb);
      if (per_b < 1) return -1;
    }
    const int noct = (nblk + 7) / 8;
    const int Cb = std::max(1, std::min(noct, num_sms() * per_b / S));
    kb<<<S * Cb, 9 * 32, smb, s>>>(static_cast<const __nv_bfloat16*>(x), m, m0, g, k, S, Cb, z.part, z.cnt, kKfChunk);
    return (int)cudaGetLastError();
  }
  static const int probe = getenv("AGENS_W4F_PROBE") ? atoi(getenv("AGENS_W4F_PROBE")) : 0;
  kern<<<S * C, 256, smem, s>>>(static_cast<const __nv_bfloat16*>(x), m, m0, g, k, S, C, z.part, z.cnt, kKfChunk, probe);
  return (int)cudaGetLastError();
}
bool kf_segs_ok(const KtSegs& g) {
  for (int i = 0; i < g.nseg; ++i)
    if (!g_kf_set.count(g.wt[i])) return false;
  return true;
}
int kf_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s) {
  static std::map<std::pair<int, cudaStream_t>, KfScratch> scr;
  int dev = 0;
  cudaGetDevice(&dev);
  KfScratch& z = scr[{dev, s}];
  static int ntmax = -1;
  if (ntmax < 0) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    const int per = (kf_chunk() / 16) * 32 * 8 + (kf_chunk() / 128) * 32 * 17;  // smem per 8-token tile
    ntmax = std::max(1, std::min(4, (optin - 8 * 256 * 36) / per));
  }
  for (int m0 = 0; m0 < m; m0 += 8 * ntmax) {  // passes of up to ntmax 8-token tiles (same bits per column)
    const int nt = std::min(ntmax, (m - m0 + 7) / 8);
    const int e = nt == 1 ? kf_launch<1>(x, m, m0, g, k, s, z) : nt == 2 ? kf_launch<2>(x, m, m0, g, k, s, z)
                : nt == 3 ? kf_launch<3>(x, m, m0, g, k, s, z) : kf_launch<4>(x, m, m0, g, k, s, z);
    if (e) return e;
  }
  return 0;
}

int mma_segs(const void* x, int m, const KtSegs& g, int k, cudaStream_t s, int group, int bits) {
  if (bits == 4 && group == 128 && !g_kf_set.empty() && kf_segs_ok(g)) {  // KF layout (AGENS_W4F)
    const int e = kf_segs(x, m, g, k, s);
    return e < 0 ? (int)cudaErrorInvalidValue : e;
  }
  switch (bits * 1000 + group) {
    case 4064: return mma64_segs<4>(x, m, g, k, s);
    case 3064: return mma64_segs<3>(x, m, g, k, s);
    case 2064: return mma64_segs<2>(x, m, g, k, s);
    case 4128: return mma128_segs<4>(x, m, g, k, s);
    case 3128: return mma128_segs<3>(x, m, g, k, s);
    case 2128: return mma128_segs<2>(x, m, g, k, s);
    default: return (int)cudaErrorInvalidValue;
  }
}
}  // namespace

// Tensor-core W4 GEMV (g128 / g64), 1 <= m <= 16 tokens; rows bit-identical for any m. Used by ak_w4_gemv_kt / _multi /
// _batched by default (bf16w = 0; AGENS_W4_EXACT=1 or ak_w4_set_mma(0) select the exact path); also direct.
extern "C" int ak_w4_mma_gemv(const void* x, const void* wt, const void* st, void* y, int m, int n, int k, int group,
                              void* stream) {
  if (!group_ok(k, group) || m < 1 || m > 128) return (int)cudaErrorInvalidValue;
  if (const int e = w4h_route(x, m, 1, &wt, &st, &y, &n, nullptr, k, group, 0, stream); e >= 0) return e;
  return mma_segs(x, m, make_segs(1, &wt, &st, &y, &n, 16), k, static_cast<cudaStream_t>(stream), group, 4);
}
extern "C" int ak_w4_set_lean(int rb) {  // lean register path: -1 auto (env / arch), 0 off, 1 / 2 / 4 row blocks
  g_lean = rb < 0 ? -1 : rb;
  return 0;
}
extern "C" int ak_w4_set_tma(int stages) {  // sm90 bulk-copy path: -1 auto (env), 0 off, 2 / 4 / 8 stages
  g_tma = stages < 0 ? -1 : stages;
  return 0;
}
extern "C" int ak_w4_set_mma(int mode) {  // -1 auto, 0 off, 1 on
  g_mma_mode = mode < 0 ? -1 : (mode ? 1 : 0);
  return 0;
}

// ---------------------------------------------------------------------------------------------- KF entry points
int agens_kf_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, cudaStream_t s) {
  if (!kf_on() || n % 16 != 0 || k % 128 != 0) return 1;
  const int kp = (k + 1023) & ~1023;
  const size_t wb = (size_t)n * kp / 2, sb = (size_t)n * kp / 128 * 2;
  void *tw = nullptr, *ts = nullptr;
  if (cudaMalloc(&tw, wb) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
  if (cudaMalloc(&ts, sb) != cudaSuccess) return cudaFree(tw), (int)cudaErrorMemoryAllocation;
  repack_kt_kernel<128><<<1024, 256, 0, s>>>(reinterpret_cast<const uint4*>(packed), static_cast<const __nv_bfloat16*>(scales),
                                             static_cast<uint4*>(tw), static_cast<__nv_bfloat16*>(ts), n, k);
  kt_to_kf_kernel<<<num_sms() * 8, 256, 0, s>>>(static_cast<const uint32_t*>(tw), static_cast<const __nv_bfloat16*>(ts),
                                                static_cast<uint32_t*>(wt), static_cast<uint32_t*>(st), n, kp);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(s);
  cudaFree(tw), cudaFree(ts);
  if (e != cudaSuccess) return (int)e;
  g_kf_set.insert(wt);
  return 0;
}
bool agens_kf_is(const void* wt) { return !g_kf_set.empty() && g_kf_set.count(wt); }
void agens_kf_forget(const void* wt) { g_kf_set.erase(wt); }
extern "C" int ak_w4_kf_forget(const void* wt) { return agens_kf_forget(wt), 0; }
int agens_kf_dequant_rows(const void* wt, const void* st, void* w, int n, int k, int r0, int nr, cudaStream_t s) {
  dequant_kf_kernel<<<num_sms() * 8, 256, 0, s>>>(static_cast<const uint32_t*>(wt), static_cast<const uint32_t*>(st),
                                                  static_cast<__nv_bfloat16*>(w), n, k, r0, nr);
  return (int)cudaGetLastError();
}
// KF on / off for weights repacked from now on (tests; the engine reads AGENS_W4F); returns the previous setting
extern "C" int ak_w4_set_kf(int on) {
  const int prev = kf_on();
  g_kf = on ? 1 : 0;
  return prev;
}
extern "C" int ak_w4_is_kf(const void* wt) { return agens_kf_is(wt); }

