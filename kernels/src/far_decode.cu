// BCSA far branch, decode (one query token), on engine kernels: replaces the ATen bridge (ak_bridge_far_decode,
// ~55 ATen kernels per BCSA layer) with 6 kernels + 3 cuBLAS calls, no allocation, no host sync, graph-capturable.
// Numerics follow sglv23's decode path (volundr_fused.far_sparse_fused + VolundrAttention._merge) op for op:
//  * the three einsums are cuBLAS GEMMs issued as torch's einsum -> bmm issues them (same transposes, leading dims,
//    batch strides, compute type, algo, workspace, math mode), so the same cuBLAS build picks the same kernels;
//  * topk(512) of the bf16 block scores: torch's gatherTopK + stable radix sort order = (radix key desc, index asc),
//    reproduced exactly by a radix select on (key << 16 | ~index) + a sort of the 512 winners;
//  * _merge in fp32 with torch's roundings (expf, div.rn, separate mul/add, no contraction) and its 512-wide sum in
//    ATen's reduction order for this shape (Reduce.cuh: vectorized input, one warp per row, 4 accumulators per lane
//    over x[4 * (lane + 32 j) + i], combined 0+1+2+3, then shuffle-down 16..1);
//  * means of 4 bf16 values in fp32 (exact), rounding points as sglv23's Triton kernels.
// Layout of the workspace (ak_far_decode_ws_bytes): see Ws below; tests read the intermediates from it.
#include "agens_kernels.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {
using bf16 = __nv_bfloat16;
constexpr int C = 4, W = 4096, D = 256, G = 6, DI = 128, IDXH = 4, KSEL = 512;
// q / kv heads of this process (tensor parallel: a rank's share, ak_far_set_heads; the full model: 24 / 4, G = 6 always)
__constant__ int g_far_h_dev = 24, g_far_kv_dev = 4;
int g_far_h_host = 24, g_far_kv_host = 4;
__host__ __device__ __forceinline__ int far_h() {
#ifdef __CUDA_ARCH__
  return g_far_h_dev;
#else
  return g_far_h_host;
#endif
}
__host__ __device__ __forceinline__ int far_kv() {
#ifdef __CUDA_ARCH__
  return g_far_kv_dev;
#else
  return g_far_kv_host;
#endif
}
#define H far_h()
#define KV far_kv()

struct Ws {  // carved from one device buffer, 256-byte aligned pieces
  bf16* kbi;   // [NB][128]
  int8_t* valid;  // [NB]
  bf16* e;     // [4][NB]  einsum("bhd,bnd->bhn")
  bf16* s;     // [NB]     block scores
  long long* top;  // [KSEL]
  bf16 *kb, *vb;   // [KSEL][4][256]
  int8_t* ok;      // [KSEL]
  bf16* e2;    // [4][6][KSEL] einsum("bkgd,bnkd->bkgn")
  bf16* pf;    // [24][KSEL]   softmax weights (bf16, as torch's pf.to(bf16))
  float* wz;   // [24]         w / z
  bf16* ofar;  // [4][6][256]  einsum("tkgn,tnkd->tkgd")
};
size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
Ws carve(void* base, int NB) {
  char* p = static_cast<char*>(base);
  Ws w;
  auto take = [&](size_t bytes) { char* q = p; p += al(bytes); return q; };
  w.kbi = (bf16*)take((size_t)NB * DI * 2);
  w.valid = (int8_t*)take(NB);
  w.e = (bf16*)take((size_t)IDXH * NB * 2);
  w.s = (bf16*)take((size_t)NB * 2);
  w.top = (long long*)take(KSEL * 8);
  w.kb = (bf16*)take((size_t)KSEL * KV * D * 2);
  w.vb = (bf16*)take((size_t)KSEL * KV * D * 2);
  w.ok = (int8_t*)take(KSEL);
  w.e2 = (bf16*)take((size_t)H * KSEL * 2);
  w.pf = (bf16*)take((size_t)H * KSEL * 2);
  w.wz = (float*)take(H * 4);
  w.ofar = (bf16*)take((size_t)H * D * 2);
  return w;
}
size_t ws_bytes(int NB) {
  return al((size_t)NB * DI * 2) + al(NB) + al((size_t)IDXH * NB * 2) + al((size_t)NB * 2) + al(KSEL * 8) +
         2 * al((size_t)KSEL * KV * D * 2) + al(KSEL) + 2 * al((size_t)H * KSEL * 2) + al(H * 4) + al((size_t)H * D * 2);
}

__device__ __forceinline__ long long nmax_of(long long pos) {  // last far-valid block (floor division), -1 if none
  const long long num = pos - W - (C - 1);
  return num >= 0 ? num / C : -1;
}

// kbi[n] = blk[r2t[n*C + C-1] / C] (block keys per 4-row page) for far-valid n, zeros elsewhere; valid[n]. One
// thread per 16 bytes.
__global__ void kbi_kernel(const long long* pos, const int* r2t, int Wctx, const uint4* blk, uint4* kbi, int8_t* valid,
                           int NB, int hi) {
  const long long nm = nmax_of(pos[0]);
  const int per = DI * 2 / 16;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NB * per; i += gridDim.x * blockDim.x) {
    const int n = i / per, c = i % per;
    const bool v = n <= nm;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (v) {
      int slot = r2t[(long long)n * C + (C - 1)];
      slot = slot < 0 ? 0 : (slot > hi ? hi : slot);
      x = blk[(size_t)(slot / C) * per + c];  // block keys are stored per 4-row page (row / 4)
    }
    kbi[i] = x;
    if (c == 0) valid[n] = v;
  }
}

// s[n] = bf16(bf16(((e0+e1)+e2)+e3) * 0.25) * scale) or -inf where invalid
// pos != nullptr: validity n <= nmax(pos) instead of valid[] (the multi-query path gathers kbi once for the last query;
// rows past a query's own nmax only feed masked scores)
__global__ void score_kernel(const bf16* e, const int8_t* valid, bf16* s, int NB, float scale, const long long* pos) {
  const long long nm = pos ? nmax_of(pos[0]) : 0;
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < NB; n += gridDim.x * blockDim.x) {
    const float e0 = __bfloat162float(e[n]), e1 = __bfloat162float(e[NB + n]), e2 = __bfloat162float(e[2 * NB + n]),
                e3 = __bfloat162float(e[3 * NB + n]);
    const float m = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3), 0.25f)));
    const bool v = pos ? n <= nm : valid[n] != 0;
    s[n] = v ? __float2bfloat16_rn(__fmul_rn(m, scale)) : __float2bfloat16_rn(-INFINITY);
  }
}

// torch's radix key of a bf16 (TopKTypeConfig<BFloat16>::convert): order-preserving, -0 below +0
__device__ __forceinline__ uint32_t radix_key(bf16 v) {
  const uint32_t x = __bfloat16_as_ushort(v);
  return x ^ ((x & 0x8000u) ? 0xFFFFu : 0x8000u);
}

// top KSEL of s by (key desc, index asc), output sorted that way (= torch.topk(sorted=True) on CUDA). One block.
// Composite keys (key << IB) | (IMAX - n) are unique (NB <= IMAX): radix-select the KSEL-th largest in byte passes,
// gather the >= winners, bitonic-sort them descending: order (key desc, index asc). IB = 16 (32-bit keys, 4 passes) for
// NB < 65536; IB = 32 (64-bit keys, 6 passes) above it (Wctx > 262143, i.e. 262K context): the same order.
template <typename K, int IB>
__global__ void __launch_bounds__(1024) topk_kernel(const bf16* s, int NB, long long* top) {
  constexpr K IMAX = (K)((1ull << IB) - 1);
  __shared__ int hist[256];
  __shared__ K prefix, mask;
  __shared__ int need, cnt;
  __shared__ K sel[KSEL];
  const int tid = threadIdx.x;
  if (tid == 0) prefix = 0, mask = 0, need = KSEL, cnt = 0;
  for (int shift = 16 + IB - 8; shift >= 0; shift -= 8) {
    for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int n = tid; n < NB; n += blockDim.x) {
      const K k = ((K)radix_key(s[n]) << IB) | (IMAX - (K)n);
      if ((k & mask) == prefix) atomicAdd(&hist[(int)(k >> shift) & 255], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int cum = 0, b = 255;
      for (; b > 0 && cum + hist[b] < need; --b) cum += hist[b];
      prefix |= (K)b << shift, mask |= (K)255 << shift, need -= cum;
    }
    __syncthreads();
  }
  for (int n = tid; n < NB; n += blockDim.x) {
    const K k = ((K)radix_key(s[n]) << IB) | (IMAX - (K)n);
    if (k >= prefix) sel[atomicAdd(&cnt, 1)] = k;
  }
  __syncthreads();
  for (int j = cnt + tid; j < KSEL; j += blockDim.x) sel[j] = 0;  // NB < KSEL (not with Volundr's Wctx)
  // bitonic sort, descending
  for (int size = 2; size <= KSEL; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int i = tid; i < KSEL; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const K a = sel[i], b = sel[j];
          if (desc ? a < b : a > b) sel[i] = b, sel[j] = a;
        }
      }
    }
  __syncthreads();
  for (int i = tid; i < KSEL; i += blockDim.x) top[i] = (long long)(IMAX - (sel[i] & IMAX));
}

// Same output as topk_kernel (top KSEL by (key desc, index asc), sorted that way), faster: one read of s into
// registers (thread i holds blocks [i*EPT, (i+1)*EPT), its 16-bit radix keys), two byte passes find the KSEL-th
// largest key K*, every key > K* is selected and keys == K* in ascending block order (a block-wide scan) until KSEL;
// the KSEL winners are bitonic-sorted as composites (key << 32 | ~index): the order topk_kernel produces.
template <int EPT>
__device__ __forceinline__ void topk2_body(const bf16* s, int NB, long long* top) {
  __shared__ int hist[256];
  __shared__ int sel_cnt, thr_key, need_eq;
  __shared__ int wsum[32];
  __shared__ unsigned long long sel[KSEL];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t kk[(EPT + 1) / 2];  // two 16-bit keys per word
  const int base = tid * EPT;
#pragma unroll
  for (int e = 0; e < EPT; e += 2) {
    const uint32_t a = base + e < NB ? radix_key(s[base + e]) : 0u;
    const uint32_t b = (e + 1 < EPT && base + e + 1 < NB) ? radix_key(s[base + e + 1]) : 0u;
    kk[e / 2] = a | (b << 16);
  }
  auto key = [&](int e) -> uint32_t { return (kk[e / 2] >> (16 * (e & 1))) & 0xFFFFu; };
  auto valid = [&](int e) { return base + e < NB; };
  // histogram counts are added per run of equal buckets within a thread (one atomic per run: the -inf scores of the
  // blocks past the context are one bucket, so per-element atomics serialized on it); counts are exact either way
  auto count_runs = [&](auto bucket_of) {
    int cur = -1, n = 0;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
      const int b = valid(e) ? bucket_of(key(e)) : -1;
      if (b != cur) {
        if (cur >= 0) atomicAdd(&hist[cur], n);
        cur = b, n = 0;
      }
      n += b >= 0;
    }
    if (cur >= 0) atomicAdd(&hist[cur], n);
  };
  // the serial scan "for (b = 255; b > 0 && cum + hist[b] < target; --b) cum += hist[b]" by warp 0: lane L sums buckets
  // 255 - 8L .. 248 - 8L, a lane scan finds the lane where the cumulative count reaches target, that lane walks its 8
  auto scan = [&](int target, int* b_out, int* cum_out) {
    if (warp != 0) return;
    const int hi = 255 - 8 * lane;
    int g = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) g += hist[hi - i];
    int incl = g;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xFFFFFFFFu, incl, o);
      if (lane >= o) incl += y;
    }
    const unsigned hit = __ballot_sync(0xFFFFFFFFu, incl >= target);
    const int lf = hit ? __ffs(hit) - 1 : 31;
    if (lane == lf) {
      int cum = incl - g, b = hi;
      for (; b > 0 && b > hi - 8 && cum + hist[b] < target; --b) cum += hist[b];
      *b_out = b, *cum_out = cum;
    }
  };
  // pass 1: high byte
  for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
  __syncthreads();
  count_runs([](uint32_t k) { return (int)(k >> 8); });
  __syncthreads();
  {
    __shared__ int sb, sc;
    scan(KSEL, &sb, &sc);
    __syncwarp();
    if (tid == 0) thr_key = sb << 8, need_eq = KSEL - sc;  // need_eq: still needed from bucket b
  }
  __syncthreads();
  const uint32_t hb = (uint32_t)thr_key >> 8;
  const int need1 = need_eq;
  for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
  __syncthreads();
  count_runs([hb](uint32_t k) { return (k >> 8) == hb ? (int)(k & 255) : -1; });
  __syncthreads();
  {
    __shared__ int sb, sc;
    scan(need1, &sb, &sc);
    __syncwarp();
    if (tid == 0) thr_key = (int)((hb << 8) | (uint32_t)sb), need_eq = need1 - sc, sel_cnt = 0;
  }
  __syncthreads();
  const uint32_t K = (uint32_t)thr_key;
  const int need = need_eq;
  // rank of this thread's == K keys among all == K keys in index order: exclusive block scan of per-thread counts
  int eq = 0;
#pragma unroll
  for (int e = 0; e < EPT; ++e) eq += valid(e) && key(e) == K;
  int x = eq;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = lane < (int)(blockDim.x >> 5) ? wsum[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xFFFFFFFFu, w, o);
      if (lane >= o) w += y;
    }
    wsum[lane] = w;  // inclusive
  }
  __syncthreads();
  int rank = x - eq + (warp ? wsum[warp - 1] : 0);  // exclusive prefix of == K keys before this thread
#pragma unroll
  for (int e = 0; e < EPT; ++e) {
    if (!valid(e)) continue;
    const uint32_t k = key(e);
    bool take = k > K;
    if (k == K) take = rank++ < need;
    if (take) sel[atomicAdd(&sel_cnt, 1)] = ((unsigned long long)k << 32) | (0xFFFFFFFFull - (unsigned long long)(base + e));
  }
  __syncthreads();
  for (int size = 2; size <= KSEL; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int i = tid; i < KSEL; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const unsigned long long a = sel[i], b = sel[j];
          if (desc ? a < b : a > b) sel[i] = b, sel[j] = a;
        }
      }
    }
  __syncthreads();
  for (int i = tid; i < KSEL; i += blockDim.x) top[i] = (long long)(0xFFFFFFFFull - (sel[i] & 0xFFFFFFFFull));
}
template <int EPT>
__global__ void __launch_bounds__(1024) topk2_kernel(const bf16* s, int NB, long long* top) {
  topk2_body<EPT>(s, NB, top);
}
bool topk2_ok(int NB);
// top-k dispatch: topk2 for NB <= 1024 * 72 (Wctx <= 294912), else the radix-select kernel. AGENS_FAR_TOPK=radix: always
// the radix-select kernel (reference).
int g_topk2 = -1;
bool topk2_ok(int NB) {
  if (g_topk2 < 0) {
    const char* e = getenv("AGENS_FAR_TOPK");
    g_topk2 = !(e && !strcmp(e, "radix"));
  }
  return g_topk2 && NB >= KSEL && NB <= 1024 * 72;
}
void topk_launch(const bf16* s, int NB, long long* top, cudaStream_t st) {
  if (topk2_ok(NB)) {
    // keys per thread: the smallest tier that holds NB (Wctx = 4096 n + 4 gives NB = 1024 n + 1); output independent
    if (NB <= 1024 * 8) topk2_kernel<8><<<1, 1024, 0, st>>>(s, NB, top);
    else if (NB <= 1024 * 24) topk2_kernel<24><<<1, 1024, 0, st>>>(s, NB, top);
    else if (NB <= 1024 * 26) topk2_kernel<26><<<1, 1024, 0, st>>>(s, NB, top);
    else if (NB <= 1024 * 34) topk2_kernel<34><<<1, 1024, 0, st>>>(s, NB, top);
    else if (NB <= 1024 * 50) topk2_kernel<50><<<1, 1024, 0, st>>>(s, NB, top);
    else if (NB <= 1024 * 66) topk2_kernel<66><<<1, 1024, 0, st>>>(s, NB, top);
    else topk2_kernel<72><<<1, 1024, 0, st>>>(s, NB, top);
  } else if (NB < 65536) {
    topk_kernel<uint32_t, 16><<<1, 1024, 0, st>>>(s, NB, top);
  } else {
    topk_kernel<unsigned long long, 32><<<1, 1024, 0, st>>>(s, NB, top);
  }
}

// kb/vb[j] = mean of the C K/V cache rows of block top[j] (serial fp32 sum, x 1/C, rounded); ok[j] = far-valid
__global__ void pool_kernel(const long long* top, const long long* pos, const int* r2t, int Wctx, const bf16* kc,
                            const bf16* vc, bf16* kb, bf16* vb, int8_t* ok, int hi,
                                  const int* nvalid) {  // nvalid: only blocks j < *nvalid (union fetch); ok may be null
  const int j = blockIdx.x;
  if (nvalid && j >= *nvalid) return;
  const long long t = top[j];
  if (ok && threadIdx.x == 0) ok[j] = t <= nmax_of(pos[0]);
  int slot[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    long long col = t * C + c;
    col = col < 0 ? 0 : (col > Wctx - 1 ? Wctx - 1 : col);
    const int sr = r2t[col];
    slot[c] = sr < 0 ? 0 : (sr > hi ? hi : sr);
  }
  // 8 consecutive values per thread with 16-byte loads (same per-element arithmetic; fewer, wider transactions, which
  // matters when kc/vc are host-mapped pools read over PCIe)
  for (int r8 = threadIdx.x; r8 < KV * D / 8; r8 += blockDim.x) {
    float ak[8] = {}, av[8] = {};
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const uint4 ku = reinterpret_cast<const uint4*>(kc + (size_t)slot[c] * KV * D)[r8];
      const uint4 vu = reinterpret_cast<const uint4*>(vc + (size_t)slot[c] * KV * D)[r8];
      const bf16* ke = reinterpret_cast<const bf16*>(&ku);
      const bf16* ve = reinterpret_cast<const bf16*>(&vu);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        ak[e] = __fadd_rn(ak[e], __bfloat162float(ke[e]));
        av[e] = __fadd_rn(av[e], __bfloat162float(ve[e]));
      }
    }
    alignas(16) bf16 ko[8], vo[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      ko[e] = __float2bfloat16_rn(__fmul_rn(ak[e], 1.0f / C));
      vo[e] = __float2bfloat16_rn(__fmul_rn(av[e], 1.0f / C));
    }
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(ko);
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(vo);
  }
}

// KV tiering (kernels/m1/m1_kv.cu) variants of pool_kernel, same output: (1) the block means already stored per 4-row
// page (ak_kv_block_mean == the serial mean above, bitwise): kb[j] = km[r2t[top*C + C-1] / C]; (2) int8 rows + fp32
// scale per (row, kv head): each value dequantized to bf16(q * s) first, then the same mean. Pools may be host-mapped
// (the gather is the PCIe fetch of the selected blocks, inside the graph).
__global__ void pool_means_kernel(const long long* top, const long long* pos, const int* r2t, int Wctx, const bf16* km,
                                  const bf16* vm, bf16* kb, bf16* vb, int8_t* ok, int hi,
                                  const int* nvalid) {  // nvalid: only blocks j < *nvalid (union fetch); ok may be null
  const int j = blockIdx.x;
  if (nvalid && j >= *nvalid) return;
  const long long t = top[j];
  if (ok && threadIdx.x == 0) ok[j] = t <= nmax_of(pos[0]);
  long long col = t * C + C - 1;
  col = col < 0 ? 0 : (col > Wctx - 1 ? Wctx - 1 : col);
  const int sr = r2t[col], page = (sr < 0 ? 0 : (sr > hi ? hi : sr)) / C;
  const uint4* ks = reinterpret_cast<const uint4*>(km + (size_t)page * KV * D);
  const uint4* vs = reinterpret_cast<const uint4*>(vm + (size_t)page * KV * D);
  for (int r = threadIdx.x; r < KV * D / 8; r += blockDim.x) {
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r] = ks[r];
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r] = vs[r];
  }
}
// (3) AGENS_KV_MEANS_INT8: block means stored per page as int8 + fp32 scale per (page, kv head); read as bf16(q * s)
// (what ak_kv_dequant_rows gives the non-graph paths)
__global__ void pool_means_i8_kernel(const long long* top, const long long* pos, const int* r2t, int Wctx,
                                     const int8_t* km, const int8_t* vm, const float* ksc, const float* vsc, bf16* kb,
                                     bf16* vb, int8_t* ok, int hi, const int* nvalid) {
  const int j = blockIdx.x;
  if (nvalid && j >= *nvalid) return;
  const long long t = top[j];
  if (ok && threadIdx.x == 0) ok[j] = t <= nmax_of(pos[0]);
  long long col = t * C + C - 1;
  col = col < 0 ? 0 : (col > Wctx - 1 ? Wctx - 1 : col);
  const int sr = r2t[col], page = (sr < 0 ? 0 : (sr > hi ? hi : sr)) / C;
  for (int r8 = threadIdx.x; r8 < KV * D / 8; r8 += blockDim.x) {
    const int h = r8 * 8 / D;
    const uint2 ku = reinterpret_cast<const uint2*>(km + (size_t)page * KV * D)[r8];
    const uint2 vu = reinterpret_cast<const uint2*>(vm + (size_t)page * KV * D)[r8];
    const int8_t* ke = reinterpret_cast<const int8_t*>(&ku);
    const int8_t* ve = reinterpret_cast<const int8_t*>(&vu);
    const float ks = ksc[(size_t)page * KV + h], vs = vsc[(size_t)page * KV + h];
    alignas(16) bf16 ko[8], vo[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      ko[e] = __float2bfloat16_rn(__fmul_rn((float)ke[e], ks));
      vo[e] = __float2bfloat16_rn(__fmul_rn((float)ve[e], vs));
    }
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(ko);
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(vo);
  }
}
__global__ void pool_int8_kernel(const long long* top, const long long* pos, const int* r2t, int Wctx, const int8_t* kq,
                                 const int8_t* vq, const float* ksc, const float* vsc, bf16* kb, bf16* vb, int8_t* ok,
                                 int hi,
                                  const int* nvalid) {  // nvalid: only blocks j < *nvalid (union fetch); ok may be null
  const int j = blockIdx.x;
  if (nvalid && j >= *nvalid) return;
  const long long t = top[j];
  if (ok && threadIdx.x == 0) ok[j] = t <= nmax_of(pos[0]);
  int slot[C];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    long long col = t * C + c;
    col = col < 0 ? 0 : (col > Wctx - 1 ? Wctx - 1 : col);
    const int sr = r2t[col];
    slot[c] = sr < 0 ? 0 : (sr > hi ? hi : sr);
  }
  for (int r8 = threadIdx.x; r8 < KV * D / 8; r8 += blockDim.x) {
    const int h = r8 * 8 / D;
    float ak[8] = {}, av[8] = {};
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const uint2 ku = reinterpret_cast<const uint2*>(kq + (size_t)slot[c] * KV * D)[r8];
      const uint2 vu = reinterpret_cast<const uint2*>(vq + (size_t)slot[c] * KV * D)[r8];
      const int8_t* ke = reinterpret_cast<const int8_t*>(&ku);
      const int8_t* ve = reinterpret_cast<const int8_t*>(&vu);
      const float ks = ksc[(size_t)slot[c] * KV + h], vs = vsc[(size_t)slot[c] * KV + h];
#pragma unroll
      for (int e = 0; e < 8; ++e) {  // bf16(q * s) first (what the dequantized staging held), then the mean
        ak[e] = __fadd_rn(ak[e], __bfloat162float(__float2bfloat16_rn(__fmul_rn((float)ke[e], ks))));
        av[e] = __fadd_rn(av[e], __bfloat162float(__float2bfloat16_rn(__fmul_rn((float)ve[e], vs))));
      }
    }
    alignas(16) bf16 ko[8], vo[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      ko[e] = __float2bfloat16_rn(__fmul_rn(ak[e], 1.0f / C));
      vo[e] = __float2bfloat16_rn(__fmul_rn(av[e], 1.0f / C));
    }
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(ko);
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r8] = *reinterpret_cast<const uint4*>(vo);
  }
}

// _merge up to the softmax weights, one warp per head row r (24 rows):
//   sf = ok ? float(bf16(e2 * scale)) : -inf;  m = max(lse, max sf);  w = exp(lse - m);  pf = exp(sf - m);
//   z = w + pf.sum() (ATen order);  pf_bf16 = bf16(pf / z);  wz = w / z
__global__ void softmax_kernel(const bf16* e2, const int8_t* ok, const float* lse, float scale, bf16* pf_out, float* wz) {
  const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (r >= H) return;
  float sf[16];  // this lane's elements x[4 * (lane + 32 j) + i], j < 4, i < 4 (ATen's vectorized-input layout)
  float mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = 4 * (lane + 32 * j) + i;
      const float v = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(e2[r * KSEL + n]), scale)));
      sf[4 * j + i] = ok[n] ? v : -INFINITY;
      mx = fmaxf(mx, sf[4 * j + i]);
    }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
  const float l = lse[r];
  const float m = fmaxf(l, mx);  // torch.maximum (no NaN here)
  const float w = expf(__fsub_rn(l, m));
  float pf[16], acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pf[4 * j + i] = expf(__fsub_rn(sf[4 * j + i], m));
      acc[i] = __fadd_rn(acc[i], pf[4 * j + i]);
    }
  float v = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
  const float z = __fadd_rn(w, __shfl_sync(0xFFFFFFFFu, v, 0));
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      pf_out[r * KSEL + 4 * (lane + 32 * j) + i] = __float2bfloat16_rn(__fdiv_rn(pf[4 * j + i], z));
  if (lane == 0) wz[r] = __fdiv_rn(w, z);
}

// out = bf16(wz * float(o_loc) + float(o_far)), rows r = kv head * 6 + g (einsum output [4][6][256] = [24][256])
__global__ void combine_kernel(const float* wz, const bf16* o_loc, const bf16* o_far, bf16* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H * D; i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16_rn(
        __fadd_rn(__fmul_rn(wz[i / D], __bfloat162float(o_loc[i])), __bfloat162float(o_far[i])));
}

// ---- fused attend (AGENS_FAR_ATTEND=fused; product numerics, the same for decode and every verify query) ----
// One block per (query t, kv head k): e2[g][j] = bf16(q[k*6+g] . kb[j][k]) (fp32, fixed order: lane-strided partial
// sums over 256 dims, then a butterfly), the softmax_kernel row math on those e2, o_far[g][d] = bf16(sum_j pf[g][j] *
// vb[j][k][d]) (fp32, j ascending), then combine_kernel's out. Rows j come from kb/vb (UNION = false, one query's
// [KSEL][4][256] blocks, ok[]) or from the union buffers through mark[top[j]] (UNION = true, ok from top <= nmax(pos)).
// No cuBLAS, no staging copies; an output depends only on its own query's data, never on T.
// MODE 0: kb / vb are one query's KSEL rows, ok_in; 1: union buffers via mark[top[j]]; 2: staged far blocks (the
// prefill extend: kb / vb hold block n at row n), row top[j]. pos == nullptr: query t's position is seq0 + t.
// The G = 6 heads of a kv head are split over kFaSub blocks of kFaHG heads (every head's math is independent of the
// others: bit-identical to one block of 6; 3x the blocks: T * KV = 32 CTAs left most of the GPU idle in verify).
constexpr int kFaHG = 2, kFaSub = G / kFaHG;
template <int MODE>
__global__ void __launch_bounds__(256) fused_attend_kernel(const bf16* __restrict__ qh, const bf16* __restrict__ o_loc,
                                                           const float* __restrict__ lse, const long long* pos,
                                                           const bf16* __restrict__ kb, const bf16* __restrict__ vb,
                                                           const int8_t* ok_in, const long long* tops,
                                                           const int* mark, float scale, bf16* __restrict__ out,
                                                           int seq0) {
  const int t = blockIdx.x / (KV * kFaSub), k = (blockIdx.x / kFaSub) % KV, g0 = (blockIdx.x % kFaSub) * kFaHG;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ bf16 e2s[kFaHG][KSEL];
  __shared__ bf16 pfs[kFaHG][KSEL];
  __shared__ int8_t oks[KSEL];
  __shared__ int rowu[KSEL];
  __shared__ float wzs[kFaHG];
  const bf16* q = qh + (size_t)t * H * D + (size_t)(k * G + g0) * D;
  const long long* top = tops + (size_t)t * KSEL;
  const long long nm = nmax_of(pos ? pos[t] : (long long)seq0 + t);
  for (int j = tid; j < KSEL; j += blockDim.x) {
    if (MODE == 1) {
      const long long b = top[j];
      rowu[j] = mark[b];
      oks[j] = b <= nm;
    } else if (MODE == 2) {
      const long long b = top[j];
      rowu[j] = (int)b;
      oks[j] = b <= nm;
    } else {
      rowu[j] = j;
      oks[j] = ok_in[j];
    }
  }
  // this lane's 8 dims (d = 8 * lane .. 8 * lane + 7) of the 6 query rows
  float qf[kFaHG][8];
#pragma unroll
  for (int g = 0; g < kFaHG; ++g) {
    const uint4 v = *reinterpret_cast<const uint4*>(q + (size_t)g * D + 8 * lane);
    const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) qf[g][i] = __bfloat162float(e[i]);
  }
  __syncthreads();
  // rows gathered kQ at a time (their loads in flight together; every value computed as before)
  constexpr int kQ = 4, kNW = 256 / 32;
  for (int j0 = warp; j0 < KSEL; j0 += kNW * kQ) {
    uint4 kv4[kQ];
#pragma unroll
    for (int u = 0; u < kQ; ++u) {
      const int j = j0 + u * kNW;
      if (j < KSEL) kv4[u] = *reinterpret_cast<const uint4*>(kb + ((size_t)rowu[j] * KV + k) * D + 8 * lane);
    }
#pragma unroll
    for (int u = 0; u < kQ; ++u) {
      const int j = j0 + u * kNW;
      if (j >= KSEL) break;
      const bf16* e = reinterpret_cast<const bf16*>(&kv4[u]);
      float kf[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) kf[i] = __bfloat162float(e[i]);
#pragma unroll
      for (int g = 0; g < kFaHG; ++g) {
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) a = fmaf(qf[g][i], kf[i], a);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
        if (lane == 0) e2s[g][j] = __float2bfloat16_rn(a);
      }
    }
  }
  __syncthreads();
  if (warp < kFaHG) {  // softmax_kernel's row math, row g = query head k*6+g
    const int g = warp, r = k * G + g0 + g;
    float sf[16];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int n = 4 * (lane + 32 * j) + i;
        const float v = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(e2s[g][n]), scale)));
        sf[4 * j + i] = oks[n] ? v : -INFINITY;
        mx = fmaxf(mx, sf[4 * j + i]);
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    const float l = lse[(size_t)t * H + r];
    const float m = fmaxf(l, mx);
    const float w = expf(__fsub_rn(l, m));
    float pf[16], acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pf[4 * j + i] = expf(__fsub_rn(sf[4 * j + i], m));
        acc[i] = __fadd_rn(acc[i], pf[4 * j + i]);
      }
    float v = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    const float z = __fadd_rn(w, __shfl_sync(0xFFFFFFFFu, v, 0));
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) pfs[g][4 * (lane + 32 * j) + i] = __float2bfloat16_rn(__fdiv_rn(pf[4 * j + i], z));
    if (lane == 0) wzs[g] = __fdiv_rn(w, z);
  }
  __syncthreads();
  {  // thread d: o_far[g][d] for the 6 rows, j ascending; then the combine
    const int d = tid;
    float acc[kFaHG];
#pragma unroll
    for (int g = 0; g < kFaHG; ++g) acc[g] = 0.f;
    // j ascending as before, 16 rows' loads issued together (the gather was one dependent latency per row)
    constexpr int kP = 16;
    static_assert(KSEL % kP == 0, "KSEL");
    for (int j0 = 0; j0 < KSEL; j0 += kP) {
      bf16 vr[kP];
#pragma unroll
      for (int u = 0; u < kP; ++u) vr[u] = vb[((size_t)rowu[j0 + u] * KV + k) * D + d];
#pragma unroll
      for (int u = 0; u < kP; ++u) {
        const float vv = __bfloat162float(vr[u]);
#pragma unroll
        for (int g = 0; g < kFaHG; ++g) acc[g] = fmaf(__bfloat162float(pfs[g][j0 + u]), vv, acc[g]);
      }
    }
#pragma unroll
    for (int g = 0; g < kFaHG; ++g) {
      const size_t i = (size_t)t * H * D + (size_t)(k * G + g0 + g) * D + d;
      const float of = __bfloat162float(__float2bfloat16_rn(acc[g]));
      out[i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(wzs[g], __bfloat162float(o_loc[i])), of));
    }
  }
}
// fused index scores (AGENS_FAR_ATTEND=fused; decode and every verify query alike): for block n, its key row
// blk[r2t[4n+3] / 4] (as kbi_kernel reads it) against the T queries' 4 index heads: e_h = bf16(sum_d q[h][d] * key[d])
// (fp32, d ascending), then score_kernel's rounding of their mean, -inf past the query's last far-valid block. One
// thread per block n: the key table is read once for all T queries (no kbi copy, no GEMM).
__global__ void __launch_bounds__(256) score_fused_kernel(const bf16* __restrict__ qi, int T, const long long* pos,
                                                          const int* r2t, int Wctx, const uint4* __restrict__ blk,
                                                          int NB, int hi, float scale, bf16* __restrict__ s) {
  __shared__ float qs[8 * IDXH * DI];  // T <= 8
  for (int i = threadIdx.x; i < T * IDXH * DI; i += blockDim.x) qs[i] = __bfloat162float(qi[i]);
  __shared__ long long nm[8];
  if (threadIdx.x < T) nm[threadIdx.x] = nmax_of(pos[threadIdx.x]);
  __syncthreads();
  const long long nlast = nm[T - 1];
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < NB; n += gridDim.x * blockDim.x) {
    if (n > nlast) {  // invalid for every query: no key read
      for (int t = 0; t < T; ++t) s[(size_t)t * NB + n] = __float2bfloat16_rn(-INFINITY);
      continue;
    }
    int slot = r2t[(long long)n * C + (C - 1)];
    slot = slot < 0 ? 0 : (slot > hi ? hi : slot);
    const uint4* kr = blk + (size_t)(slot / C) * (DI * 2 / 16);
    float acc[8][IDXH];
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int h = 0; h < IDXH; ++h) acc[t][h] = 0.f;
#pragma unroll 2
    for (int c = 0; c < DI / 8; ++c) {
      const uint4 v = kr[c];
      const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float kf = __bfloat162float(e[i]);
        const int d = 8 * c + i;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if (t >= T) break;
#pragma unroll
          for (int h = 0; h < IDXH; ++h) acc[t][h] = fmaf(qs[(t * IDXH + h) * DI + d], kf, acc[t][h]);
        }
      }
    }
    for (int t = 0; t < T; ++t) {
      const float e0 = __bfloat162float(__float2bfloat16_rn(acc[t][0])), e1 = __bfloat162float(__float2bfloat16_rn(acc[t][1])),
                  e2 = __bfloat162float(__float2bfloat16_rn(acc[t][2])), e3 = __bfloat162float(__float2bfloat16_rn(acc[t][3]));
      const float m = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3), 0.25f)));
      s[(size_t)t * NB + n] = n <= nm[t] ? __float2bfloat16_rn(__fmul_rn(m, scale)) : __float2bfloat16_rn(-INFINITY);
    }
  }
}
// score_fused_kernel for T > 1 (verify): thread (block n, query t) instead of thread n, the same per-(t, head) fma chain
// over the 128 dims (d ascending) and the same roundings, so every score is bit-identical to score_fused_kernel's (and
// to the decode's T = 1 launch); 8x the threads (the T = 8 chains of one thread were serial). q in shared memory as
// [d][t][head] (the 8 queries of a warp's n read consecutive words).
__global__ void __launch_bounds__(256) score_fused_t_kernel(const bf16* __restrict__ qi, int T, const long long* pos,
                                                            const int* r2t, int Wctx, const uint4* __restrict__ blk,
                                                            int NB, int hi, float scale, bf16* __restrict__ s) {
  __shared__ float qs[DI][8][IDXH];
  for (int i = threadIdx.x; i < T * IDXH * DI; i += blockDim.x) {
    const int t = i / (IDXH * DI), h = (i / DI) % IDXH, d = i % DI;
    qs[d][t][h] = __bfloat162float(qi[i]);
  }
  __shared__ long long nm[8];
  if (threadIdx.x < T) nm[threadIdx.x] = nmax_of(pos[threadIdx.x]);
  __syncthreads();
  const int t = threadIdx.x & 7, n = blockIdx.x * 32 + (threadIdx.x >> 3);
  if (t >= T || n >= NB) return;
  if (n > nm[T - 1]) {
    s[(size_t)t * NB + n] = __float2bfloat16_rn(-INFINITY);
    return;
  }
  int slot = r2t[(long long)n * C + (C - 1)];
  slot = slot < 0 ? 0 : (slot > hi ? hi : slot);
  const uint4* kr = blk + (size_t)(slot / C) * (DI * 2 / 16);
  float acc[IDXH] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
  for (int c = 0; c < DI / 8; ++c) {
    const uint4 v = kr[c];
    const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float kf = __bfloat162float(e[i]);
      const int d = 8 * c + i;
#pragma unroll
      for (int h = 0; h < IDXH; ++h) acc[h] = fmaf(qs[d][t][h], kf, acc[h]);
    }
  }
  const float e0 = __bfloat162float(__float2bfloat16_rn(acc[0])), e1 = __bfloat162float(__float2bfloat16_rn(acc[1])),
              e2 = __bfloat162float(__float2bfloat16_rn(acc[2])), e3 = __bfloat162float(__float2bfloat16_rn(acc[3]));
  const float m = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3), 0.25f)));
  s[(size_t)t * NB + n] = n <= nm[t] ? __float2bfloat16_rn(__fmul_rn(m, scale)) : __float2bfloat16_rn(-INFINITY);
}
bool far_fused() {
  static const int on = [] { const char* e = getenv("AGENS_FAR_ATTEND"); return e && !strcmp(e, "fused"); }();
  return on;
}
}  // namespace
extern "C" int ak_far_fused_on() { return far_fused() ? 1 : 0; }  // tests: which attend path this process takes
namespace {

// cuBLAS handle configured like torch's for this device (workspace size, default math = reduced-precision reduction
// allowed), separate from the bf16_linear handle so the two never share a stream binding.
cublasHandle_t handle() {
  static cublasHandle_t h[16] = {};
  int dev;
  cudaGetDevice(&dev);
  if (!h[dev & 15]) {
    cublasCreate(&h[dev & 15]);
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    const size_t wsb = major >= 9 ? (size_t)4096 * 8 * 1024 : (size_t)4096 * 1024 * 2 + 16 * 1024 * 8;
    void* ws = nullptr;
    cudaMalloc(&ws, wsb);
    cublasSetWorkspace(h[dev & 15], ws, wsb);
    cublasSetMathMode(h[dev & 15], CUBLAS_DEFAULT_MATH);
  }
  return h[dev & 15];
}
int bgemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void* A, int lda, long long sa,
          const void* B, int ldb, long long sb, void* Cm, int ldc, long long sc, int batch, cudaStream_t st) {
  cublasHandle_t h = handle();
  cublasSetStream(h, st);
  const float alpha = 1.f, beta = 0.f;
  const cublasStatus_t r = cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, &alpha, A, CUDA_R_16BF, lda, sa, B, CUDA_R_16BF,
                                                      ldb, sb, &beta, Cm, CUDA_R_16BF, ldc, sc, batch,
                                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return r == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}
}  // namespace

extern "C" size_t ak_far_decode_ws_bytes(int Wctx) { return ws_bytes(Wctx / C); }

extern "C" int ak_far_decode_src(const void* qi, const void* qh, const void* o_loc, const float* lse,
                                 const long long* pos, const int* r2t, int Wctx, int src, const void* kc, const void* vc,
                                 const float* ksc, const float* vsc, const void* blk, int rows, void* ws, void* out,
                                 void* stream);
extern "C" int ak_far_decode(const void* qi, const void* qh, const void* o_loc, const float* lse, const long long* pos,
                             const int* r2t, int Wctx, const void* kc, const void* vc, const void* blk, int rows,
                             void* ws, void* out, void* stream) {
  return ak_far_decode_src(qi, qh, o_loc, lse, pos, r2t, Wctx, 0, kc, vc, nullptr, nullptr, blk, rows, ws, out, stream);
}

// src 0: kc/vc = bf16 K/V rows; 1: kc/vc = bf16 block means per 4-row page; 2: kc/vc = int8 rows, ksc/vsc = scales;
// 3: kc/vc = int8 block means per 4-row page, ksc/vsc = scales per (page, kv head) (AGENS_KV_MEANS_INT8).
extern "C" int ak_far_decode_src(const void* qi, const void* qh, const void* o_loc, const float* lse,
                                 const long long* pos, const int* r2t, int Wctx, int src, const void* kc, const void* vc,
                                 const float* ksc, const float* vsc, const void* blk, int rows, void* ws, void* out,
                                 void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int NB = Wctx / C, hi = rows;
  if (NB < KSEL) return (int)cudaErrorInvalidValue;
  const Ws w = carve(ws, NB);
  if (far_fused()) {  // product numerics: fused scores, then the same top-k / pools / fused attend
    score_fused_kernel<<<(NB + 255) / 256, 256, 0, st>>>(static_cast<const bf16*>(qi), 1, pos, r2t, Wctx,
                                                         static_cast<const uint4*>(blk), NB, hi, 0.08838834764831845f, w.s);
    topk_launch(w.s, NB, w.top, st);
    if (src == 1)
      pool_means_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const bf16*>(kc),
                                              static_cast<const bf16*>(vc), w.kb, w.vb, w.ok, hi, nullptr);
    else if (src == 3)
      pool_means_i8_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                                 static_cast<const int8_t*>(vc), ksc, vsc, w.kb, w.vb, w.ok, hi, nullptr);
    else if (src == 2)
      pool_int8_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                             static_cast<const int8_t*>(vc), ksc, vsc, w.kb, w.vb, w.ok, hi, nullptr);
    else
      pool_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const bf16*>(kc), static_cast<const bf16*>(vc),
                                        w.kb, w.vb, w.ok, hi, nullptr);
    fused_attend_kernel<0><<<KV * kFaSub, 256, 0, st>>>(static_cast<const bf16*>(qh), static_cast<const bf16*>(o_loc), lse, pos,
                                                   w.kb, w.vb, w.ok, w.top, nullptr, 0.0625f, static_cast<bf16*>(out), 0);
    return (int)cudaGetLastError();
  }
  kbi_kernel<<<(NB * 16 + 255) / 256, 256, 0, st>>>(pos, r2t, Wctx, static_cast<const uint4*>(blk), (uint4*)w.kbi,
                                                     w.valid, NB, hi);
  // e = einsum("bhd,bnd->bhn", qi [1,4,128], kbi [1,NB,128]): row-major [4][NB] = col-major [NB][4] = kbi^T-op x qi
  int e = bgemm(CUBLAS_OP_T, CUBLAS_OP_N, NB, IDXH, DI, w.kbi, DI, (long long)NB * DI, qi, DI, (long long)IDXH * DI,
                w.e, NB, (long long)IDXH * NB, 1, st);
  if (e) return e;
  score_kernel<<<(NB + 255) / 256, 256, 0, st>>>(w.e, w.valid, w.s, NB, 0.08838834764831845f, nullptr);
  topk_launch(w.s, NB, w.top, st);
  if (src == 1)
    pool_means_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const bf16*>(kc),
                                            static_cast<const bf16*>(vc), w.kb, w.vb, w.ok, hi, nullptr);
  else if (src == 3)
    pool_means_i8_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                               static_cast<const int8_t*>(vc), ksc, vsc, w.kb, w.vb, w.ok, hi, nullptr);
  else if (src == 2)
    pool_int8_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                           static_cast<const int8_t*>(vc), ksc, vsc, w.kb, w.vb, w.ok, hi, nullptr);
  else
    pool_kernel<<<KSEL, 128, 0, st>>>(w.top, pos, r2t, Wctx, static_cast<const bf16*>(kc), static_cast<const bf16*>(vc),
                                      w.kb, w.vb, w.ok, hi, nullptr);
  if (far_fused()) {
    fused_attend_kernel<0><<<KV * kFaSub, 256, 0, st>>>(static_cast<const bf16*>(qh), static_cast<const bf16*>(o_loc), lse, pos,
                                                   w.kb, w.vb, w.ok, w.top, nullptr, 0.0625f, static_cast<bf16*>(out), 0);
    return (int)cudaGetLastError();
  }
  // e2 = einsum("bkgd,bnkd->bkgn", q [1,4,6,256], kb [1,512,4,256]): per kv head k, [6][512] = q_k [6][256] x kb_k^T
  e = bgemm(CUBLAS_OP_T, CUBLAS_OP_N, KSEL, G, D, w.kb, KV * D, D, qh, D, (long long)G * D, w.e2, KSEL,
            (long long)G * KSEL, KV, st);
  if (e) return e;
  softmax_kernel<<<(H + 7) / 8, 256, 0, st>>>(w.e2, w.ok, lse, 0.0625f, w.pf, w.wz);
  // o_far = einsum("tkgn,tnkd->tkgd", pf [1,4,6,512], vb [1,512,4,256]): per k, [6][256] = pf_k [6][512] x vb_k [512][256]
  e = bgemm(CUBLAS_OP_N, CUBLAS_OP_N, D, G, KSEL, w.vb, KV * D, D, w.pf, KSEL, (long long)G * KSEL, w.ofar, D,
            (long long)G * D, KV, st);
  if (e) return e;
  combine_kernel<<<(H * D + 255) / 256, 256, 0, st>>>(w.wz, static_cast<const bf16*>(o_loc), w.ofar,
                                                       static_cast<bf16*>(out));
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------- multi-query (verify)
// T consecutive query tokens of one sequence (a speculative verify block), each with exactly the math of its own
// ak_far_decode_src call -- so every output is bit-identical to T separate calls -- but the data each would read again
// is read once: block keys gathered once (for the last query; a query's own validity masks the rest of its scores),
// the T top-k selections in one launch (one CTA per query), and the union of the T selections fetched once from the
// pools (host-mapped when tiered: the PCIe fetch), then copied on the device into each query's selection order.
// Three phases so the caller can overlap the fetch (side stream) with other work: select (kbi, scores, top-k, union
// list), fetch (the union's block means / rows), attend (per query: its blocks in order, then the attend half).
namespace {
constexpr int kMaxT = 8;
struct WsM {
  bf16* kbi;
  int8_t* valid;
  int *mark, *cnt;           // mark[NB]: -1, or the union slot of block n during a call (reset by attend)
  long long *uniq, *tops;    // union block ids [T*KSEL]; per-query top-k [T][KSEL]
  bf16 *ukb, *uvb;           // union K/V block means [T*KSEL][4][256]
  bf16 *e, *s;               // per query: e [T][4][NB], scores [T][NB]
  Ws q;                      // per-query attend scratch, reused query by query
};
WsM carve_m(void* base, int NB, int T) {
  char* p = static_cast<char*>(base);
  auto take = [&](size_t bytes) { char* r = p; p += al(bytes); return r; };
  WsM w;
  w.kbi = (bf16*)take((size_t)NB * DI * 2);
  w.valid = (int8_t*)take(NB);
  w.mark = (int*)take((size_t)NB * 4);
  w.cnt = (int*)take(16);
  w.uniq = (long long*)take((size_t)T * KSEL * 8);
  w.tops = (long long*)take((size_t)T * KSEL * 8);
  w.ukb = (bf16*)take((size_t)T * KSEL * KV * D * 2);
  w.uvb = (bf16*)take((size_t)T * KSEL * KV * D * 2);
  w.e = (bf16*)take((size_t)T * IDXH * NB * 2);
  w.s = (bf16*)take((size_t)T * NB * 2);
  w.q.kbi = w.kbi, w.q.valid = w.valid, w.q.top = nullptr, w.q.e = nullptr, w.q.s = nullptr;
  w.q.kb = (bf16*)take((size_t)KSEL * KV * D * 2);
  w.q.vb = (bf16*)take((size_t)KSEL * KV * D * 2);
  w.q.ok = (int8_t*)take(KSEL);
  w.q.e2 = (bf16*)take((size_t)H * KSEL * 2);
  w.q.pf = (bf16*)take((size_t)H * KSEL * 2);
  w.q.wz = (float*)take(H * 4);
  w.q.ofar = (bf16*)take((size_t)H * D * 2);
  return w;
}
size_t wsm_bytes(int NB, int T) {
  return al((size_t)NB * DI * 2) + al(NB) + al((size_t)NB * 4) + al(16) + 2 * al((size_t)T * KSEL * 8) +
         2 * al((size_t)T * KSEL * KV * D * 2) + al((size_t)T * IDXH * NB * 2) + al((size_t)T * NB * 2) +
         2 * al((size_t)KSEL * KV * D * 2) + al(KSEL) + 2 * al((size_t)H * KSEL * 2) + al(H * 4) + al((size_t)H * D * 2);
}
__global__ void claim_kernel(const long long* tops, int n, int* mark, int* cnt, long long* uniq) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const long long b = tops[i];
    if (atomicCAS(&mark[b], -1, -2) == -1) {
      const int u = atomicAdd(cnt, 1);
      uniq[u] = b;
      mark[b] = u;
    }
  }
}
// query's selection order: kb[j] = union[mark[top[j]]]; ok[j] as pool_*_kernel computes it
__global__ void gather_kernel(const long long* top, const long long* pos, const int* mark, const bf16* ukb,
                              const bf16* uvb, bf16* kb, bf16* vb, int8_t* ok) {
  const int j = blockIdx.x;
  const long long b = top[j];
  if (threadIdx.x == 0) ok[j] = b <= nmax_of(pos[0]);
  const int u = mark[b];
  const uint4* ks = reinterpret_cast<const uint4*>(ukb + (size_t)u * KV * D);
  const uint4* vs = reinterpret_cast<const uint4*>(uvb + (size_t)u * KV * D);
  for (int r = threadIdx.x; r < KV * D / 8; r += blockDim.x) {
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r] = ks[r];
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r] = vs[r];
  }
}
__global__ void unmark_kernel(const long long* tops, int n, int* mark, int* cnt) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) mark[tops[i]] = -1;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cnt = 0;
}
__global__ void fill_kernel(int* p, int n, int v) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = v;
}
// T queries' scores in one launch (blockIdx.y = query): score_kernel's math per element
__global__ void score_multi_kernel(const bf16* e, bf16* s, int NB, float scale, const long long* pos, int seq0) {
  const int t = blockIdx.y;
  const long long nm = nmax_of(pos ? pos[t] : (long long)seq0 + t);
  const bf16* et = e + (size_t)t * IDXH * NB;
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < NB; n += gridDim.x * blockDim.x) {
    const float e0 = __bfloat162float(et[n]), e1 = __bfloat162float(et[NB + n]), e2 = __bfloat162float(et[2 * NB + n]),
                e3 = __bfloat162float(et[3 * NB + n]);
    const float m = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fadd_rn(e0, e1), e2), e3), 0.25f)));
    s[(size_t)t * NB + n] = n <= nm ? __float2bfloat16_rn(__fmul_rn(m, scale)) : __float2bfloat16_rn(-INFINITY);
  }
}
// T top-k selections in one launch: CTA t = topk2_kernel on query t's scores (same code path, same output)
template <int EPT>
__global__ void __launch_bounds__(1024) topk2_multi_kernel(const bf16* s, int NB, long long* top) {
  topk2_body<EPT>(s + (size_t)blockIdx.x * NB, NB, top + (size_t)blockIdx.x * KSEL);
}
}  // namespace

extern "C" size_t ak_far_decode_multi_ws_bytes(int Wctx, int T) { return wsm_bytes(Wctx / C, T); }
// once after allocation: union marks empty, count 0
extern "C" int ak_far_decode_multi_init(void* ws, int Wctx, int T, void* stream) {
  const int NB = Wctx / C;
  WsM w = carve_m(ws, NB, T);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  fill_kernel<<<(NB + 255) / 256, 256, 0, st>>>(w.mark, NB, -1);
  fill_kernel<<<1, 32, 0, st>>>(w.cnt, 4, 0);
  return (int)cudaGetLastError();
}
// phase 1 (query t: qi + t*4*128, pos + t; positions pos[0] < ... < pos[T-1]): block keys, scores, top-k, union list
extern "C" int ak_far_decode_multi_select(int T, const void* qi, const long long* pos, const int* r2t, int Wctx,
                                          const void* blk, int rows, void* ws, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int NB = Wctx / C, hi = rows;
  if (NB < KSEL || T < 1 || T > kMaxT) return (int)cudaErrorInvalidValue;
  WsM w = carve_m(ws, NB, T);
  if (far_fused()) {
    if (T > 1)  // thread per (block, query): bit-identical scores, 8x the parallelism
      score_fused_t_kernel<<<(NB + 31) / 32, 256, 0, st>>>(static_cast<const bf16*>(qi), T, pos, r2t, Wctx,
                                                           static_cast<const uint4*>(blk), NB, hi, 0.08838834764831845f,
                                                           w.s);
    else
      score_fused_kernel<<<(NB + 255) / 256, 256, 0, st>>>(static_cast<const bf16*>(qi), T, pos, r2t, Wctx,
                                                           static_cast<const uint4*>(blk), NB, hi, 0.08838834764831845f,
                                                           w.s);
  } else {
  kbi_kernel<<<(NB * 16 + 255) / 256, 256, 0, st>>>(pos + T - 1, r2t, Wctx, static_cast<const uint4*>(blk), (uint4*)w.kbi,
                                                     w.valid, NB, hi);
  for (int t = 0; t < T; ++t) {  // per query: e (the same cuBLAS call as ak_far_decode_src)
    const bf16* qit = static_cast<const bf16*>(qi) + (size_t)t * IDXH * DI;
    int e = bgemm(CUBLAS_OP_T, CUBLAS_OP_N, NB, IDXH, DI, w.kbi, DI, (long long)NB * DI, qit, DI, (long long)IDXH * DI,
                  w.e + (size_t)t * IDXH * NB, NB, (long long)IDXH * NB, 1, st);
    if (e) return e;
  }
  score_multi_kernel<<<dim3((NB + 255) / 256, T), 256, 0, st>>>(w.e, w.s, NB, 0.08838834764831845f, pos, 0);
  }
  if (topk2_ok(NB)) {
    if (NB <= 1024 * 8) topk2_multi_kernel<8><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else if (NB <= 1024 * 24) topk2_multi_kernel<24><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else if (NB <= 1024 * 26) topk2_multi_kernel<26><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else if (NB <= 1024 * 34) topk2_multi_kernel<34><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else if (NB <= 1024 * 50) topk2_multi_kernel<50><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else if (NB <= 1024 * 66) topk2_multi_kernel<66><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
    else topk2_multi_kernel<72><<<T, 1024, 0, st>>>(w.s, NB, w.tops);
  } else {
    for (int t = 0; t < T; ++t) topk_launch(w.s + (size_t)t * NB, NB, w.tops + (size_t)t * KSEL, st);
  }
  const int n = T * KSEL;
  claim_kernel<<<(n + 255) / 256, 256, 0, st>>>(w.tops, n, w.mark, w.cnt, w.uniq);
  return (int)cudaGetLastError();
}
// phase 2: the union's blocks from the pools (src as ak_far_decode_src; may run on another stream after select)
extern "C" int ak_far_decode_multi_fetch(int T, const long long* pos, const int* r2t, int Wctx, int src, const void* kc,
                                         const void* vc, const float* ksc, const float* vsc, int rows, void* ws,
                                         void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int NB = Wctx / C, hi = rows, n = T * KSEL;
  WsM w = carve_m(ws, NB, T);
  if (src == 1)
    pool_means_kernel<<<n, 128, 0, st>>>(w.uniq, pos, r2t, Wctx, static_cast<const bf16*>(kc), static_cast<const bf16*>(vc),
                                         w.ukb, w.uvb, nullptr, hi, w.cnt);
  else if (src == 3)
    pool_means_i8_kernel<<<n, 128, 0, st>>>(w.uniq, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                            static_cast<const int8_t*>(vc), ksc, vsc, w.ukb, w.uvb, nullptr, hi, w.cnt);
  else if (src == 2)
    pool_int8_kernel<<<n, 128, 0, st>>>(w.uniq, pos, r2t, Wctx, static_cast<const int8_t*>(kc),
                                        static_cast<const int8_t*>(vc), ksc, vsc, w.ukb, w.uvb, nullptr, hi, w.cnt);
  else
    pool_kernel<<<n, 128, 0, st>>>(w.uniq, pos, r2t, Wctx, static_cast<const bf16*>(kc), static_cast<const bf16*>(vc),
                                   w.ukb, w.uvb, nullptr, hi, w.cnt);
  return (int)cudaGetLastError();
}
// ---- cross-step cache of the union (block means, src 1): one per layer, owned by the caller ----
// layout: tag (long long), prev (int), ids [cap] (long long), mark [NB] (int, -1 = not cached), data K / V [cap][4][256]
// A union block found in the cache (same owner tag: same session and epoch) is copied on the device, the rest is read
// from the pools; afterwards the cache holds this step's union (the likeliest blocks of the next step). Data movement
// only: the values are the pool's.
namespace {
struct FCache {
  long long* tag;
  int* prev;
  long long* ids;
  int* mark;
  bf16 *k, *v;
};
FCache carve_c(void* base, int NB, int cap) {
  char* p = static_cast<char*>(base);
  auto take = [&](size_t bytes) { char* r = p; p += al(bytes); return r; };
  FCache c;
  c.tag = (long long*)take(16);
  c.prev = (int*)take(16);
  c.ids = (long long*)take((size_t)cap * 8);
  c.mark = (int*)take((size_t)NB * 4);
  c.k = (bf16*)take((size_t)cap * KV * D * 2);
  c.v = (bf16*)take((size_t)cap * KV * D * 2);
  return c;
}
__global__ void pool_means_cached_kernel(const long long* uniq, const int* cnt, const int* r2t, int Wctx,
                                         const bf16* km, const bf16* vm, bf16* kb, bf16* vb, int hi, FCache c,
                                         const long long* tag) {
  const int j = blockIdx.x;
  if (j >= *cnt) return;
  const long long b = uniq[j];
  const int s = *c.tag == *tag ? c.mark[b] : -1;
  const uint4 *ks, *vs;
  if (s >= 0) {
    ks = reinterpret_cast<const uint4*>(c.k + (size_t)s * KV * D);
    vs = reinterpret_cast<const uint4*>(c.v + (size_t)s * KV * D);
  } else {
    long long col = b * C + C - 1;
    col = col < 0 ? 0 : (col > Wctx - 1 ? Wctx - 1 : col);
    const int sr = r2t[col], page = (sr < 0 ? 0 : (sr > hi ? hi : sr)) / C;
    ks = reinterpret_cast<const uint4*>(km + (size_t)page * KV * D);
    vs = reinterpret_cast<const uint4*>(vm + (size_t)page * KV * D);
  }
  for (int r = threadIdx.x; r < KV * D / 8; r += blockDim.x) {
    reinterpret_cast<uint4*>(kb + (size_t)j * KV * D)[r] = ks[r];
    reinterpret_cast<uint4*>(vb + (size_t)j * KV * D)[r] = vs[r];
  }
}
__global__ void cache_clear_kernel(FCache c) {
  const int n = *c.prev;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) c.mark[c.ids[i]] = -1;
}
__global__ void cache_fill_kernel(const long long* uniq, const int* cnt, const bf16* ukb, const bf16* uvb, FCache c,
                                  int cap, const long long* tag) {
  const int j = blockIdx.x, n = *cnt < cap ? *cnt : cap;
  if (j == 0 && threadIdx.x == 0) *c.prev = n, *c.tag = *tag;  // read by the next step only (stream order)
  if (j >= n) return;
  const long long b = uniq[j];
  if (threadIdx.x == 0) c.ids[j] = b, c.mark[b] = j;
  for (int r = threadIdx.x; r < KV * D / 8; r += blockDim.x) {
    reinterpret_cast<uint4*>(c.k + (size_t)j * KV * D)[r] = reinterpret_cast<const uint4*>(ukb + (size_t)j * KV * D)[r];
    reinterpret_cast<uint4*>(c.v + (size_t)j * KV * D)[r] = reinterpret_cast<const uint4*>(uvb + (size_t)j * KV * D)[r];
  }
}
}  // namespace
extern "C" size_t ak_far_cache_bytes(int Wctx, int cap) {
  const int NB = Wctx / C;
  return al(16) + al(16) + al((size_t)cap * 8) + al((size_t)NB * 4) + 2 * al((size_t)cap * KV * D * 2);
}
extern "C" int ak_far_cache_init(void* cache, int Wctx, int cap, void* stream) {
  const int NB = Wctx / C;
  FCache c = carve_c(cache, NB, cap);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  fill_kernel<<<(NB + 255) / 256, 256, 0, st>>>(c.mark, NB, -1);
  fill_kernel<<<1, 32, 0, st>>>(c.prev, 4, 0);
  fill_kernel<<<1, 32, 0, st>>>(reinterpret_cast<int*>(c.tag), 4, -1);
  return (int)cudaGetLastError();
}
// phase 2 with the layer's cache (block means only): hits from the cache, misses from the pools, then the cache := union
extern "C" int ak_far_decode_multi_fetch_cached(int T, const int* r2t, int Wctx, const void* km, const void* vm, int rows,
                                                void* ws, void* cache, int cap, const long long* tag, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int NB = Wctx / C, n = T * KSEL;
  WsM w = carve_m(ws, NB, T);
  FCache c = carve_c(cache, NB, cap);
  pool_means_cached_kernel<<<n, 128, 0, st>>>(w.uniq, w.cnt, r2t, Wctx, static_cast<const bf16*>(km),
                                              static_cast<const bf16*>(vm), w.ukb, w.uvb, rows, c, tag);
  cache_clear_kernel<<<(cap + 255) / 256, 256, 0, st>>>(c);
  cache_fill_kernel<<<cap, 128, 0, st>>>(w.uniq, w.cnt, w.ukb, w.uvb, c, cap, tag);
  return (int)cudaGetLastError();
}

// phase 3 (query t: qh / o_loc / out + t*24*256, lse + t*24): per query its blocks in its order, the attend half of
// ak_far_decode_src; then the union marks are reset
extern "C" int ak_far_decode_multi_attend(int T, const void* qh, const void* o_loc, const float* lse,
                                          const long long* pos, int Wctx, void* ws, void* out, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int NB = Wctx / C, n = T * KSEL;
  WsM w = carve_m(ws, NB, T);
  if (far_fused()) {  // all T queries in one launch, straight from the union buffers
    fused_attend_kernel<1><<<T * KV * kFaSub, 256, 0, st>>>(static_cast<const bf16*>(qh), static_cast<const bf16*>(o_loc), lse,
                                                      pos, w.ukb, w.uvb, nullptr, w.tops, w.mark, 0.0625f,
                                                      static_cast<bf16*>(out), 0);
    unmark_kernel<<<(n + 255) / 256, 256, 0, st>>>(w.tops, n, w.mark, w.cnt);
    return (int)cudaGetLastError();
  }
  for (int t = 0; t < T; ++t) {
    const bf16* qht = static_cast<const bf16*>(qh) + (size_t)t * H * D;
    gather_kernel<<<KSEL, 128, 0, st>>>(w.tops + (size_t)t * KSEL, pos + t, w.mark, w.ukb, w.uvb, w.q.kb, w.q.vb, w.q.ok);
    int e = bgemm(CUBLAS_OP_T, CUBLAS_OP_N, KSEL, G, D, w.q.kb, KV * D, D, qht, D, (long long)G * D, w.q.e2, KSEL,
                  (long long)G * KSEL, KV, st);
    if (e) return e;
    softmax_kernel<<<(H + 7) / 8, 256, 0, st>>>(w.q.e2, w.q.ok, lse + (size_t)t * H, 0.0625f, w.q.pf, w.q.wz);
    e = bgemm(CUBLAS_OP_N, CUBLAS_OP_N, D, G, KSEL, w.q.vb, KV * D, D, w.q.pf, KSEL, (long long)G * KSEL, w.q.ofar, D,
              (long long)G * D, KV, st);
    if (e) return e;
    combine_kernel<<<(H * D + 255) / 256, 256, 0, st>>>(w.q.wz, static_cast<const bf16*>(o_loc) + (size_t)t * H * D,
                                                         w.q.ofar, static_cast<bf16*>(out) + (size_t)t * H * D);
  }
  unmark_kernel<<<(n + 255) / 256, 256, 0, st>>>(w.tops, n, w.mark, w.cnt);
  return (int)cudaGetLastError();
}
// all three phases on one stream
extern "C" int ak_far_decode_multi_src(int T, const void* qi, const void* qh, const void* o_loc, const float* lse,
                                       const long long* pos, const int* r2t, int Wctx, int src, const void* kc,
                                       const void* vc, const float* ksc, const float* vsc, const void* blk, int rows,
                                       void* ws, void* out, void* stream) {
  int e = ak_far_decode_multi_select(T, qi, pos, r2t, Wctx, blk, rows, ws, stream);
  if (!e) e = ak_far_decode_multi_fetch(T, pos, r2t, Wctx, src, kc, vc, ksc, vsc, rows, ws, stream);
  if (!e) e = ak_far_decode_multi_attend(T, qh, o_loc, lse, pos, Wctx, ws, out, stream);
  return e;
}

// ---------------------------------------------------------------------------------------------- prefill extend (engine)
// The far branch of a prefill chunk (queries at positions seq0 .. seq0 + T - 1) without ATen (AGENS_FAR_EXTEND_ENGINE,
// budget mode): the chunk's closed blocks' keys written (pooled indexer keys, as the bridge), the far block keys
// gathered once, then per tile of 64 queries one GEMM of their index scores, score_multi_kernel's rounding with each
// query's own validity, the top-k of each query in one launch, and the fused attend over the staged far block means
// (kst / vst: block n at row n, as m1_kv's far-extend staging holds them). Engine-owned temporaries only.
namespace {
constexpr int kExtTile = 64;
__global__ void blk_extend_kernel(const int* r2t, const bf16* idx, bf16* blk, long long n_lo, int hi) {
  const long long b = n_lo + blockIdx.x;
  const int d = threadIdx.x;
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    int slot = r2t[b * C + c];
    slot = slot < 0 ? 0 : (slot > hi ? hi : slot);
    acc = __fadd_rn(acc, __bfloat162float(idx[(size_t)slot * DI + d]));
  }
  int last = r2t[b * C + C - 1];
  last = last < 0 ? 0 : (last > hi ? hi : last);
  blk[(size_t)(last / C) * DI + d] = __float2bfloat16_rn(__fmul_rn(acc, 1.0f / C));
}
// kbi[n] = blk[r2t[4n+3] / 4] for n < nb, zeros for nb <= n < nbuf
__global__ void kbi_n_kernel(const int* r2t, const uint4* blk, uint4* kbi, int nb, int nbuf, int hi) {
  const int per = DI * 2 / 16;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbuf * per; i += gridDim.x * blockDim.x) {
    const int n = i / per, c = i % per;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (n < nb) {
      int slot = r2t[(long long)n * C + (C - 1)];
      slot = slot < 0 ? 0 : (slot > hi ? hi : slot);
      x = blk[(size_t)(slot / C) * per + c];
    }
    kbi[i] = x;
  }
}
}  // namespace
extern "C" size_t ak_far_extend_engine_ws_bytes(int Wctx) {
  const size_t nbm = (size_t)Wctx / C + KSEL;
  return al(nbm * DI * 2) + al((size_t)kExtTile * IDXH * nbm * 2) + al((size_t)kExtTile * nbm * 2) +
         al((size_t)kExtTile * KSEL * 8);
}
extern "C" int ak_far_extend_engine(int T, int seq0, const void* qi, const void* qh, const void* o_loc, const float* lse,
                                    const int* r2t, int Wctx, const void* kst, const void* vst, const void* idx,
                                    void* blk, int rows, void* ws, void* out, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  const int hi = rows, seq_len = seq0 + T;
  {  // the chunk's closed blocks: their keys (the bridge's index_copy_ of pooled indexer keys)
    const long long n_lo = (long long)seq0 / C, n_hi = (long long)seq_len / C;
    if (n_hi > n_lo)
      blk_extend_kernel<<<(int)(n_hi - n_lo), DI, 0, st>>>(r2t, static_cast<const bf16*>(idx), static_cast<bf16*>(blk),
                                                           n_lo, hi);
  }
  const long long m = (long long)seq_len - 1 - W - (C - 1);
  const int nb = m >= 0 ? (int)(m / C + 1) : 0;
  if (nb == 0) {  // no far block for any query: the local output
    return (int)cudaMemcpyAsync(out, o_loc, (size_t)T * H * D * 2, cudaMemcpyDeviceToDevice, st);
  }
  const int nbuf = nb < KSEL ? KSEL : nb;
  const size_t nbm = (size_t)Wctx / C + KSEL;
  if ((size_t)nbuf > nbm) return (int)cudaErrorInvalidValue;
  char* p = static_cast<char*>(ws);
  bf16* kbi = (bf16*)p;
  p += al(nbm * DI * 2);
  bf16* e = (bf16*)p;
  p += al((size_t)kExtTile * IDXH * nbm * 2);
  bf16* sc = (bf16*)p;
  p += al((size_t)kExtTile * nbm * 2);
  long long* tops = (long long*)p;
  kbi_n_kernel<<<(nbuf * 16 + 255) / 256, 256, 0, st>>>(r2t, static_cast<const uint4*>(blk), (uint4*)kbi, nb, nbuf, hi);
  for (int t0 = 0; t0 < T; t0 += kExtTile) {
    const int tq = T - t0 < kExtTile ? T - t0 : kExtTile;
    const bf16* qit = static_cast<const bf16*>(qi) + (size_t)t0 * IDXH * DI;
    int er = bgemm(CUBLAS_OP_T, CUBLAS_OP_N, nbuf, IDXH * tq, DI, kbi, DI, (long long)nbuf * DI, qit, DI,
                   (long long)IDXH * tq * DI, e, nbuf, (long long)IDXH * tq * nbuf, 1, st);
    if (er) return er;
    score_multi_kernel<<<dim3((nbuf + 255) / 256, tq), 256, 0, st>>>(e, sc, nbuf, 0.08838834764831845f, nullptr,
                                                                      seq0 + t0);
    if (!topk2_ok(nbuf)) return (int)cudaErrorInvalidValue;
    if (nbuf <= 1024 * 8) topk2_multi_kernel<8><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else if (nbuf <= 1024 * 24) topk2_multi_kernel<24><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else if (nbuf <= 1024 * 26) topk2_multi_kernel<26><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else if (nbuf <= 1024 * 34) topk2_multi_kernel<34><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else if (nbuf <= 1024 * 50) topk2_multi_kernel<50><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else if (nbuf <= 1024 * 66) topk2_multi_kernel<66><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    else topk2_multi_kernel<72><<<tq, 1024, 0, st>>>(sc, nbuf, tops);
    fused_attend_kernel<2><<<tq * KV * kFaSub, 256, 0, st>>>(static_cast<const bf16*>(qh) + (size_t)t0 * H * D,
                                                    static_cast<const bf16*>(o_loc) + (size_t)t0 * H * D,
                                                    lse + (size_t)t0 * H, nullptr, static_cast<const bf16*>(kst),
                                                    static_cast<const bf16*>(vst), nullptr, tops, nullptr, 0.0625f,
                                                    static_cast<bf16*>(out) + (size_t)t0 * H * D, seq0 + t0);
  }
  return (int)cudaGetLastError();
}
extern "C" int ak_far_set_heads(int qh, int kvh) {
  if (kvh < 1 || qh != G * kvh) return (int)cudaErrorInvalidValue;
  g_far_h_host = qh, g_far_kv_host = kvh;
  cudaError_t e = cudaMemcpyToSymbol(g_far_h_dev, &qh, 4);
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_far_kv_dev, &kvh, 4);
  return (int)e;
}

