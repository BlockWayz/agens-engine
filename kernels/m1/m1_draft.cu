// Draft step ported from SGLang (Apache-2.0). See THIRD_PARTY_NOTICES.md.
// DFlash2 drafter for M1 (sglang's DFlash2DraftModel + DFlashWorkerV2's draft step, one sequence):
//   context KV  : per committed target token, ctx = hidden_norm(fc(concat of the 5 captured target features)); per
//                 draft layer k = rope(k_norm(k_proj(ctx))), v = v_proj(ctx) into a ring of kRing positions
//   draft block : [anchor, mask x 7] at positions P..P+7; per layer: residual RMSNorm, grouped dynamic conv (prepare),
//                 attention (q/k norm, rope theta 1e7; is_causal = false: every block query sees the whole block, and
//                 the context from its own sliding window, window_left 2047), conv (finish), residual RMSNorm, conv,
//                 SwiGLU MLP, conv; final norm
//   selector    : top-16 of the target lm_head per block slot 1..7 (unary scores), then the greedy walk through the
//                 candidate lattice score[e][p][c] = unary[e][c] + <A[pred] * proj(h[e]), B[cand[e][c]]>, pred = the
//                 anchor for slot 0 and the walked candidate of the previous slot after (only the walked rows are
//                 scored: same argmax path as the full lattice).
// The drafter only proposes; its numerics (fp32 accumulate, bf16 roundings, not bit-matched to sglang) change the
// acceptance rate, never the output tokens.
#include "m1_draft.h"
#include "agens_kernels.h"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace m1draft {
namespace {
using bf16 = __nv_bfloat16;
constexpr int HID = 5120, NL = 5, QH = 32, KVH = 8, HD = 128, INTER = 17408, GROUPS = 320, GSZ = 16;
constexpr int RANK = 256, TOPK = 16, WINDOW = 2048, kRing = 4096;
constexpr float EPS = 1e-6f;

#define DCK(x)                                                                     \
  do {                                                                             \
    int e_ = (int)(x);                                                             \
    if (e_) {                                                                      \
      fprintf(stderr, "m1draft: %s failed (%d) at %s:%d\n", #x, e_, __FILE__, __LINE__); \
      return e_;                                                                   \
    }                                                                              \
  } while (0)

std::map<std::string, void*> Wd;
struct W4 {
  const void *wt, *st;
  int n, k, group;
  int bits = 4;  // W{bits} KT (AGENS_DRAFT_BITS: 3 or 2 -> the mixed-precision kernels, ak_wq_*)
};
std::map<std::string, W4> Wq;  // W4 linears (k-tiled layout), by the name of their bf16 weight
struct Lin {                   // a linear: bf16 [n][k], or W4
  const bf16* w = nullptr;
  const W4* q = nullptr;
  bool ok() const { return w || q; }
};
struct Layer {
  const bf16 *in_ln, *post_ln, *ac_base, *mc_base, *qn, *kn;
  Lin ac_proj, mc_proj, q, k, v, o, gu, down;
};
struct Drafter {
  bool ready = false;
  int max_ctx = 0, mask_id = 0, max_pos = 0;
  const bf16 *hidden_norm = nullptr, *norm = nullptr, *sel_A = nullptr, *sel_B = nullptr;
  Lin fc, sel_proj;
  bf16* wdq = nullptr;  // W4 linears with m > 16 (context materialize): dequantized row tiles of kWdq bytes
  size_t wdq_bytes = 0;
  Layer L[NL];
  bf16 *ring_k = nullptr, *ring_v = nullptr;  // [n_rings][NL][kRing][KVH][HD]
  int n_rings = 1, ring = 0, max_rows = kBlock;  // rings, the single-sequence calls' ring, batched rows (B * kBlock)
  bf16* hsel = nullptr;                       // batched: the B * 7 selector rows gathered
  float* rope = nullptr;                      // [max_pos][HD] cos|sin
  // block buffers [kBlock][...]
  int* ids = nullptr;
  long long* pos = nullptr;
  bf16 *h, *r, *x, *t, *delta, *q, *kv, *ao, *gu, *act, *logits, *hp;
  float *apart = nullptr, *aml = nullptr;  // split-key attention partials (draft_attn_split)
  float *uval, *pval;
  int *cid, *pid;
  // materialize buffers [max_ctx][...]
  bf16 *ctx, *ck, *cv;
  size_t own = 0;
} D;

template <typename T>
T* alloc(size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T) > 0 ? n * sizeof(T) : 16) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  D.own += n * sizeof(T);
  return (T*)p;
}
const bf16* wp(const std::string& n) {
  auto it = Wd.find(n);
  if (it == Wd.end()) {
    fprintf(stderr, "m1draft: missing weight %s\n", n.c_str());
    return nullptr;
  }
  return (const bf16*)it->second;
}
Lin lw(const std::string& n) {
  Lin l;
  auto it = Wq.find(n);
  if (it != Wq.end()) l.q = &it->second;
  else l.w = wp(n);
  return l;
}

__global__ void block_prep(const int* step, int mask_id, int* ids, long long* pos) {
  const int j = threadIdx.x;
  if (j >= kBlock) return;
  ids[j] = j == 0 ? step[2] : mask_id;
  pos[j] = (long long)step[1] + j;
}
// y[t][c] = coef0 * x[t][c] + (t > 0 ? coef1 * x[t-1][c] : 0), coef_tap = base[side][tap][c] + delta[t][side][tap][c/16]
__global__ void grouped_conv(const bf16* __restrict__ x, const bf16* __restrict__ delta, const bf16* __restrict__ base,
                             int side, bf16* __restrict__ y) {
  const int t = blockIdx.y;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < HID; c += gridDim.x * blockDim.x) {
    const int g = c / GSZ;
    const float c0 = __bfloat162float(base[(side * 2 + 0) * HID + c]) + __bfloat162float(delta[t * 2 * 2 * GROUPS + (side * 2 + 0) * GROUPS + g]);
    float v = c0 * __bfloat162float(x[t * HID + c]);
    if (t % kBlock > 0) {  // within the token's draft block (batched: rows of B blocks)
      const float c1 = __bfloat162float(base[(side * 2 + 1) * HID + c]) + __bfloat162float(delta[t * 2 * 2 * GROUPS + (side * 2 + 1) * GROUPS + g]);
      v += c1 * __bfloat162float(x[(t - 1) * HID + c]);
    }
    y[t * HID + c] = __float2bfloat16_rn(v);
  }
}
// ring[(pos % kRing)] = k/v of n tokens (n_dev nullable)
// batched (rec != nullptr): rows of B sequences of kBlock; row t belongs to sequence t / kBlock, written iff its index
// in the block is < that sequence's count n_dev[seq * n_stride] (nullable: all), into ring rec[seq * kRecInts + 19]
__global__ void store_ring(const bf16* __restrict__ k, const bf16* __restrict__ v, int ld, const long long* pos, int n_max,
                           const int* n_dev, bf16* rk, bf16* rv, const int* rec = nullptr, int n_stride = 0,
                           size_t rstride = 0, int bk = kBlock) {
  const int per = KVH * HD;
  const int n = rec ? n_max : (n_dev ? min(*n_dev, n_max) : n_max);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * per; i += gridDim.x * blockDim.x) {
    const int t = i / per, e = i % per;
    bf16 *dk = rk, *dv = rv;
    if (rec) {
      const int sq = t / bk;
      if (n_dev && t % bk >= n_dev[sq * n_stride]) continue;
      const size_t off = (size_t)rec[sq * kRecInts + 19] * rstride;
      dk += off, dv += off;
    }
    const size_t slot = (size_t)(pos[t] % kRing) * per + e;
    dk[slot] = k[(size_t)t * ld + e];
    dv[slot] = v[(size_t)t * ld + e];
  }
}
// One block per (query j, q head h): keys at positions [max(0, p-WINDOW+1), P+7] (non-causal: the whole draft block;
// sglang's ENCODER_ONLY attention with window_left = WINDOW - 1). Warp w takes keys in runs of 8 (w*8 + 32i ..), its
// lanes 4 dims each: 8 independent row loads in flight per warp (latency), scores by warp shuffles, output partials
// of the 4 warps summed in shared memory.
constexpr int kAttnKeys = WINDOW + kBlock;
__global__ void __launch_bounds__(128) draft_attn(const bf16* __restrict__ q, int q_ld, const bf16* __restrict__ rk,
                                                  const bf16* __restrict__ rv, const long long* __restrict__ pos,
                                                  bf16* __restrict__ out) {
  __shared__ float sc[kAttnKeys];
  __shared__ float part[4][HD];
  __shared__ float red[4];
  const int j = blockIdx.x, h = blockIdx.y, g = h / (QH / KVH), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long p = pos[j], hi = pos[0] + kBlock - 1;
  const long long lo = p - WINDOW + 1 > 0 ? p - WINDOW + 1 : 0;
  const int nk = (int)(hi - lo + 1);
  float qv[4];
#pragma unroll
  for (int d = 0; d < 4; ++d) qv[d] = __bfloat162float(q[(size_t)j * q_ld + h * HD + 4 * lane + d]);
  const float scale = rsqrtf((float)HD);
  const int lo32 = (int)(lo & (kRing - 1));  // ring slot of key 0 (kRing is a power of 2)
  float mx = -INFINITY;
  for (int k0 = 8 * w; k0 < nk; k0 += 32) {
    uint2 kr[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (k0 + u < nk) kr[u] = *(const uint2*)(rk + ((size_t)((lo32 + k0 + u) & (kRing - 1)) * KVH + g) * HD + 4 * lane);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (k0 + u >= nk) break;
      const __nv_bfloat162 k01 = *(const __nv_bfloat162*)&kr[u].x, k23 = *(const __nv_bfloat162*)&kr[u].y;
      float s = qv[0] * __low2float(k01) + qv[1] * __high2float(k01) + qv[2] * __low2float(k23) + qv[3] * __high2float(k23);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
      s *= scale;
      if (lane == 0) sc[k0 + u] = s;
      mx = fmaxf(mx, s);
    }
  }
  if (lane == 0) red[w] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  float sum = 0.f;
  for (int kk = threadIdx.x; kk < nk; kk += 128) {
    const float e = __expf(sc[kk] - mx);
    sc[kk] = e;
    sum += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  if (lane == 0) red[w] = sum;
  __syncthreads();
  sum = red[0] + red[1] + red[2] + red[3];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k0 = 8 * w; k0 < nk; k0 += 32) {
    uint2 vr[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (k0 + u < nk) vr[u] = *(const uint2*)(rv + ((size_t)((lo32 + k0 + u) & (kRing - 1)) * KVH + g) * HD + 4 * lane);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (k0 + u >= nk) break;
      const __nv_bfloat162 v01 = *(const __nv_bfloat162*)&vr[u].x, v23 = *(const __nv_bfloat162*)&vr[u].y;
      const float pk = sc[k0 + u];
      acc[0] += pk * __low2float(v01), acc[1] += pk * __high2float(v01);
      acc[2] += pk * __low2float(v23), acc[3] += pk * __high2float(v23);
    }
  }
#pragma unroll
  for (int d = 0; d < 4; ++d) part[w][4 * lane + d] = acc[d];
  __syncthreads();
  const int d = threadIdx.x;
  out[(size_t)j * QH * HD + h * HD + d] = __float2bfloat16_rn((part[0][d] + part[1][d] + part[2][d] + part[3][d]) / sum);
}
// Split-key GQA variant (default; AGENS_DRAFT_ATTN=1 keeps draft_attn): block (query j, kv head g, key chunk c) scores
// its kAttnCh keys against the G = QH / KVH heads sharing them (each K / V row read once per block instead of once per
// q head: the ring is L2-resident and draft_attn reads it 32x per layer), then draft_attn_merge combines the chunks'
// (max, sum, unnormalised output) per (j, head). Drafter numerics only move which tokens are proposed (acceptance),
// never the greedy output.
constexpr int kAttnCh = 256, kAttnNch = (kAttnKeys + kAttnCh - 1) / kAttnCh, kG = QH / KVH;
__global__ void __launch_bounds__(128) draft_attn_split(const bf16* __restrict__ q, int q_ld, const bf16* __restrict__ rk,
                                                        const bf16* __restrict__ rv, const long long* __restrict__ pos,
                                                        float* __restrict__ opart, float* __restrict__ ml,
                                                        const int* rec = nullptr, size_t rstride = 0) {
  __shared__ float sc[kG][kAttnCh];
  __shared__ float part[4][kG][HD];
  __shared__ float red[4][kG];
  const int j = blockIdx.x, g = blockIdx.y, c = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j0 = j - j % kBlock;  // the first row of this query's draft block (batched: B blocks)
  if (rec) {
    const size_t off = (size_t)rec[(j / kBlock) * kRecInts + 19] * rstride;
    rk += off, rv += off;
  }
  const long long p = pos[j], hi = pos[j0] + kBlock - 1;
  const long long lo = p - WINDOW + 1 > 0 ? p - WINDOW + 1 : 0;
  const int nk = (int)(hi - lo + 1), k_lo = c * kAttnCh, k_hi = min(nk, k_lo + kAttnCh);
  const size_t ob = ((size_t)j * QH + g * kG) * kAttnNch + c;  // (j, head g*kG + h, chunk c) at ob + h * kAttnNch
  if (k_lo >= k_hi) {
    if (threadIdx.x < kG) ml[2 * (ob + threadIdx.x * kAttnNch)] = -INFINITY, ml[2 * (ob + threadIdx.x * kAttnNch) + 1] = 0.f;
    for (int i = threadIdx.x; i < kG * HD; i += 128) opart[(ob + (i / HD) * kAttnNch) * HD + i % HD] = 0.f;
    return;
  }
  float qv[kG][4];
#pragma unroll
  for (int h = 0; h < kG; ++h)
#pragma unroll
    for (int d = 0; d < 4; ++d) qv[h][d] = __bfloat162float(q[(size_t)j * q_ld + (g * kG + h) * HD + 4 * lane + d]);
  const float scale = rsqrtf((float)HD);
  const int lo32 = (int)(lo & (kRing - 1));
  float mx[kG];
#pragma unroll
  for (int h = 0; h < kG; ++h) mx[h] = -INFINITY;
  for (int k0 = k_lo + 8 * w; k0 < k_hi; k0 += 32) {
    uint2 kr[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (k0 + u < k_hi) kr[u] = *(const uint2*)(rk + ((size_t)((lo32 + k0 + u) & (kRing - 1)) * KVH + g) * HD + 4 * lane);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (k0 + u >= k_hi) break;
      const __nv_bfloat162 k01 = *(const __nv_bfloat162*)&kr[u].x, k23 = *(const __nv_bfloat162*)&kr[u].y;
      const float a0 = __low2float(k01), a1 = __high2float(k01), a2 = __low2float(k23), a3 = __high2float(k23);
#pragma unroll
      for (int h = 0; h < kG; ++h) {
        float sv = qv[h][0] * a0 + qv[h][1] * a1 + qv[h][2] * a2 + qv[h][3] * a3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(0xFFFFFFFFu, sv, o);
        sv *= scale;
        if (lane == 0) sc[h][k0 + u - k_lo] = sv;
        mx[h] = fmaxf(mx[h], sv);
      }
    }
  }
  if (lane == 0)
#pragma unroll
    for (int h = 0; h < kG; ++h) red[w][h] = mx[h];
  __syncthreads();
#pragma unroll
  for (int h = 0; h < kG; ++h) mx[h] = fmaxf(fmaxf(red[0][h], red[1][h]), fmaxf(red[2][h], red[3][h]));
  __syncthreads();
  const int n = k_hi - k_lo;
  float sum[kG];
#pragma unroll
  for (int h = 0; h < kG; ++h) {
    sum[h] = 0.f;
    for (int kk = threadIdx.x; kk < n; kk += 128) {
      const float e = __expf(sc[h][kk] - mx[h]);
      sc[h][kk] = e;
      sum[h] += e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum[h] += __shfl_xor_sync(0xFFFFFFFFu, sum[h], o);
  }
  if (lane == 0)
#pragma unroll
    for (int h = 0; h < kG; ++h) red[w][h] = sum[h];
  __syncthreads();
  float acc[kG][4];
#pragma unroll
  for (int h = 0; h < kG; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
  for (int k0 = k_lo + 8 * w; k0 < k_hi; k0 += 32) {
    uint2 vr[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (k0 + u < k_hi) vr[u] = *(const uint2*)(rv + ((size_t)((lo32 + k0 + u) & (kRing - 1)) * KVH + g) * HD + 4 * lane);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (k0 + u >= k_hi) break;
      const __nv_bfloat162 v01 = *(const __nv_bfloat162*)&vr[u].x, v23 = *(const __nv_bfloat162*)&vr[u].y;
      const float b0 = __low2float(v01), b1 = __high2float(v01), b2 = __low2float(v23), b3 = __high2float(v23);
#pragma unroll
      for (int h = 0; h < kG; ++h) {
        const float pk = sc[h][k0 + u - k_lo];
        acc[h][0] += pk * b0, acc[h][1] += pk * b1, acc[h][2] += pk * b2, acc[h][3] += pk * b3;
      }
    }
  }
#pragma unroll
  for (int h = 0; h < kG; ++h)
#pragma unroll
    for (int d = 0; d < 4; ++d) part[w][h][4 * lane + d] = acc[h][d];
  __syncthreads();
  for (int i = threadIdx.x; i < kG * HD; i += 128) {
    const int h = i / HD, d = i % HD;
    opart[(ob + h * kAttnNch) * HD + d] = part[0][h][d] + part[1][h][d] + part[2][h][d] + part[3][h][d];
  }
  if (threadIdx.x < kG) {
    const int h = threadIdx.x;
    ml[2 * (ob + h * kAttnNch)] = mx[h];
    ml[2 * (ob + h * kAttnNch) + 1] = red[0][h] + red[1][h] + red[2][h] + red[3][h];
  }
}
__global__ void __launch_bounds__(HD) draft_attn_merge(const float* __restrict__ opart, const float* __restrict__ ml,
                                                       bf16* __restrict__ out) {
  const int j = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const size_t b = ((size_t)j * QH + h) * kAttnNch;
  float M = -INFINITY;
#pragma unroll
  for (int c = 0; c < kAttnNch; ++c) M = fmaxf(M, ml[2 * (b + c)]);
  float num = 0.f, den = 0.f;
#pragma unroll
  for (int c = 0; c < kAttnNch; ++c) {
    const float wgt = __expf(ml[2 * (b + c)] - M);
    num += wgt * opart[(b + c) * HD + d];
    den += wgt * ml[2 * (b + c) + 1];
  }
  out[(size_t)j * QH * HD + h * HD + d] = __float2bfloat16_rn(num / den);
}
// Top TOPK by (value desc, index asc) in two stages: topk_part -- block (row, chunk) keeps the top TOPK of its chunk
// of logits[row][0..nv) -> (vals, ids)[row][chunk][TOPK]; topk_merge -- one block per row over those candidates.
constexpr int kTopkChunks = 64;
__device__ __forceinline__ bool topk_before(float v, int i, float w, int j) { return v > w || (v == w && i < j); }
// the block's top TOPK of its threads' sorted lists (lv/li, TOPK each) -> vals/ids (thread 0 writes)
__device__ void topk_block_merge(const float* lv, const int* li, float* vals, int* ids) {
  __shared__ float sv[256 * TOPK];
  __shared__ int si[256 * TOPK];
  __shared__ float bv[256];
  __shared__ int bi[256], bslot[256];
  const int tid = threadIdx.x;
  for (int i = 0; i < TOPK; ++i) sv[tid * TOPK + i] = lv[i], si[tid * TOPK + i] = li[i];
  __syncthreads();
  for (int r = 0; r < TOPK; ++r) {
    float v = -INFINITY;
    int id = 0x7FFFFFFF, sl = -1;
    for (int i = tid; i < 256 * TOPK; i += 256)
      if (topk_before(sv[i], si[i], v, id)) v = sv[i], id = si[i], sl = i;
    bv[tid] = v, bi[tid] = id, bslot[tid] = sl;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (tid < s && topk_before(bv[tid + s], bi[tid + s], bv[tid], bi[tid]))
        bv[tid] = bv[tid + s], bi[tid] = bi[tid + s], bslot[tid] = bslot[tid + s];
      __syncthreads();
    }
    if (tid == 0) {
      vals[r] = bv[0], ids[r] = bi[0];
      if (bslot[0] >= 0) sv[bslot[0]] = -INFINITY, si[bslot[0]] = 0x7FFFFFFF;
    }
    __syncthreads();
  }
}
__device__ void topk_insert(float* lv, int* li, float v, int c) {
  if (!topk_before(v, c, lv[TOPK - 1], li[TOPK - 1])) return;
  int k = TOPK - 1;
  while (k > 0 && topk_before(v, c, lv[k - 1], li[k - 1])) lv[k] = lv[k - 1], li[k] = li[k - 1], --k;
  lv[k] = v, li[k] = c;
}
__global__ void __launch_bounds__(256) topk_part(const bf16* __restrict__ logits, int ld, int nv, float* vals, int* ids) {
  const int row = blockIdx.x, ch = blockIdx.y, per = (nv + kTopkChunks - 1) / kTopkChunks;
  const int c0 = ch * per, c1 = c0 + per < nv ? c0 + per : nv;
  float lv[TOPK];
  int li[TOPK];
  for (int i = 0; i < TOPK; ++i) lv[i] = -INFINITY, li[i] = 0x7FFFFFFF;
  for (int c = c0 + threadIdx.x; c < c1; c += 256) topk_insert(lv, li, __bfloat162float(logits[(size_t)row * ld + c]), c);
  const size_t o = ((size_t)row * kTopkChunks + ch) * TOPK;
  topk_block_merge(lv, li, vals + o, ids + o);
}
__global__ void __launch_bounds__(256) topk_merge(const float* __restrict__ pv, const int* __restrict__ pi, float* vals,
                                                  int* ids) {
  const int row = blockIdx.x;
  float lv[TOPK];
  int li[TOPK];
  for (int i = 0; i < TOPK; ++i) lv[i] = -INFINITY, li[i] = 0x7FFFFFFF;
  for (int c = threadIdx.x; c < kTopkChunks * TOPK; c += 256)
    topk_insert(lv, li, pv[(size_t)row * kTopkChunks * TOPK + c], pi[(size_t)row * kTopkChunks * TOPK + c]);
  topk_block_merge(lv, li, vals + row * TOPK, ids + row * TOPK);
}
__global__ void map_ids(int* cid, int n, const int* ids) {  // shortlist indices -> token ids
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) cid[i] = ids[cid[i]];
}
// greedy walk: warp c scores candidate c of slot e; writes the 7 drafts to step[3..9]
__global__ void __launch_bounds__(TOPK * 32) selector_walk(const float* __restrict__ uval, const int* __restrict__ cid,
                                                          const bf16* __restrict__ hp, const bf16* __restrict__ A,
                                                          const bf16* __restrict__ B, int* step) {
  __shared__ float s[TOPK];
  __shared__ int pred;
  const int c = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (gridDim.x > 1) {  // batched: sequence blockIdx.x's 7 slots and record
    const int b = blockIdx.x;
    uval += (size_t)b * (kBlock - 1) * TOPK, cid += (size_t)b * (kBlock - 1) * TOPK, hp += (size_t)b * (kBlock - 1) * RANK;
    step += b * kRecInts;
  }
  if (threadIdx.x == 0) pred = step[2];
  __syncthreads();
  for (int e = 0; e < kBlock - 1; ++e) {
    const int cand = cid[e * TOPK + c];
    const bf16* a = A + (size_t)pred * RANK;
    const bf16* b = B + (size_t)cand * RANK;
    const bf16* h = hp + (size_t)e * RANK;
    float acc = 0.f;
    for (int r = lane; r < RANK; r += 32)
      acc += __bfloat162float(a[r]) * __bfloat162float(h[r]) * __bfloat162float(b[r]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) s[c] = uval[e * TOPK + c] + acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      int best = 0;
      for (int i = 1; i < TOPK; ++i)
        if (s[i] > s[best]) best = i;  // first max, as torch.argmax
      pred = cid[e * TOPK + best];
      step[3 + e] = pred;
    }
    __syncthreads();
  }
}

// bf16: cuBLAS (the drafter needs no batch invariance: its numerics move acceptance, never the output). W4: the
// engine's GEMV for m <= 16, else dequantized row tiles (bounded scratch) + cuBLAS.
int linear(const bf16* x, const Lin& w, bf16* y, int m, int n, int k, cudaStream_t s) {
  if (w.w) return ak_bf16_linear(x, w.w, y, m, n, k, s);
  const W4& q = *w.q;
  if (q.n != n || q.k != k) {
    fprintf(stderr, "m1draft: W4 [%d, %d] called as [%d, %d]\n", q.n, q.k, n, k);
    return (int)cudaErrorInvalidValue;
  }
  if (m == 1 && q.bits == 4) return ak_w4_gemv_kt(x, q.wt, q.st, y, n, k, q.group, 0, s);
  if (m <= 128)  // m > 8: the batched-verify kernels (one pass over the weights)
    return ak_wq_gemv_kt_multi(q.bits, x, m, 1, &q.wt, &q.st, reinterpret_cast<void* const*>(&y), &n, nullptr, k, q.group,
                               0, s);
  const int tn = (int)(D.wdq_bytes / 2 / k) & ~63;
  for (int n0 = 0; n0 < n; n0 += tn) {
    const int nt = n - n0 < tn ? n - n0 : tn;
    DCK(ak_wq_dequant_kt_rows(q.bits, q.wt, q.st, D.wdq, n, k, q.group, n0, nt, s));
    DCK(ak_bf16_linear_f32reduce_ld(x, D.wdq, y + n0, m, nt, k, n, s));
  }
  return 0;
}
}  // namespace

int register_weight(const char* name, void* ptr) {
  Wd[name] = ptr;
  return 0;
}
int register_w4(const char* name, const void* wt, const void* st, int n, int k, int group, int bits) {
  Wq[name] = W4{wt, st, n, k, group, bits};
  return 0;
}
bool ready() { return D.ready; }
size_t bytes() { return D.own; }

void set_ring(int ring) { D.ring = ring >= 0 && ring < D.n_rings ? ring : 0; }
int rings() { return D.n_rings; }
int copy_ring(int dst, int src, cudaStream_t s) {  // a forked session's drafter context (the source slot's ring)
  if (!D.ready || dst == src || dst <= 0 || src <= 0 || dst >= D.n_rings || src >= D.n_rings) return 0;
  const size_t n = (size_t)NL * kRing * KVH * HD;
  DCK(cudaMemcpyAsync(D.ring_k + dst * n, D.ring_k + src * n, n * sizeof(bf16), cudaMemcpyDeviceToDevice, s));
  DCK(cudaMemcpyAsync(D.ring_v + dst * n, D.ring_v + src * n, n * sizeof(bf16), cudaMemcpyDeviceToDevice, s));
  return 0;
}
size_t ring_bytes() { return D.ready ? (size_t)2 * NL * kRing * KVH * HD * sizeof(bf16) : 0; }
int ring_io(int ring, void* buf, bool to_buf, cudaStream_t s) {  // ring <-> buf (ring_bytes(): K then V)
  if (!D.ready || ring < 0 || ring >= D.n_rings) return 0;
  const size_t n = (size_t)NL * kRing * KVH * HD;
  bf16 *k = D.ring_k + ring * n, *v = D.ring_v + ring * n, *b = static_cast<bf16*>(buf);
  DCK(cudaMemcpyAsync(to_buf ? b : k, to_buf ? k : b, n * sizeof(bf16), cudaMemcpyDefault, s));
  DCK(cudaMemcpyAsync(to_buf ? b + n : v, to_buf ? v : b + n, n * sizeof(bf16), cudaMemcpyDefault, s));
  return 0;
}
int setup(int max_ctx, int mask_token_id, int max_pos, int n_rings, int max_seqs) {
  D.n_rings = n_rings < 1 ? 1 : n_rings;
  D.max_rows = (max_seqs < 1 ? 1 : max_seqs) * kBlock;
  D.max_ctx = max_ctx < kBlock ? kBlock : max_ctx;
  D.mask_id = mask_token_id;
  D.fc = lw("fc.weight"), D.hidden_norm = wp("hidden_norm.weight"), D.norm = wp("norm.weight");
  D.sel_proj = lw("candidate_selector.hidden_projection.weight");
  D.sel_A = wp("candidate_selector.predecessor_codebook"), D.sel_B = wp("candidate_selector.successor_codebook");
  if (!D.fc.ok() || !D.hidden_norm || !D.norm || !D.sel_proj.ok() || !D.sel_A || !D.sel_B) return 1;
  for (int l = 0; l < NL; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Layer& L = D.L[l];
    L.in_ln = wp(p + "input_layernorm.weight"), L.post_ln = wp(p + "post_attention_layernorm.weight");
    L.ac_base = wp(p + "attention_conv.base_kernel"), L.ac_proj = lw(p + "attention_conv.kernel_projection.weight");
    L.mc_base = wp(p + "mlp_conv.base_kernel"), L.mc_proj = lw(p + "mlp_conv.kernel_projection.weight");
    L.q = lw(p + "self_attn.q_proj.weight"), L.k = lw(p + "self_attn.k_proj.weight"), L.v = lw(p + "self_attn.v_proj.weight");
    L.o = lw(p + "self_attn.o_proj.weight"), L.qn = wp(p + "self_attn.q_norm.weight"), L.kn = wp(p + "self_attn.k_norm.weight");
    L.gu = lw(p + "mlp.gate_up_proj.weight"), L.down = lw(p + "mlp.down_proj.weight");
    for (const bf16* x : {L.in_ln, L.post_ln, L.ac_base, L.mc_base, L.qn, L.kn})
      if (!x) return 1;
    for (const Lin* x : {&L.ac_proj, &L.mc_proj, &L.q, &L.k, &L.v, &L.o, &L.gu, &L.down})
      if (!x->ok()) return 1;
  }
  D.ring_k = alloc<bf16>((size_t)D.n_rings * NL * kRing * KVH * HD);
  D.ring_v = alloc<bf16>((size_t)D.n_rings * NL * kRing * KVH * HD);
  D.max_pos = std::max(1 << 16, (max_pos + 4095) & ~4095);  // positions < the target's Wctx (was fixed 65536)
  if (getenv("AGENS_ROPE_CACHE") && getenv("AGENS_ROPE_CACHE")[0] == '1') {  // table (reference); default: on the fly
    D.rope = alloc<float>((size_t)D.max_pos * HD);
    DCK(ak_rope_cache(D.rope, D.max_pos, HD, 10000000.0f, nullptr));
  }
  const int T = D.max_rows;
  D.ids = alloc<int>(T), D.pos = alloc<long long>(T);
  if (T > kBlock) D.hsel = alloc<bf16>((size_t)T * HID);
  D.h = alloc<bf16>(T * HID), D.r = alloc<bf16>(T * HID), D.x = alloc<bf16>(T * HID), D.t = alloc<bf16>(T * HID);
  D.delta = alloc<bf16>(T * 4 * GROUPS), D.q = alloc<bf16>(T * (QH + 2 * KVH) * HD), D.kv = alloc<bf16>(T * 2 * KVH * HD);
  if (!(getenv("AGENS_DRAFT_ATTN") && getenv("AGENS_DRAFT_ATTN")[0] == '1'))  // split-key attention scratch
    D.apart = alloc<float>((size_t)T * QH * kAttnNch * HD), D.aml = alloc<float>((size_t)T * QH * kAttnNch * 2);
  D.ao = alloc<bf16>(T * QH * HD), D.gu = alloc<bf16>((size_t)T * 2 * INTER), D.act = alloc<bf16>((size_t)T * INTER);
  D.logits = alloc<bf16>((size_t)T * 248320), D.hp = alloc<bf16>(T * RANK);
  D.uval = alloc<float>(T * TOPK), D.cid = alloc<int>(T * TOPK);
  D.pval = alloc<float>(T * kTopkChunks * TOPK), D.pid = alloc<int>(T * kTopkChunks * TOPK);
  D.ctx = alloc<bf16>((size_t)D.max_ctx * HID), D.ck = alloc<bf16>((size_t)D.max_ctx * KVH * HD);
  D.cv = alloc<bf16>((size_t)D.max_ctx * KVH * HD);
  if (!Wq.empty()) D.wdq_bytes = (size_t)16 << 20, D.wdq = alloc<bf16>(D.wdq_bytes / 2);  // tiles of >= 64 rows
  if (!D.ring_k || !D.ring_v || !D.ids || !D.logits || !D.ctx || !D.ck || !D.cv || (D.wdq_bytes && !D.wdq))
    return 2;
  DCK(cudaDeviceSynchronize());
  D.ready = true;
  return 0;
}

// rec == nullptr: one sequence, ring D.ring; else B sequences (rows b * kBlock .., count n_dev[b * n_stride], ring
// from each record)
int materialize_impl(const Target& t, const bf16* cap, int n_max, const int* n_dev, int n_stride, const long long* pos,
                     const int* rec, int bk = kBlock) {
  if (!D.ready || n_max > D.max_ctx) return 1;
  cudaStream_t s = t.s;
  DCK(linear(cap, D.fc, D.ctx, n_max, HID, 5 * HID, s));
  DCK(ak_rmsnorm(D.ctx, D.hidden_norm, D.ctx, n_max, HID, HID, HID, EPS, 0, s));
  const size_t ring = (size_t)kRing * KVH * HD, rstride = (size_t)NL * ring;
  const size_t r0 = rec ? 0 : (size_t)D.ring * rstride;
  for (int l = 0; l < NL; ++l) {
    DCK(linear(D.ctx, D.L[l].k, D.ck, n_max, KVH * HD, HID, s));
    DCK(linear(D.ctx, D.L[l].v, D.cv, n_max, KVH * HD, HID, s));
    DCK(ak_rmsnorm(D.ck, D.L[l].kn, D.ck, n_max * KVH, HD, HD, HD, EPS, 0, s));
    DCK(D.rope ? ak_rope_neox(D.ck, D.ck, D.rope, (const int64_t*)pos, n_max, 0, KVH, HD, HD, 0, KVH * HD, s)
               : ak_rope_neox_fly(D.ck, D.ck, (const int64_t*)pos, n_max, 0, KVH, HD, HD, 0, KVH * HD, 10000000.0f, s));
    store_ring<<<(n_max * KVH * HD + 255) / 256, 256, 0, s>>>(D.ck, D.cv, KVH * HD, pos, n_max, n_dev,
                                                              D.ring_k + r0 + l * ring, D.ring_v + r0 + l * ring, rec,
                                                              n_stride, rstride, bk);
  }
  return (int)cudaGetLastError();
}
int materialize(const Target& t, const bf16* cap, int n_max, const int* n_dev, const long long* pos) {
  return materialize_impl(t, cap, n_max, n_dev, 0, pos, nullptr);
}
int materialize_b(const Target& t, const bf16* cap, int B, const int* n_dev, int n_stride, const long long* pos,
                  const int* rec, int rows_per) {
  return materialize_impl(t, cap, B * rows_per, n_dev, n_stride, pos, rec, rows_per);
}

__global__ void block_prep_b(const int* rec, int mask_id, int* ids, long long* pos) {  // B blocks of kBlock rows
  const int b = blockIdx.x, j = threadIdx.x;
  if (j >= kBlock) return;
  const int* r = rec + b * kRecInts;
  ids[b * kBlock + j] = j == 0 ? r[2] : mask_id;
  pos[b * kBlock + j] = (long long)r[1] + j;
}
__global__ void gather_sel_rows(const bf16* __restrict__ x, int B, bf16* __restrict__ y) {  // rows 1..7 of each block
  const long n = (long)B * (kBlock - 1) * HID / 8;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long r = i / (HID / 8), c = i % (HID / 8), b = r / (kBlock - 1), e = r % (kBlock - 1);
    reinterpret_cast<uint4*>(y)[i] = reinterpret_cast<const uint4*>(x)[(b * kBlock + 1 + e) * (HID / 8) + c];
  }
}
// B = 0: one sequence (step = the verify staging buffer, ring D.ring); else B sequences, records rec
int propose_impl(const Target& t, int B, int* step) {
  if (!D.ready) return 1;
  cudaStream_t s = t.s;
  const int T = B ? B * kBlock : kBlock;
  if (T > D.max_rows) return 2;
  const size_t ring = (size_t)kRing * KVH * HD, rstride = (size_t)NL * ring;
  const size_t r0 = B ? 0 : (size_t)D.ring * rstride;
  const int* rec = B ? step : nullptr;
  if (B) block_prep_b<<<B, 32, 0, s>>>(step, D.mask_id, D.ids, D.pos);
  else block_prep<<<1, 32, 0, s>>>(step, D.mask_id, D.ids, D.pos);
  DCK(ak_embed(D.ids, t.embed, D.h, T, HID, s));
  for (int l = 0; l < NL; ++l) {
    const Layer& L = D.L[l];
    if (l == 0) DCK(cudaMemcpyAsync(D.r, D.h, (size_t)T * HID * 2, cudaMemcpyDeviceToDevice, s));
    else DCK(ak_bf16_binary(D.h, D.r, D.r, (long)T * HID, 0, s));
    DCK(ak_rmsnorm(D.r, L.in_ln, D.x, T, HID, HID, HID, EPS, 0, s));
    DCK(linear(D.x, L.ac_proj, D.delta, T, 4 * GROUPS, HID, s));
    grouped_conv<<<dim3(20, T), 256, 0, s>>>(D.x, D.delta, L.ac_base, 0, D.t);
    bf16 *qb = D.q, *kb = D.q + (size_t)T * QH * HD, *vb = kb + (size_t)T * KVH * HD;
    DCK(linear(D.t, L.q, qb, T, QH * HD, HID, s));
    DCK(linear(D.t, L.k, kb, T, KVH * HD, HID, s));
    DCK(linear(D.t, L.v, vb, T, KVH * HD, HID, s));
    DCK(ak_rmsnorm(qb, L.qn, qb, T * QH, HD, HD, HD, EPS, 0, s));
    DCK(ak_rmsnorm(kb, L.kn, kb, T * KVH, HD, HD, HD, EPS, 0, s));
    DCK(D.rope ? ak_rope_neox(qb, kb, D.rope, (const int64_t*)D.pos, T, QH, KVH, HD, HD, QH * HD, KVH * HD, s)
               : ak_rope_neox_fly(qb, kb, (const int64_t*)D.pos, T, QH, KVH, HD, HD, QH * HD, KVH * HD, 10000000.0f, s));
    store_ring<<<(T * KVH * HD + 255) / 256, 256, 0, s>>>(kb, vb, KVH * HD, D.pos, T, nullptr, D.ring_k + r0 + l * ring,
                                                          D.ring_v + r0 + l * ring, rec, 0, rstride);
    if (D.apart) {
      draft_attn_split<<<dim3(T, KVH, kAttnNch), 128, 0, s>>>(qb, QH * HD, D.ring_k + r0 + l * ring,
                                                             D.ring_v + r0 + l * ring, D.pos, D.apart, D.aml, rec,
                                                             rstride);
      draft_attn_merge<<<dim3(T, QH), HD, 0, s>>>(D.apart, D.aml, D.ao);
    } else {
      if (B) return 3;  // the split attention is the batched path
      draft_attn<<<dim3(T, QH), 128, 0, s>>>(qb, QH * HD, D.ring_k + r0 + l * ring, D.ring_v + r0 + l * ring, D.pos,
                                              D.ao);
    }
    DCK(linear(D.ao, L.o, D.x, T, HID, QH * HD, s));
    grouped_conv<<<dim3(20, T), 256, 0, s>>>(D.x, D.delta, L.ac_base, 1, D.h);
    DCK(ak_bf16_binary(D.h, D.r, D.r, (long)T * HID, 0, s));
    DCK(ak_rmsnorm(D.r, L.post_ln, D.x, T, HID, HID, HID, EPS, 0, s));
    DCK(linear(D.x, L.mc_proj, D.delta, T, 4 * GROUPS, HID, s));
    grouped_conv<<<dim3(20, T), 256, 0, s>>>(D.x, D.delta, L.mc_base, 0, D.t);
    DCK(linear(D.t, L.gu, D.gu, T, 2 * INTER, HID, s));
    DCK(ak_silu_mul(D.gu, D.act, T, INTER, s));
    DCK(linear(D.act, L.down, D.x, T, HID, INTER, s));
    grouped_conv<<<dim3(20, T), 256, 0, s>>>(D.x, D.delta, L.mc_base, 1, D.h);
  }
  DCK(ak_bf16_binary(D.h, D.r, D.r, (long)T * HID, 0, s));
  DCK(ak_rmsnorm(D.r, D.norm, D.x, T, HID, HID, HID, EPS, 0, s));
  // selector over slots 1..7 of each block
  const int ns = B ? B * (kBlock - 1) : kBlock - 1;
  const bf16* hpred = D.x + HID;
  if (B) {
    gather_sel_rows<<<256, 256, 0, s>>>(D.x, B, D.hsel);
    hpred = D.hsel;
  }
  if (t.nsl > 0 && t.sl_drafter) {  // shortlist head: candidates among its rows, then their token ids
    DCK(ak_bf16_linear(hpred, t.sl_head, D.logits, ns, t.nsl, HID, s));
    topk_part<<<dim3(ns, kTopkChunks), 256, 0, s>>>(D.logits, t.nsl, t.nsl, D.pval, D.pid);
    topk_merge<<<ns, 256, 0, s>>>(D.pval, D.pid, D.uval, D.cid);
    map_ids<<<(ns * TOPK + 255) / 256, 256, 0, s>>>(D.cid, ns * TOPK, t.sl_ids);
    DCK(linear(hpred, D.sel_proj, D.hp, ns, RANK, HID, s));
    selector_walk<<<B ? B : 1, TOPK * 32, 0, s>>>(D.uval, D.cid, D.hp, D.sel_A, D.sel_B, step);
    return (int)cudaGetLastError();
  }
  if (t.lm_head_w4) {
    const int nv = 248320;
    void* y = D.logits;
    DCK(ak_wq_gemv_kt_multi(4, hpred, ns, 1, &t.lm_head_w4, &t.lm_head_w4s, &y, &nv, nullptr, HID, 128, 0, s));
  } else if (t.lm_head) DCK(linear(hpred, Lin{(const bf16*)t.lm_head}, D.logits, ns, 248320, HID, s));
  else  // the INT8 head takes up to 16 rows per call (batched: B x 7)
    for (int r0 = 0; r0 < ns; r0 += 16)
      DCK(ak_i8_gemv(hpred + (size_t)r0 * HID, t.lm_head_i8, t.lm_head_scale, D.logits + (size_t)r0 * 248320,
                     ns - r0 < 16 ? ns - r0 : 16, 248320, HID, 248320, s));
  topk_part<<<dim3(ns, kTopkChunks), 256, 0, s>>>(D.logits, 248320, t.vocab_valid, D.pval, D.pid);
  topk_merge<<<ns, 256, 0, s>>>(D.pval, D.pid, D.uval, D.cid);
  DCK(linear(hpred, D.sel_proj, D.hp, ns, RANK, HID, s));
  selector_walk<<<B ? B : 1, TOPK * 32, 0, s>>>(D.uval, D.cid, D.hp, D.sel_A, D.sel_B, step);
  return (int)cudaGetLastError();
}
int propose(const Target& t, int* step) { return propose_impl(t, 0, step); }
int propose_b(const Target& t, int B, int* rec) { return propose_impl(t, B, rec); }
static_assert(TOPK == kCandK, "m1_draft.h kCandK");
const int* cand_ids() { return D.ready ? D.cid : nullptr; }
}  // namespace m1draft
