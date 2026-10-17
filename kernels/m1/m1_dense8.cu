// INT8 KV for the dense attention layer (AGENS_KV_INT8_DENSE=1; one device, personal mode): K/V stored int8 + one fp32
// scale per (token, kv head) with the INT8 KV quantizer of the BCSA layers (ak_kv_quant_store: scale = amax / 127,
// q = rint(x / scale)), read as bf16(q * scale). 2,080 instead of 4,096 bytes per token (0.53 GB at 262K).
// flashinfer reads bf16 K/V, so attention runs over tiles of TILE positions: a tile is dequantized into a bf16 staging
// buffer, attended (decode / verify: the engine's multi-query attention, one bs = 1 plan per query -- every tile is <=
// 8192 rows, so the bs = 1 chunk size is 128 for any length and the multi launch never stands down; prefill: the host
// planned prefill over the tile, causal for the chunk's own keys), and the tiles' (o, lse) are merged in fp32 in tile
// order. Decode and every verify query run the same per-query tile partition and merge order: verify == decode bitwise.
// Quality gate (vs bf16 KV, teacher-forced): tools/compare_logits.py.
#include "m1_dense8.h"
#include "agens_kernels.h"
#include "agens_kv.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

extern "C" {
int ak_attn_decode_multi_plan_mode(int mslot, int T, const int* rng, int stride, const int* indices, int max_len,
                                   int qo_heads, int kv_heads, const int* disable_slots, int mode, void* stream);
int ak_attn_decode_multi(int mslot, void* q, void* kc, void* vc, int qo_heads, int kv_heads, float sm_scale, void* o,
                         float* lse, void* stream);
int ak_prefill_staged_lse(void* q, void* kc, void* vc, int* indices, int T, int kv_len, int causal, int qo_heads,
                          int kv_heads, float sm_scale, void* o, float* lse, void* stream);
}

namespace m1d8 {
namespace {
using bf16 = __nv_bfloat16;
constexpr int KVH = 4, HD = 256, QH = 24, MAXT = 1024;  // MAXT: prefill chunk rows
constexpr int MSLOT1 = 3, MSLOT8 = 4;                     // multi-attention slots: decode (T = 1) / verify (T <= 8)

#define CK(x)                                                                          \
  do {                                                                                 \
    int e_ = (int)(x);                                                                 \
    if (e_) {                                                                          \
      fprintf(stderr, "m1d8: %s failed (%d) at %s:%d\n", #x, e_, __FILE__, __LINE__); \
      return e_;                                                                       \
    }                                                                                  \
  } while (0)

struct State {
  int8_t *kq = nullptr, *vq = nullptr;
  float *ks = nullptr, *vs = nullptr;
  bf16 *kst = nullptr, *vst = nullptr;  // staging [TILE][4][256]
  int* iota = nullptr;                  // 0 .. TILE-1 (staging rows)
  int* rng = nullptr;                   // per-tile query ranges [8][4]
  bf16* to = nullptr;                   // tile output [MAXT][24][256]
  float *tl = nullptr, *acc = nullptr, *al = nullptr;  // tile lse [MAXT][24]; merged o [MAXT][24][256] fp32, lse
  size_t bytes = 0;
};
State g;

template <typename T>
T* dal(size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T) > 0 ? n * sizeof(T) : 16) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  g.bytes += n * sizeof(T);
  return (T*)p;
}

__global__ void iota_k(int* p, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = i;
}
// staging[j] = dequant(row r2t[k0 + j]) for j < n, n = clamp(len - k0, 0, TILE) with len = *len_dev or len_host
__global__ void dequant_tile_k(const int8_t* q, const float* sc, const int* r2t, int k0, const long long* last_pos,
                               int len_host, bf16* dst) {
  const int len = last_pos ? (int)(*last_pos + 1) : len_host;
  const int n = min(max(len - k0, 0), TILE);
  const int per_row = KVH * HD / 8;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < (long)n * per_row; i += (long)gridDim.x * blockDim.x) {
    const long j = i / per_row, c8 = i % per_row, h = c8 * 8 / HD;
    const long r = r2t[k0 + j];
    alignas(16) bf16 o[8];
    const float s = sc[r * KVH + h];
    const uint2 u = *reinterpret_cast<const uint2*>(q + r * KVH * HD + c8 * 8);
    const int8_t* e = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16_rn(__fmul_rn((float)e[k], s));  // = ak_kv_dequant_rows
    *reinterpret_cast<uint4*>(dst + j * KVH * HD + c8 * 8) = *reinterpret_cast<const uint4*>(o);
  }
}
// query t (position pos[t]) sees tile positions [k0, pos[t] + 1): staging rows [0, clamp(pos[t] + 1 - k0, 0, TILE))
__global__ void tile_ranges_k(const long long* pos, int T, int k0, int* rng) {
  const int t = threadIdx.x;
  if (t >= T) return;
  const int n = (int)min(max(pos[t] + 1 - (long long)k0, 0LL), (long long)TILE);
  rng[4 * t] = 0, rng[4 * t + 1] = n, rng[4 * t + 2] = 0, rng[4 * t + 3] = 0;
}
// running merge of normalized tile results: (acc, al) <- merge((acc, al), (o_k, l_k)) in fp32, tile order; first: copy.
// An empty tile (l_k = -inf) leaves the state as it is.
__global__ void merge_k(const bf16* to, const float* tl, float* acc, float* al, int rows, int first) {
  const long n = (long)rows * QH * HD;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long r = i / HD;  // (token, head)
    const float lk = tl[r], ok = __bfloat162float(to[i]);
    if (first) {
      acc[i] = isinf(lk) && lk < 0 ? 0.f : ok;
      if (i % HD == 0) al[r] = lk;
      continue;
    }
    const float la = al[r];
    if (isinf(lk) && lk < 0) continue;
    if (isinf(la) && la < 0) {
      acc[i] = ok;
    } else {
      const float m = fmaxf(la, lk), wa = exp2f(la - m), wb = exp2f(lk - m);
      acc[i] = (acc[i] * wa + ok * wb) / (wa + wb);
    }
  }
}
// lse update after the o update of the same tile (one thread per (token, head) row)
__global__ void merge_lse_k(const float* tl, float* al, int rows) {
  const int n = rows * QH;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
    const float la = al[r], lk = tl[r];
    if (isinf(lk) && lk < 0) continue;
    if (isinf(la) && la < 0) { al[r] = lk; continue; }
    const float m = fmaxf(la, lk);
    al[r] = m + log2f(exp2f(la - m) + exp2f(lk - m));
  }
}
__global__ void final_k(const float* acc, bf16* o, long n) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    o[i] = __float2bfloat16_rn(acc[i]);
}
int grid(long n) { return (int)std::min<long>((n + 255) / 256, 4096); }
int merge(int rows, bool first, cudaStream_t s) {
  merge_k<<<grid((long)rows * QH * HD), 256, 0, s>>>(g.to, g.tl, g.acc, g.al, rows, first ? 1 : 0);
  if (!first) merge_lse_k<<<grid((long)rows * QH), 256, 0, s>>>(g.tl, g.al, rows);
  return (int)cudaGetLastError();
}
}  // namespace

bool on() {
  static const int v = [] { const char* e = getenv("AGENS_KV_INT8_DENSE"); return e && e[0] == '1'; }();
  return v;
}
size_t bytes() { return g.bytes; }
size_t scratch_bytes() {
  return (size_t)2 * TILE * KVH * HD * 2 + (size_t)TILE * 4 + (size_t)MAXT * QH * HD * (2 + 4) + (size_t)MAXT * QH * 8 + 4096;
}

int alloc(int rows) {
  if (g.kq) return 0;
  const size_t n = (size_t)(rows + 1) * KVH;
  if (!(g.kq = dal<int8_t>(n * HD)) || !(g.vq = dal<int8_t>(n * HD)) || !(g.ks = dal<float>(n)) ||
      !(g.vs = dal<float>(n)) || !(g.kst = dal<bf16>((size_t)TILE * KVH * HD)) ||
      !(g.vst = dal<bf16>((size_t)TILE * KVH * HD)) || !(g.iota = dal<int>(TILE)) || !(g.rng = dal<int>(64)) ||
      !(g.to = dal<bf16>((size_t)MAXT * QH * HD)) || !(g.tl = dal<float>((size_t)MAXT * QH)) ||
      !(g.acc = dal<float>((size_t)MAXT * QH * HD)) || !(g.al = dal<float>((size_t)MAXT * QH)))
    return (int)cudaErrorMemoryAllocation;
  iota_k<<<grid(TILE), 256>>>(g.iota, TILE);
  return (int)cudaDeviceSynchronize();
}

int store(const void* k, const void* v, const int* loc, int T, cudaStream_t s) {
  CK(ak_kv_quant_store(k, T, KVH, HD, loc, g.kq, g.ks, s));
  return ak_kv_quant_store(v, T, KVH, HD, loc, g.vq, g.vs, s);
}

int attend_dec(void* q, const long long* pos, const int* r2t, int T, int ntiles, float scale, void* o, cudaStream_t s) {
  if (T < 1 || T > 8) return (int)cudaErrorInvalidValue;
  const int ms = T == 1 ? MSLOT1 : (T == 8 ? MSLOT8 : MSLOT8 + 1);  // a multi slot is sized for one T
  for (int k = 0; k < ntiles; ++k) {
    const int k0 = k * TILE;
    dequant_tile_k<<<grid((long)TILE * KVH * HD / 8), 256, 0, s>>>(g.kq, g.ks, r2t, k0, pos + T - 1, 0, g.kst);
    dequant_tile_k<<<grid((long)TILE * KVH * HD / 8), 256, 0, s>>>(g.vq, g.vs, r2t, k0, pos + T - 1, 0, g.vst);
    tile_ranges_k<<<1, 32, 0, s>>>(pos, T, k0, g.rng);
    CK(ak_attn_decode_multi_plan_mode(ms, T, g.rng, 4, g.iota, TILE, QH, KVH, nullptr, 1, s));
    CK(ak_attn_decode_multi(ms, q, g.kst, g.vst, QH, KVH, scale, g.to, g.tl, s));
    CK(merge(T, k == 0, s));
  }
  final_k<<<grid((long)T * QH * HD), 256, 0, s>>>(g.acc, static_cast<bf16*>(o), (long)T * QH * HD);
  return (int)cudaGetLastError();
}

int attend_prefill(void* q, int seq0, int T, const int* r2t, float scale, void* o, cudaStream_t s) {
  if (T < 1 || T > MAXT) return (int)cudaErrorInvalidValue;
  bool first = true;
  for (int k0 = 0; k0 < seq0; k0 += TILE) {  // keys before the chunk: every query sees all of them (no mask)
    const int n = std::min(TILE, seq0 - k0);
    dequant_tile_k<<<grid((long)n * KVH * HD / 8), 256, 0, s>>>(g.kq, g.ks, r2t, k0, nullptr, seq0, g.kst);
    dequant_tile_k<<<grid((long)n * KVH * HD / 8), 256, 0, s>>>(g.vq, g.vs, r2t, k0, nullptr, seq0, g.vst);
    CK(ak_prefill_staged_lse(q, g.kst, g.vst, g.iota, T, n, 0, QH, KVH, scale, g.to, g.tl, s));
    CK(merge(T, first, s));
    first = false;
  }
  // the chunk's own keys (positions seq0 .. seq0 + T - 1): causal
  dequant_tile_k<<<grid((long)T * KVH * HD / 8), 256, 0, s>>>(g.kq, g.ks, r2t, seq0, nullptr, seq0 + T, g.kst);
  dequant_tile_k<<<grid((long)T * KVH * HD / 8), 256, 0, s>>>(g.vq, g.vs, r2t, seq0, nullptr, seq0 + T, g.vst);
  CK(ak_prefill_staged_lse(q, g.kst, g.vst, g.iota, T, T, 1, QH, KVH, scale, g.to, g.tl, s));
  CK(merge(T, first, s));
  final_k<<<grid((long)T * QH * HD), 256, 0, s>>>(g.acc, static_cast<bf16*>(o), (long)T * QH * HD);
  return (int)cudaGetLastError();
}
}  // namespace m1d8
