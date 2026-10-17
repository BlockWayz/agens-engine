// KV tiering + INT8 KV kernels (docs/kv_tiering.md). Row gathers/scatters between device pools, pinned host pools
// (mapped, UVA) and staging buffers; window/ring row lists; per-(token, head) symmetric INT8 quantize/dequantize.
// The copies are pure data movement, so a tiered attention that reads staged rows is bit-identical to reading the
// pool directly. INT8 changes numerics by design (quality-gated).
#include "agens_kv.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {
using bf16 = __nv_bfloat16;
int grid(long n) { return (int)std::min<long>(8192, (n + 255) / 256); }

__global__ void copy_rows_k(const uint4* src, const int32_t* src_rows, uint4* dst, const int32_t* dst_rows, int n, int r16) {
  const long m = (long)n * r16;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
    const long j = i / r16, c = i % r16;
    const long sr = src_rows ? src_rows[j] : j, dr = dst_rows ? dst_rows[j] : j;
    dst[dr * r16 + c] = sr >= 0 ? src[sr * r16 + c] : make_uint4(0, 0, 0, 0);
  }
}

__global__ void window_rows_k(int p0, int n, int lo, int zero_row, int ring_base, int ring, int chunk_start, int chunk_base,
                              int32_t* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const int p = p0 + j;
  out[j] = p < lo ? zero_row : (p >= chunk_start ? chunk_base + (p - chunk_start) : ring_base + p % ring);
}

// One warp per (token, head) row; each lane owns 8 consecutive values per 256-wide slab.
__global__ void quant_store_k(const bf16* x, int T, int heads, int hd, const int32_t* rows, int8_t* q, float* scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= T * heads) return;
  const int t = warp / heads, h = warp % heads;
  const bf16* xr = x + ((long)t * heads + h) * hd;
  float amax = 0.f;
  for (int c = lane * 8; c < hd; c += 256) {
    const uint4 u = *reinterpret_cast<const uint4*>(xr + c);
    const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(e[k])));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = __fdiv_rn(amax, 127.f);
  const long row = rows ? rows[t] : t;
  int8_t* qr = q + (row * heads + h) * hd;
  for (int c = lane * 8; c < hd; c += 256) {
    const uint4 u = *reinterpret_cast<const uint4*>(xr + c);
    const bf16* e = reinterpret_cast<const bf16*>(&u);
    alignas(8) int8_t o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int v = s > 0.f ? __float2int_rn(__fdiv_rn(__bfloat162float(e[k]), s)) : 0;
      o[k] = (int8_t)max(-127, min(127, v));
    }
    *reinterpret_cast<uint2*>(qr + c) = *reinterpret_cast<const uint2*>(o);
  }
  if (lane == 0) scale[row * heads + h] = s;
}

// One thread per 8 values: 8 B of int8 in, 16 B of bf16 out.
__global__ void dequant_rows_k(const int8_t* q, const float* scale, const int32_t* rows, int n, int heads, int hd, bf16* dst) {
  const int per_row = heads * hd / 8;
  const long m = (long)n * per_row;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
    const long j = i / per_row, c8 = i % per_row, h = c8 * 8 / hd;
    const long r = rows ? rows[j] : j;
    alignas(16) bf16 o[8];
    if (r >= 0) {
      const float s = scale[r * heads + h];
      const uint2 u = *reinterpret_cast<const uint2*>(q + r * heads * hd + c8 * 8);
      const int8_t* e = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16_rn(__fmul_rn((float)e[k], s));
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16_rn(0.f);
    }
    *reinterpret_cast<uint4*>(dst + j * heads * hd + c8 * 8) = *reinterpret_cast<const uint4*>(o);
  }
}
// Block means of 4 rows as ATen computes x.view(.., 4, heads, hd).mean(dim of the 4) for bf16: fp32 accumulation in
// row order, times 0.25f, one bf16 rounding (checked bitwise in tools/test_kv_tier.py).
__global__ void block_mean_k(const bf16* src, const int32_t* rows, int nblk, int width, bf16* dst, const int32_t* drows,
                             int dstride, int dshift) {
  const long m = (long)nblk * width;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
    const long b = i / width, c = i % width;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const long r = rows ? rows[b * 4 + k] : b * 4 + k;
      acc = __fadd_rn(acc, __bfloat162float(src[r * width + c]));
    }
    const long d = drows ? drows[b * dstride] >> dshift : b;
    dst[d * width + c] = __float2bfloat16_rn(__fmul_rn(acc, 0.25f));
  }
}
__global__ void pick_rows_k(const int32_t* in, int offset, int stride, int n, int shift, int32_t* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) out[j] = in[offset + (long)j * stride] >> shift;
}
// Block cache: row j comes from the GPU cache when its key was staged last step, else from the (host) pool.
__global__ void cached_gather_k(const uint4* pool, const uint4* cache, const int32_t* slot_of, const int32_t* keys, int n,
                                int r16, uint4* dst, int* misses) {
  const long m = (long)n * r16;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x) {
    const long j = i / r16, c = i % r16;
    const long key = keys[j];
    const int s = slot_of[key];
    dst[i] = s >= 0 ? cache[(long)s * r16 + c] : pool[key * r16 + c];
    if (misses && c == 0 && s < 0) atomicAdd(misses, 1);
  }
}
__global__ void slot_map_k(int32_t* slot_of, const int32_t* keys, int n, int set) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n && keys[j] >= 0) slot_of[keys[j]] = set ? j : -1;
}
}  // namespace

extern "C" {
int ak_kv_cached_gather(const void* pool, const void* cache, const int32_t* slot_of, const int32_t* keys, int n,
                        int row_bytes, void* dst, int* misses, void* stream) {
  if (row_bytes % 16) return (int)cudaErrorInvalidValue;
  if (n <= 0) return 0;
  const int r16 = row_bytes / 16;
  cached_gather_k<<<grid((long)n * r16), 256, 0, (cudaStream_t)stream>>>((const uint4*)pool, (const uint4*)cache, slot_of,
                                                                         keys, n, r16, (uint4*)dst, misses);
  return (int)cudaGetLastError();
}
int ak_kv_slot_map(int32_t* slot_of, const int32_t* keys, int n, int set, void* stream) {
  if (n <= 0) return 0;
  slot_map_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(slot_of, keys, n, set);
  return (int)cudaGetLastError();
}

int ak_kv_block_mean(const void* src, const int32_t* rows, int nblk, int width, void* dst, const int32_t* dst_rows,
                     int dst_stride, int dst_shift, void* stream) {
  if (nblk <= 0) return 0;
  block_mean_k<<<grid((long)nblk * width), 256, 0, (cudaStream_t)stream>>>((const bf16*)src, rows, nblk, width, (bf16*)dst,
                                                                          dst_rows, dst_stride, dst_shift);
  return (int)cudaGetLastError();
}

int ak_kv_pick_rows(const int32_t* in, int offset, int stride, int n, int shift, int32_t* out, void* stream) {
  if (n <= 0) return 0;
  pick_rows_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(in, offset, stride, n, shift, out);
  return (int)cudaGetLastError();
}

int ak_kv_copy_rows(const void* src, const int32_t* src_rows, void* dst, const int32_t* dst_rows, int n, int row_bytes,
                    void* stream) {
  if (row_bytes % 16) return (int)cudaErrorInvalidValue;
  if (n <= 0) return 0;
  const int r16 = row_bytes / 16;
  copy_rows_k<<<grid((long)n * r16), 256, 0, (cudaStream_t)stream>>>((const uint4*)src, src_rows, (uint4*)dst, dst_rows,
                                                                     n, r16);
  return (int)cudaGetLastError();
}

int ak_kv_window_rows(int p0, int n, int lo, int zero_row, int ring_base, int ring, int chunk_start, int chunk_base,
                      int32_t* out, void* stream) {
  if (n <= 0) return 0;
  window_rows_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(p0, n, lo, zero_row, ring_base, ring, chunk_start,
                                                                   chunk_base, out);
  return (int)cudaGetLastError();
}

int ak_kv_quant_store(const void* x, int T, int heads, int head_dim, const int32_t* rows, int8_t* q, float* scale,
                      void* stream) {
  if (head_dim % 8 || T <= 0) return T < 0 ? (int)cudaErrorInvalidValue : 0;
  const long threads = (long)T * heads * 32;
  quant_store_k<<<(int)((threads + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, T, heads, head_dim, rows,
                                                                                q, scale);
  return (int)cudaGetLastError();
}

int ak_kv_dequant_rows(const int8_t* q, const float* scale, const int32_t* rows, int n, int heads, int head_dim,
                       void* dst, void* stream) {
  if (head_dim % 8) return (int)cudaErrorInvalidValue;
  if (n <= 0) return 0;
  dequant_rows_k<<<grid((long)n * heads * head_dim / 8), 256, 0, (cudaStream_t)stream>>>(q, scale, rows, n, heads,
                                                                                         head_dim, (bf16*)dst);
  return (int)cudaGetLastError();
}
}  // extern "C"
