// C ABI of the KV tiering / INT8 KV kernels (kernels/src/kv_tier.cu), part of libagens_kernels.
// Design: docs/kv_tiering.md. All functions return 0 or a cudaError_t value; `stream` is a cudaStream_t.
// Row pointers may be device memory or pinned host memory mapped into the device (UVA): a kernel that reads a
// host-mapped pool pulls exactly the rows it touches over PCIe (zero-copy), so a tiered fetch is a plain gather.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// dst[dst_rows ? dst_rows[j] : j] = src[src_rows ? src_rows[j] : j] for j < n; row_bytes % 16 == 0.
// A negative source row writes zeros. Pure data movement (bit-exact by construction).
int ak_kv_copy_rows(const void* src, const int32_t* src_rows, void* dst, const int32_t* dst_rows, int n, int row_bytes,
                    void* stream);

// Pool / window row of positions p0 .. p0+n-1 (one int32 per position):
//   p < lo                 -> zero_row
//   p >= chunk_start       -> chunk_base + (p - chunk_start)          (prefill chunk area)
//   otherwise              -> ring_base + p % ring                    (per-session window ring)
// With ring > every p and ring_base = -lo this is an identity list into a compact staging buffer.
int ak_kv_window_rows(int p0, int n, int lo, int zero_row, int ring_base, int ring, int chunk_start, int chunk_base,
                      int32_t* out, void* stream);

// INT8 KV, symmetric per (token, kv head): scale = amax(|x|) / 127 over head_dim values, q = rint(x / scale).
// x [T][heads][head_dim] bf16 -> q[rows[t]] [heads][head_dim] int8, scale[rows[t]] [heads] fp32 (rows may be host-mapped).
int ak_kv_quant_store(const void* x, int T, int heads, int head_dim, const int32_t* rows, int8_t* q, float* scale,
                      void* stream);
// dst[j] = bf16(q[rows[j]] * scale[rows[j]]) [heads][head_dim]; rows[j] < 0 -> zeros. rows == NULL -> identity.
int ak_kv_dequant_rows(const int8_t* q, const float* scale, const int32_t* rows, int n, int heads, int head_dim,
                       void* dst, void* stream);

// Compressed far blocks: block b = mean of rows rows[4b..4b+3] (identity when NULL), width bf16 values per row, computed
// as ATen's bf16 .mean() over the 4 rows (fp32 sum in order, * 0.25f, one rounding), written to
// dst[dst_rows ? dst_rows[b * dst_stride] >> dst_shift : b] (dst_shift 2: per-page pools). The far pool of
// AGENS_KV_TIER=2 (one mean row per 4-row page).
int ak_kv_block_mean(const void* src, const int32_t* rows, int nblk, int width, void* dst, const int32_t* dst_rows,
                     int dst_stride, int dst_shift, void* stream);
// out[j] = in[offset + j * stride] >> shift, j < n (e.g. the page of each block's last token from r2t or a selection).
int ak_kv_pick_rows(const int32_t* in, int offset, int stride, int n, int shift, int32_t* out, void* stream);

// GPU block cache (delta fetch): dst[j] = cache[slot_of[keys[j]]] when that is >= 0, else pool[keys[j]] (pool may be
// host-mapped: only the misses cross PCIe); *misses (nullable) += rows taken from the pool. slot_of covers every key.
int ak_kv_cached_gather(const void* pool, const void* cache, const int32_t* slot_of, const int32_t* keys, int n,
                        int row_bytes, void* dst, int* misses, void* stream);
// slot_of[keys[j]] = set ? j : -1 for keys[j] >= 0 (build / clear the key -> cache slot map of the previous step).
int ak_kv_slot_map(int32_t* slot_of, const int32_t* keys, int n, int set, void* stream);

#ifdef __cplusplus
}
#endif
