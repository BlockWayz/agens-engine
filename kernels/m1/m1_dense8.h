// INT8 KV of the dense attention layer (AGENS_KV_INT8_DENSE=1): storage, tiled attention. See m1_dense8.cu.
#pragma once
#include <cuda_runtime.h>
#include <stddef.h>

namespace m1d8 {
constexpr int TILE = 8192;  // positions per attention tile (<= 8192: the bs = 1 chunk size is always 128)
bool on();
int alloc(int rows);                         // K/V int8 [rows + 1][kv heads][256] + scales, staging, merge buffers
size_t bytes();                              // device bytes allocated
int set_heads(int qh, int kvh);              // this rank's q / kv heads (tensor parallel), before alloc
size_t scratch_bytes();                      // staging + merge buffers (the memory plan's share)
int store(const void* k, const void* v, const int* loc, int T, cudaStream_t s);  // T new tokens -> rows loc[t]
// decode / verify: T <= 8 queries at positions pos[0..T) (keys up to pos[T-1] stored), ntiles = ceil((pos[T-1]+1) / TILE)
int attend_dec(void* q, const long long* pos, const int* r2t, int T, int ntiles, float scale, void* o, cudaStream_t s);
// prefill chunk: T <= 1024 queries at positions seq0 .. seq0 + T - 1 (their keys stored)
int attend_prefill(void* q, int seq0, int T, const int* r2t, float scale, void* o, cudaStream_t s);
}  // namespace m1d8
