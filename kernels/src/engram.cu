// Engram decode lookup with HOST-resident tables (zero-copy over PCIe/UVA): per request, hash the 2-gram and
// 3-gram ending at the current token (sglv23 VolundrModel._engram_row_indices, decode branch, bit-identical
// int64 arithmetic), gather the two 512-wide bf16 rows into e [B][2*dim] (= torch.cat of the two embeddings),
// and rotate the per-request prev-token pair. Tables may live in pinned host memory (cudaHostRegister /
// cudaHostAlloc, mapped) or in device memory: the kernel only dereferences the pointer.
// One block per request: thread 0 hashes, then 16-byte copies of both rows (128 threads -> 2 KB).
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__constant__ unsigned long long kPrimes[5] = {1000003ull, 1000033ull, 1000037ull, 1000039ull, 1000081ull};
constexpr unsigned long long kMask61 = 0x1FFFFFFFFFFFFFFFull;

__device__ __forceinline__ unsigned long long fold(const long long* ids, int n) {
  unsigned long long code = 0;
  for (int j = 0; j < n; ++j) code = (code * kPrimes[j % 5] + (unsigned long long)(ids[j] + 1)) & kMask61;
  return code;
}

__global__ void engram_kernel(const int32_t* __restrict__ cur, int32_t* __restrict__ prev, int prev_rows,
                              const int32_t* __restrict__ slots, const uint4* __restrict__ t2,
                              const uint4* __restrict__ t3, long long rows, uint4* __restrict__ e, int dim,
                              long long* __restrict__ rows_out) {
  __shared__ long long r2s, r3s;
  const int bi = blockIdx.x;
  int slot = slots[bi];
  if (slot < 0) slot = prev_rows - 1;  // trash row, as sglv23's -1 wrap
  if (threadIdx.x == 0) {
    const long long p2 = prev[2 * slot], p1 = prev[2 * slot + 1], c = cur[bi];
    const long long g2[2] = {p1, c}, g3[3] = {p2, p1, c};
    r2s = (long long)(fold(g2, 2) % (unsigned long long)rows);
    r3s = (long long)(fold(g3, 3) % (unsigned long long)rows);
    if (rows_out) {
      rows_out[2 * bi] = r2s;
      rows_out[2 * bi + 1] = r3s;
    }
  }
  __syncthreads();
  const int per_row = dim * 2 / 16;  // 16-byte vectors per table row (dim=512 -> 64)
  uint4* out = e + (size_t)bi * 2 * per_row;
  for (int i = threadIdx.x; i < 2 * per_row; i += blockDim.x) {
    out[i] = i < per_row ? t2[(size_t)r2s * per_row + i] : t3[(size_t)r3s * per_row + (i - per_row)];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    prev[2 * slot] = prev[2 * slot + 1];
    prev[2 * slot + 1] = cur[bi];
  }
}
}  // namespace

extern "C" int ak_engram_decode(const int32_t* cur, int32_t* prev, int prev_rows, const int32_t* slots,
                                const void* table2, const void* table3, long long rows, void* e, int dim,
                                long long* rows_out, int batch, void* stream) {
  if ((dim * 2) % 16 != 0) return (int)cudaErrorInvalidValue;
  engram_kernel<<<batch, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      cur, prev, prev_rows, slots, static_cast<const uint4*>(table2), static_cast<const uint4*>(table3), rows,
      static_cast<uint4*>(e), dim, rows_out);
  return (int)cudaGetLastError();
}

// Pin an existing host range (e.g. an mmap'd copy of the tables) for zero-copy device access.
extern "C" int ak_host_register(void* ptr, size_t bytes) {
  return (int)cudaHostRegister(ptr, bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly);
}
extern "C" int ak_host_unregister(void* ptr) { return (int)cudaHostUnregister(ptr); }
