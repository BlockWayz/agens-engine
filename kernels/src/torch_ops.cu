// Small ops on sglv23's decode path that are plain torch tensor ops there, reproduced as torch's CUDA kernels compute
// them (opmath = fp32, precise expf / IEEE div, result rounded once to bf16), plus the two cuBLAS calls issued like
// torch: bmm (mHC res·H) and addmm (linear with bias: Engram gate).
#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>

namespace {
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }

// op: 0 add (a + b), 1 mul (a * b), 2 a * sigmoid(b)  [torch: out * torch.sigmoid(gate), sigmoid rounded to bf16 first]
__global__ void binary_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, long n, int op) {
  ak_pdl_entry();
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const float x = f(a[i]), z = f(b[i]);
    float r;
    if (op == 0) r = x + z;
    else if (op == 1) r = x * z;
    else {
      const float s = __bfloat162float(__float2bfloat16(1.0f / (1.0f + expf(-z))));
      r = x * s;
    }
    y[i] = __float2bfloat16(r);
  }
}

// g [T] bf16 = bf16(sigmoid(float(x)))   (Engram: torch.sigmoid(gate(h).float()).to(bf16))
__global__ void sigmoid_f32_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, int n) {
  ak_pdl_entry();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(1.0f / (1.0f + expf(-f(x[i]))));
}

// y[t, :] = g[t] * v[t, :]  (bf16 * bf16 broadcast)
__global__ void scale_rows_kernel(const __nv_bfloat16* g, const __nv_bfloat16* v, __nv_bfloat16* y, int rows, int d) {
  ak_pdl_entry();
  const long n = (long)rows * d;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    y[i] = __float2bfloat16(f(g[i / d]) * f(v[i]));
}

// Greedy sampling: argmax over bf16 logits (as float), lowest index on ties (torch.argmax), ids >= vocab_valid masked.
__global__ void argmax_kernel(const __nv_bfloat16* logits, int vocab, int vocab_valid, int* out) {
  const __nv_bfloat16* row = logits + (size_t)blockIdx.x * vocab;
  float best = -FLT_MAX;
  int bi = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < vocab_valid; i += blockDim.x) {
    const float v = f(row[i]);
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
    if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
  }
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = blockDim.x >> 5;
    best = threadIdx.x < nw ? sv[threadIdx.x] : -FLT_MAX;
    bi = threadIdx.x < nw ? si[threadIdx.x] : 0x7FFFFFFF;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
      const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi;
  }
}

// The same argmax in two launches (rows x kAmChunks blocks, then one warp per row): each block takes the best of its
// chunk of the row (16-byte loads), the row's chunk results are combined with the same rule. The rule (larger value;
// ties: lower index; the start (-FLT_MAX, INT_MAX)) is a max over a total order, so any grouping gives argmax_kernel's
// result exactly.
constexpr int kAmChunks = 64;
__device__ __forceinline__ void am_take(float& best, int& bi, float v, int i) {
  if (v > best || (v == best && i < bi)) best = v, bi = i;
}
__device__ __forceinline__ void am_warp(float& best, int& bi) {
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
    am_take(best, bi, ov, oi);
  }
}
// indices [base, vocab_valid) of each row (base % 8 == 0; the tensor-parallel head: this rank's vocabulary share)
__global__ void __launch_bounds__(256) argmax_part_kernel(const __nv_bfloat16* logits, int vocab, int base, int vocab_valid,
                                                          float* pv, int* pi) {
  ak_pdl_entry();
  const int r = blockIdx.y, c = blockIdx.x;
  const __nv_bfloat16* row = logits + (size_t)r * vocab;
  const int per = ((vocab_valid - base + kAmChunks - 1) / kAmChunks + 7) & ~7;  // a multiple of 8: 16-byte pieces
  const int lo = base + c * per, hi = min(vocab_valid, lo + per);
  float best = -FLT_MAX;
  int bi = 0x7FFFFFFF;
  const bool vec = (vocab & 7) == 0;
  for (int i = lo + 8 * (int)threadIdx.x; i < hi; i += 8 * blockDim.x) {
    if (vec && i + 8 <= hi) {
      const uint4 u = *reinterpret_cast<const uint4*>(row + i);
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int j = 0; j < 8; ++j) am_take(best, bi, f(e[j]), i + j);
    } else {
      for (int j = i; j < min(hi, i + 8); ++j) am_take(best, bi, f(row[j]), j);
    }
  }
  am_warp(best, bi);
  __shared__ float sv[8];
  __shared__ int si[8];
  if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = best, si[threadIdx.x >> 5] = bi;
  __syncthreads();
  if (threadIdx.x < 32) {
    best = threadIdx.x < 8 ? sv[threadIdx.x] : -FLT_MAX;
    bi = threadIdx.x < 8 ? si[threadIdx.x] : 0x7FFFFFFF;
    am_warp(best, bi);
    if (threadIdx.x == 0) pv[r * kAmChunks + c] = best, pi[r * kAmChunks + c] = bi;
  }
}
__global__ void argmax_final_kernel(const float* pv, const int* pi, int* out) {
  ak_pdl_entry();
  const int r = blockIdx.x;
  float best = -FLT_MAX;
  int bi = 0x7FFFFFFF;
  for (int c = threadIdx.x; c < kAmChunks; c += 32) am_take(best, bi, pv[r * kAmChunks + c], pi[r * kAmChunks + c]);
  am_warp(best, bi);
  if (threadIdx.x == 0) out[r] = bi;
}
// tensor-parallel greedy head: this rank's best of its share, per row, into its slot of hx[row][nx] (zeros elsewhere):
// the logit as bf16 (it is one: exact) and the index as three base-64 digits (each exact in bf16); no candidate: index
// 262143 (> any vocabulary here), value -inf. The all-reduce sums the slots (x + 0 is exact), argmax_combine_kernel takes
// the best slot with the same rule: argmax_kernel's result over the whole row.
__global__ void argmax_final_enc_kernel(const float* pv, const int* pi, int rank, int nx, __nv_bfloat16* hx) {
  ak_pdl_entry();
  const int r = blockIdx.x;
  float best = -FLT_MAX;
  int bi = 0x7FFFFFFF;
  for (int c = threadIdx.x; c < kAmChunks; c += 32) am_take(best, bi, pv[r * kAmChunks + c], pi[r * kAmChunks + c]);
  am_warp(best, bi);
  const int id = bi == 0x7FFFFFFF ? 262143 : bi;
  const float val[4] = {bi == 0x7FFFFFFF ? -INFINITY : best, (float)(id & 63), (float)((id >> 6) & 63), (float)(id >> 12)};
  for (int j = threadIdx.x; j < nx; j += 32)
    hx[(size_t)r * nx + j] = j >> 2 == rank ? __float2bfloat16_rn(val[j & 3]) : __float2bfloat16_rn(0.f);
}
__global__ void argmax_combine_kernel(const __nv_bfloat16* hx, int rows, int nx, int tp, int* out) {
  ak_pdl_entry();
  const int r = threadIdx.x;
  if (r >= rows) return;
  float best = -FLT_MAX;
  int bi = 0x7FFFFFFF;
  for (int q = 0; q < tp; ++q) {
    const __nv_bfloat16* e = hx + (size_t)r * nx + 4 * q;
    const int id = (int)f(e[1]) + 64 * (int)f(e[2]) + 4096 * (int)f(e[3]);
    if (id != 262143) am_take(best, bi, f(e[0]), id);
  }
  out[r] = bi;
}

__global__ void embed_kernel(const int32_t* ids, const uint4* table, uint4* out, int d16) {
  ak_pdl_entry();
  const int t = blockIdx.x;
  const uint4* row = table + (size_t)ids[t] * d16;
  for (int i = threadIdx.x; i < d16; i += blockDim.x) out[(size_t)t * d16 + i] = row[i];
}

cublasHandle_t handle() {
  static cublasHandle_t h[16] = {};
  int dev;
  cudaGetDevice(&dev);
  if (!h[dev & 15]) {
    cublasCreate(&h[dev & 15]);
    void* ws;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    const size_t wsb = major >= 9 ? (size_t)4096 * 8 * 1024 : (size_t)4096 * 1024 * 2 + 16 * 1024 * 8;
    cudaMalloc(&ws, wsb);
    cublasSetWorkspace(h[dev & 15], ws, wsb);
  }
  return h[dev & 15];
}
}  // namespace

extern "C" int ak_bf16_binary(const void* a, const void* b, void* y, long n, int op, void* stream) {
  binary_kernel<<<(int)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(y), n, op);
  return (int)cudaGetLastError();
}
extern "C" int ak_sigmoid_to_bf16(const void* x, void* y, int n, void* stream) {
  sigmoid_f32_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y), n);
  return (int)cudaGetLastError();
}
extern "C" int ak_scale_rows(const void* g, const void* v, void* y, int rows, int d, void* stream) {
  scale_rows_kernel<<<(rows * d + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(g), static_cast<const __nv_bfloat16*>(v), static_cast<__nv_bfloat16*>(y), rows, d);
  return (int)cudaGetLastError();
}
// the chunk results' scratch of the two-launch argmax, per device, grown outside graph capture only (false: too small)
bool am_scratch(int rows, cudaStream_t st, float** v_out, int** i_out) {
  struct Scr { float* v = nullptr; int* i = nullptr; int rows = 0; };
  static Scr scr[16];
  int dev = 0;
  cudaGetDevice(&dev);
  Scr& z = scr[dev & 15];
  if (rows > z.rows) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    if (cs == cudaStreamCaptureStatusNone) {
      const int nr = std::max(rows, 128);
      float* v = nullptr;
      int* i = nullptr;
      if (cudaMalloc(&v, (size_t)nr * kAmChunks * 4) == cudaSuccess && cudaMalloc(&i, (size_t)nr * kAmChunks * 4) == cudaSuccess) {
        cudaDeviceSynchronize();
        if (z.v) cudaFree(z.v), cudaFree(z.i);
        z.v = v, z.i = i, z.rows = nr;
      } else {
        cudaGetLastError();
      }
    }
  }
  *v_out = z.v, *i_out = z.i;
  return rows <= z.rows;
}
// tensor-parallel greedy head, rank `rank` of tp: logits rows (row stride vocab) hold this rank's share
// [base, base_end) (base % 8 == 0); hx [rows][nx] (nx = 4 tp rounded up to 8) gets the rank's encoded best per row;
// the caller all-reduces hx (rows * nx elements), then ak_argmax_tp_combine -> out[rows]
extern "C" int ak_argmax_tp_part(const void* logits, int rows, int vocab, int base, int base_end, int rank, int nx,
                                 void* hx, void* stream) {
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  float* v;
  int* i;
  if ((base & 7) || base_end - base < 16 * kAmChunks || 4 * rank + 4 > nx || !am_scratch(rows, st, &v, &i))
    return (int)cudaErrorInvalidValue;
  argmax_part_kernel<<<dim3(kAmChunks, rows), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(logits), vocab, base,
                                                             base_end, v, i);
  argmax_final_enc_kernel<<<rows, 32, 0, st>>>(v, i, rank, nx, static_cast<__nv_bfloat16*>(hx));
  return (int)cudaGetLastError();
}
extern "C" int ak_argmax_tp_combine(const void* hx, int rows, int nx, int tp, int* out, void* stream) {
  if (rows > 32 || 4 * tp > nx) return (int)cudaErrorInvalidValue;
  argmax_combine_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(hx), rows, nx, tp,
                                                                         out);
  return (int)cudaGetLastError();
}
extern "C" int ak_argmax(const void* logits, int rows, int vocab, int vocab_valid, int* out, void* stream) {
  // two-launch form for long rows (AGENS_ARGMAX_SPLIT=0: off): scratch per device, grown outside graph capture only
  static const bool split = !(getenv("AGENS_ARGMAX_SPLIT") && getenv("AGENS_ARGMAX_SPLIT")[0] == '0');
  if (split && vocab_valid >= 16 * kAmChunks) {
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    float* v;
    int* i;
    if (am_scratch(rows, st, &v, &i)) {
      argmax_part_kernel<<<dim3(kAmChunks, rows), 256, 0, st>>>(static_cast<const __nv_bfloat16*>(logits), vocab, 0,
                                                                 vocab_valid, v, i);
      argmax_final_kernel<<<rows, 32, 0, st>>>(v, i, out);
      return (int)cudaGetLastError();
    }
  }
  argmax_kernel<<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(logits), vocab,
                                                                       vocab_valid, out);
  return (int)cudaGetLastError();
}
extern "C" int ak_embed(const int32_t* ids, const void* table, void* out, int tokens, int d, void* stream) {
  embed_kernel<<<tokens, 128, 0, static_cast<cudaStream_t>(stream)>>>(ids, static_cast<const uint4*>(table),
                                                                      static_cast<uint4*>(out), d * 2 / 16);
  return (int)cudaGetLastError();
}
namespace {
// ak_bf16_bmm for the mHC res . H shape (m = k = 4): the tensor-core product cuBLAS computes (one bf16 MMA over the
// four k terms in k slots 0..3, the other slots zero, fp32 accumulator from 0, one rounding), without its 32x32 WMMA
// tile kernel (sm80 kernel, ~1.4 ms per 4096-token call on sm90; this one is memory bound). Warp = 16 output columns of
// one token per mma.m16n8k16: A rows = columns c (A[c][k] = B_in[t][k][c]), B cols = output rows r (B[k][r] = A_in[t][r][k]).
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__global__ void bmm44_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C, int T, int n) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int groups = n / 16, per_tok = groups;
  const long warps = (long)gridDim.x * (blockDim.x / 32);
  const __nv_bfloat16 z = __ushort_as_bfloat16(0);
  for (long w = blockIdx.x * (long)(blockDim.x / 32) + threadIdx.x / 32; w < (long)T * per_tok; w += warps) {
    const int t = (int)(w / per_tok), c0 = (int)(w % per_tok) * 16;
    const __nv_bfloat16* b = B + (size_t)t * 4 * n;
    const __nv_bfloat16* a = A + (size_t)t * 16;
    uint32_t a0 = 0, a1 = 0, b0 = 0;
    if (t4 < 2) {
      const int k = 2 * t4;
      a0 = pack2(b[(size_t)k * n + c0 + g], b[(size_t)(k + 1) * n + c0 + g]);
      a1 = pack2(b[(size_t)k * n + c0 + g + 8], b[(size_t)(k + 1) * n + c0 + g + 8]);
      if (g < 4) b0 = pack2(a[g * 4 + k], a[g * 4 + k + 1]);
    }
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(0u), "r"(0u), "r"(b0), "r"(0u));
    if (t4 < 2) {
      __nv_bfloat16* c = C + (size_t)t * 4 * n;
      const int r = 2 * t4;
      c[(size_t)r * n + c0 + g] = __float2bfloat16_rn(d[0]);
      c[(size_t)(r + 1) * n + c0 + g] = __float2bfloat16_rn(d[1]);
      c[(size_t)r * n + c0 + g + 8] = __float2bfloat16_rn(d[2]);
      c[(size_t)(r + 1) * n + c0 + g + 8] = __float2bfloat16_rn(d[3]);
    }
    (void)z;
  }
}
}  // namespace
// C [T][4][n] = A [T][4][4] . B [T][4][n] (n % 16 == 0), bit-identical to ak_bf16_bmm (tools/test_bmm44.py)
extern "C" int ak_bf16_bmm44(const void* A, const void* B, void* C, int T, int n, void* stream) {
  if (n % 16) return (int)cudaErrorInvalidValue;
  const long warps = (long)T * (n / 16);
  const int blocks = (int)std::min<long>((warps + 7) / 8, 132 * 16);
  bmm44_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(A),
                                                                      static_cast<const __nv_bfloat16*>(B),
                                                                      static_cast<__nv_bfloat16*>(C), T, n);
  return (int)cudaGetLastError();
}
// torch.bmm(A [T][m][k], B [T][k][n]) -> C [T][m][n], all bf16 row-major contiguous, issued like at::cuda::blas::bgemm
extern "C" int ak_bf16_bmm(const void* A, const void* B, void* C, int T, int m, int k, int n, void* stream) {
  cublasHandle_t h = handle();
  cublasSetStream(h, static_cast<cudaStream_t>(stream));
  const float alpha = 1.f, beta = 0.f;
  // row-major C = A B  <=>  col-major C^T = B^T A^T
  cublasStatus_t s = cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16BF, n,
                                                (long long)k * n, A, CUDA_R_16BF, k, (long long)m * k, &beta, C,
                                                CUDA_R_16BF, n, (long long)m * n, T, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}
