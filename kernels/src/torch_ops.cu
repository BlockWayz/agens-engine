// Small ops on sglv23's decode path that are plain torch tensor ops there, reproduced as torch's CUDA kernels compute
// them (opmath = fp32, precise expf / IEEE div, result rounded once to bf16), plus the two cuBLAS calls issued like
// torch: bmm (mHC res·H) and addmm (linear with bias: Engram gate).
#include "agens_kernels.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }

// op: 0 add (a + b), 1 mul (a * b), 2 a * sigmoid(b)  [torch: out * torch.sigmoid(gate), sigmoid rounded to bf16 first]
__global__ void binary_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, long n, int op) {
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
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(1.0f / (1.0f + expf(-f(x[i]))));
}

// y[t, :] = g[t] * v[t, :]  (bf16 * bf16 broadcast)
__global__ void scale_rows_kernel(const __nv_bfloat16* g, const __nv_bfloat16* v, __nv_bfloat16* y, int rows, int d) {
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

__global__ void embed_kernel(const int32_t* ids, const uint4* table, uint4* out, int d16) {
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
extern "C" int ak_argmax(const void* logits, int rows, int vocab, int vocab_valid, int* out, void* stream) {
  argmax_kernel<<<rows, 1024, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(logits), vocab,
                                                                       vocab_valid, out);
  return (int)cudaGetLastError();
}
extern "C" int ak_embed(const int32_t* ids, const void* table, void* out, int tokens, int d, void* stream) {
  embed_kernel<<<tokens, 128, 0, static_cast<cudaStream_t>(stream)>>>(ids, static_cast<const uint4*>(table),
                                                                      static_cast<uint4*>(out), d * 2 / 16);
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
