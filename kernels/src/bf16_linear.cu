// bf16 linear for the bit-exact reference path: y[m, n] = x[m, k] · W[n, k]^T through cuBLAS, issued exactly the
// way torch's F.linear -> mm -> at::cuda::blas::gemm<BFloat16> issues it (cublasGemmEx, OP_T/OP_N on the
// column-major view, fp32 compute, default tensor-op algo, reduced-precision reduction allowed = torch default),
// so the same cuBLAS build picks the same kernel and produces the same bits as sglv23's UnquantizedLinearMethod.
// Verified bitwise against torch by tools/test_bf16_linear.py.
#include "agens_kernels.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
struct Ctx {
  int dev = -1;
  cublasHandle_t h = nullptr;
  cublasHandle_t hp = nullptr;  // fp32 reduction (reduced-precision split-K reduction disallowed)
  void* ws = nullptr;
};
Ctx g_ctx[16];
size_t g_ws_bytes = 0;  // 0 = auto (torch default for the device)

size_t torch_default_ws(int dev) {
  // torch (c10/cuda/CUDABlasHandlePool.cpp, 2.11): sm90+ 32 MiB, else 4 MiB * 2 + 16 KiB * 8.
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  return major >= 9 ? (size_t)4096 * 8 * 1024 : (size_t)4096 * 1024 * 2 + 16 * 1024 * 8;
}

int ctx(Ctx** out) {
  int dev;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return (int)e;
  Ctx& c = g_ctx[dev & 15];
  if (c.h == nullptr) {
    if (cublasCreate(&c.h) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    const size_t ws = g_ws_bytes ? g_ws_bytes : torch_default_ws(dev);
    if ((e = cudaMalloc(&c.ws, ws)) != cudaSuccess) return (int)e;
    cublasSetWorkspace(c.h, c.ws, ws);
    cublasSetMathMode(c.h, CUBLAS_DEFAULT_MATH);
    if (cublasCreate(&c.hp) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
    cublasSetWorkspace(c.hp, c.ws, ws);  // shared: both handles are only used on the caller's stream, in order
    cublasSetMathMode(c.hp, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    c.dev = dev;
  }
  *out = &c;
  return 0;
}
}  // namespace

extern "C" int ak_bf16_set_workspace_bytes(size_t bytes) {
  g_ws_bytes = bytes;
  return 0;
}

extern "C" int ak_bf16_linear(const void* x, const void* w, void* y, int m, int n, int k, void* stream) {
  Ctx* c;
  int e = ctx(&c);
  if (e) return e;
  cublasSetStream(c->h, static_cast<cudaStream_t>(stream));
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t s = cublasGemmEx(c->h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF,
                                  k, &beta, y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}

// Same GEMM with the reduction kept in fp32 (no bf16 split-K partials): the W4 prefill path (dequantized weights in
// a scratch buffer), where there is no torch result to match and Marlin reduces in fp32.
// y [m][ldy] (columns 0..n of each row written): one column tile of a wider output (tiled W4 prefill)
extern "C" int ak_bf16_linear_f32reduce_ld(const void* x, const void* w, void* y, int m, int n, int k, int ldy,
                                           void* stream) {
  Ctx* c;
  int e = ctx(&c);
  if (e) return e;
  cublasSetStream(c->hp, static_cast<cudaStream_t>(stream));
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t s = cublasGemmEx(c->hp, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF,
                                  k, &beta, y, CUDA_R_16BF, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}

extern "C" int ak_bf16_linear_f32reduce(const void* x, const void* w, void* y, int m, int n, int k, void* stream) {
  Ctx* c;
  int e = ctx(&c);
  if (e) return e;
  cublasSetStream(c->hp, static_cast<cudaStream_t>(stream));
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t s = cublasGemmEx(c->hp, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF,
                                  k, &beta, y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}

// Linear with bias as torch's addmm issues it when cuBLASLt is not used: out = bias broadcast, then GEMM with beta = 1
// (the bias joins the fp32 accumulator, one bf16 rounding). Verified against F.linear(x, W, b) by test_torch_ops.py.
namespace {
__global__ void bcast_rows(const __nv_bfloat16* b, __nv_bfloat16* y, int m, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += gridDim.x * blockDim.x) y[i] = b[i % n];
}
}  // namespace

extern "C" int ak_bf16_linear_bias(const void* x, const void* w, const void* bias, void* y, int m, int n, int k,
                                   void* stream) {
  Ctx* c;
  int e = ctx(&c);
  if (e) return e;
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  bcast_rows<<<(m * n + 255) / 256 < 1024 ? (m * n + 255) / 256 : 1024, 256, 0, st>>>(
      static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(y), m, n);
  cublasSetStream(c->h, st);
  const float alpha = 1.f, beta = 1.f;
  cublasStatus_t s = cublasGemmEx(c->h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF,
                                  k, &beta, y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s == CUBLAS_STATUS_SUCCESS ? 0 : (int)cudaErrorUnknown;
}
