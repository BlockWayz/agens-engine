// ORACLE TREE: FlashInfer's own norm templates (third_party/flashinfer), instantiated as sgl_kernel does, so the
// engine's tests can compare its own kernels against the exact code sglv23 runs. Not used by the engine itself.
#include <cuda_bf16.h>
#include <flashinfer/norm.cuh>

extern "C" int ak_oracle_rmsnorm(void* x, void* w, void* y, int batch, int d, int stride_in, int stride_out, float eps,
                                 int gemma, void* stream) {
  auto* X = static_cast<__nv_bfloat16*>(x);
  auto* W = static_cast<__nv_bfloat16*>(w);
  auto* Y = static_cast<__nv_bfloat16*>(y);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t e = gemma ? flashinfer::norm::GemmaRMSNorm(X, W, Y, batch, d, stride_in, stride_out, eps, false, s)
                        : flashinfer::norm::RMSNorm(X, W, Y, batch, d, stride_in, stride_out, eps, false, s);
  return (int)e;
}
