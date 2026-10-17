#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <cstring>

extern "C" {
int ak_device_count(int* n) { return (int)cudaGetDeviceCount(n); }
int ak_set_device(int dev) { return (int)cudaSetDevice(dev); }
int ak_device_name(int dev, char* buf, int len) {
  cudaDeviceProp p;
  cudaError_t e = cudaGetDeviceProperties(&p, dev);
  if (e != cudaSuccess) return (int)e;
  strncpy(buf, p.name, len - 1);
  buf[len - 1] = 0;
  return 0;
}
int ak_malloc(void** ptr, size_t bytes) { return (int)cudaMalloc(ptr, bytes); }
int ak_free(void* ptr) { return (int)cudaFree(ptr); }
int ak_memcpy_h2d(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int ak_memcpy_d2h(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int ak_synchronize(void) { return (int)cudaDeviceSynchronize(); }
int ak_mem_info(size_t* f, size_t* t) { return (int)cudaMemGetInfo(f, t); }
const char* ak_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }
// Pinned, mapped host memory: the returned pointer is valid on host AND device (UVA), for weights that do not fit.
int ak_host_alloc(void** ptr, size_t bytes) { return (int)cudaHostAlloc(ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable); }
int ak_host_free(void* ptr) { return (int)cudaFreeHost(ptr); }
}
