// Read-bandwidth roofline: which access pattern gets closest to peak DRAM bandwidth on this card?
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint4 ldnc(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
// grid-stride, U independent loads per thread per step
template <int U>
__global__ void k_stride(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  uint32_t acc = 0;
  const size_t T = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += T * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = (i + u * T < n16) ? ldnc(p + i + u * T) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
// each block owns a contiguous segment (like a warp/row-owning GEMV), U loads in flight per thread
template <int U>
__global__ void k_segment(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  uint32_t acc = 0;
  const size_t seg = (n16 + gridDim.x - 1) / gridDim.x;
  const size_t b = blockIdx.x * seg, e = min(n16, b + seg);
  for (size_t i = b + threadIdx.x; i < e; i += (size_t)blockDim.x * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = (i + u * blockDim.x < e) ? ldnc(p + i + u * blockDim.x) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}

template <typename K>
void run(const char* name, K kern, int blocks, int threads, const uint4* buf, size_t bytes, uint32_t* sink) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  kern<<<blocks, threads>>>(buf, bytes / 16, sink);
  cudaEventRecord(e0);
  for (int i = 0; i < 10; ++i) kern<<<blocks, threads>>>(buf, bytes / 16, sink);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("%-14s blocks %5d x %3d : %6.1f GB/s\n", name, blocks, threads, bytes * 10.0 / (ms * 1e6));
}

int main() {
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  const int sm = prop.multiProcessorCount;
  printf("# %s, mem clock %d kHz, bus %d bit -> theoretical %.0f GB/s\n", prop.name, prop.memoryClockRate,
         prop.memoryBusWidth, 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8 / 1e9);
  const size_t bytes = 1ull << 30;
  uint4* buf; uint32_t* sink;
  cudaMalloc(&buf, bytes); cudaMalloc(&sink, 4); cudaMemset(buf, 1, bytes);
  for (int m : {4, 8, 16}) {
    run("stride U1", k_stride<1>, sm * m, 256, buf, bytes, sink);
    run("stride U4", k_stride<4>, sm * m, 256, buf, bytes, sink);
    run("segment U1", k_segment<1>, sm * m, 256, buf, bytes, sink);
    run("segment U4", k_segment<4>, sm * m, 256, buf, bytes, sink);
    run("segment U8", k_segment<8>, sm * m, 256, buf, bytes, sink);
  }
  run("segment U4", k_segment<4>, sm * 64, 256, buf, bytes, sink);
  run("segment U4", k_segment<4>, sm * 6, 1024, buf, bytes, sink);
  // cudaMemcpy D2D as a reference point (reads + writes)
  void* dst; cudaMalloc(&dst, bytes / 2);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < 10; ++i) cudaMemcpy(dst, buf, bytes / 2, cudaMemcpyDeviceToDevice);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("memcpy D2D     read+write      : %6.1f GB/s\n", bytes * 10.0 / (ms * 1e6));
  return 0;
}
