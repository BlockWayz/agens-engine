#include <cstdio>
#include <cstdint>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  uint32_t s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    s ^= v.x ^ v.w;
  }
  if (s == 0x1234567u) out[0] = s;
}
int main() {
  const size_t bytes = 89ull << 20, n = bytes / 16;
  uint4* b[4];
  for (auto& x : b) cudaMalloc(&x, bytes), cudaMemset(x, 1, bytes);
  unsigned* o; cudaMalloc(&o, 4);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int bpm : {1, 2, 4, 8}) for (int th : {256, 512, 1024}) {
    for (int i = 0; i < 10; ++i) rd<<<sms * bpm, th>>>(b[i % 4], n, o);
    cudaEventRecord(e0);
    for (int i = 0; i < 100; ++i) rd<<<sms * bpm, th>>>(b[i % 4], n, o);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("blocks/SM %d threads %4d: %.2f TB/s\n", bpm, th, bytes / (ms / 100 * 1e-3) / 1e12);
  }
  cudaEventRecord(e0);
  for (int i = 0; i < 100; ++i) cudaMemcpyAsync(b[(i + 1) % 4], b[i % 4], bytes, cudaMemcpyDeviceToDevice);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("memcpy D2D: %.2f TB/s (read + write)\n", 2 * bytes / (ms / 100 * 1e-3) / 1e12);
}
