// cp.async.bulk streaming ceiling on this GPU: a producer lane per block streams CHUNK-byte pieces into a STAGES-deep
// shared ring (mbarrier complete_tx), NC consumer warps wait / release (no math). Pieces are assigned to blocks
// grid-stride (piece p -> block p % grid) or contiguously (block b takes pieces [b * P / grid, (b + 1) * P / grid)).
// Usage: bw_bulk <MB per buffer> ; prints GB/s per (chunk, stages, blocks/SM, assignment)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_(unsigned a, unsigned par) {
  unsigned done = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(a), "r"(par) : "memory");
  } while (!done);
}
__global__ void __launch_bounds__(288) bulk_kernel(const unsigned char* src, long bytes, int chunk, int stages, int contig,
                                                   unsigned* sink, int stride, int off) {
  extern __shared__ __align__(128) unsigned char sm[];
  __shared__ __align__(8) unsigned long long full[64], empty[64];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nc = blockDim.x / 32 - 1;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(su(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(&empty[i])), "r"(nc));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const long P = bytes / chunk;
  long p0, p1, ps;
  if (contig) p0 = P * blockIdx.x / gridDim.x, p1 = P * (blockIdx.x + 1) / gridDim.x, ps = 1;
  else p0 = blockIdx.x, p1 = P, ps = gridDim.x;
  if (warp == nc) {
    if (lane == 0) {
      int seq = 0;
      for (long p = p0; p < p1; p += ps, ++seq) {
        const int s = seq % stages;
        wait_(su(&empty[s]), ((seq / stages) & 1) ^ 1);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(&full[s])), "r"(chunk) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         su(sm) + off + s * stride),
                     "l"(src + p * chunk), "r"(chunk), "r"(su(&full[s]))
                     : "memory");
      }
    }
    return;
  }
  unsigned acc = 0;
  int seq = 0;
  for (long p = p0; p < p1; p += ps, ++seq) {
    const int s = seq % stages;
    wait_(su(&full[s]), (seq / stages) & 1);
    acc += *reinterpret_cast<const unsigned*>(sm + off + s * stride + lane * 4 + warp * 128);
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(&empty[s])) : "memory");
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

__global__ void fill_kernel(unsigned* p, long n, unsigned seed) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed;
    x ^= x >> 15, x *= 2246822519u, x ^= x >> 13, x *= 3266489917u, x ^= x >> 16;
    p[i] = x;
  }
}
int main(int argc, char** argv) {
  const long mb = argc > 1 ? atol(argv[1]) : 44;
  const long bytes = mb << 20;
  const int copies = std::max(2L, (long)(480L << 20) / bytes + 1);
  std::vector<unsigned char*> buf(copies);
  const bool rnd = getenv("BW_RANDOM") != nullptr;
  for (auto& b : buf) {
    cudaMalloc(&b, bytes);
    if (rnd) fill_kernel<<<1024, 256>>>(reinterpret_cast<unsigned*>(b), bytes / 4, (unsigned)(size_t)b);
    else cudaMemset(b, 1, bytes);
  }
  const int only = getenv("BW_QUICK") != nullptr;
  unsigned* sink;
  cudaMalloc(&sink, 4);
  cudaFuncSetAttribute(bulk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 225 * 1024);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaStream_t st;
  cudaStreamCreate(&st);
  printf("# %ld MB buffers x %d copies\n", mb, copies);
  for (int chunk : {1024, 2048, 4096, 8192, 16384, 32768})
    for (int ring : {32768, 65536, 131072, 196608})
      for (int bps : {1, 2})
        for (int contig : {0, 1})
          for (int nc : {1, 4, 8}) {
            const int stages = ring / chunk / bps;
            if (stages < 2 || stages > 64) continue;
            const int stride = getenv("BW_STRIDE") ? atoi(getenv("BW_STRIDE")) : chunk;
            const int off = getenv("BW_OFF") ? atoi(getenv("BW_OFF")) : 0;
            const int smem = stages * stride + off;
            if (smem * bps > 225 * 1024) continue;
            if (only && !(chunk <= 8192 && ring == 65536 && bps == 1 && nc == 4 && contig == 0)) continue;
            cudaGraph_t g;
            cudaGraphExec_t ge;
            const int iters = 100;
            cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            for (int i = 0; i < iters; ++i)
              bulk_kernel<<<sms * bps, 32 * (nc + 1), smem, st>>>(buf[i % copies], bytes, chunk, stages, contig, sink, stride, off);
            cudaStreamEndCapture(st, &g);
            if (cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) { printf("inst fail\n"); continue; }
            cudaGraphLaunch(ge, st);
            cudaStreamSynchronize(st);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0), cudaEventCreate(&e1);
            float best = 1e9;
            for (int r = 0; r < 3; ++r) {
              cudaEventRecord(e0, st);
              cudaGraphLaunch(ge, st);
              cudaEventRecord(e1, st);
              cudaEventSynchronize(e1);
              float ms;
              cudaEventElapsedTime(&ms, e0, e1);
              best = std::min(best, ms);
            }
            const double us = best * 1e3 / iters;
            printf("chunk %6d ring %6d bps %d contig %d nc %d stages %2d: %8.2f us %7.0f GB/s %s\n", chunk, ring, bps, contig,
                   nc, stages, us, bytes / us * 1e-3, cudaGetErrorString(cudaGetLastError()));
            fflush(stdout);
            cudaGraphExecDestroy(ge), cudaGraphDestroy(g);
          }
  return 0;
}
