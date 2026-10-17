// Sweep of the W4 k-tiled GEMV launch shape on this GPU: rows/warp x unroll x warps/block x blocks-per-SM cap, on the
// Volundr decode shapes (weights rotated past L2). Every configuration must give the reference bits (checked); prints
// the best few per shape. Usage: tune_w4 [iters]
#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                             \
  do {                                                                                    \
    cudaError_t e_ = (cudaError_t)(x);                                                    \
    if (e_ != cudaSuccess) {                                                              \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

__global__ void fill(uint32_t* p, size_t n, uint32_t seed, uint32_t mask, uint32_t or_) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t a = (uint32_t)i * 2654435761u + seed;
    a ^= a >> 16, a *= 0x7feb352dU, a ^= a >> 15, a *= 0x846ca68bU, a ^= a >> 16;
    p[i] = (a & mask) | or_;
  }
}

struct Cfg { int r, u, w, bps; double us; };

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 100;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("# %s, %d SMs\n", prop.name, prop.multiProcessorCount);
  struct Shape { const char* name; int n, k; } shapes[] = {
      {"gate_up", 34816, 5120}, {"down", 5120, 17408}, {"kda qkv+z+fdown", 16512, 5120}, {"attn q+k+v", 14336, 5120},
      {"kda/attn out", 5120, 6144}, {"kda in_proj_z", 6144, 5120}};
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  for (const Shape& sh : shapes) {
    const int n = sh.n, k = sh.k;
    const size_t wb = ak_w4_kt_weight_bytes(n, k), sb = ak_w4_kt_scale_bytes(n, k, 128), bytes = wb + sb;
    const int copies = (int)std::max<size_t>(2, (1024ull << 20) / bytes + 1);
    std::vector<void*> WT(copies), ST(copies);
    for (int c = 0; c < copies; ++c) {
      CK(cudaMalloc(&WT[c], wb));
      CK(cudaMalloc(&ST[c], sb));
      fill<<<1024, 256>>>((uint32_t*)WT[c], wb / 4, 11 + c, 0xFFFFFFFFu, 0);
      fill<<<1024, 256>>>((uint32_t*)ST[c], sb / 4, 77 + c, 0x03FF03FFu, 0x3C003C00u);  // bf16 scales in [0.0078, 0.0156)
    }
    void *X, *Y, *Yref;
    CK(cudaMalloc(&X, (size_t)k * 2));
    fill<<<64, 256>>>((uint32_t*)X, k / 2, 5, 0xBFFFBFFFu, 0x3C003C00u);
    CK(cudaMalloc(&Y, (size_t)n * 2));
    CK(cudaMalloc(&Yref, (size_t)n * 2));
    CK(ak_w4g128_gemv_kt_variant(21, X, WT[0], ST[0], Yref, n, k, nullptr));
    std::vector<uint16_t> ref(n), got(n);
    CK(cudaMemcpy(ref.data(), Yref, n * 2, cudaMemcpyDeviceToHost));
    std::vector<Cfg> res;
    for (int r : {1, 2, 4})
      for (int u : {1, 2, 4})
        for (int w : {4, 8, 16})
          for (int bps : {0, 1, 2, 4}) {
            if (ak_w4g128_gemv_kt_tune(r, u, w, bps, X, WT[0], ST[0], Y, n, k, nullptr)) { cudaGetLastError(); continue; }
            CK(cudaMemcpy(got.data(), Y, n * 2, cudaMemcpyDeviceToHost));
            if (got != ref) { printf("  r%d u%d w%d bps%d: DIFFERENT BITS\n", r, u, w, bps); continue; }
            for (int i = 0; i < 5; ++i) ak_w4g128_gemv_kt_tune(r, u, w, bps, X, WT[i % copies], ST[i % copies], Y, n, k, nullptr);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i)
              ak_w4g128_gemv_kt_tune(r, u, w, bps, X, WT[i % copies], ST[i % copies], Y, n, k, nullptr);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            res.push_back({r, u, w, bps, ms * 1000.0 / iters});
          }
    std::sort(res.begin(), res.end(), [](const Cfg& a, const Cfg& b) { return a.us < b.us; });
    const Cfg* cur = nullptr;  // the launch shape production uses today on this arch
    printf("%-16s n=%5d k=%5d (%.1f MB):", sh.name, n, k, bytes / 1e6);
    for (int i = 0; i < 4 && i < (int)res.size(); ++i)
      printf("  r%du%dw%dbps%d %.1fus %.0fGB/s |", res[i].r, res[i].u, res[i].w, res[i].bps, res[i].us, bytes / res[i].us / 1e3);
    for (const Cfg& c : res)
      if (c.r == 2 && c.u == 1 && c.w == 8 && c.bps == 0) cur = &c;
    if (cur) printf("  [r2u1w8: %.1fus]", cur->us);
    printf("\n");
    for (int c = 0; c < copies; ++c) cudaFree(WT[c]), cudaFree(ST[c]);
    cudaFree(X), cudaFree(Y), cudaFree(Yref);
  }
  return 0;
}
