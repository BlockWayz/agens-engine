// Bandwidth probe for the W4 KT layout at 8 tokens: where does the lean kernel lose bandwidth?
// mode 0: stream the weights only (lean access pattern: warp = group, lane 16 B, rows g / g + 8), XOR-sink
// mode 1: + dequant + MMA with constant x (no x loads)
// mode 2: + x loads from global as the lean kernel (__ldg, 4 x 16 B per lane per tile)
// mode 3: mode 1 with "row-major per warp" assignment (warp takes a row block across all 8 groups of a tile)
// U: (row blocks per warp item) loads in flight; persistent grid = SMs x occupancy
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t deq(uint32_t w, int j) {
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}

template <int D>
__global__ void __launch_bounds__(256) probe_kf(const uint4* __restrict__ w, int n, int k, float* out) {
  // group-major KF stream: warp-global id walks row blocks; per row block all groups, D groups of loads in flight
  const int lane = threadIdx.x & 31, wg = blockIdx.x * 8 + (threadIdx.x >> 5), nw = gridDim.x * 8;
  const int nrb = n >> 4, ng = k >> 7;
  uint32_t sink = 0;
  for (int rb = wg; rb < nrb; rb += nw) {
    for (int G = 0; G < ng; G += D) {
      uint4 a[D], b[D];
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const uint4* p = w + ((size_t)min(G + d, ng - 1) * nrb + rb) * 64 + lane;
        a[d] = ld_stream(p), b[d] = ld_stream(p + 32);
      }
#pragma unroll
      for (int d = 0; d < D; ++d) sink ^= a[d].x ^ a[d].w ^ b[d].y ^ b[d].z;
    }
  }
  if (sink == 0x12345678u) out[0] = 1.f;
}
template <int D>
void run_kf(std::vector<uint4*>& W, int n, int k, float* out, int sms) {
  int per = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, probe_kf<D>, 256, 0);
  const int blocks = sms * per;
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) probe_kf<D><<<blocks, 256>>>(W[i % W.size()], n, k, out);
  cudaEventRecord(a);
  for (int i = 0; i < 100; ++i) probe_kf<D><<<blocks, 256>>>(W[i % W.size()], n, k, out);
  cudaEventRecord(b);
  CK(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  const double us = ms * 1e3 / 100, bytes = (double)n * k / 2;
  printf("%-28s n %6d k %6d D %d occ %d: %7.1f us %6.2f TB/s\n", "KF group-major stream", n, k, D, per, us, bytes / us / 1e6);
}
template <int MODE, int RB>
__global__ void __launch_bounds__(256) probe(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x, int n, int k,
                                             float* out) {
  const int ntile = k >> 10, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = n >> 4, nitem = (nblk + RB - 1) / RB;
  uint32_t sink = 0;
  float acc[RB][4] = {};
  const uint4* xr0 = reinterpret_cast<const uint4*>(x + (size_t)g * k + (size_t)(4 * warp + t4) * 32);
  for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
    for (int t = 0; t < ntile; ++t) {
      uint4 w0[RB], w1[RB];
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const int b = min(it * RB + r, nblk - 1);
        const uint4* p = MODE == 6 ? w + ((size_t)t * n + b * 16 + (lane >> 5) + 2 * warp) * 32 + lane  // 512 B per warp
                       : MODE == 3 ? w + ((size_t)t * n + b * 16 + g) * 32 + 4 * (lane & 7)  // any 8 groups: same bytes
                                   : w + ((size_t)t * n + b * 16 + g) * 32 + 4 * warp + t4;
        w0[r] = ld_stream(p);
        w1[r] = ld_stream(p + 8 * 32);
      }
      if (MODE == 0 || MODE == 6) {
#pragma unroll
        for (int r = 0; r < RB; ++r) sink ^= w0[r].x ^ w0[r].y ^ w0[r].z ^ w0[r].w ^ w1[r].x ^ w1[r].y ^ w1[r].z ^ w1[r].w;
        acc[0][0] += (float)(sink & 1);
        continue;
      }
      if (MODE == 5) {  // dequant only
#pragma unroll
        for (int r = 0; r < RB; ++r) {
          const uint32_t a[4] = {w0[r].x, w0[r].y, w0[r].z, w0[r].w}, b[4] = {w1[r].x, w1[r].y, w1[r].z, w1[r].w};
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
            sink ^= deq(a[wd], j0) ^ deq(b[wd], j0) ^ deq(a[wd], j1) ^ deq(b[wd], j1);
          }
        }
        acc[0][0] += (float)(sink & 1);
        continue;
      }
      uint32_t xb[16];
      if (MODE == 2) {
        uint4 xv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = __ldg(xr0 + (size_t)t * 128 + q);
        const uint32_t xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                                 xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
#pragma unroll
        for (int q = 0; q < 16; ++q) xb[q] = xw[q];
      } else {
#pragma unroll
        for (int q = 0; q < 16; ++q) xb[q] = 0x3f803f80u + q;
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const uint32_t a[4] = {w0[r].x, w0[r].y, w0[r].z, w0[r].w}, b[4] = {w1[r].x, w1[r].y, w1[r].z, w1[r].w};
        float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int wd = q >> 1, j0 = 2 * (q & 1), j1 = j0 + 1;
          if (MODE == 4) mma16816(cf, a[wd] >> j0, b[wd] >> j0, a[wd] >> j1, b[wd] >> j1, xb[2 * q], xb[2 * q + 1]);
          else mma16816(cf, deq(a[wd], j0), deq(b[wd], j0), deq(a[wd], j1), deq(b[wd], j1), xb[2 * q], xb[2 * q + 1]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = fmaf(cf[j], 0.01f, acc[r][j]);
      }
    }
  }
  float s = (float)sink;
#pragma unroll
  for (int r = 0; r < RB; ++r) s += acc[r][0] + acc[r][1] + acc[r][2] + acc[r][3];
  if (s == 12345.678f) out[0] = s;
}

template <int MODE, int RB>
void run(const char* name, std::vector<uint4*>& W, const __nv_bfloat16* X, int n, int k, float* out, int sms) {
  int per = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, probe<MODE, RB>, 256, 0);
  const int want = (n / 16 + RB - 1) / RB, blocks = std::min(want, sms * per);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) probe<MODE, RB><<<blocks, 256>>>(W[i % W.size()], X, n, k, out);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  cudaEventRecord(a);
  const int it = 100;
  for (int i = 0; i < it; ++i) probe<MODE, RB><<<blocks, 256>>>(W[i % W.size()], X, n, k, out);
  cudaEventRecord(b);
  CK(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  const double us = ms * 1e3 / it, bytes = (double)n * k / 2;
  printf("%-28s n %6d k %6d RB %d occ %d: %7.1f us %6.2f TB/s\n", name, n, k, RB, per, us, bytes / us / 1e6);
}

int main() {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  float* out;
  CK(cudaMalloc(&out, 4));
  const int shapes[4][2] = {{34816, 5120}, {5120, 17408}, {10240, 5120}, {2048, 5120}};
  __nv_bfloat16* X;
  CK(cudaMalloc(&X, (size_t)8 * 17408 * 2));
  cudaMemset(X, 0, (size_t)8 * 17408 * 2);
  for (auto& sh : shapes) {
    const int n = sh[0], k = sh[1];
    std::vector<uint4*> W(n == 2048 ? 1 : 4);  // n = 2048: one 5 MB copy, L2-resident (compute bound)
    for (auto& p : W) { CK(cudaMalloc(&p, (size_t)n * k / 2)); cudaMemset(p, 0x11, (size_t)n * k / 2); }
    run_kf<1>(W, n, k, out, sms);
    run_kf<2>(W, n, k, out, sms);
    run_kf<4>(W, n, k, out, sms);
    run<6, 1>("stream contiguous 512B", W, X, n, k, out, sms);
    run<6, 2>("stream contiguous 512B", W, X, n, k, out, sms);
    run<0, 1>("stream only", W, X, n, k, out, sms);
    run<0, 2>("stream only", W, X, n, k, out, sms);
    run<0, 4>("stream only", W, X, n, k, out, sms);
    run<1, 1>("deq+mma, const x", W, X, n, k, out, sms);
    run<1, 2>("deq+mma, const x", W, X, n, k, out, sms);
    run<1, 4>("deq+mma, const x", W, X, n, k, out, sms);
    run<4, 1>("mma raw, const x", W, X, n, k, out, sms);
    run<4, 2>("mma raw, const x", W, X, n, k, out, sms);
    run<5, 1>("deq only", W, X, n, k, out, sms);
    run<5, 2>("deq only", W, X, n, k, out, sms);
    run<2, 1>("deq+mma, x ldg", W, X, n, k, out, sms);
    run<2, 2>("deq+mma, x ldg", W, X, n, k, out, sms);
    run<2, 4>("deq+mma, x ldg", W, X, n, k, out, sms);
    for (auto p : W) cudaFree(p);
  }
  return 0;
}
