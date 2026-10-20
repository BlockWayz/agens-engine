// Compute ceiling of the W4 g128 tensor-core group step (no memory traffic): per group, the KT path's 32 deq pairs
// (lop3 + bf16x2 sub), 8 m16n8k16 MMAs (one 8-token tile) and 4 scaled adds, from registers. Reports groups per SM
// cycle and the equivalent weight bandwidth (1 KB per group) for W warps per SM and U independent groups in flight
// per warp. Usage: w4_alu
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>

__device__ __forceinline__ uint32_t deq_pair(uint32_t w, int j) {
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
// variant: the right shift as mul.hi (IMAD.HI: FMA pipe instead of the integer ALU pipe)
__device__ __forceinline__ uint32_t deq_pair_mh(uint32_t w, int j) {
  uint32_t sh = w, v;
  if (j) asm("mul.hi.u32 %0, %1, %2;" : "=r"(sh) : "r"(w), "r"(1u << (32 - 4 * j)));
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(sh), "r"(0x000F000Fu), "r"(0x43004300u));
  const __nv_bfloat162 r = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&v), __floats2bfloat162_rn(136.f, 136.f));
  return *reinterpret_cast<const uint32_t*>(&r);
}
// variant 2: no subtract (A = 128 + n; the bias 136 * sum(x) would be removed per group in fp32)
__device__ __forceinline__ uint32_t deq_pair_nb(uint32_t w, int j) {
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(v) : "r"(w >> (4 * j)), "r"(0x000F000Fu), "r"(0x43004300u));
  return v;
}
template <int V>
__device__ __forceinline__ uint32_t deq(uint32_t w, int j) {
  return V == 2 ? deq_pair_nb(w, j) : V ? deq_pair_mh(w, j) : deq_pair(w, j);
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int U, int NT, int V>
__global__ void alu_kernel(int iters, float* out, long long* clk) {
  uint32_t w[U][8], xb[NT][16];
  for (int u = 0; u < U; ++u)
    for (int q = 0; q < 8; ++q) w[u][q] = 0x89ABCDEFu * (threadIdx.x + q + u);
  for (int t = 0; t < NT; ++t)
    for (int q = 0; q < 16; ++q) xb[t][q] = 0x3F803F80u ^ (q + t);
  float acc[U][NT][4] = {};
  const long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float cf[NT][4] = {};
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int wd = q >> 1, j0 = 2 * (q & 1);
        const uint32_t a0 = deq<V>(w[u][wd], j0), a1 = deq<V>(w[u][4 + wd], j0), a2 = deq<V>(w[u][wd], j0 + 1),
                       a3 = deq<V>(w[u][4 + wd], j0 + 1);
#pragma unroll
        for (int t = 0; t < NT; ++t) mma16816(cf[t], a0, a1, a2, a3, xb[t][2 * q], xb[t][2 * q + 1]);
      }
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[u][t][e] = fmaf(cf[t][e], 0.5f, acc[u][t][e]);
#pragma unroll
      for (int q = 0; q < 8; ++q) w[u][q] ^= (uint32_t)it;  // new weights every iteration
    }
  }
  const long long c1 = clock64();
  float s = 0;
  for (int u = 0; u < U; ++u)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 4; ++e) s += acc[u][t][e];
  if (s == 1234.5f) out[threadIdx.x] = s;
  if (threadIdx.x == 0) clk[blockIdx.x] = c1 - c0;
}
// F = 1: per group the operands come from shared memory (2 x LDS.128 + LDS.32, rotating over 16 KB) then __syncwarp;
// F = 2: also an mbarrier try_wait (on a completed phase) + arrive per group
template <int F>
__global__ void ldsalu_kernel(int iters, float* out, long long* clk) {
  __shared__ __align__(16) uint4 buf[1024];
  __shared__ __align__(8) unsigned long long bar[2];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) buf[i] = make_uint4(i * 77u, i * 91u, i * 13u, i * 5u);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(&bar[0])));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((unsigned)__cvta_generic_to_shared(&bar[0])) : "memory");
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  uint32_t xb[16];
  for (int q = 0; q < 16; ++q) xb[q] = 0x3F803F80u ^ q;
  float acc[4] = {};
  const long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
    const int off = ((it * 8 + (threadIdx.x >> 5)) & 15) * 64;
    if (F == 2) {
      unsigned done = 0, a = (unsigned)__cvta_generic_to_shared(&bar[0]);
      do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(a), "r"(0u) : "memory");
      } while (!done);
    }
    uint4 w0, w1;
    uint32_t sc;
    {
      const unsigned b = (unsigned)__cvta_generic_to_shared(&buf[off + lane]);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w0.x), "=r"(w0.y), "=r"(w0.z), "=r"(w0.w) : "r"(b));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w1.x), "=r"(w1.y), "=r"(w1.z), "=r"(w1.w) : "r"(b + 512));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sc) : "r"(b));
    }
    __syncwarp();
    const uint32_t ww0[4] = {w0.x, w0.y, w0.z, w0.w}, ww1[4] = {w1.x, w1.y, w1.z, w1.w};
    float cf[4] = {};
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int wd = q >> 1, j0 = 2 * (q & 1);
      mma16816(cf, deq_pair(ww0[wd], j0), deq_pair(ww1[wd], j0), deq_pair(ww0[wd], j0 + 1), deq_pair(ww1[wd], j0 + 1),
               xb[2 * q], xb[2 * q + 1]);
    }
    const float s = __uint_as_float(sc & 0x3F80FFFFu);
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = fmaf(cf[e], s, acc[e]);
  }
  const long long c1 = clock64();
  if (acc[0] + acc[1] + acc[2] + acc[3] == 1234.5f) out[threadIdx.x] = acc[0];
  if (threadIdx.x == 0) clk[blockIdx.x] = c1 - c0;
}
template <int F>
void run_lds(int warps) {
  float* out;
  long long* clk;
  cudaMalloc(&out, 4096 * 4);
  cudaMalloc(&clk, 132 * 8);
  const int iters = 4000;
  ldsalu_kernel<F><<<132, warps * 32>>>(iters, out, clk);
  cudaDeviceSynchronize();
  long long c;
  ldsalu_kernel<F><<<132, warps * 32>>>(iters, out, clk);
  cudaMemcpy(&c, clk, 8, cudaMemcpyDeviceToHost);
  printf("F %d warps %2d: %.1f clk/group/SM %s\n", F, warps, c / ((double)iters * warps), cudaGetErrorString(cudaGetLastError()));
}
template <int U, int NT, int V = 0>
void run(int warps) {
  float* out;
  long long* clk;
  cudaMalloc(&out, 4096 * 4);
  cudaMalloc(&clk, 132 * 8);
  const int iters = 4000;
  alu_kernel<U, NT, V><<<132, warps * 32>>>(iters, out, clk);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  cudaEventRecord(e0);
  alu_kernel<U, NT, V><<<132, warps * 32>>>(iters, out, clk);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  long long c;
  cudaMemcpy(&c, clk, 8, cudaMemcpyDeviceToHost);
  const double groups = (double)iters * U * warps;  // per SM
  printf("V %d warps %2d U %d NT %d: %.1f clk/group/SM, %.0f GB/s equivalent (%.3f ms) %s\n", V, warps, U, NT, c / groups,
         groups * 132 * 1024 / (ms * 1e-3) * 1e-9, ms, cudaGetErrorString(cudaGetLastError()));
}
int main() {
  for (int w : {8, 16}) run_lds<1>(w), run_lds<2>(w);
  run<1, 1, 0>(8);
  for (int w : {8, 16}) {
    run<2, 1, 0>(w);
    run<2, 1, 2>(w);
    run<2, 2, 0>(w);
    run<2, 2, 2>(w);
  }
  return 0;
}
