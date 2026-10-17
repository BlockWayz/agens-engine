// Batch-invariant small-m kernels for speculative decoding: a verify of m tokens must give every token the bits a
// one-token decode step gives it, so decode and verify run these same kernels (cuBLAS picks different kernels, and
// reduction orders, for different m).
//  * ak_bf16_gemv_bi: y[m][n] = x[m][k] . W[n][k]^T (+ bias), bf16 weights row-major, 1 <= m <= 16, on tensor cores
//    (mma.m16n8k16, fp32 accumulate). A block owns 16 rows; its warps take interleaved 32-input k-steps and their
//    partial sums are added in warp order. MMA columns are independent, so row i of the result does not depend on m.
//    Thread (g = lane/4, t4 = lane%4) holds, per k-step, inputs [32s + 8 t4, +8) of rows g and g+8 (one 16-byte load
//    per row) and of token g; MMA j < 2 of the step takes inputs 8 t4 + 4j + {0,1} (k slots 2 t4, +1) and {2,3}
//    (slots 2 t4 + 8, +9) -- any input-to-slot map works as long as A and B use the same one.
//  * ak_bf16_bmm_bi: per-token C[t] = A[t] [mr x kk] . B[t] [kk x nc], fp32 sequential sum over kk, one rounding
//    (mHC res . H: kk = 4).
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <map>

namespace {
using bf16 = __nv_bfloat16;
constexpr int kWarps = 8;

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// m <= 8 per launch (MMA width); x [m][k], y [m][n] (row stride n)
template <int kU>
__global__ void __launch_bounds__(kWarps * 32)
bf16_gemv_bi_kernel(const bf16* __restrict__ x, int m, const bf16* __restrict__ w, const bf16* __restrict__ bias,
                    bf16* __restrict__ y, int n, int k) {
  __shared__ float red[kWarps][32][4];
  const int steps = k >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = (n + 15) >> 4;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int r0 = (rb << 4) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // the loads of kU steps are issued together, then their MMAs in step order (the same chain: bit-identical for any
    // kU); small n (mHC dyn n = 24, in_proj_b, indexer projections) leaves few blocks, so the step loop is latency-bound
    for (int s0 = warp; s0 < steps; s0 += kWarps * kU) {
      uint4 w0[kU], w1[kU], xv[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int s = s0 + u * kWarps, c = 32 * s + 8 * t4;
        const bool in = s < steps;
        w0[u] = in && ok0 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r0 * k + c)) : make_uint4(0, 0, 0, 0);
        w1[u] = in && ok1 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r1 * k + c)) : make_uint4(0, 0, 0, 0);
        xv[u] = in && g < m ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)g * k + c)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        if (s0 + u * kWarps >= steps) break;
        mma16816(acc, w0[u].x, w1[u].x, w0[u].y, w1[u].y, xv[u].x, xv[u].y);  // inputs c + 0..3
        mma16816(acc, w0[u].z, w1[u].z, w0[u].w, w1[u].w, xv[u].z, xv[u].w);  // inputs c + 4..7
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp][lane][j] = acc[j];
    __syncthreads();
    if (warp == 0) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int ww = 0; ww < kWarps; ++ww)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[ww][lane][j];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * n + r0] = __float2bfloat16_rn(bias ? v[cc] + __bfloat162float(bias[r0]) : v[cc]);
          if (ok1) y[(size_t)col * n + r1] = __float2bfloat16_rn(bias ? v[2 + cc] + __bfloat162float(bias[r1]) : v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}

// m > 8 (batched speculative verify): NT tiles of 8 tokens per weight pass; each token's MMA chain over the k-steps,
// the warp-order sum and the bias add are exactly bf16_gemv_bi_kernel's (MMA columns are independent), so every row is
// bit-identical to its <= 8-token launch.
template <int NT>
__global__ void __launch_bounds__(kWarps * 32)
bf16_gemv_bi_nt_kernel(const bf16* __restrict__ x, int m, const bf16* __restrict__ w, const bf16* __restrict__ bias,
                       bf16* __restrict__ y, int n, int k) {
  __shared__ float red[kWarps][32][4 * NT];
  const int steps = k >> 5;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = (n + 15) >> 4;
  for (int rb = blockIdx.x; rb < nblk; rb += gridDim.x) {
    const int r0 = (rb << 4) + g, r1 = r0 + 8;
    const bool ok0 = r0 < n, ok1 = r1 < n;
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    constexpr int kU = 2;
    for (int s0 = warp; s0 < steps; s0 += kWarps * kU) {
      uint4 w0[kU], w1[kU];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int s = s0 + u * kWarps, c = 32 * s + 8 * t4;
        const bool in = s < steps;
        w0[u] = in && ok0 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r0 * k + c)) : make_uint4(0, 0, 0, 0);
        w1[u] = in && ok1 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r1 * k + c)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int s = s0 + u * kWarps, c = 32 * s + 8 * t4;
        if (s >= steps) break;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
          const int tok = tt * 8 + g;
          const uint4 xv = tok < m ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)tok * k + c)) : make_uint4(0, 0, 0, 0);
          mma16816(acc[tt], w0[u].x, w1[u].x, w0[u].y, w1[u].y, xv.x, xv.y);
          mma16816(acc[tt], w0[u].z, w1[u].z, w0[u].w, w1[u].w, xv.z, xv.w);
        }
      }
    }
#pragma unroll
    for (int tt = 0; tt < NT; ++tt)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[warp][lane][4 * tt + j] = acc[tt][j];
    __syncthreads();
    for (int tt = warp; tt < NT; tt += kWarps) {
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int ww = 0; ww < kWarps; ++ww)
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[ww][lane][4 * tt + j];
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int col = tt * 8 + 2 * t4 + cc;
        if (col < m) {
          if (ok0) y[(size_t)col * n + r0] = __float2bfloat16_rn(bias ? v[cc] + __bfloat162float(bias[r0]) : v[cc]);
          if (ok1) y[(size_t)col * n + r1] = __float2bfloat16_rn(bias ? v[2 + cc] + __bfloat162float(bias[r1]) : v[2 + cc]);
        }
      }
    }
    __syncthreads();
  }
}
// Small n, m > 8: block = (16-row block, k-step slot w of the 8, 32 tokens), warp v = token tile v; each warp runs
// slot w's chain (steps w, w + 8, ..; the 2 MMAs per step of bf16_gemv_bi_kernel) and stores it; the last of the 8
// slot blocks adds them in slot order (from 0) and the bias: bit-identical to the <= 8-token launches.
__global__ void __launch_bounds__(128)
bf16_bi_split_kernel(const bf16* __restrict__ x, int m, const bf16* __restrict__ w, const bf16* __restrict__ bias,
                     bf16* __restrict__ y, int n, int k, float* __restrict__ part, unsigned* __restrict__ cnt) {
  __shared__ int last_s;
  const int steps = k >> 5;
  const int lane = threadIdx.x & 31, tv = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nb32 = (m + 31) / 32, slot = blockIdx.x & 7, rest = blockIdx.x >> 3, tb = rest % nb32, rb = rest / nb32;
  const int r0 = (rb << 4) + g, r1 = r0 + 8;
  const bool ok0 = r0 < n, ok1 = r1 < n;
  const int tt = tb * 4 + tv, tok = tt * 8 + g;  // this warp's token tile and lane's token
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  constexpr int kU = 4;
  for (int s0 = slot; s0 < steps; s0 += 8 * kU) {
    uint4 w0[kU], w1[kU], xv[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int st = s0 + u * 8, c = 32 * st + 8 * t4;
      const bool in = st < steps;
      w0[u] = in && ok0 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r0 * k + c)) : make_uint4(0, 0, 0, 0);
      w1[u] = in && ok1 ? ld_nc(reinterpret_cast<const uint4*>(w + (size_t)r1 * k + c)) : make_uint4(0, 0, 0, 0);
      xv[u] = in && tok < m ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)tok * k + c)) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (s0 + u * 8 >= steps) break;
      mma16816(acc, w0[u].x, w1[u].x, w0[u].y, w1[u].y, xv[u].x, xv[u].y);
      mma16816(acc, w0[u].z, w1[u].z, w0[u].w, w1[u].w, xv[u].z, xv[u].w);
    }
  }
  const int ntt = nb32 * 4;
  reinterpret_cast<float4*>(part)[(((size_t)rb * 8 + slot) * ntt + tt) * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_s = (atomicAdd(&cnt[rb * nb32 + tb], 1u) & 7u) == 7u;
  __syncthreads();
  if (!last_s) return;
  __threadfence();
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sl = 0; sl < 8; ++sl) {
    const float4 q = __ldcg(reinterpret_cast<const float4*>(part) + (((size_t)rb * 8 + sl) * ntt + tt) * 32 + lane);
    v[0] += q.x, v[1] += q.y, v[2] += q.z, v[3] += q.w;
  }
#pragma unroll
  for (int cc = 0; cc < 2; ++cc) {
    const int col = tt * 8 + 2 * t4 + cc;
    if (col < m) {
      if (ok0) y[(size_t)col * n + r0] = __float2bfloat16_rn(bias ? v[cc] + __bfloat162float(bias[r0]) : v[cc]);
      if (ok1) y[(size_t)col * n + r1] = __float2bfloat16_rn(bias ? v[2 + cc] + __bfloat162float(bias[r1]) : v[2 + cc]);
    }
  }
}
// m > 8, large n (lm_head): block = 4 row blocks x 32 tokens, 8 warps = the 8 k-step slots; 256-input k-tiles of the
// weights (64 rows) and of x staged through a STAGES-deep cp.async ring; warp w runs step 8 t + w of tile t for all
// row blocks and token tiles (bf16_gemv_bi_kernel's 2 MMAs per step into its chain), then the slot-order sum + bias:
// bit-identical to the <= 8-token launches. 16-byte pieces XOR-swizzled by row parity (conflict-free reads).
__device__ __forceinline__ void cp16(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src),
               "r"(bytes) : "memory");
}
template <int STAGES>
__global__ void __launch_bounds__(256, 1)
bf16_bi_gemm_kernel(const bf16* __restrict__ x, int m, const bf16* __restrict__ w, const bf16* __restrict__ bias,
                    bf16* __restrict__ y, int n, int k) {
  constexpr int RB = 4, NTT = 4, TOK = 32, WB = RB * 16 * 512, XB = TOK * 512, STG = WB + XB;
  extern __shared__ __align__(16) unsigned char bsm[];
  const int steps = k >> 5, ntile = (steps + 7) >> 3;
  const int lane = threadIdx.x & 31, slot = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int nblk = (n + 15) >> 4, ntt = (m + TOK - 1) / TOK;
  const int item = blockIdx.x / ntt, tok0 = (blockIdx.x % ntt) * TOK;
  auto sw = [](int row, int p) { return (p & ~7) | ((p & 7) ^ ((row & 1) << 2)); };  // 32 pieces per 512-byte row
  auto stage = [&](int t, int sl) {
    unsigned char* S0 = bsm + (size_t)sl * STG;
    if (t < ntile) {
      for (int P = threadIdx.x; P < (WB + XB) / 16; P += 256) {
        if (P < WB / 16) {  // weight row (local 0..63), piece
          const int rl = P >> 5, p = P & 31, row = item * RB * 16 + rl, col = t * 256 + p * 8;
          const bool in = row < n && col < k;
          cp16(S0 + ((size_t)rl * 32 + sw(rl, p)) * 16, w + (size_t)(in ? row : 0) * k + (in ? col : 0), in ? 16 : 0);
        } else {
          const int Q = P - WB / 16, tok = Q >> 5, p = Q & 31, row = tok0 + tok, col = t * 256 + p * 8;
          const bool in = row < m && col < k;
          cp16(S0 + WB + ((size_t)tok * 32 + sw(tok, p)) * 16, x + (size_t)(in ? row : 0) * k + (in ? col : 0), in ? 16 : 0);
        }
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  float acc[RB][NTT][4];
#pragma unroll
  for (int r = 0; r < RB; ++r)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt) acc[r][tt][0] = acc[r][tt][1] = acc[r][tt][2] = acc[r][tt][3] = 0.f;
#pragma unroll
  for (int sl = 0; sl < STAGES - 1; ++sl) stage(sl, sl);
  for (int t = 0; t < ntile; ++t) {
    stage(t + STAGES - 1, (t + STAGES - 1) % STAGES);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1) : "memory");
    __syncthreads();
    const uint4* W = reinterpret_cast<const uint4*>(bsm + (size_t)(t % STAGES) * STG);
    const uint4* X = reinterpret_cast<const uint4*>(bsm + (size_t)(t % STAGES) * STG + WB);
    if (8 * t + slot < steps) {
      const int p = slot * 4 + t4;  // this lane's 8 inputs of the step's 32
      uint4 xb[NTT];
#pragma unroll
      for (int tt = 0; tt < NTT; ++tt) {
        const int tok = tt * 8 + g;
        xb[tt] = X[tok * 32 + sw(tok, p)];
      }
#pragma unroll
      for (int r = 0; r < RB; ++r) {
        const int ra = r * 16 + g, rb2 = ra + 8;
        const uint4 w0 = W[ra * 32 + sw(ra, p)], w1 = W[rb2 * 32 + sw(rb2, p)];
#pragma unroll
        for (int tt = 0; tt < NTT; ++tt) {
          mma16816(acc[r][tt], w0.x, w1.x, w0.y, w1.y, xb[tt].x, xb[tt].y);
          mma16816(acc[r][tt], w0.z, w1.z, w0.w, w1.w, xb[tt].z, xb[tt].w);
        }
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  float* red = reinterpret_cast<float*>(bsm);  // [8 slots][32 lanes][RB * NTT * 4]
#pragma unroll
  for (int r = 0; r < RB; ++r)
#pragma unroll
    for (int tt = 0; tt < NTT; ++tt)
#pragma unroll
      for (int j = 0; j < 4; ++j) red[((size_t)slot * 32 + lane) * (RB * NTT * 4) + (r * NTT + tt) * 4 + j] = acc[r][tt][j];
  __syncthreads();
  for (int pr = slot; pr < RB * NTT; pr += 8) {
    const int r = pr / NTT, tt = pr % NTT, rbk = item * RB + r;
    if (rbk >= nblk) continue;
    const int r0 = (rbk << 4) + g, r1 = r0 + 8;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ww = 0; ww < 8; ++ww)
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] += red[((size_t)ww * 32 + lane) * (RB * NTT * 4) + pr * 4 + j];
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int col = tok0 + tt * 8 + 2 * t4 + cc;
      if (col < m) {
        if (r0 < n) y[(size_t)col * n + r0] = __float2bfloat16_rn(bias ? v[cc] + __bfloat162float(bias[r0]) : v[cc]);
        if (r1 < n) y[(size_t)col * n + r1] = __float2bfloat16_rn(bias ? v[2 + cc] + __bfloat162float(bias[r1]) : v[2 + cc]);
      }
    }
  }
}
__global__ void bmm_bi_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ Cm, int T,
                              int mr, int kk, int nc) {
  const long total = (long)T * mr * nc;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int t = (int)(i / ((long)mr * nc)), rem = (int)(i % ((long)mr * nc)), r = rem / nc, c = rem % nc;
    const bf16* a = A + ((size_t)t * mr + r) * kk;
    const bf16* b = B + (size_t)t * kk * nc + c;
    float s = 0.f;
    for (int j = 0; j < kk; ++j) s = __fmaf_rn(__bfloat162float(a[j]), __bfloat162float(b[(size_t)j * nc]), s);
    Cm[i] = __float2bfloat16_rn(s);
  }
}

int sms() {
  static int v = 0;
  if (!v) {
    int d;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
  }
  return v;
}
}  // namespace

extern "C" int ak_bf16_gemv_bi(const void* x, const void* w, const void* bias, void* y, int m, int n, int k,
                               void* stream) {
  if (m < 1 || m > 128 || k % 32 != 0) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  static const bool nt = [] { const char* e = getenv("AGENS_W4_NT"); return !(e && e[0] == '0'); }();
  // batched verify, few row blocks (n < 32 x SMs: mHC dyn n = 24, in_proj_b n = 48, indexer projections): the slot split
  // (AGENS_BI_SPLIT8=1: the split for m <= 8 too, bit-identical; off by default: slower on H200 at 8 tokens)
  static const bool split8 = [] { const char* e = getenv("AGENS_BI_SPLIT8"); return e && e[0] == '1'; }();
  if ((m > 8 || split8) && nt && (n + 15) / 16 < 2 * sms()) {
    static std::map<std::pair<int, cudaStream_t>, std::pair<float*, unsigned*>> scr;
    int dev = 0;
    cudaGetDevice(&dev);
    auto& z = scr[{dev, s}];
    const int nblk = (n + 15) / 16, nb32 = (m + 31) / 32;
    if (!z.first) {
      cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(s, &cs);
      if (cs == cudaStreamCaptureStatusNone) {
        const int maxblk = 2 * sms();
        cudaMalloc(&z.first, (size_t)maxblk * 8 * 16 * 32 * 16);  // [rb][8][<= 16 tiles][32][4] floats
        cudaMalloc(&z.second, (size_t)maxblk * 4 * 4);
        cudaMemset(z.second, 0, (size_t)maxblk * 4 * 4);
        cudaDeviceSynchronize();
      }
    }
    if (z.first) {
      bf16_bi_split_kernel<<<nblk * nb32 * 8, 128, 0, s>>>(static_cast<const bf16*>(x), m, static_cast<const bf16*>(w),
                                                          static_cast<const bf16*>(bias), static_cast<bf16*>(y), n, k,
                                                          z.first, z.second);
      return (int)cudaGetLastError();
    }
  }
  if (m > 8 && nt && k % 256 == 0) {  // batched verify, large n: the pipelined GEMM (shared memory 96 KB: sm80+)
    constexpr int STG = 4 * 16 * 512 + 32 * 512;
    int dev = 0, smax = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    const bool three = smax >= 3 * STG;
    static bool set[2] = {false, false};
    auto kern = three ? bf16_bi_gemm_kernel<3> : bf16_bi_gemm_kernel<2>;
    const int smem = (three ? 3 : 2) * STG;
    if (!set[three]) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), set[three] = true;
    const int items = ((n + 15) / 16 + 3) / 4, ntt = (m + 31) / 32;
    kern<<<items * ntt, 256, smem, s>>>(static_cast<const bf16*>(x), m, static_cast<const bf16*>(w),
                                         static_cast<const bf16*>(bias), static_cast<bf16*>(y), n, k);
    return (int)cudaGetLastError();
  }
  if (m > 8 && nt) {  // batched verify: 8-token tiles per weight pass (64 tokens per launch)
    const int want = (n + 15) >> 4;
    for (int m0 = 0; m0 < m; m0 += 64) {
      const int mm = m - m0 < 64 ? m - m0 : 64, t = (mm + 7) / 8;
      const bf16* xx = static_cast<const bf16*>(x) + (size_t)m0 * k;
      bf16* yy = static_cast<bf16*>(y) + (size_t)m0 * n;
      auto go = [&](auto kern) {
        int per = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kern, kWarps * 32, 0);
        if (per < 1) per = 1;
        const int blocks = want < sms() * per ? want : sms() * per;
        kern<<<blocks, kWarps * 32, 0, s>>>(xx, mm, static_cast<const bf16*>(w), static_cast<const bf16*>(bias), yy, n, k);
      };
      if (t <= 2) go(bf16_gemv_bi_nt_kernel<2>);
      else if (t <= 4) go(bf16_gemv_bi_nt_kernel<4>);
      else go(bf16_gemv_bi_nt_kernel<8>);
    }
    return (int)cudaGetLastError();
  }
  // few 16-row blocks (n < 16 x SMs): 10 steps' loads in flight per warp; else 5 (more blocks per SM)
  const int want = (n + 15) >> 4;
  const bool small = want < sms();
  static int per_sm[2] = {0, 0};
  if (!per_sm[small]) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[small], small ? bf16_gemv_bi_kernel<10> : bf16_gemv_bi_kernel<5>,
                                                  kWarps * 32, 0);
    if (per_sm[small] < 1) per_sm[small] = 1;
  }
  const int blocks = want < sms() * per_sm[small] ? want : sms() * per_sm[small];
  auto kern = small ? bf16_gemv_bi_kernel<10> : bf16_gemv_bi_kernel<5>;
  for (int m0 = 0; m0 < m; m0 += 8) {
    kern<<<blocks, kWarps * 32, 0, s>>>(static_cast<const bf16*>(x) + (size_t)m0 * k,
                                                       m - m0 < 8 ? m - m0 : 8, static_cast<const bf16*>(w),
                                                       static_cast<const bf16*>(bias),
                                                       static_cast<bf16*>(y) + (size_t)m0 * n, n, k);
  }
  return (int)cudaGetLastError();
}

extern "C" int ak_bf16_bmm_bi(const void* A, const void* B, void* C, int T, int m, int k, int n, void* stream) {
  const long total = (long)T * m * n;
  bmm_bi_kernel<<<(int)((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096), 256, 0,
                  static_cast<cudaStream_t>(stream)>>>(static_cast<const bf16*>(A), static_cast<const bf16*>(B),
                                                       static_cast<bf16*>(C), T, m, k, n);
  return (int)cudaGetLastError();
}
