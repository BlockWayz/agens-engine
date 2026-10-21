// SiLU(gate) * up for the merged gate_up output [T][2*I] -> [T][I] bf16. Compiled with --use_fast_math (see
// CMakeLists) because sglv23's JIT act_and_mul_kernel is (sglang activation.py: fast-math on sm < 100), and the
// expression is written exactly as theirs so nvcc emits the same instructions: x / (1 + expf(-x)) * up.
#include "agens_kernels.h"
#include "agens_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__global__ void silu_mul_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, unsigned num_vecs,
                                unsigned tokens) {
  ak_pdl_entry();
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned t = tid / num_vecs;
  if (t >= tokens) return;
  const unsigned off = tid % num_vecs;
  const uint4 g4 = in[t * num_vecs * 2 + off], u4 = in[t * num_vecs * 2 + num_vecs + off];
  const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g4);
  const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&u4);
  uint4 o4;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&o4);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float x = __bfloat162float(g[i]);
    const float up = __bfloat162float(u[i]);
    o[i] = __float2bfloat16(x / (1.0f + expf(-x)) * up);
  }
  out[tid] = o4;
}
// one tensor-parallel share of the merged gate_up: in = [gate share | up share] (np each), out = the full [inter] row
// with the share's SiLU product at [lo, lo + np) (silu_mul_kernel's expression and flags: the same bits) and zeros
// elsewhere (summed over the ranks by an all-reduce: x + 0 is exact)
__global__ void silu_mul_part_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, unsigned inter8, unsigned lo8,
                                     unsigned np8) {
  ak_pdl_entry();
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= inter8) return;
  if (tid < lo8 || tid >= lo8 + np8) {
    out[tid] = make_uint4(0, 0, 0, 0);
    return;
  }
  const unsigned off = tid - lo8;
  const uint4 g4 = in[off], u4 = in[np8 + off];
  const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g4);
  const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&u4);
  uint4 o4;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&o4);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float x = __bfloat162float(g[i]);
    const float up = __bfloat162float(u[i]);
    o[i] = __float2bfloat16(x / (1.0f + expf(-x)) * up);
  }
  out[tid] = o4;
}
}  // namespace

extern "C" int ak_silu_mul_part(const void* gate_up_part, void* out, int inter, int lo, int np, void* stream) {
  if (inter % 8 || lo % 8 || np % 8 || lo + np > inter) return (int)cudaErrorInvalidValue;
  const unsigned n8 = inter / 8;
  silu_mul_part_kernel<<<(n8 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(gate_up_part), static_cast<uint4*>(out), n8, lo / 8, np / 8);
  return (int)cudaGetLastError();
}

extern "C" int ak_silu_mul(const void* gate_up, void* out, int tokens, int inter, void* stream) {
  if (inter % 8) return (int)cudaErrorInvalidValue;
  const unsigned nv = inter / 8, total = nv * tokens;
  silu_mul_kernel<<<(total + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(gate_up), static_cast<uint4*>(out), nv, tokens);
  return (int)cudaGetLastError();
}
