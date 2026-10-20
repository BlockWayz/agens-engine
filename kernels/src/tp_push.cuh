// Push all-reduce for small messages between peer GPUs (kernels/src/tp_ar.cu), as reusable device pieces. A kernel that
// consumes a tensor-parallel sum (the fused AR + add + RMSNorm in tp_ar.cu; a fused consumer elsewhere can do the same)
// is launched with grid = tp_push::grid() blocks (the same fixed count for every push launch on every rank) and runs,
// in every block (a block with nothing to push still runs all of it):
//   const unsigned e = tp_push::begin(d);          the block's epoch
//   tp_push::push(d, e, x, i)                      for each uint4 unit i of the local partial x this block owns
//   tp_push::signal_wait(d, e)                     (__syncthreads; flag mode: flag the peers, wait for theirs)
//   tp_push::sum8(d, e, x, i, acc)                 for each owned unit: rank-order fp32 sum of its 8 bf16 -> acc[8]
//   tp_push::end(d, e, units)                      units = what the call pushed in all (the same in every block)
// A block must sum exactly the units it pushed (the peers' blocks of the same index pushed the same units).
//
// Sum. Unit i's elements are summed over the ranks in rank order 0..size-1 in fp32 starting from +0.0 (own share read
// from x, the peers' from this rank's inbox) and rounded once to bf16 by the caller: the result depends only on the
// ranks' values, so it is identical on every rank, for every batch size, grid and unit-to-block mapping, and bit-identical
// to the pull kernel's (ar_kernel: same order, same rounding).
//
// Epochs. Every rank holds per-block u32 epochs (Desc::epoch[block], device) and every rank runs the same sequence of
// push launches with the same fixed grid, so epoch[b] agrees across ranks and across the blocks of a rank (every
// launch advances every block's epoch by one, in end(), with no cross-block step: no atomics, no bump launch, nothing
// from the host per call -- graph-capturable). Desc::epoch[kMaxBlk + b] keeps the unit count of the call that used
// buffer b (written by block 0).
//
// Inbox of rank r (device memory, CUDA IPC-mapped by every peer): [0, kFlagBytes) u32 flag[src][block] (flag mode),
// then data[buffer][src][elems] bf16 (kBufs buffers).
//
// Lamport mode (Desc::lamport = 1, the default; AGENS_TP_PUSH_LAMPORT=0 for flag mode). Buffer b = e % 3. Every data
// element of a buffer is the sentinel -0.0 (0x8000) until a peer pushes into it. push() stores the unit to data[b][rank]
// of every peer with -0.0 halves sent as +0.0 (exact: the sum starts from +0.0, 0 + -0 = +0 and x + -0 = x, so a -0
// contribution never changes a result); sum8() polls each peer's unit (ld.volatile: from L2, where NVLink writes land)
// until none of its 8 halves is the sentinel. No flags and no cross-block wait (a block only waits for the peers'
// blocks of its index). end() re-arms buffer (e + 2) % 3 -- the one the previous call used -- with sentinels over that
// call's unit count. Reuse: a peer writes buffer b again at its call e + 3, which it reaches only after reading this
// rank's pushes of call e + 2, issued after this rank's call e + 1 (whose end re-armed b) completed in stream order.
//
// Flag mode (lamport = 0). Buffer p = e & 1. After the pushes, __syncthreads, then thread q (q != rank) does a system
// fence and st.release.sys flag[rank][block] = e + 1 in peer q's inbox and spins (ld.acquire.sys, local memory) until
// flag[q][block] of its own inbox is >= e + 1 (wrap-safe), __syncthreads; peers' units are then read with ld.cg.
// Reuse: data[p] is written again at e + 2, which a rank reaches only after its call e + 1 saw flag e + 2 from every
// peer, i.e. every peer had started its call e + 1, so its call e (the readers of data[p]) had completed.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tp_push {
constexpr int kMaxTp = 8;
constexpr int kMaxBlk = 256;
constexpr int kBufs = 3;
constexpr size_t kFlagBytes = (size_t)kMaxTp * kMaxBlk * 4;
struct Desc {
  int rank, size;
  long elems;                     // bf16 elements per (buffer, src) data area: the largest push all-reduce
  unsigned char* inbox[kMaxTp];   // every rank's inbox as this device sees it
  unsigned* epoch;                // [b] block b's epoch; [kMaxBlk + buffer] units of the call that used the buffer
  int lamport;                    // 1: sentinel-polled data, 3 buffers; 0: flags, 2 buffers
  int grid;                       // the fixed grid of every push launch
};

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned* flag(const Desc& d, int owner, int src, int blk) {
  return reinterpret_cast<unsigned*>(d.inbox[owner]) + src * kMaxBlk + blk;
}
__device__ __forceinline__ unsigned buf(const Desc& d, unsigned e) { return d.lamport ? e % 3 : e & 1; }
__device__ __forceinline__ uint4* data_b(const Desc& d, int owner, unsigned b, int src) {
  return reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(d.inbox[owner] + kFlagBytes) +
                                  ((size_t)b * kMaxTp + src) * d.elems);
}
__device__ __forceinline__ uint4* data(const Desc& d, int owner, unsigned e, int src) { return data_b(d, owner, buf(d, e), src); }
constexpr unsigned kSent2 = 0x80008000u;  // two bf16 sentinels (-0.0)
__device__ __forceinline__ unsigned nosent(unsigned w) {  // -0.0 halves -> +0.0
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000FFFFu;
  return w;
}
__device__ __forceinline__ bool has_sent(unsigned w) { return (w & 0xFFFFu) == 0x8000u || (w >> 16) == 0x8000u; }

__device__ __forceinline__ unsigned begin(const Desc& d) { return *reinterpret_cast<volatile unsigned*>(d.epoch + blockIdx.x); }
// uint4 unit i of x (8 bf16) -> data[buffer][rank][i] of every peer
__device__ __forceinline__ void push(const Desc& d, unsigned e, const uint4* x, long i) {
  uint4 v = x[i];
  if (d.lamport) v.x = nosent(v.x), v.y = nosent(v.y), v.z = nosent(v.z), v.w = nosent(v.w);
  for (int q = 0; q < d.size; ++q)
    if (q != d.rank) data(d, q, e, d.rank)[i] = v;
}
__device__ __forceinline__ void signal_wait(const Desc& d, unsigned e) {
  if (d.lamport) return;
  __syncthreads();
  const int q = threadIdx.x;
  if (q < d.size && q != d.rank) {
    __threadfence_system();
    st_release_sys(flag(d, q, d.rank, blockIdx.x), e + 1);
    const unsigned* f = flag(d, d.rank, q, blockIdx.x);
    while ((int)(ld_acquire_sys(f) - (e + 1)) < 0) {
    }
  }
  __syncthreads();
}
// rank-order sum of uint4 unit i: own share from x, the peers' from this rank's inbox
__device__ __forceinline__ void sum8(const Desc& d, unsigned e, const uint4* x, long i, float acc[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int r = 0; r < d.size; ++r) {
    uint4 u;
    if (r == d.rank) u = x[i];
    else if (!d.lamport) u = __ldcg(data(d, d.rank, e, r) + i);
    else {
      const uint4* p = data(d, d.rank, e, r) + i;
      do u = ld_volatile(p);
      while (has_sent(u.x) || has_sent(u.y) || has_sent(u.z) || has_sent(u.w));
    }
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += __bfloat162float(b[j]);
  }
}
// units: what this call pushed in all (the same in every block). Lamport: re-arm the buffer the previous call used.
__device__ __forceinline__ void end(const Desc& d, unsigned e, long units) {
  if (d.lamport) {
    const unsigned b = (e + 2) % 3;
    const long prev = d.epoch[kMaxBlk + b];
    const uint4 s4 = make_uint4(kSent2, kSent2, kSent2, kSent2);
    for (int r = 0; r < d.size; ++r)
      if (r != d.rank) {
        uint4* p = data_b(d, d.rank, b, r);
        for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < prev; i += (long)gridDim.x * blockDim.x) p[i] = s4;
      }
    if (blockIdx.x == 0 && threadIdx.x == 0) d.epoch[kMaxBlk + e % 3] = (unsigned)units;
  }
  if (threadIdx.x == 0) d.epoch[blockIdx.x] = e + 1;  // read by this block's next launch (stream order)
}
}  // namespace tp_push

// this process's push descriptor (valid after ak_tp_setup with peer regions; null otherwise); launch consumers with
// grid = desc->grid blocks
const tp_push::Desc* ak_tp_push_desc();
