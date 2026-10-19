#include <vector>
// Tensor-parallel all-reduce for the engine's ranks (one process per GPU, kernels/m1 ak_m1_tp_config): a one-shot
// sum over every rank's exchange region, in rank order, so all ranks hold bit-identical results (the replicated parts
// of the model -- mHC, norms, Engram, lm_head, drafter -- then stay identical across ranks).
//
// Exchange region of a rank (device memory exported by CUDA IPC when the GPUs are peers, or a shared host mapping
// when they are not): [0, 128) its flag (the last epoch it has staged), [4096, ..) two slots of `slot_elems` bf16
// (parity of the epoch). The epoch is a device counter of this rank; every rank runs the same sequence of all-reduces,
// so the epochs agree. One all-reduce: every block copies its share of x into the rank's slot, the last block to
// arrive publishes flag = epoch + 1 (release, system scope), every block waits for all flags >= epoch + 1, then sums
// the ranks' slots in rank order into x; a one-thread kernel then advances the epoch. Slot p is written again at
// epoch + 2, when every peer has published epoch + 2, i.e. finished this epoch's reads (stream order).
// Graph-capturable (no host values per call).
#include "agens_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {
using bf16 = __nv_bfloat16;
constexpr int kMaxTp = 8;
constexpr size_t kSlotOff = 4096;
struct TpDev {
  int rank, size;
  long slot_elems;
  unsigned char* region[kMaxTp];  // every rank's region as this device sees it
  unsigned* epoch;                // this rank's counter (device)
  unsigned* arrive;               // [2] this rank's per-parity block counters (device)
  int local;                      // 1: this rank's own share is read from x (its slot copy may sit in host memory)
};
TpDev g_tp{};
bool g_tp_ok = false;
int g_tp_blocks = 0;

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned* flag_of(const TpDev& d, int r) { return reinterpret_cast<unsigned*>(d.region[r]); }
__device__ __forceinline__ bf16* slot_of(const TpDev& d, int r, int p) {
  return reinterpret_cast<bf16*>(d.region[r] + kSlotOff) + (size_t)p * d.slot_elems;
}

__global__ void __launch_bounds__(256) ar_kernel(bf16* __restrict__ x, long n, const __grid_constant__ TpDev d) {
  const unsigned e = *d.epoch;
  const int p = e & 1;
  bf16* my = slot_of(d, d.rank, p);
  const long stride = (long)gridDim.x * blockDim.x;
  // 16-byte pieces where aligned (n % 8 == 0: the engine's rows of 5120), else elements
  const bool v = (n & 7) == 0;
  if (v) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n / 8; i += stride)
      reinterpret_cast<uint4*>(my)[i] = reinterpret_cast<const uint4*>(x)[i];
  } else {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) my[i] = x[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned c = atomicAdd(d.arrive + p, 1u);
    if (c == gridDim.x - 1) {
      d.arrive[p] = 0;
      __threadfence_system();
      st_release_sys(flag_of(d, d.rank), e + 1);
    }
    for (int r = 0; r < d.size; ++r)
      while ((!d.local || r != d.rank) && (int)(ld_acquire_sys(flag_of(d, r)) - (e + 1)) < 0) {
      }
  }
  __syncthreads();
  if (v) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n / 8; i += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < d.size; ++r) {
        // own share: x itself (the same bf16 values as the slot copy, so the rank-order sum is unchanged)
        const uint4 u = d.local && r == d.rank ? reinterpret_cast<const uint4*>(x)[i]
                                               : __ldcv(reinterpret_cast<const uint4*>(slot_of(d, r, p)) + i);
        const bf16* b = reinterpret_cast<const bf16*>(&u);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += __bfloat162float(b[j]);
      }
      uint4 o;
      bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
      for (int j = 0; j < 8; ++j) ob[j] = __float2bfloat16_rn(acc[j]);
      reinterpret_cast<uint4*>(x)[i] = o;
    }
  } else {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
      float acc = 0.f;
      for (int r = 0; r < d.size; ++r) {
        const unsigned short u = d.local && r == d.rank ? __bfloat16_as_ushort(x[i])
                                                        : __ldcv(reinterpret_cast<const unsigned short*>(slot_of(d, r, p)) + i);
        acc += __bfloat162float(__ushort_as_bfloat16(u));
      }
      x[i] = __float2bfloat16_rn(acc);
    }
  }
}
__global__ void bump_kernel(unsigned* epoch) { *epoch += 1; }
}  // namespace

// Exchange memory. Device region exported for CUDA IPC (handle: 64 bytes) ...
extern "C" int ak_tp_region_alloc(size_t bytes, void** dev_ptr, void* ipc_handle) {
  if (cudaMalloc(dev_ptr, bytes) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
  cudaMemset(*dev_ptr, 0, bytes);
  cudaIpcMemHandle_t h;
  const cudaError_t e = cudaIpcGetMemHandle(&h, *dev_ptr);
  if (e != cudaSuccess) return (int)e;
  memcpy(ipc_handle, &h, sizeof(h));
  return 0;
}
// ... opened by a peer process (its GPU must be a peer of the exporter's)
extern "C" int ak_tp_region_open(const void* ipc_handle, void** dev_ptr) {
  cudaIpcMemHandle_t h;
  memcpy(&h, ipc_handle, sizeof(h));
  return (int)cudaIpcOpenMemHandle(dev_ptr, h, cudaIpcMemLazyEnablePeerAccess);
}
// ... or a POSIX shared-memory segment (/dev/shm/<name>) mapped into this process and registered with CUDA (GPUs without
// peer access: the exchange crosses host memory). create = 1 by the first rank (zeroed).
extern "C" int ak_tp_region_shm(const char* name, size_t bytes, int create, void** dev_ptr) {
  const int fd = shm_open(name, create ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
  if (fd < 0) return (int)cudaErrorInvalidValue;
  if (create && ftruncate(fd, (off_t)bytes) != 0) return close(fd), (int)cudaErrorInvalidValue;
  void* h = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (h == MAP_FAILED) return (int)cudaErrorMemoryAllocation;
  if (create) memset(h, 0, bytes);
  cudaError_t e = cudaHostRegister(h, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess) return (int)e;
  e = cudaHostGetDevicePointer(dev_ptr, h, 0);
  return (int)e;
}
// whether this device can read device memory of `peer` directly (CUDA IPC path) -- callers decide per pair
extern "C" int ak_tp_can_peer(int dev, int peer) {
  int ok = 0, n = 0;
  cudaGetDeviceCount(&n);
  if (dev < 0 || peer < 0 || dev >= n || peer >= n) return 0;
  if (cudaDeviceCanAccessPeer(&ok, dev, peer) != cudaSuccess) ok = 0;
  cudaGetLastError();  // never leave an error for the next launch check
  return ok;
}
// regions[r] = rank r's region as this process sees it; slot_elems = bf16 elements per slot (the largest all-reduce)
extern "C" int ak_tp_setup(int rank, int size, void* const* regions, long slot_elems) {
  if (size < 1 || size > kMaxTp || rank < 0 || rank >= size) return (int)cudaErrorInvalidValue;
  g_tp = TpDev{};
  g_tp.rank = rank, g_tp.size = size, g_tp.slot_elems = slot_elems;
  for (int r = 0; r < size; ++r) g_tp.region[r] = static_cast<unsigned char*>(regions[r]);
  void* c = nullptr;
  if (cudaMalloc(&c, 64) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
  cudaMemset(c, 0, 64);
  g_tp.epoch = static_cast<unsigned*>(c), g_tp.arrive = static_cast<unsigned*>(c) + 4;
  // AGENS_TP_AR_LOCAL=0: read the own share back from the slot as well (the pre-1.2 kernel; same results)
  g_tp.local = !(getenv("AGENS_TP_AR_LOCAL") && getenv("AGENS_TP_AR_LOCAL")[0] == '0');
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  g_tp_blocks = sms;  // every block resident (the blocks wait on each other's flags)
  g_tp_ok = true;
  return (int)cudaDeviceSynchronize();
}
extern "C" size_t ak_tp_region_bytes(long slot_elems) { return kSlotOff + 2 * (size_t)slot_elems * 2; }
extern "C" int ak_tp_allreduce(void* x, long n, void* stream) {
  if (!g_tp_ok || n > g_tp.slot_elems) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  static long calls = 0;
  static const bool trace = getenv("AGENS_TP_TRACE") != nullptr;
  // trace: per call the FNV hash of this rank's input and of the result (synchronises: debugging only)
  auto hash = [&](void) -> unsigned long long {
    static std::vector<unsigned short> h;
    h.resize(n);
    cudaStreamSynchronize(s);
    cudaMemcpy(h.data(), x, n * 2, cudaMemcpyDeviceToHost);
    unsigned long long f = 1469598103934665603ull;
    for (long i = 0; i < n; ++i) f = (f ^ h[i]) * 1099511628211ull;
    return f;
  };
  const unsigned long long pre = trace ? hash() : 0;
  ++calls;
  const long pieces = (n & 7) == 0 ? n / 8 : n;
  int blocks = (int)((pieces + 255) / 256);
  if (blocks > g_tp_blocks) blocks = g_tp_blocks;
  if (blocks < 1) blocks = 1;
  ar_kernel<<<blocks, 256, 0, s>>>(static_cast<bf16*>(x), n, g_tp);
  bump_kernel<<<1, 1, 0, s>>>(g_tp.epoch);
  if (trace) fprintf(stderr, "tp ar #%ld rank %d n %ld pre %016llx post %016llx\n", calls - 1, g_tp.rank, n, pre, hash());
  return (int)cudaGetLastError();
}
// Self-test (AGENS_TP_AR_SELFTEST): `iters` all-reduces of n elements, rank r's x[i] = bf16(((i * 31 + it) % 97) * (r + 1) / 64);
// returns the number of elements that differ from the expected rank-order sum (every rank runs the same calls)
namespace {
__global__ void st_fill(bf16* x, long n, int r, int it) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    x[i] = __float2bfloat16_rn((float)(((i * 31 + it) % 97) * (r + 1)) / 64.f);
}
__global__ void st_check(const bf16* x, long n, int size, int it, unsigned* bad) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    float acc = 0.f;
    for (int r = 0; r < size; ++r) acc += __bfloat162float(__float2bfloat16_rn((float)(((i * 31 + it) % 97) * (r + 1)) / 64.f));
    if (__bfloat16_as_ushort(x[i]) != __bfloat16_as_ushort(__float2bfloat16_rn(acc))) atomicAdd(bad, 1u);
  }
}
}  // namespace
// Timing (AGENS_TP_AR_BENCH): mean microseconds per all-reduce of n elements over `iters` back-to-back calls (after 20
// warm-up calls); every rank runs the same calls.
extern "C" double ak_tp_bench(int iters, long n) {
  bf16* x = nullptr;
  if (cudaMalloc(&x, n * 2) != cudaSuccess) return -1;
  cudaMemset(x, 0, n * 2);
  for (int i = 0; i < 20; ++i) ak_tp_allreduce(x, n, nullptr);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) ak_tp_allreduce(x, n, nullptr);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaFree(x);
  return 1000.0 * ms / iters;
}
extern "C" long ak_tp_selftest(int iters, long n) {
  bf16* x = nullptr;
  unsigned* bad = nullptr;
  cudaMalloc(&x, n * 2), cudaMalloc(&bad, 4);
  cudaMemset(bad, 0, 4);
  for (int it = 0; it < iters; ++it) {
    const long nn = it % 3 == 0 ? n : (it % 3 == 1 ? 5120 : 40960);
    st_fill<<<256, 256>>>(x, nn, g_tp.rank, it);
    if (ak_tp_allreduce(x, nn, nullptr)) return -1;
    st_check<<<256, 256>>>(x, nn, g_tp.size, it, bad);
  }
  unsigned h = 0;
  cudaMemcpy(&h, bad, 4, cudaMemcpyDeviceToHost);
  cudaFree(x), cudaFree(bad);
  return (long)h;
}

