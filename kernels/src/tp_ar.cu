#include <vector>
#include "agens_pdl.cuh"
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
// Small all-reduces (<= AGENS_TP_PUSH_ROWS rows of 5120, default 256) between peer GPUs take the push kernels instead
// (tp_push.cuh: one launch, the epoch advanced in-kernel, no bump; the same rank-order sum, so the same bits), and the
// mixer all-reduce can be fused with its residual add + RMSNorm (ak_tp_allreduce_add_rmsnorm).
#include "agens_kernels.h"
#include "tp_push.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#if defined(_MSC_VER)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
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
tp_push::Desc g_push{};
bool g_push_ok = false;
bool g_force_pull = false;  // self-test: the pull kernel as the reference
long push_elems(long slot_elems) {  // AGENS_TP_PUSH_ROWS (rows of 5120; 0: off), the same on every rank (inherited env)
  long rows = 256;
  if (const char* e = getenv("AGENS_TP_PUSH_ROWS")) rows = atol(e);
  const long n = rows * 5120;
  return n < slot_elems ? n : slot_elems;
}
size_t push_off(long slot_elems) { return (kSlotOff + 2 * (size_t)slot_elems * 2 + 4095) & ~(size_t)4095; }

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

// push all-reduce of x[n] in place (n % 8 == 0): 256 threads, uint4 units grid-strided (the same units in both phases)
__global__ void __launch_bounds__(256) push_ar_kernel(bf16* __restrict__ x, long units, const __grid_constant__ tp_push::Desc d) {
  const unsigned e = tp_push::begin(d);  // (its griddepcontrol.wait)
  ak_pdl_launch();
  uint4* xv = reinterpret_cast<uint4*>(x);
  const long stride = (long)gridDim.x * blockDim.x;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < units; i += stride) tp_push::push(d, e, xv, i);
  tp_push::signal_wait(d, e);
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < units; i += stride) {
    float acc[8];
    tp_push::sum8(d, e, xv, i, acc);
    uint4 o;
    bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) ob[j] = __float2bfloat16_rn(acc[j]);
    xv[i] = o;
  }
  tp_push::end(d, e, units);
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// push all-reduce of the mixer output mo[rows][H] fused with the residual add + RMSNorm of ak_add_rmsnorm (rmsnorm.cu,
// TPR 64; same bits): a = AR(mo); s = bf16(res + a) -> s_out; y = RMSNorm(s) * (w + bias). One block per row (rows
// strided by the grid), H / 8 threads: thread k owns uint4 k of the row; the norm's sum of squares runs as TPR 64's
// lanes do (lane l: units b * 64 + l, b ascending, one fma chain; bfly within the warp; the two warp sums bfly-added).
__global__ void __launch_bounds__(1024) push_ar_norm_kernel(const bf16* __restrict__ mo, const bf16* __restrict__ res,
                                                            bf16* __restrict__ s_out, const bf16* __restrict__ w,
                                                            bf16* __restrict__ y, int rows, int H, float bias, float eps,
                                                            const __grid_constant__ tp_push::Desc d) {
  extern __shared__ uint4 srow[];  // the row's s, H / 8 units
  __shared__ float ws[2], rs;
  const unsigned e = tp_push::begin(d);  // (its griddepcontrol.wait)
  ak_pdl_launch();
  const int U = H / 8, k = threadIdx.x;
  const uint4* mv = reinterpret_cast<const uint4*>(mo);
  for (int r = blockIdx.x; r < rows; r += gridDim.x)
    if (k < U) tp_push::push(d, e, mv, (long)r * U + k);
  tp_push::signal_wait(d, e);
  const uint4 wv = k < U ? __ldg(reinterpret_cast<const uint4*>(w) + k) : uint4{};
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const long i = (long)r * U + k;
    uint4 sv{};
    if (k < U) {
      const uint4 xr = reinterpret_cast<const uint4*>(res)[i];  // issued before the peers' units are polled
      float acc[8];
      tp_push::sum8(d, e, mv, i, acc);
      const bf16* xe = reinterpret_cast<const bf16*>(&xr);
      bf16* se = reinterpret_cast<bf16*>(&sv);
#pragma unroll
      for (int j = 0; j < 8; ++j)  // a = bf16(sum), s = bf16(res + a) (torch's bf16 add)
        se[j] = __float2bfloat16(__bfloat162float(xe[j]) + __bfloat162float(__float2bfloat16_rn(acc[j])));
      reinterpret_cast<uint4*>(s_out)[i] = sv;
      srow[k] = sv;
    }
    __syncthreads();
    if (k < 64) {
      float sum = 0.f;
      for (int b = 0; b * 64 + k < U; ++b) {
        const uint4 v = srow[b * 64 + k];
        const bf16* ve = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float f = __bfloat162float(ve[j]);
          sum = __fmaf_rn(f, f, sum);
        }
      }
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) sum = __fadd_rn(__shfl_xor_sync(0xFFFFFFFFu, sum, o, 32), sum);
      if ((k & 31) == 0) ws[k >> 5] = sum;
    }
    __syncthreads();
    if (k < 32) {
      float v = k < 2 ? ws[k] : 0.f;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
      if (k == 0) rs = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(v, (float)H), eps));
    }
    __syncthreads();
    if (k < U) {
      const float rstd = rs;
      const bf16* se = reinterpret_cast<const bf16*>(&sv);
      const bf16* we = reinterpret_cast<const bf16*>(&wv);
      uint4 o4;
      bf16* o = reinterpret_cast<bf16*>(&o4);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(se[j]), rstd), __fadd_rn(__bfloat162float(we[j]), bias)));
      reinterpret_cast<uint4*>(y)[i] = o4;
    }
    __syncthreads();  // srow / ws / rs reused by the next row
  }
  tp_push::end(d, e, (long)rows * U);
}
__global__ void push_arm(unsigned* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = tp_push::kSent2;
}
}  // namespace
const tp_push::Desc* ak_tp_push_desc() { return g_push_ok ? &g_push : nullptr; }
extern "C" int ak_tp_push_on(void) { return g_push_ok ? 1 : 0; }  // the push all-reduce (peer GPUs over CUDA IPC)

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
// ... or a shared-memory segment (/dev/shm/<name> on POSIX; a named Windows mapping) mapped into this process and
// registered with CUDA (GPUs without peer access: the exchange crosses host memory). create = 1 by the first rank (zeroed).
extern "C" int ak_tp_region_shm(const char* name, size_t bytes, int create, void** dev_ptr) {
#if defined(_MSC_VER)
  // Windows: the unnamed namespace is session-local, so all ranks must run in one logon session (cross-session would
  // need a "Global\\" prefix, which requires SeCreateGlobalPrivilege — unavailable to standard users).
  const char* n = name[0] == '/' ? name + 1 : name;
  HANDLE m = create ? CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(bytes >> 32),
                                         (DWORD)(bytes & 0xFFFFFFFF), n)
                    : OpenFileMappingA(FILE_MAP_WRITE, FALSE, n);
  if (!m) return (int)cudaErrorInvalidValue;
  void* h = MapViewOfFile(m, FILE_MAP_WRITE, 0, 0, (SIZE_T)bytes);
  CloseHandle(m);
  if (!h) return (int)cudaErrorMemoryAllocation;
#else
  const int fd = shm_open(name, create ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
  if (fd < 0) return (int)cudaErrorInvalidValue;
  if (create && ftruncate(fd, (off_t)bytes) != 0) return close(fd), (int)cudaErrorInvalidValue;
  void* h = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (h == MAP_FAILED) return (int)cudaErrorMemoryAllocation;
  if (create) memset(h, 0, bytes);
#endif
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
  // push path: every region device memory (CUDA IPC between peer GPUs; not the host shared-memory fallback)
  g_push = tp_push::Desc{};
  g_push_ok = false;
  bool dev_all = push_elems(slot_elems) > 0;
  for (int r = 0; r < size && dev_all; ++r) {
    cudaPointerAttributes a{};
    dev_all = cudaPointerGetAttributes(&a, regions[r]) == cudaSuccess && a.type == cudaMemoryTypeDevice;
  }
  cudaGetLastError();
  if (dev_all) {
    g_push.rank = rank, g_push.size = size, g_push.elems = push_elems(slot_elems);
    for (int r = 0; r < size; ++r) g_push.inbox[r] = static_cast<unsigned char*>(regions[r]) + push_off(slot_elems);
    const size_t eb = (tp_push::kMaxBlk + tp_push::kBufs) * 4;
    if (cudaMalloc(&c, eb) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
    cudaMemset(c, 0, eb);
    g_push.epoch = static_cast<unsigned*>(c);
    // the fixed grid of every push launch: one block per SM (AGENS_TP_PUSH_BLOCKS; the same on every rank)
    // (a multiple of 8: the fused mHC boundary, ak_mhc_fused, joins the MLP all-reduce as 8-block clusters per row)
    g_push.grid = sms >= 8 ? sms / 8 * 8 : sms;
    if (const char* e = getenv("AGENS_TP_PUSH_BLOCKS")) g_push.grid = atoi(e);
    if (g_push.grid < 1) g_push.grid = 1;
    if (g_push.grid > tp_push::kMaxBlk) g_push.grid = tp_push::kMaxBlk;
    g_push.lamport = !(getenv("AGENS_TP_PUSH_LAMPORT") && getenv("AGENS_TP_PUSH_LAMPORT")[0] == '0');
    if (g_push.lamport) {  // this rank's inbox: every data element the sentinel (before the setup barrier: no peer writes yet)
      const size_t n = (size_t)tp_push::kBufs * tp_push::kMaxTp * g_push.elems / 2;
      push_arm<<<1024, 256>>>(reinterpret_cast<unsigned*>(g_push.inbox[rank] + tp_push::kFlagBytes), n);
    }
    g_push_ok = true;
  }
  if (rank == 0)
    fprintf(stderr, "tp: all-reduce push path %s%s (%ld elements, %d blocks)\n", g_push_ok ? "on" : "off", g_push.lamport ? ", lamport" : "", g_push_ok ? g_push.elems : 0L, g_push.grid);
  return (int)cudaDeviceSynchronize();
}
extern "C" size_t ak_tp_region_bytes(long slot_elems) {
  return push_off(slot_elems) + tp_push::kFlagBytes + tp_push::kBufs * (size_t)tp_push::kMaxTp * push_elems(slot_elems) * 2;
}
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
  if (g_push_ok && !g_force_pull && (n & 7) == 0 && n <= g_push.elems) {
    const long units = n / 8;
    static const int th = getenv("AGENS_TP_PUSH_THREADS") ? atoi(getenv("AGENS_TP_PUSH_THREADS")) : 256;
    push_ar_kernel<<<g_push.grid, th, 0, s>>>(static_cast<bf16*>(x), units, g_push);
    if (trace) fprintf(stderr, "tp ar #%ld rank %d n %ld pre %016llx post %016llx\n", calls - 1, g_tp.rank, n, pre, hash());
    return (int)cudaGetLastError();
  }
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


// The mixer all-reduce fused with the residual add + RMSNorm after it: mo [rows][d] (this rank's head share, summed
// over the ranks; mo itself is left as it was), s_out = bf16(res + sum), y = RMSNorm(s_out) * (w + gemma) -- the bits of
// ak_tp_allreduce(mo) + ak_add_rmsnorm(res, mo, s_out, w, y). Returns cudaErrorNotSupported when the push path cannot take
// it (no peer regions, too many rows, d not a TPR 64 shape): the caller then runs the two calls.
extern "C" int ak_tp_allreduce_add_rmsnorm(const void* mo, const void* res, void* s_out, const void* w, void* y, int rows,
                                           int d, float eps, int gemma, void* stream) {
  if (!g_push_ok || d % 8 || d <= 3072 || d > 6144 || (long)rows * d > g_push.elems || rows < 1)
    return (int)cudaErrorNotSupported;
  if (getenv("AGENS_TP_TRACE")) return (int)cudaErrorNotSupported;  // the trace hashes the unfused all-reduce
  const int U = d / 8, threads = (U + 31) / 32 * 32;
  push_ar_norm_kernel<<<g_push.grid, threads < 64 ? 64 : threads, (size_t)U * 16, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(mo), static_cast<const bf16*>(res), static_cast<bf16*>(s_out), static_cast<const bf16*>(w),
      static_cast<bf16*>(y), rows, d, gemma ? 1.f : 0.f, eps, g_push);
  return (int)cudaGetLastError();
}

// Push self-test (AGENS_TP_AR_SELFTEST, peer regions): per round a row count (1, 2, 8, 16, 64, 128, 256, ... within the
// push capacity), varied bf16 data per rank (signs, magnitudes, ties), and
//   (a) push all-reduce vs the pull kernel, bitwise; (b) fused AR + add + RMSNorm vs pull AR + ak_add_rmsnorm, bitwise
//   (s_out and y). Returns the number of differing elements (every rank runs the same calls), -1 when not set up.
namespace {
__global__ void pt_fill(bf16* x, long n, int r, int it, float scale) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)(i * 2654435761u) ^ (unsigned)(r * 40503u + it * 97u + 12345u);
    h ^= h >> 13, h *= 0x5bd1e995u, h ^= h >> 15;
    const float v = ((float)(h & 0xFFFF) / 65536.f - 0.5f) * scale * (float)(1 + (h >> 16) % 7);
    // -0.0 on some ranks and on every rank (the Lamport sentinel value), +0.0 elsewhere
    x[i] = i % 997 == r || i % 991 == 0 ? __ushort_as_bfloat16(0x8000) : (i % 983 == 0 ? __float2bfloat16_rn(0.f) : __float2bfloat16_rn(v));
  }
}
// PDL bench: a memory-bound producer (reads wb bytes, grid-strided, then writes x[n]) -- the shape of a W4 GEMV feeding
// the all-reduce; pdl: it waits on its own predecessor (griddepcontrol) and lets its dependents launch at the start
__global__ void __launch_bounds__(256) pt_prod(const uint4* w, long wn, bf16* x, long n, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (pdl) asm volatile("griddepcontrol.launch_dependents;");
#endif
  uint4 acc = {0, 0, 0, 0};
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < wn; i += (long)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(w + i);
    acc.x ^= v.x, acc.y ^= v.y, acc.z ^= v.z, acc.w ^= v.w;
  }
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    x[i] = __float2bfloat16_rn((float)((acc.x ^ acc.w) & 7) * 0.f + 0.125f);
}
__global__ void pt_nop(bf16* x) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && x == nullptr) x[0] = bf16{};
}
__global__ void pt_diff(const bf16* a, const bf16* b, long n, unsigned* bad) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    if (__bfloat16_as_ushort(a[i]) != __bfloat16_as_ushort(b[i])) atomicAdd(bad, 1u);
}
}  // namespace
extern "C" long ak_tp_push_selftest(int iters) {
  if (!g_push_ok) return -1;
  const int H = 5120;
  const long cap_rows = g_push.elems / H;
  const long n = cap_rows * H;
  bf16 *x0, *x1, *mo, *res, *w, *s0, *s1, *y0, *y1;
  unsigned* bad;
  bf16** all[] = {&x0, &x1, &mo, &res, &s0, &s1, &y0, &y1};
  for (auto p : all) cudaMalloc(p, n * 2);
  cudaMalloc(&w, H * 2), cudaMalloc(&bad, 4);
  cudaMemset(bad, 0, 4);
  pt_fill<<<64, 256>>>(w, H, 0, 7, 0.5f);  // replicated: the same weights on every rank
  const int rset[] = {1, 2, 3, 8, 16, 24, 64, 128, 256, 100, 7};
  for (int it = 0; it < iters; ++it) {
    long rows = rset[it % 11];
    if (rows > cap_rows) rows = cap_rows;
    const long m = rows * H;
    const float scale = it % 3 == 0 ? 1.f : (it % 3 == 1 ? 64.f : 1.f / 256.f);
    pt_fill<<<256, 256>>>(x0, m, g_push.rank, it, scale);
    pt_fill<<<256, 256>>>(res, m, 99, it, 4.f * scale);  // replicated residual
    cudaMemcpyAsync(x1, x0, m * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(mo, x0, m * 2, cudaMemcpyDeviceToDevice);
    g_force_pull = true;
    if (ak_tp_allreduce(x0, m, nullptr)) return -1;
    g_force_pull = false;
    if (ak_tp_allreduce(x1, m, nullptr)) return -1;
    pt_diff<<<256, 256>>>(x0, x1, m, bad);
    // reference: pull-summed x0 + add_rmsnorm; fused from the unsummed mo
    if (ak_add_rmsnorm(res, x0, s0, w, y0, (int)rows, H, 1e-6f, 1, nullptr)) return -1;
    if (ak_tp_allreduce_add_rmsnorm(mo, res, s1, w, y1, (int)rows, H, 1e-6f, 1, nullptr)) return -1;
    pt_diff<<<256, 256>>>(s0, s1, m, bad);
    pt_diff<<<256, 256>>>(y0, y1, m, bad);
  }
  unsigned h = 0;
  cudaMemcpy(&h, bad, 4, cudaMemcpyDeviceToHost);
  for (auto p : all) cudaFree(*p);
  cudaFree(w), cudaFree(bad);
  return cudaGetLastError() == cudaSuccess ? (long)h : -1;
}
// Timing of the fused path (AGENS_TP_AR_BENCH): mean us per ak_tp_allreduce_add_rmsnorm of `rows` rows, and per
// pull AR + ak_add_rmsnorm, as CUDA graphs of `per` back-to-back calls (the engine's verify graphs) when graph = 1.
extern "C" double ak_tp_bench2(int iters, long rows, int which, int graph) {
  const int H = 5120;
  const long m = rows * H;
  bf16 *mo, *res, *w, *s, *y;
  cudaMalloc(&mo, m * 2), cudaMalloc(&res, m * 2), cudaMalloc(&s, m * 2), cudaMalloc(&y, m * 2), cudaMalloc(&w, H * 2);
  cudaMemset(mo, 0, m * 2), cudaMemset(res, 0, m * 2), cudaMemset(w, 0, H * 2);
  cudaStream_t st;
  cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  // the producer's weight read: AGENS_TP_PDL_MB (default 13: a 5120 x 2 x 13824 / tp2 W4 share, ~ the MLP down)
  const long pwn = (long)(getenv("AGENS_TP_PDL_MB") ? atoi(getenv("AGENS_TP_PDL_MB")) : 13) << 16;  // uint4s
  uint4* pw = nullptr;
  if (which >= 5) cudaMalloc(&pw, pwn * 16), cudaMemset(pw, 0, pwn * 16);
  auto one = [&]() {
    switch (which) {
      case 0: g_force_pull = true; ak_tp_allreduce(mo, m, st); g_force_pull = false; break;   // pull AR
      case 1: ak_tp_allreduce(mo, m, st); break;                                               // push AR
      case 2: g_force_pull = true; ak_tp_allreduce(mo, m, st); g_force_pull = false;            // pull AR + add_rmsnorm
              ak_add_rmsnorm(res, mo, s, w, y, (int)rows, H, 1e-6f, 1, st); break;
      case 3: ak_tp_allreduce_add_rmsnorm(mo, res, s, w, y, (int)rows, H, 1e-6f, 1, st); break;  // fused
      case 5: case 6: case 7: case 8: {  // producer (W4-GEMV-sized read) -> push AR; 6: with PDL on both; 7: producer
        // alone; 8: PDL attribute on the AR launch only (the producer unchanged: no trigger, implicit at its end)
        const int pdl = which == 6;
        cudaLaunchConfig_t c{};
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization, at[0].val.programmaticStreamSerializationAllowed = 1;
        c.stream = st, c.attrs = at, c.numAttrs = pdl;
        c.gridDim = dim3(132), c.blockDim = dim3(256);
        cudaLaunchKernelEx(&c, pt_prod, (const uint4*)pw, pwn, mo, m, pdl);
        if (which == 7) break;
        c.gridDim = dim3(g_push.grid), c.blockDim = dim3(256);
        c.numAttrs = which == 6 || which == 8;
        cudaLaunchKernelEx(&c, push_ar_kernel, mo, m / 8, g_push);
        break;
      }
      default: pt_nop<<<1, 32, 0, st>>>(mo); break;                                                // launch floor
    }
  };
  const int per = 144;
  cudaGraphExec_t ge = nullptr;
  if (graph) {
    cudaGraph_t g;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < per; ++i) one();
    cudaStreamEndCapture(st, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaGraphDestroy(g);
  }
  auto run = [&]() { if (graph) cudaGraphLaunch(ge, st); else for (int i = 0; i < per; ++i) one(); };
  for (int i = 0; i < 3; ++i) run();
  cudaStreamSynchronize(st);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  cudaEventRecord(a, st);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(b, st);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  if (ge) cudaGraphExecDestroy(ge);
  cudaStreamDestroy(st);
  cudaFree(mo), cudaFree(res), cudaFree(s), cudaFree(y), cudaFree(w);
  if (pw) cudaFree(pw);
  return cudaGetLastError() == cudaSuccess ? 1000.0 * ms / ((double)iters * per) : -1;
}
