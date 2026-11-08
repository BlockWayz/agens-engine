// Float workspace of the flashinfer paged attention plans (split-KV partials: batch_prefill_tmp_v / tmp_s).
// 1.3.7 fixed it at 128 MiB (sglang's _FI_WORKSPACE_BYTES); the split plan of a long prefill chunk can need more
// (e.g. 144 MB), and flashinfer's allocator then threw across the C ABI, which aborted the process. 1.3.8: the buffer
// starts at AGENS_FI_WORKSPACE_MIB (default 128, what the memory plan reserves) and grows to what a plan needs, up to
// AGENS_FI_WORKSPACE_MAX_MIB (default 1024). The size never changes a plan (the split depends on the SM count and the
// lengths only), so outputs are the same bits. A plan that still fails (past the cap, out of memory, or any
// flashinfer exception) fails the engine call with an error instead of aborting.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace agens_fi_ws {
inline size_t mib_env(const char* n, size_t d) {
  const char* e = getenv(n);
  const long v = e ? atol(e) : 0;
  return v > 0 ? (size_t)v : d;
}
inline size_t init_bytes() {
  static const size_t b = mib_env("AGENS_FI_WORKSPACE_MIB", 128) << 20;
  return b;
}
inline size_t max_bytes() {
  static const size_t b = std::max(mib_env("AGENS_FI_WORKSPACE_MAX_MIB", 1024) << 20, init_bytes());
  return b;
}

// *buf holds at least `need` bytes on cudaSuccess (first call: max(need, init_bytes()); growth rounds up to 64 MiB)
inline cudaError_t ensure(void** buf, size_t* cap, size_t need, const char* who, cudaStream_t s) {
  if (*buf && need <= *cap) return cudaSuccess;
  if (need > max_bytes()) {
    fprintf(stderr, "attn: %s needs a %.0f MiB flashinfer workspace, over AGENS_FI_WORKSPACE_MAX_MIB (%zu)\n", who,
            need / 1048576.0, max_bytes() >> 20);
    return cudaErrorMemoryAllocation;
  }
  size_t want = std::max(need, *buf ? *cap : init_bytes());
  const size_t step = size_t(64) << 20;
  if (*buf) want = std::min((want + step - 1) / step * step, max_bytes());
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (*buf && cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
    fprintf(stderr, "attn: %s needs a %.0f MiB flashinfer workspace (have %zu MiB) inside a CUDA graph capture\n", who,
            need / 1048576.0, *cap >> 20);
    return cudaErrorMemoryAllocation;
  }
  void* nb = nullptr;
  if (*buf) cudaStreamSynchronize(s);  // the old buffer may still be read by work on the stream
  if (cudaMalloc(&nb, want) != cudaSuccess) {
    cudaGetLastError();
    fprintf(stderr, "attn: %s: a %zu MiB flashinfer workspace could not be allocated (out of device memory)\n", who,
            want >> 20);
    return cudaErrorMemoryAllocation;
  }
  if (want > init_bytes())
    fprintf(stderr, "attn: %s flashinfer workspace %zu -> %zu MiB (a plan needs %.0f MiB; AGENS_FI_WORKSPACE_MIB)\n",
            who, *buf ? *cap >> 20 : 0, want >> 20, need / 1048576.0);
  if (*buf) cudaFree(*buf);
  *buf = nb;
  *cap = want;
  return cudaSuccess;
}

// runs an oracle body; an exception (flashinfer's planner throws on a workspace or shape it cannot handle) becomes an
// error code, so the engine call fails instead of the process aborting
template <typename F>
int guarded(const char* who, F&& body) {
  try {
    return body();
  } catch (const std::exception& x) {
    fprintf(stderr, "attn: %s failed: %s\n", who, x.what());
  } catch (...) {
    fprintf(stderr, "attn: %s failed (exception)\n", who);
  }
  return (int)cudaErrorUnknown;
}
}  // namespace agens_fi_ws
