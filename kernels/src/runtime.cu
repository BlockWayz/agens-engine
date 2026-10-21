#include "agens_kernels.h"
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
int ak_device_count(int* n) { return (int)cudaGetDeviceCount(n); }
int ak_set_device(int dev) { return (int)cudaSetDevice(dev); }
int ak_device_name(int dev, char* buf, int len) {
  cudaDeviceProp p;
  cudaError_t e = cudaGetDeviceProperties(&p, dev);
  if (e != cudaSuccess) return (int)e;
  strncpy(buf, p.name, len - 1);
  buf[len - 1] = 0;
  return 0;
}
int ak_malloc(void** ptr, size_t bytes) { return (int)cudaMalloc(ptr, bytes); }
int ak_free(void* ptr) { return (int)cudaFree(ptr); }
int ak_memcpy_h2d(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int ak_memcpy_d2h(void* dst, const void* src, size_t bytes) {
  return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int ak_synchronize(void) { return (int)cudaDeviceSynchronize(); }
int ak_mem_info(size_t* f, size_t* t) { return (int)cudaMemGetInfo(f, t); }
const char* ak_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }
// Pinned, mapped host memory: the returned pointer is valid on host AND device (UVA), for weights that do not fit.
int ak_host_alloc(void** ptr, size_t bytes) { return (int)cudaHostAlloc(ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable); }
int ak_host_free(void* ptr) { return (int)cudaFreeHost(ptr); }
}

// ---- programmatic dependent launch in captured graphs (agens_pdl.cuh: the contract of the listed kernels)
namespace {
// kernels that run ak_pdl_wait() before touching memory of earlier kernels, in every block (agens_pdl.cuh)
const char* const kPdlKernels[] = {
    "w4_mma_kernel", "w4_lean_kernel", "w4_gsplit_kernel", "w4h_xr_kernel", "w4h_kernel", "w4h_bt_kernel",
    "rmsnorm_cute_kernel", "silu_mul_kernel", "silu_mul_part_kernel", "bf16_gemv_bi_kernel", "bf16_gemv_bi_nt_kernel", "push_ar_kernel",
    "push_ar_norm_kernel", "mhc_fused_kernel", "kda_multi_pre_w_kernel", "conv_multi_kernel", "gated_rmsnorm_kernel",
    "qk_prep_kernel", "sl_argmax_part", "sl_combine", "binary_kernel", "sigmoid_f32_kernel", "scale_rows_kernel",
    "embed_kernel", "argmax_part_kernel", "argmax_final_kernel", "rmsnorm_pair_embed_kernel",
    "argmax_final_enc_kernel", "argmax_combine_kernel",
    // BCSA / dense attention plans (oracle/bcsa_local_oracle.cu)
    "decode_plan_kernel", "multi_plan_kernel", "multi_idx_kernel", "single_disable_kernel", "packed_plan_kernel",
    // m1: MTP chain / context, verify step prep / accept / commit, Engram, KV stores
    "mat_prep", "gather_hidden", "store_rows", "set_pending", "chain_prep", "gather_pend", "split_gate",
    "store_step0", "mtp_attn", "mtp_attn_merge", "write_slot", "sl_argmax_bf", "sl_argmax", "sl_gemv", "cand_dot",
    "broadcast_streams", "engram_rows_extend", "gather_engram", "block_keys_decode", "apply_hint", "verify_step_prep",
    "accept_greedy", "engram_prev_commit", "bv_prep", "bv_apply_hint", "bv_engram", "bv_r2t_stage",
    "bv_engram_commit", "samp_chain", "split_q_gate", "store_kv", "decode_step_prep", "engram_prev_update",
    "kda_commit_all_kernel", "kda_verify_fused_kernel", "verify_step_pos", "verify_step_ids",
};
// push all-reduce kernels read their block epoch before the wait (tp_push::begin): only after a listed non-push kernel
// (it lets them launch once it is past its own wait, i.e. once the kernel before it -- maybe the previous push -- is done)
const char* const kPushKernels[] = {"push_ar_kernel", "push_ar_norm_kernel", "mhc_fused_kernel"};
// the kernel's own identifier in a demangled name: "void ns::name<...>(...)" / "name(...)"
bool is_ident(const std::string& dn, const char* id) {
  const size_t n = strlen(id);
  for (size_t p = dn.find(id); p != std::string::npos; p = dn.find(id, p + 1)) {
    const char b = p ? dn[p - 1] : ' ', a = p + n < dn.size() ? dn[p + n] : '(';
    const bool lb = !(b == '_' || (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z'));
    if (lb && (a == '(' || a == '<')) return true;
  }
  return false;
}
// 0: not ours / unknown, 1: PDL kernel, 3: PDL push kernel
int kernel_class(const void* func) {
  static std::mutex mu;
  static std::unordered_map<const void*, int> cache;
  std::lock_guard<std::mutex> l(mu);
  auto it = cache.find(func);
  if (it != cache.end()) return it->second;
  int c = 0;
  const char* name = nullptr;
  if (func && cudaFuncGetName(&name, func) == cudaSuccess && name) {
    int st = 0;
    char* d = abi::__cxa_demangle(name, nullptr, nullptr, &st);
    const std::string dn = d && st == 0 ? std::string(d) : std::string(name);
    free(d);
    // never a library kernel of the same name (torch, flashinfer, CUTLASS, CUB, cuBLAS)
    const bool lib = dn.find("at::") != std::string::npos || dn.find("flashinfer") != std::string::npos ||
                     dn.find("cutlass") != std::string::npos || dn.find("cub::") != std::string::npos ||
                     dn.find("cublas") != std::string::npos;
    for (const char* k : kPdlKernels)
      if (!lib && is_ident(dn, k)) c = 1;
    // flashinfer's merge kernel waits (griddepcontrol.wait) before touching the partial states; before it, it reads
    // only the plan's sequence length, written by the step's first kernels (complete before its predecessor started)
    if (dn.find("flashinfer::") != std::string::npos && is_ident(dn, "PersistentVariableLengthMergeStatesKernel")) c = 1;
    if (c)
      for (const char* k : kPushKernels)
        if (is_ident(dn, k)) c = 3;
  }
  cudaGetLastError();
  cache[func] = c;
  return c;
}
int node_class(cudaGraphNode_t nd, bool* kernel) {
  cudaGraphNodeType t;
  *kernel = false;
  if (cudaGraphNodeGetType(nd, &t) != cudaSuccess || t != cudaGraphNodeTypeKernel) return cudaGetLastError(), 0;
  *kernel = true;
  cudaKernelNodeParams p{};
  if (cudaGraphKernelNodeGetParams(nd, &p) != cudaSuccess) return cudaGetLastError(), 0;
  return kernel_class(p.func);
}
}  // namespace

extern "C" {
// the device is compute capability 9.0 (where the PDL / side-stream / split defaults were measured)
int ak_cc90(void) {
  static const int v = [] {
    int dev = 0, ma = 0, mi = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    return ma == 9 && mi == 0 ? 1 : 0;
  }();
  return v;
}
// AGENS_PDL: 0 off; 1 (default on compute capability 9.0 devices, where it was measured and gated): every programmatic
// edge fires at the upstream kernel's trigger (ak_pdl_launch, after its wait), so one kernel runs ahead of the one
// past its wait; 2 (opt-in, slower on that device: 10.8 vs 9.9 ms per verify step): an edge into a non-push kernel fires
// once every block of the upstream kernel has started, so a chain of kernels can be resident ahead of the running one
// (an edge into a push kernel keeps the trigger: tp_push::begin reads the epoch before its wait, which needs the kernel
// before it past its own wait)
int ak_pdl_on(void) {
  static const int on = [] {
    if (const char* e = getenv("AGENS_PDL")) return e[0] >= '1' && e[0] <= '2' ? e[0] - '0' : 0;
    return ak_cc90();
  }();
  return on;
}
// Programmatic edges for a captured graph (before instantiation): every default kernel -> kernel edge whose downstream
// kernel is on kPdlKernels (a push kernel: only after a listed non-push one). Returns the edges converted, or -(CUDA error).
int ak_graph_pdl(void* graph) {
  cudaGraph_t g = static_cast<cudaGraph_t>(graph);
  size_t n = 0;
  if (cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &n) != cudaSuccess) return -(int)cudaGetLastError();
  if (!n) return 0;
  std::vector<cudaGraphNode_t> from(n), to(n);
  std::vector<cudaGraphEdgeData> ed(n);
  if (cudaGraphGetEdges_v2(g, from.data(), to.data(), ed.data(), &n) != cudaSuccess) return -(int)cudaGetLastError();
  int conv = 0;
  for (size_t i = 0; i < n; ++i) {
    if (ed[i].type != cudaGraphDependencyTypeDefault || ed[i].from_port != cudaGraphKernelNodePortDefault) continue;
    bool kf = false, kt = false;
    const int ct = node_class(to[i], &kt);
    if (!kt || !(ct & 1)) continue;
    const int cf = node_class(from[i], &kf);
    if (!kf || (ct == 3 && cf != 1)) continue;
    cudaGraphEdgeData pe{};
    pe.from_port = ak_pdl_on() == 2 && ct != 3 ? cudaGraphKernelNodePortLaunchCompletion : cudaGraphKernelNodePortProgrammatic;
    pe.to_port = 0;
    pe.type = cudaGraphDependencyTypeProgrammatic;
    if (cudaGraphRemoveDependencies_v2(g, &from[i], &to[i], &ed[i], 1) != cudaSuccess) return -(int)cudaGetLastError();
    if (cudaGraphAddDependencies_v2(g, &from[i], &to[i], &pe, 1) != cudaSuccess) {
      const int e = (int)cudaGetLastError();
      cudaGraphAddDependencies_v2(g, &from[i], &to[i], &ed[i], 1);  // restore the plain edge
      cudaGetLastError();
      return -e;
    }
    ++conv;
  }
  return conv;
}
}
