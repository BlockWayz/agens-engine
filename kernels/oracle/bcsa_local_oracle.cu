#include <algorithm>
// ORACLE TREE: sglv23's BCSA local-branch decode = flashinfer 0.6.15 BatchDecodeWithPagedKVCacheWrapper(NHD, fa2,
// use_tensor_cores=True) -> the batch_prefill fa2 module (AOT build flags: -use_fast_math -O3 -DNDEBUG). This file
// re-creates that module's plan + paged run for Volundr's shape (bf16, head_dim 256, mask none, no posenc/window/
// softcap) from the vendored templates, so tests can compare bits. fi_batch_prefill_config.inc is the generated
// config of that module, copied unmodified (its ffi macros are never expanded here).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <vector>

#include "fi_batch_prefill_config.inc"
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill.cuh>

using namespace flashinfer;
using Variant = DefaultAttention</*custom mask*/ false, false, false, false>;

namespace {
struct Ws {
  void* f = nullptr;
  void* i = nullptr;
  void* pinned = nullptr;
  size_t fbytes = 0, ibytes = 0;
  // plan resident in the int workspace (decode): reused while its inputs are unchanged, like flashinfer's
  // plan-once-per-forward (the kernels only read it). A dense prefill plan overwrites the workspace and clears it.
  std::vector<int> key;
  PrefillPlanInfo plan;
};
Ws g_wsd[16];  // per device
Ws& cur_ws() {
  int d = 0;
  cudaGetDevice(&d);
  return g_wsd[d & 15];
}
}  // namespace
#define g_ws cur_ws()

// q [bs][24][256] bf16; k/v cache [pool][1][kvh][256] bf16 (NHD, page 1); kv_indptr / indices / last_page_len device
// int32; kv_indptr_host + kv_len_host on host. o [bs][24][256] bf16; lse [bs][24] fp32 (log2 domain).
extern "C" int ak_oracle_bcsa_local(void* q, void* kc, void* vc, int pool, int* kv_indptr, int* kv_indices,
                                    int* last_page_len, const int* kv_indptr_host, const int* qo_indptr_host,
                                    int* qo_indptr_dev, int bs, int qo_heads, int kv_heads, float sm_scale, void* o,
                                    float* lse, int cuda_graph, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (!g_ws.f) {
    g_ws.fbytes = 128ull << 20;  // sglv23 _FI_WORKSPACE_BYTES
    g_ws.ibytes = 8ull << 20;    // flashinfer wrapper int workspace
    cudaMalloc(&g_ws.f, g_ws.fbytes);
    cudaMalloc(&g_ws.i, g_ws.ibytes);
    cudaMallocHost(&g_ws.pinned, g_ws.ibytes);
  }
  std::vector<int> key = {bs, qo_heads, kv_heads, cuda_graph};
  key.insert(key.end(), qo_indptr_host, qo_indptr_host + bs + 1);
  key.insert(key.end(), kv_indptr_host, kv_indptr_host + bs + 1);
  if (key != g_ws.key) {
    g_ws.key.clear();
    cudaError_t e = PrefillPlan<int32_t>(g_ws.f, g_ws.fbytes, g_ws.i, g_ws.pinned, g_ws.ibytes, g_ws.plan,
                                         const_cast<int*>(qo_indptr_host), const_cast<int*>(kv_indptr_host),
                                         /*total_num_rows=*/bs, bs, qo_heads, kv_heads, 256, 256, /*page=*/1,
                                         cuda_graph != 0, /*sizeof_dtype_o=*/2, /*window_left=*/-1,
                                         /*fixed_split_size=*/-1, /*disable_split_kv=*/false, 0, s);
    if (e != cudaSuccess) return (int)e;
    // PrefillPlan stages the plan through ONE pinned host buffer with an async H2D copy; the next plan would
    // overwrite it before that copy runs. Same inputs (all BCSA layers of a decode step) reuse the resident plan.
    cudaStreamSynchronize(s);
    g_ws.key = key;
  }
  const PrefillPlanInfo& plan = g_ws.plan;
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, bs, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, kv_indices, kv_indptr, last_page_len);
  p.q_indptr = qo_indptr_dev;
  p.o = static_cast<DTypeO*>(o);
  p.lse = lse;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.logits_soft_cap = 0.0;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  char* ib = static_cast<char*>(g_ws.i);
  char* fb = static_cast<char*>(g_ws.f);
  p.request_indices = reinterpret_cast<IdType*>(ib + plan.request_indices_offset);
  p.qo_tile_indices = reinterpret_cast<IdType*>(ib + plan.qo_tile_indices_offset);
  p.kv_tile_indices = reinterpret_cast<IdType*>(ib + plan.kv_tile_indices_offset);
  p.o_indptr = reinterpret_cast<IdType*>(ib + plan.o_indptr_offset);
  p.kv_chunk_size_ptr = reinterpret_cast<IdType*>(ib + plan.kv_chunk_size_ptr_offset);
  if (plan.split_kv) {
    p.merge_indptr = reinterpret_cast<IdType*>(ib + plan.merge_indptr_offset);
    tmp_v = reinterpret_cast<DTypeO*>(fb + plan.v_offset);
    tmp_s = reinterpret_cast<float*>(fb + plan.s_offset);
    if (plan.enable_cuda_graph) p.block_valid_mask = reinterpret_cast<bool*>(ib + plan.block_valid_mask_offset);
  }
  p.padded_batch_size = plan.padded_batch_size;
  p.max_total_num_rows = plan.total_num_rows;
  if (plan.enable_cuda_graph) p.total_num_rows = reinterpret_cast<uint32_t*>(ib + plan.total_num_rows_offset);
  p.partition_kv = plan.split_kv;
  switch (plan.cta_tile_q) {
    case 16:
      return (int)BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kNone,
                                                         Variant, PagedParams>(p, tmp_v, tmp_s, false, s);
    case 64:
      return (int)BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, false, MaskMode::kNone,
                                                         Variant, PagedParams>(p, tmp_v, tmp_s, false, s);
    case 128:
      return (int)BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, false,
                                                         MaskMode::kNone, Variant, PagedParams>(p, tmp_v, tmp_s,
                                                                                                false, s);
    default:
      return (int)cudaErrorInvalidValue;
  }
}

// Dense layer (71) prefill: sglang's own FlashInferAttnBackend paged prefill wrapper (use_ragged=False for this
// multimodal model class), causal, no window, eager plan. Same non-SWA module as the decode above (mask_1 variant).
extern "C" int ak_oracle_dense_prefill(void* q, void* kc, void* vc, int* qo_indptr_dev, int* kv_indptr, int* kv_indices,
                                       int* last_page_len, const int* qo_indptr_host, const int* kv_indptr_host,
                                       int bs, int qo_heads, int kv_heads, float sm_scale, void* o, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (!g_ws.f) {
    g_ws.fbytes = 128ull << 20;
    g_ws.ibytes = 8ull << 20;
    cudaMalloc(&g_ws.f, g_ws.fbytes);
    cudaMalloc(&g_ws.i, g_ws.ibytes);
    cudaMallocHost(&g_ws.pinned, g_ws.ibytes);
  }
  PrefillPlanInfo plan;
  g_ws.key.clear();  // overwrites the resident decode plan
  cudaError_t e = PrefillPlan<int32_t>(g_ws.f, g_ws.fbytes, g_ws.i, g_ws.pinned, g_ws.ibytes, plan,
                                       const_cast<int*>(qo_indptr_host), const_cast<int*>(kv_indptr_host),
                                       qo_indptr_host[bs], bs, qo_heads, kv_heads, 256, 256, 1, false, 2, -1, -1, false,
                                       0, s);
  if (e != cudaSuccess) return (int)e;
  // PrefillPlan stages the plan through ONE pinned host buffer with an async H2D copy; the next call would
  // overwrite it before that copy runs. (flashinfer plans once per forward; this oracle plans per call.)
  cudaStreamSynchronize(s);
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, bs, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, kv_indices, kv_indptr, last_page_len);
  p.q_indptr = qo_indptr_dev;
  p.o = static_cast<DTypeO*>(o);
  p.lse = nullptr;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  char* ib = static_cast<char*>(g_ws.i);
  char* fb = static_cast<char*>(g_ws.f);
  p.request_indices = reinterpret_cast<IdType*>(ib + plan.request_indices_offset);
  p.qo_tile_indices = reinterpret_cast<IdType*>(ib + plan.qo_tile_indices_offset);
  p.kv_tile_indices = reinterpret_cast<IdType*>(ib + plan.kv_tile_indices_offset);
  p.o_indptr = reinterpret_cast<IdType*>(ib + plan.o_indptr_offset);
  p.kv_chunk_size_ptr = reinterpret_cast<IdType*>(ib + plan.kv_chunk_size_ptr_offset);
  if (plan.split_kv) {
    p.merge_indptr = reinterpret_cast<IdType*>(ib + plan.merge_indptr_offset);
    tmp_v = reinterpret_cast<DTypeO*>(fb + plan.v_offset);
    tmp_s = reinterpret_cast<float*>(fb + plan.s_offset);
  }
  p.padded_batch_size = plan.padded_batch_size;
  p.max_total_num_rows = plan.total_num_rows;
  p.partition_kv = plan.split_kv;
  switch (plan.cta_tile_q) {
    case 16:
      return (int)BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal,
                                                         Variant, PagedParams>(p, tmp_v, tmp_s, false, s);
    case 64:
      return (int)BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal,
                                                         Variant, PagedParams>(p, tmp_v, tmp_s, false, s);
    case 128:
      return (int)BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, false,
                                                         MaskMode::kCausal, Variant, PagedParams>(p, tmp_v, tmp_s,
                                                                                                  false, s);
    default:
      return (int)cudaErrorInvalidValue;
  }
}

// ---------------------------------------------------------------- graph-safe decode attention (M1 decode graphs)
// The decode call above, planned ON DEVICE: one query token (bs = 1), page 1, CUDA-graph plan (what sglv23's decode
// graphs run), KV = indices[kv_range[0] .. kv_range[1]) of a slot map read from device memory. For bs = 1 the
// graph-mode PrefillPlan is integer arithmetic on kv_len alone -- cta_tile_q 16, padded_batch_size = max grid (fixed),
// kv_chunk_size by the same binary search, request/qo_tile = 0, kv_tile = i, block_valid_mask = i < chunks,
// o/merge_indptr = {0, chunks}, total_num_rows = 1 -- so a one-block kernel writes the same plan into the same
// workspace layout and the same kernels run: bit-identical to ak_oracle_bcsa_local(cuda_graph = 1), with no host plan
// or sync per step (capturable). Plans live in slots (e.g. BCSA window, dense); a slot's plan is written by
// ak_attn_decode_plan and read by every ak_attn_decode of that slot until the next plan.
namespace agens_dev_attn {  // named: nvcc's stub file cannot reference kernels in a second anonymous namespace
struct DevPlanSlot {
  bool init = false;
  void *f = nullptr, *i = nullptr, *pin = nullptr;
  int* qo_indptr = nullptr;  // {0, 1}
  int* last = nullptr;       // {1}
  PrefillPlanInfo plan;
  int max_batch_if_split = 0, qo_heads = 0, kv_heads = 0;
};
constexpr int kDevSlots = 32;  // decode: 2 (window, dense); speculative verify: 2 per query
DevPlanSlot g_dev_slot[16][kDevSlots];
// float workspace (split-KV partials) shared by all slots of a device: a call writes and merges it before the next
// call on the stream starts; only the plan (int workspace) is per slot
void* g_dev_fws[16] = {};
constexpr size_t kDevFws = 64ull << 20;

__global__ void decode_plan_kernel(const int* kv_range, int max_batch_if_split, int padded, int group_size,
                                   int* request_indices, int* qo_tile_indices, int* kv_tile_indices, int* o_indptr,
                                   int* kv_chunk_size, int* merge_indptr, bool* block_valid_mask,
                                   uint32_t* total_num_rows) {
  __shared__ int chunk, nchunks;
  if (threadIdx.x == 0) {
    const int64_t kv_len = (int64_t)kv_range[1] - kv_range[0];
    // PrefillBinarySearchKVChunkSize (one request; packed qo len = group_size, cta_tile_q = 16; min chunk 128 / page)
    const int64_t tiles_q = (group_size + 15) / 16;
    int64_t low = 128, high = kv_len > 1 ? kv_len : 1;
    while (low < high) {
      const int64_t mid = (low + high) / 2;
      const int64_t nb = tiles_q * (((kv_len > 1 ? kv_len : 1) + mid - 1) / mid);
      if (nb > max_batch_if_split) low = mid + 1;
      else high = mid;
    }
    chunk = (int)low;
    const int64_t kl = kv_len > 1 ? kv_len : 1;  // PrefillSplitQOKVIndptr: max(int(effective_kv_len), 1)
    nchunks = (int)((kl + low - 1) / low);
    *kv_chunk_size = chunk;
    o_indptr[0] = 0, o_indptr[1] = nchunks;
    merge_indptr[0] = 0, merge_indptr[1] = nchunks;
    *total_num_rows = 1;
  }
  __syncthreads();
  for (int b = threadIdx.x; b < padded; b += blockDim.x) {
    request_indices[b] = 0;
    qo_tile_indices[b] = 0;
    kv_tile_indices[b] = b;
    block_valid_mask[b] = b < nchunks;
  }
}

DevPlanSlot* dev_slot(int slot, int qo_heads, int kv_heads) {
  int d = 0;
  cudaGetDevice(&d);
  if (slot < 0 || slot >= kDevSlots) return nullptr;
  DevPlanSlot& ps = g_dev_slot[d & 15][slot];
  if (!ps.init) {
    ps.qo_heads = qo_heads, ps.kv_heads = kv_heads;
    const size_t fb = kDevFws, ib = 256ull << 10;
    if (!g_dev_fws[d & 15] && cudaMalloc(&g_dev_fws[d & 15], fb)) return nullptr;
    ps.f = g_dev_fws[d & 15];
    if (cudaMalloc(&ps.i, ib) || cudaMallocHost(&ps.pin, ib) || cudaMalloc(&ps.qo_indptr, 8) || cudaMalloc(&ps.last, 4))
      return nullptr;
    const int qo_h[2] = {0, 1}, one = 1, kv_h[2] = {0, 4096};
    cudaMemcpy(ps.qo_indptr, qo_h, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(ps.last, &one, 4, cudaMemcpyHostToDevice);
    // the host plan fixes the layout (offsets depend on padded_batch_size / total_num_rows only, both constant here)
    if (PrefillPlan<int32_t>(ps.f, fb, ps.i, ps.pin, ib, ps.plan, const_cast<int*>(qo_h), const_cast<int*>(kv_h), 1, 1,
                             qo_heads, kv_heads, 256, 256, 1, true, 2, -1, -1, false, 0, nullptr) != cudaSuccess)
      return nullptr;
    cudaDeviceSynchronize();
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d);
    ps.max_batch_if_split = (2 * sms) / kv_heads;  // PrefillPlanImpl: 2 blocks per SM, / num_kv_heads
    if (!ps.plan.split_kv || ps.plan.cta_tile_q != 16 || ps.plan.padded_batch_size != ps.max_batch_if_split) return nullptr;
    ps.init = true;
  }
  return ps.qo_heads == qo_heads && ps.kv_heads == kv_heads ? &ps : nullptr;
}
}  // namespace agens_dev_attn
using namespace agens_dev_attn;

extern "C" int ak_attn_decode_plan(int slot, const int* kv_range, int qo_heads, int kv_heads, void* stream) {
  DevPlanSlot* ps = dev_slot(slot, qo_heads, kv_heads);
  if (!ps) return (int)cudaErrorInvalidValue;
  char* ib = static_cast<char*>(ps->i);
  const PrefillPlanInfo& pl = ps->plan;
  decode_plan_kernel<<<1, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      kv_range, ps->max_batch_if_split, (int)pl.padded_batch_size, qo_heads / kv_heads,
      reinterpret_cast<int*>(ib + pl.request_indices_offset), reinterpret_cast<int*>(ib + pl.qo_tile_indices_offset),
      reinterpret_cast<int*>(ib + pl.kv_tile_indices_offset), reinterpret_cast<int*>(ib + pl.o_indptr_offset),
      reinterpret_cast<int*>(ib + pl.kv_chunk_size_ptr_offset), reinterpret_cast<int*>(ib + pl.merge_indptr_offset),
      reinterpret_cast<bool*>(ib + pl.block_valid_mask_offset), reinterpret_cast<uint32_t*>(ib + pl.total_num_rows_offset));
  return (int)cudaGetLastError();
}

// q [24][256] bf16 (one token); kc/vc [pool][kv_heads][256]; indices = slot map row; kv_range device int32 {lo, hi}
// (used as the paged indptr); o [24][256] bf16; lse [24] fp32 (log2 domain).
extern "C" int ak_attn_decode(int slot, void* q, void* kc, void* vc, int* indices, int* kv_range, int qo_heads,
                              int kv_heads, float sm_scale, void* o, float* lse, void* stream) {
  DevPlanSlot* ps = dev_slot(slot, qo_heads, kv_heads);
  if (!ps) return (int)cudaErrorInvalidValue;
  const PrefillPlanInfo& plan = ps->plan;
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, 1, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, indices, kv_range, ps->last);
  p.q_indptr = ps->qo_indptr;
  p.o = static_cast<DTypeO*>(o);
  p.lse = lse;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.logits_soft_cap = 0.0;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  char* ib = static_cast<char*>(ps->i);
  char* fb = static_cast<char*>(ps->f);
  p.request_indices = reinterpret_cast<IdType*>(ib + plan.request_indices_offset);
  p.qo_tile_indices = reinterpret_cast<IdType*>(ib + plan.qo_tile_indices_offset);
  p.kv_tile_indices = reinterpret_cast<IdType*>(ib + plan.kv_tile_indices_offset);
  p.o_indptr = reinterpret_cast<IdType*>(ib + plan.o_indptr_offset);
  p.kv_chunk_size_ptr = reinterpret_cast<IdType*>(ib + plan.kv_chunk_size_ptr_offset);
  p.merge_indptr = reinterpret_cast<IdType*>(ib + plan.merge_indptr_offset);
  DTypeO* tmp_v = reinterpret_cast<DTypeO*>(fb + plan.v_offset);
  float* tmp_s = reinterpret_cast<float*>(fb + plan.s_offset);
  p.block_valid_mask = reinterpret_cast<bool*>(ib + plan.block_valid_mask_offset);
  p.padded_batch_size = plan.padded_batch_size;
  p.max_total_num_rows = plan.total_num_rows;
  p.total_num_rows = reinterpret_cast<uint32_t*>(ib + plan.total_num_rows_offset);
  p.partition_kv = true;
  return (int)BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kNone,
                                                     Variant, PagedParams>(p, tmp_v, tmp_s, false,
                                                                           static_cast<cudaStream_t>(stream));
}

// ---------------------------------------------------------------- multi-query decode attention (speculative verify)
// T query tokens of one sequence (positions p .. p+T-1), each with its own KV range, in ONE launch of the same kernel,
// each query with exactly its bs = 1 plan: the bs = 1 chunk size (same binary search) and its own chunks, partials
// written per (query, chunk) and merged per query by the same merge kernel -- every output is bit-identical to the
// query's ak_attn_decode. The CTAs are ordered chunk-major (the T queries' tiles of one KV chunk are neighbours), so
// the K/V rows the T queries share are read from DRAM about once (L2). flashinfer takes one kv_chunk_size per launch:
// when the T bs = 1 chunk sizes differ (the length crosses a multiple of the split at this step) the plan disables the
// multi launch on device (no blocks, no merge rows) and leaves the per-query plans of disable_slots enabled;
// otherwise it disables those. KV lists: request t = indices[lo_t, hi_t) copied into a concatenated list.
namespace agens_dev_attn {
struct MultiSlot {
  bool init = false;
  int T = 0, max_len = 0, padded = 0, mbis = 0, qo_heads = 0, kv_heads = 0;
  int *req = nullptr, *qtile = nullptr, *ktile = nullptr, *o_indptr = nullptr, *chunk = nullptr, *merge = nullptr;
  int *q_indptr = nullptr, *kv_indptr = nullptr, *last = nullptr, *idx = nullptr, *flag = nullptr, *lo = nullptr;
  bool* valid = nullptr;
  uint32_t* rows = nullptr;
  void* fws = nullptr;  // T > 8 (batched verify): its own partial-output workspace (padded rows exceed kDevFws)
  int Tcur = 0;         // requests of the current plan (<= T: a batched slot is sized for kMaxT)
};
constexpr int kMultiSlots = 8, kMaxT = 128;
MultiSlot g_multi[16][kMultiSlots];

__device__ __forceinline__ int bs1_chunk(int64_t kv_len, int group_size, int mbis) {  // = decode_plan_kernel's
  const int64_t tiles_q = (group_size + 15) / 16;
  int64_t low = 128, high = kv_len > 1 ? kv_len : 1;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    const int64_t nb = tiles_q * (((kv_len > 1 ? kv_len : 1) + mid - 1) / mid);
    if (nb > mbis) low = mid + 1;
    else high = mid;
  }
  return (int)low;
}
// mode 1 (INT8 dense tiles): T = 1 allowed, and a zero-length request gets no chunks (its merged row: zeros, lse -inf)
__global__ void multi_plan_kernel(int T, const int* rng, int stride, int mbis, int padded, int group_size, int* req,
                                  int* qtile, int* ktile, int* o_indptr, int* chunk, int* merge, int* q_indptr,
                                  int* kv_indptr, int* last, bool* valid, uint32_t* rows, int* flag, int* lo_out,
                                  int mode) {
  __shared__ int nch[kMaxT], ok;
  if (threadIdx.x == 0) {
    // mode 2 (batched verify): every request with its own bs = 1 chunk size (chunk[0] = 0, chunk[1 + t]; the paged
    // kernel reads it per request), so the launch never stands down
    int c0 = -1, all = T > 1 || mode == 1 || mode == 2, off = 0, ooff = 0;
    for (int t = 0; t < T; ++t) {
      const int lo = rng[t * stride], hi = rng[t * stride + 1];
      const int64_t len = hi - lo;
      const int c = bs1_chunk(len, group_size, mbis);
      const bool empty = mode == 1 && len <= 0;
      if (!empty && mode != 2) {
        if (c0 < 0) c0 = c;
        all &= c == c0;
      }
      if (mode == 2) chunk[1 + t] = c;
      const int64_t kl = len > 1 ? len : 1;
      nch[t] = empty ? 0 : (int)((kl + c - 1) / c);
      kv_indptr[t] = off, o_indptr[t] = ooff, merge[t] = ooff, q_indptr[t] = t, last[t] = 1, lo_out[t] = lo;
      off += (int)len, ooff += nch[t];
    }
    kv_indptr[T] = off, o_indptr[T] = ooff, merge[T] = ooff, q_indptr[T] = T;
    if (c0 < 0) c0 = 128;
    *chunk = mode == 2 ? 0 : c0, *flag = all, *rows = all ? T : 0, ok = all;
  }
  __syncthreads();
  for (int b = threadIdx.x; b < padded; b += blockDim.x) {
    const int c = b / T, r = b % T;
    req[b] = r, qtile[b] = 0, ktile[b] = c;
    valid[b] = ok && c < nch[r];
  }
}
__global__ void multi_idx_kernel(int T, const int* flag, const int* kv_indptr, const int* lo, const int* indices, int* idx) {
  if (!*flag) return;
  for (int t = 0; t < T; ++t) {
    const int n = kv_indptr[t + 1] - kv_indptr[t];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
      idx[kv_indptr[t] + i] = indices[lo[t] + i];
  }
}
__global__ void single_disable_kernel(const int* flag, bool* valid, int padded, uint32_t* rows) {
  if (!*flag) return;
  for (int b = threadIdx.x; b < padded; b += blockDim.x) valid[b] = false;
  if (threadIdx.x == 0) *rows = 0;
}
// rows of the batched-verify slots: 8 x AGENS_SPEC_BATCH_MAX sequences (default 16), at most kMaxT
int bv_rows() {
  static const int r = [] {
    const char* e = getenv("AGENS_SPEC_BATCH_MAX");
    const int b = e ? atoi(e) : 16;
    return std::max(16, std::min(kMaxT, 8 * b));
  }();
  return r;
}
MultiSlot* multi_slot(int mslot, int T, int max_len, int qo_heads, int kv_heads) {
  int d = 0;
  cudaGetDevice(&d);
  if (mslot < 0 || mslot >= kMultiSlots || T < 1 || T > kMaxT) return nullptr;
  MultiSlot& m = g_multi[d & 15][mslot];
  const int Treq = T;
  if (!m.init) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d);
    if (T > 8 || mslot >= 6) T = bv_rows();  // batched verify slots (6, 7): one size for every batch
    m.mbis = (2 * sms) / kv_heads, m.T = T, m.max_len = max_len, m.padded = T * m.mbis;
    m.qo_heads = qo_heads, m.kv_heads = kv_heads;
    const size_t P = m.padded;
    if (cudaMalloc(&m.req, P * 4) || cudaMalloc(&m.qtile, P * 4) || cudaMalloc(&m.ktile, P * 4) ||
        cudaMalloc(&m.valid, P) || cudaMalloc(&m.o_indptr, (T + 1) * 4) || cudaMalloc(&m.merge, (T + 1) * 4) ||
        cudaMalloc(&m.q_indptr, (T + 1) * 4) || cudaMalloc(&m.kv_indptr, (T + 1) * 4) || cudaMalloc(&m.last, T * 4) ||
        cudaMalloc(&m.lo, T * 4) || cudaMalloc(&m.chunk, (T + 1) * 4) || cudaMalloc(&m.flag, 4) || cudaMalloc(&m.rows, 4) ||
        cudaMalloc(&m.idx, (size_t)T * max_len * 4))
      return nullptr;
    if (T > 8 && cudaMalloc(&m.fws, P * qo_heads * 256 * 2 + P * qo_heads * 4 + 4096)) return nullptr;
    m.init = true;
  }
  const bool fits = m.T == Treq || (m.T > 8 && Treq <= m.T);
  if (fits) m.Tcur = Treq;
  return fits && m.max_len >= max_len && m.qo_heads == qo_heads && m.kv_heads == kv_heads ? &m : nullptr;
}
}  // namespace agens_dev_attn

extern "C" size_t ak_attn_decode_multi_bytes(int T, int max_len) {
  return (size_t)T * max_len * 4 + (size_t)T * 128 * 16 + 4096;
}
// rng: T ranges {lo, hi} at rng[t * stride]; indices: the slot map the ranges index; disable_slots: the T per-query
// plan slots (ak_attn_decode_plan, already written for this step) to switch off when the multi launch is on, or null.
extern "C" int ak_attn_decode_multi_plan_mode(int mslot, int T, const int* rng, int stride, const int* indices,
                                              int max_len, int qo_heads, int kv_heads, const int* disable_slots, int mode,
                                              void* stream);
extern "C" int ak_attn_decode_multi_plan(int mslot, int T, const int* rng, int stride, const int* indices, int max_len,
                                         int qo_heads, int kv_heads, const int* disable_slots, void* stream) {
  return ak_attn_decode_multi_plan_mode(mslot, T, rng, stride, indices, max_len, qo_heads, kv_heads, disable_slots, 0,
                                        stream);
}
// mode 1: the INT8 dense-layer tiles (T = 1 allowed, empty requests)
extern "C" int ak_attn_decode_multi_plan_mode(int mslot, int T, const int* rng, int stride, const int* indices,
                                              int max_len, int qo_heads, int kv_heads, const int* disable_slots, int mode,
                                              void* stream) {
  MultiSlot* m = multi_slot(mslot, T, max_len, qo_heads, kv_heads);
  if (!m) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  multi_plan_kernel<<<1, 256, 0, s>>>(T, rng, stride, m->mbis, m->padded, qo_heads / kv_heads, m->req, m->qtile,
                                      m->ktile, m->o_indptr, m->chunk, m->merge, m->q_indptr, m->kv_indptr, m->last,
                                      m->valid, m->rows, m->flag, m->lo, mode);
  multi_idx_kernel<<<256, 256, 0, s>>>(T, m->flag, m->kv_indptr, m->lo, indices, m->idx);
  if (disable_slots)
    for (int t = 0; t < T; ++t) {
      DevPlanSlot* ps = dev_slot(disable_slots[t], qo_heads, kv_heads);
      if (!ps) return (int)cudaErrorInvalidValue;
      char* ib = static_cast<char*>(ps->i);
      single_disable_kernel<<<1, 128, 0, s>>>(m->flag, reinterpret_cast<bool*>(ib + ps->plan.block_valid_mask_offset),
                                              (int)ps->plan.padded_batch_size,
                                              reinterpret_cast<uint32_t*>(ib + ps->plan.total_num_rows_offset));
    }
  return (int)cudaGetLastError();
}
// q [T][24][256], o [T][24][256], lse [T][24]: as T ak_attn_decode calls with the planned ranges (no-op when disabled)
extern "C" int ak_attn_decode_multi(int mslot, void* q, void* kc, void* vc, int qo_heads, int kv_heads, float sm_scale,
                                    void* o, float* lse, void* stream) {
  int d = 0;
  cudaGetDevice(&d);
  if (mslot < 0 || mslot >= kMultiSlots) return (int)cudaErrorInvalidValue;
  MultiSlot* m = &g_multi[d & 15][mslot];
  if (!m->init || !(m->fws || g_dev_fws[d & 15])) return (int)cudaErrorInvalidValue;
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, m->Tcur, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, m->idx, m->kv_indptr, m->last);
  p.q_indptr = m->q_indptr;
  p.o = static_cast<DTypeO*>(o);
  p.lse = lse;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.logits_soft_cap = 0.0;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  p.request_indices = m->req;
  p.qo_tile_indices = m->qtile;
  p.kv_tile_indices = m->ktile;
  p.o_indptr = m->o_indptr;
  p.kv_chunk_size_ptr = m->chunk;
  p.merge_indptr = m->merge;
  char* fb = static_cast<char*>(m->fws ? m->fws : g_dev_fws[d & 15]);
  DTypeO* tmp_v = reinterpret_cast<DTypeO*>(fb);                                       // <= T * 64 partial rows
  float* tmp_s = reinterpret_cast<float*>(fb + (size_t)m->padded * qo_heads * 256 * 2);
  p.block_valid_mask = m->valid;
  // the CTAs of the planned rows only: CTA b = (chunk b / Tcur, request b % Tcur) is valid only for chunk < nch <= mbis,
  // so CTAs >= Tcur * mbis never are -- a batched slot sized for kMaxT rows launched all T_slot * mbis of them (16896 at
  // tp2 for one sequence's 8 rows: ~50 us per BCSA window layer). Same CTAs do the work, same bits.
  p.padded_batch_size = (m->Tcur > 0 && m->Tcur < m->T) ? m->Tcur * m->mbis : m->padded;
  p.max_total_num_rows = m->Tcur;
  p.total_num_rows = m->rows;
  p.partition_kv = true;
  return (int)BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kNone,
                                                     Variant, PagedParams>(p, tmp_v, tmp_s, false,
                                                                           static_cast<cudaStream_t>(stream));
}

// ---------------------------------------------------------------- packed verify attention (dense layer)
// The T verify queries of one sequence as ONE causal request (qo_len = T, kv = indices[lo, hi_{T-1}) with lo = 0 for
// every query): each CTA holds 16 packed (query, head) rows, so a KV chunk is read ceil(T*group/16) times instead of T.
// Bit-identical to the T bs = 1 decode calls when the T bs = 1 chunk sizes agree (checked on device; otherwise this
// stands down and the per-query plans of disable_slots stay on): the chunks start at 0 for every query (no window),
// each row's arithmetic does not depend on the other rows of its tile, and positions past a row's own end are masked
// to exact zeros (an extra fully masked chunk merges with weight 0).
namespace agens_dev_attn {
__global__ void packed_plan_kernel(int T, const int* rng, int stride, int mbis, int qtiles, int padded, int group_size,
                                   int* req, int* qtile, int* ktile, int* o_indptr, int* chunk, int* merge,
                                   int* q_indptr, int* kv_indptr, int* last, bool* valid, uint32_t* rows, int* flag) {
  __shared__ int nch, ok;
  if (threadIdx.x == 0) {
    int c0 = 0, all = T > 1;
    for (int t = 0; t < T; ++t) {
      const int lo = rng[t * stride], hi = rng[t * stride + 1];
      const int c = bs1_chunk((int64_t)hi - lo, group_size, mbis);
      if (t == 0) c0 = c;
      all &= c == c0 && lo == rng[0];
    }
    const int lo = rng[(T - 1) * stride], hi = rng[(T - 1) * stride + 1];
    const int kl = hi - lo > 1 ? hi - lo : 1;
    nch = (kl + c0 - 1) / c0;
    all &= nch * qtiles <= padded;
    kv_indptr[0] = lo, kv_indptr[1] = hi, q_indptr[0] = 0, q_indptr[1] = T, last[0] = 1;
    o_indptr[0] = 0, o_indptr[1] = T * nch;
    for (int t = 0; t <= T; ++t) merge[t] = t * nch;
    *chunk = c0, *flag = all, *rows = all ? T : 0, ok = all;
  }
  __syncthreads();
  for (int b = threadIdx.x; b < padded; b += blockDim.x) {
    const int c = b / qtiles, qt = b % qtiles;  // chunk-major: a chunk's q tiles are neighbours (L2)
    req[b] = 0, qtile[b] = qt, ktile[b] = c;
    valid[b] = ok && c < nch;
  }
}
}  // namespace agens_dev_attn

// indices: the slot map the ranges index (the dense layer: S.r2t); rng: T ranges {0, p_t + 1} at rng[t * stride]
extern "C" int ak_attn_decode_packed_plan(int mslot, int T, const int* rng, int stride, int qo_heads, int kv_heads,
                                          const int* disable_slots, void* stream) {
  MultiSlot* m = multi_slot(mslot, T, 1, qo_heads, kv_heads);
  if (!m) return (int)cudaErrorInvalidValue;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int group = qo_heads / kv_heads, qtiles = (T * group + 15) / 16;
  packed_plan_kernel<<<1, 256, 0, s>>>(T, rng, stride, m->mbis, qtiles, m->padded, group, m->req, m->qtile, m->ktile,
                                       m->o_indptr, m->chunk, m->merge, m->q_indptr, m->kv_indptr, m->last, m->valid,
                                       m->rows, m->flag);
  if (disable_slots)
    for (int t = 0; t < T; ++t) {
      DevPlanSlot* ps = dev_slot(disable_slots[t], qo_heads, kv_heads);
      if (!ps) return (int)cudaErrorInvalidValue;
      char* ib = static_cast<char*>(ps->i);
      single_disable_kernel<<<1, 128, 0, s>>>(m->flag, reinterpret_cast<bool*>(ib + ps->plan.block_valid_mask_offset),
                                              (int)ps->plan.padded_batch_size,
                                              reinterpret_cast<uint32_t*>(ib + ps->plan.total_num_rows_offset));
    }
  return (int)cudaGetLastError();
}
// q [T][24][256] (the request's qo rows), o / lse as T ak_attn_decode calls (no-op when the plan stood down)
extern "C" int ak_attn_decode_packed(int mslot, void* q, void* kc, void* vc, int* indices, int qo_heads, int kv_heads,
                                     float sm_scale, void* o, float* lse, void* stream) {
  int d = 0;
  cudaGetDevice(&d);
  if (mslot < 0 || mslot >= kMultiSlots) return (int)cudaErrorInvalidValue;
  MultiSlot* m = &g_multi[d & 15][mslot];
  if (!m->init || !g_dev_fws[d & 15]) return (int)cudaErrorInvalidValue;
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, 1, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, indices, m->kv_indptr, m->last);
  p.q_indptr = m->q_indptr;
  p.o = static_cast<DTypeO*>(o);
  p.lse = lse;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.logits_soft_cap = 0.0;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  p.request_indices = m->req;
  p.qo_tile_indices = m->qtile;
  p.kv_tile_indices = m->ktile;
  p.o_indptr = m->o_indptr;
  p.kv_chunk_size_ptr = m->chunk;
  p.merge_indptr = m->merge;
  char* fb = static_cast<char*>(g_dev_fws[d & 15]);
  DTypeO* tmp_v = reinterpret_cast<DTypeO*>(fb);  // <= T * 64 partial rows of 24 x 256
  float* tmp_s = reinterpret_cast<float*>(fb + (size_t)m->T * m->mbis * qo_heads * 256 * 2);
  p.block_valid_mask = m->valid;
  p.padded_batch_size = m->padded;
  p.max_total_num_rows = m->T;
  p.total_num_rows = m->rows;
  p.partition_kv = true;
  return (int)BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal,
                                                     Variant, PagedParams>(p, tmp_v, tmp_s, false,
                                                                           static_cast<cudaStream_t>(stream));
}

// ---------------------------------------------------------------- prefill over a staged KV list, with lse
// q [T][24][256] against kv rows indices[0, kv_len) of kc/vc (one request; host plan), causal (the chunk's own keys,
// kv_len == T) or no mask (keys before the chunk); o [T][24][256] bf16, lse [T][24] (log2). INT8 dense-layer prefill.
extern "C" int ak_prefill_staged_lse(void* q, void* kc, void* vc, int* indices, int T, int kv_len, int causal,
                                     int qo_heads, int kv_heads, float sm_scale, void* o, float* lse, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (!g_ws.f) {
    g_ws.fbytes = 128ull << 20;
    g_ws.ibytes = 8ull << 20;
    cudaMalloc(&g_ws.f, g_ws.fbytes);
    cudaMalloc(&g_ws.i, g_ws.ibytes);
    cudaMallocHost(&g_ws.pinned, g_ws.ibytes);
  }
  static int *qd = nullptr, *kd = nullptr, *ld = nullptr, *hp = nullptr;
  if (!qd) {
    cudaMalloc(&qd, 8), cudaMalloc(&kd, 8), cudaMalloc(&ld, 4);
    cudaMallocHost(&hp, 32);
  }
  int qh[2] = {0, T}, kh[2] = {0, kv_len};
  cudaStreamSynchronize(s);  // hp is reused per call
  hp[0] = 0, hp[1] = T, hp[2] = 0, hp[3] = kv_len, hp[4] = 1;
  cudaMemcpyAsync(qd, hp, 8, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(kd, hp + 2, 8, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(ld, hp + 4, 4, cudaMemcpyHostToDevice, s);
  PrefillPlanInfo plan;
  g_ws.key.clear();
  cudaError_t e = PrefillPlan<int32_t>(g_ws.f, g_ws.fbytes, g_ws.i, g_ws.pinned, g_ws.ibytes, plan, qh, kh, T, 1,
                                       qo_heads, kv_heads, 256, 256, 1, false, 2, -1, -1, false, 0, s);
  if (e != cudaSuccess) return (int)e;
  cudaStreamSynchronize(s);
  PagedParams p;
  memset(&p, 0, sizeof(p));
  p.q = static_cast<DTypeQ*>(q);
  const int64_t strides[4] = {(int64_t)kv_heads * 256, (int64_t)kv_heads * 256, 256, 1};
  p.paged_kv = paged_kv_t<DTypeKV, IdType>(kv_heads, 1, 256, 1, QKVLayout::kNHD, static_cast<DTypeKV*>(kc),
                                           static_cast<DTypeKV*>(vc), strides, indices, kd, ld);
  p.q_indptr = qd;
  p.o = static_cast<DTypeO*>(o);
  p.lse = lse;
  p.num_qo_heads = qo_heads;
  p.group_size = uint_fastdiv(qo_heads / kv_heads);
  p.q_stride_n = qo_heads * 256;
  p.q_stride_h = 256;
  p.window_left = -1;
  p.sm_scale = sm_scale;
  p.rope_rcp_scale = 1.0;
  p.rope_rcp_theta = 1.0 / 1e4;
  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  char* ib = static_cast<char*>(g_ws.i);
  char* fb = static_cast<char*>(g_ws.f);
  p.request_indices = reinterpret_cast<IdType*>(ib + plan.request_indices_offset);
  p.qo_tile_indices = reinterpret_cast<IdType*>(ib + plan.qo_tile_indices_offset);
  p.kv_tile_indices = reinterpret_cast<IdType*>(ib + plan.kv_tile_indices_offset);
  p.o_indptr = reinterpret_cast<IdType*>(ib + plan.o_indptr_offset);
  p.kv_chunk_size_ptr = reinterpret_cast<IdType*>(ib + plan.kv_chunk_size_ptr_offset);
  if (plan.split_kv) {
    p.merge_indptr = reinterpret_cast<IdType*>(ib + plan.merge_indptr_offset);
    tmp_v = reinterpret_cast<DTypeO*>(fb + plan.v_offset);
    tmp_s = reinterpret_cast<float*>(fb + plan.s_offset);
  }
  p.padded_batch_size = plan.padded_batch_size;
  p.max_total_num_rows = plan.total_num_rows;
  p.partition_kv = plan.split_kv;
#define AK_PF(TQ, MASK) BatchPrefillWithPagedKVCacheDispatched<TQ, 256, 256, PosEncodingMode::kNone, false, MASK, Variant, \
                                                               PagedParams>(p, tmp_v, tmp_s, false, s)
  switch (plan.cta_tile_q * 2 + (causal ? 1 : 0)) {
    case 32: return (int)AK_PF(16, MaskMode::kNone);
    case 33: return (int)AK_PF(16, MaskMode::kCausal);
    case 128: return (int)AK_PF(64, MaskMode::kNone);
    case 129: return (int)AK_PF(64, MaskMode::kCausal);
    case 256: return (int)AK_PF(128, MaskMode::kNone);
    case 257: return (int)AK_PF(128, MaskMode::kCausal);
    default: return (int)cudaErrorInvalidValue;
  }
#undef AK_PF
}
