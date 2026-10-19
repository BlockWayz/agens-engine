// KV tiering + INT8 KV for the M1 engine's BCSA layers (see m1_kv.h, docs/kv_tiering.md). Only reached when
// AGENS_KV_TIER=1 or AGENS_KV_INT8=1.
#include "m1_kv.h"
#include "agens_kv.h"
#include <cuda_bf16.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <map>
#include <vector>

extern "C" {
int ak_oracle_bcsa_local(void* q, void* kc, void* vc, int pool, int* kv_indptr, int* kv_indices, int* last_page_len,
                         const int* kv_indptr_host, const int* qo_indptr_host, int* qo_indptr_dev, int bs, int qo_heads,
                         int kv_heads, float sm_scale, void* o, float* lse, int cuda_graph, void* stream);
int ak_oracle_bcsa_prefill(void* q, void* kc, void* vc, int* qo_indptr_dev, int* kv_indptr, int* kv_indices,
                           int* last_page_len, const int* qo_indptr_host, const int* kv_indptr_host, int bs, int qo_heads,
                           int kv_heads, int window_left, float sm_scale, void* o, float* lse, void* stream);
int ak_bridge_far_select_decode(void* qi, void* pos, void* reqs, void* r2t, int R, int Wctx, int rows, void* blk, int bs,
                                void* sel_rows, void* sel_ok, void* stream);
int ak_bridge_far_attend_decode(void* qh, void* o_loc, void* lse, void* kst, void* vst, void* sel_ok, int bs, int ksel,
                                int means, void* out, void* stream);
int ak_attn_decode(int slot, void* q, void* kc, void* vc, int* indices, int* kv_range, int qo_heads, int kv_heads,
                   float sm_scale, void* o, float* lse, void* stream);
int ak_attn_decode_multi(int mslot, void* q, void* kc, void* vc, int qo_heads, int kv_heads, float sm_scale, void* o,
                         float* lse, void* stream);
int ak_far_decode_src(const void* qi, const void* qh, const void* o_loc, const float* lse, const long long* pos,
                      const int* r2t, int Wctx, int src, const void* kc, const void* vc, const float* ksc,
                      const float* vsc, const void* blk, int rows, void* ws, void* out, void* stream);
int ak_far_decode_multi_select(int T, const void* qi, const long long* pos, const int* r2t, int Wctx, const void* blk,
                               int rows, void* ws, void* stream);
int ak_far_decode_multi_fetch(int T, const long long* pos, const int* r2t, int Wctx, int src, const void* kc,
                              const void* vc, const float* ksc, const float* vsc, int rows, void* ws, void* stream);
int ak_far_decode_multi_fetch_cached(int T, const int* r2t, int Wctx, const void* km, const void* vm, int rows,
                                     void* ws, void* cache, int cap, const long long* tag, void* stream);
size_t ak_far_cache_bytes(int Wctx, int cap);
int ak_far_cache_init(void* cache, int Wctx, int cap, void* stream);
int ak_far_decode_multi_attend(int T, const void* qh, const void* o_loc, const float* lse, const long long* pos, int Wctx,
                               void* ws, void* out, void* stream);
int ak_far_decode_multi_src(int T, const void* qi, const void* qh, const void* o_loc, const float* lse,
                            const long long* pos, const int* r2t, int Wctx, int src, const void* kc, const void* vc,
                            const float* ksc, const float* vsc, const void* blk, int rows, void* ws, void* out,
                            void* stream);
size_t ak_far_extend_engine_ws_bytes(int Wctx);
int ak_far_extend_engine(int T, int seq0, const void* qi, const void* qh, const void* o_loc, const float* lse,
                         const int* r2t, int Wctx, const void* kst, const void* vst, const void* idx, void* blk, int rows,
                         void* ws, void* out, void* stream);
int ak_bridge_far_extend_staged(void* qi_all, void* qh, void* o_loc, void* lse, int seq_len, int T, int req, void* r2t,
                                int R, int Wctx, void* kst, void* vst, int n_rows, int means, void* idx, void* blk,
                                int rows, void* out, void* stream);
}

namespace m1kv {
namespace {
constexpr int QH = 24, KVH = 4, HD = 256, WIN = 4096, CHUNK = 4096, C = 4, KSEL_MAX = 512;
// positions kept per session per layer on the GPU: the window + a speculative verify block (8 tokens stored before
// the first of them attends: none may overwrite a position in the window of an earlier one)
constexpr int RING = WIN + 8;
constexpr int STAGE = WIN + CHUNK;           // local staging rows (int8): window + chunk, then one zero row
constexpr int BF_ROW = KVH * HD * 2, I8_ROW = KVH * HD, SC_ROW = KVH * 4;

#define CK(x)                                                                         \
  do {                                                                                \
    int e_ = (int)(x);                                                                \
    if (e_) {                                                                         \
      fprintf(stderr, "m1kv: %s failed (%d) at %s:%d\n", #x, e_, __FILE__, __LINE__); \
      return e_;                                                                      \
    }                                                                                 \
  } while (0)

int env_on(const char* n) {
  const char* e = getenv(n);
  return e && e[0] && e[0] != '0';
}

struct Pool {        // one K or V pool / window: bf16 rows, or int8 rows + fp32 scales
  void* rows = nullptr;
  float* scale = nullptr;
};
struct Layer {
  int dev = 0, li = 0;  // li = index among the device's BCSA layers (ring block)
  Pool k, v;            // bulk pool [rows+1]
  Pool km, vm;          // AGENS_KV_TIER=2: 4-row block means, at the pool row of each block's last token (host)
};
struct Cache {  // AGENS_KV_CACHE=1: last step's staged block means per (layer, session), double-buffered
  __nv_bfloat16 *k[2] = {}, *v[2] = {};
  int* keys[2] = {};
  int cur = 0, last_seq = -2, gen = -1;
  bool valid = false;
};
struct Dev {
  int nb = 0, nslots = 1, wrows = 0, ksel = 0, wctx = 0;
  std::map<long, Cache> cache;
  int *slot_of = nullptr, *misses = nullptr;  // pool row -> cache slot (-1), miss counter
  __nv_bfloat16 *att_k = nullptr, *att_v = nullptr;  // what far_attend reads (staging or a cache buffer)
  long miss_rows = 0, req_rows = 0;
  Pool kw, vw;                       // tier: rings [nb][nslots][RING] + chunk area [CHUNK] + zero row
  __nv_bfloat16 *kst = nullptr, *vst = nullptr;  // int8: local staging [STAGE + 1]
  __nv_bfloat16 *kf = nullptr, *vf = nullptr;    // far decode staging [ksel * C]
  __nv_bfloat16 *ke = nullptr, *ve = nullptr;    // far extend staging, grown on demand
  size_t erows = 0;
  int *tmp = nullptr, *tmp2 = nullptr, *sel_rows = nullptr, *sel_last = nullptr;
  int* ringidx = nullptr;  // graph-safe decode: ringidx[p] = p % RING (the window as paged indices into a ring slot)
  int* dtmp = nullptr;     // graph-safe decode: device-computed row lists (ring row, block rows + page)
  int* ringmap_b = nullptr;  // batched decode: per row, position -> row of the row's ring slot (layer-relative)
  int* brows = nullptr;      // batched decode: per row {ring row, 4 block rows, mean page} x kMaxBatch
  bool* sel_ok = nullptr;
  cudaStream_t side = nullptr;
  cudaEvent_t ev_sel = nullptr, ev_fetch = nullptr, ev_t0 = nullptr;
  cudaEvent_t ev_ms = nullptr, ev_mf = nullptr;  // verify union fetch: fork / join (capturable, no timing)
  size_t dev_bytes = 0, host_bytes = 0;
  // AGENS_KV_STATS=1: fetch time + overlap of each step's selection with the previous step's (same layer + slot)
  std::map<long, std::vector<int>> prev_sel;
  double fetch_ms = 0, overlap = 0;
  long fetches = 0, overlap_n = 0;
};
std::map<int, Layer> g_layer;
Dev g_dev[2];

template <typename T>
T* dzero(size_t n, Dev& d) {
  void* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T) > 0 ? n * sizeof(T) : 16) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  d.dev_bytes += n * sizeof(T);
  return (T*)p;
}
void* hzero(size_t bytes, Dev& d) {  // pinned + mapped + portable: the pointer is valid on the host and every device
  void* p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
  memset(p, 0, bytes);
  d.host_bytes += bytes;
  return p;
}
int alloc_pool(Pool& p, size_t nrows, bool host, Dev& d) {
  const size_t rb = int8() ? I8_ROW : BF_ROW;
  p.rows = host ? hzero(nrows * rb, d) : (void*)dzero<char>(nrows * rb, d);
  if (!p.rows) return (int)cudaErrorMemoryAllocation;
  if (int8()) {
    p.scale = host ? (float*)hzero(nrows * SC_ROW, d) : dzero<float>(nrows * KVH, d);
    if (!p.scale) return (int)cudaErrorMemoryAllocation;
  }
  return 0;
}
// rows of `src` (list or identity) -> rows of `dst` (list or identity), bf16 or int8 (+ scales): data movement
int move_rows(const Pool& src, const int* srows, const Pool& dst, const int* drows, int n, cudaStream_t s) {
  if (!int8()) return ak_kv_copy_rows(src.rows, srows, dst.rows, drows, n, BF_ROW, s);
  CK(ak_kv_copy_rows(src.rows, srows, dst.rows, drows, n, I8_ROW, s));
  return ak_kv_copy_rows(src.scale, srows, dst.scale, drows, n, SC_ROW, s);
}
// rows of a pool -> bf16 staging (identity order): a gather (bf16) or a gather + dequant (int8)
int fetch_rows(const Pool& src, const int* srows, int n, __nv_bfloat16* dst, cudaStream_t s) {
  if (!int8()) return ak_kv_copy_rows(src.rows, srows, dst, nullptr, n, BF_ROW, s);
  return ak_kv_dequant_rows((const int8_t*)src.rows, src.scale, srows, n, KVH, HD, dst, s);
}
int write_rows(const void* x, int T, const int* drows, const Pool& dst, cudaStream_t s) {  // new tokens -> pool rows
  if (!int8()) return ak_kv_copy_rows(x, nullptr, dst.rows, drows, T, BF_ROW, s);
  return ak_kv_quant_store(x, T, KVH, HD, drows, (int8_t*)dst.rows, dst.scale, s);
}
int ring_base(const Dev& d, int li, int kslot) { return (li * d.nslots + (kslot - 1)) * RING; }
int chunk_base(const Dev& d) { return d.nb * d.nslots * RING; }
int zero_row(const Dev& d) { return chunk_base(d) + CHUNK; }

void report() {
  for (int k = 0; k < 2; ++k) {
    Dev& d = g_dev[k];
    if (!d.nb) continue;
    fprintf(stderr, "m1kv dev %d: %d BCSA layers, tier %d int8 %d: GPU %.1f MB, host %.1f MB", k, d.nb, tier(), int8(),
            d.dev_bytes / 1e6, d.host_bytes / 1e6);
    if (d.fetches)
      fprintf(stderr, "; far fetch %.3f ms avg over %ld (%.2f MB each), selection overlap with previous step %.1f%%",
              d.fetch_ms / d.fetches, d.fetches,
              (double)d.ksel * (means() ? 1 : C) * (int8() ? I8_ROW + SC_ROW : BF_ROW) * 2 / 1e6,
              d.overlap_n ? 100.0 * d.overlap / d.overlap_n : 0.0);
    if (d.req_rows)
      fprintf(stderr, "; block cache: %.1f%% of selected blocks fetched over PCIe", 100.0 * d.miss_rows / d.req_rows);
    fprintf(stderr, "\n");
  }
}
}  // namespace

bool tier() {
  static int v = env_on("AGENS_KV_TIER");
  return v;
}
bool int8() {
  static int v = env_on("AGENS_KV_INT8");
  return v;
}
std::map<int, int> g_gen;  // per KDA slot: bumped when the slot's session starts over, restores or is forked into
bool cache_on();
bool means() {
  static int v = tier() && !int8() && getenv("AGENS_KV_TIER")[0] == '2';
  return v;
}

bool cache_on() {
  static int v = means() && env_on("AGENS_KV_CACHE");
  return v;
}

int alloc_layer(int dev, int layer, int rows, void** kc, void** vc) {
  Dev& d = g_dev[dev];
  Layer& L = g_layer[layer];
  L.dev = dev, L.li = d.nb++;
  CK(alloc_pool(L.k, (size_t)rows + 1, tier(), d));
  CK(alloc_pool(L.v, (size_t)rows + 1, tier(), d));
  if (means()) {
    CK(alloc_pool(L.km, (size_t)rows / C + 1, true, d));  // one mean row per 4-row page (+ trash page)
    CK(alloc_pool(L.vm, (size_t)rows / C + 1, true, d));
  }
  *kc = int8() ? nullptr : L.k.rows;
  *vc = int8() ? nullptr : L.v.rows;
  return 0;
}

int init_device(int dev, int nslots, int wctx) {
  Dev& d = g_dev[dev];
  d.nslots = nslots;
  d.ksel = std::min(KSEL_MAX, wctx / C);
  if (tier()) {
    d.wrows = zero_row(d) + 1;
    CK(alloc_pool(d.kw, d.wrows, false, d));
    CK(alloc_pool(d.vw, d.wrows, false, d));
  }
  if (int8()) {
    if (!(d.kst = dzero<__nv_bfloat16>((size_t)(STAGE + 1) * KVH * HD, d))) return (int)cudaErrorMemoryAllocation;
    if (!(d.vst = dzero<__nv_bfloat16>((size_t)(STAGE + 1) * KVH * HD, d))) return (int)cudaErrorMemoryAllocation;
  }
  d.kf = dzero<__nv_bfloat16>((size_t)d.ksel * C * KVH * HD, d);
  d.vf = dzero<__nv_bfloat16>((size_t)d.ksel * C * KVH * HD, d);
  d.tmp = dzero<int>(std::max(wctx, STAGE + 1), d), d.tmp2 = dzero<int>(STAGE + 1, d);
  d.sel_rows = dzero<int>((size_t)d.ksel * C, d);
  d.sel_last = dzero<int>(d.ksel, d);
  d.dtmp = dzero<int>(64, d);  // store_dec: 8; store_dec_multi: 6 per token (T <= 8)
  d.brows = dzero<int>(16 * 8, d);
  if (tier() && !(d.ringmap_b = dzero<int>((size_t)16 * wctx, d))) return (int)cudaErrorMemoryAllocation;
  d.wctx = wctx;
  if (tier()) {
    if (!(d.ringidx = dzero<int>(wctx, d))) return (int)cudaErrorMemoryAllocation;
    CK(ak_kv_window_rows(0, wctx, 0, 0, 0, RING, INT_MAX, 0, d.ringidx, nullptr));
  }
  d.sel_ok = dzero<bool>(d.ksel, d);
  if (!d.kf || !d.vf || !d.tmp || !d.tmp2 || !d.sel_rows || !d.sel_ok) return (int)cudaErrorMemoryAllocation;
  CK(cudaStreamCreateWithFlags(&d.side, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&d.ev_sel, cudaEventDisableTiming));
  CK(cudaEventCreate(&d.ev_fetch));
  CK(cudaEventCreateWithFlags(&d.ev_ms, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&d.ev_mf, cudaEventDisableTiming));
  CK(cudaEventCreate(&d.ev_t0));
  static bool once = false;
  if (!once) atexit(report), once = true;
  fprintf(stderr, "m1kv dev %d: tier %d int8 %d, %d BCSA layers: GPU %.1f MB (window rings %d slots), host %.1f MB\n",
          dev, tier(), int8(), d.nb, d.dev_bytes / 1e6, nslots, d.host_bytes / 1e6);
  return (int)cudaDeviceSynchronize();
}

int store(const Ctx& c, const void* k, const void* v, const int* loc) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  if (c.prefill || c.seq0 == 0) ++g_gen[c.kslot];  // block cache: the session's rows may be new or reused
  CK(write_rows(k, c.T, loc, L.k, c.s));
  CK(write_rows(v, c.T, loc, L.v, c.s));
  if (tier()) {  // decode: the ring slot of the new position; prefill: the chunk area (committed to the ring later)
    CK(ak_kv_window_rows(c.seq0, c.T, 0, zero_row(d), ring_base(d, L.li, c.kslot), RING, c.prefill ? c.seq0 : INT_MAX,
                         chunk_base(d), d.tmp, c.s));
    CK(write_rows(k, c.T, d.tmp, d.kw, c.s));
    CK(write_rows(v, c.T, d.tmp, d.vw, c.s));
  }
  if (means()) {  // blocks closed by this call (last position 4n+3 in it): mean of their 4 rows (ring / chunk area),
                  // stored at the pool row of the block's last token
    const int n0 = c.seq0 / C, n1 = (c.seq0 + c.T) / C;
    if (n1 > n0) {
      CK(ak_kv_window_rows(C * n0, C * (n1 - n0), 0, 0, ring_base(d, L.li, c.kslot), RING,
                           c.prefill ? c.seq0 : INT_MAX, chunk_base(d), d.tmp, c.s));
      CK(ak_kv_block_mean(d.kw.rows, d.tmp, n1 - n0, KVH * HD, L.km.rows, c.r2t + C * n0 + C - 1, C, 2, c.s));
      CK(ak_kv_block_mean(d.vw.rows, d.tmp, n1 - n0, KVH * HD, L.vm.rows, c.r2t + C * n0 + C - 1, C, 2, c.s));
    }
  }
  return 0;
}

int local(const Ctx& c, void* q, int* qo_dev, int* kv_indptr, int* kv_idx, int* last1, float sm_scale, void* o,
          float* lse, int graph_mode) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  const int seq_after = c.seq0 + c.T;
  // all-GPU path: decode lists the last min(seq, W) positions; prefill lists every position (window mask W-1), and
  // only positions >= seq0 - (W-1) are ever unmasked, so the rest may point at a zero row.
  const int kv_len = c.prefill ? seq_after : std::min(seq_after, WIN);
  const int list0 = seq_after - kv_len;
  const int lo = c.prefill ? std::max(0, c.seq0 - (WIN - 1)) : list0;
  const int n_real = seq_after - lo;
  const int rb = tier() ? ring_base(d, L.li, c.kslot) : 0;
  const int cs = c.prefill ? c.seq0 : INT_MAX;
  void *kc, *vc;
  int pool;
  if (!int8()) {  // tier, bf16: list straight into the window buffer
    CK(ak_kv_window_rows(list0, kv_len, lo, zero_row(d), rb, RING, cs, chunk_base(d), kv_idx, c.s));
    kc = d.kw.rows, vc = d.vw.rows, pool = d.wrows;
  } else {  // int8: dequant the real rows into staging [0, n_real), list into staging (zero row = STAGE)
    const int* src;
    if (tier()) {
      CK(ak_kv_window_rows(lo, n_real, 0, 0, rb, RING, cs, chunk_base(d), d.tmp, c.s));
      src = d.tmp;
    } else {
      src = c.r2t + lo;
    }
    CK(fetch_rows(tier() ? d.kw : L.k, src, n_real, d.kst, c.s));
    CK(fetch_rows(tier() ? d.vw : L.v, src, n_real, d.vst, c.s));
    CK(ak_kv_window_rows(list0, kv_len, lo, STAGE, -lo, INT_MAX, INT_MAX, 0, kv_idx, c.s));
    kc = d.kst, vc = d.vst, pool = STAGE + 1;
  }
  int kv_indptr_h[2] = {0, kv_len}, qo_h[2] = {0, c.T};
  cudaMemcpyAsync(kv_indptr, kv_indptr_h, sizeof(kv_indptr_h), cudaMemcpyHostToDevice, c.s);
  cudaMemcpyAsync(qo_dev, qo_h, sizeof(qo_h), cudaMemcpyHostToDevice, c.s);
  if (c.prefill)
    return ak_oracle_bcsa_prefill(q, kc, vc, qo_dev, kv_indptr, kv_idx, last1, qo_h, kv_indptr_h, 1, QH, KVH, WIN - 1,
                                  sm_scale, o, lse, c.s);
  return ak_oracle_bcsa_local(q, kc, vc, pool, kv_indptr, kv_idx, last1, kv_indptr_h, qo_h, qo_dev, 1, QH, KVH, sm_scale,
                              o, lse, graph_mode, c.s);
}

int far_prefetch(const Ctx& c, void* qi, void* pos, void* reqs, void* blk) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  const int n = d.ksel * C;
  CK(ak_bridge_far_select_decode(qi, pos, reqs, (void*)c.r2t, 1, c.wctx, c.rows, blk, 1, d.sel_rows, d.sel_ok, c.s));
  if (means()) CK(ak_kv_pick_rows(d.sel_rows, C - 1, C, d.ksel, 2, d.sel_last, c.s));  // pages
  CK(cudaEventRecord(d.ev_sel, c.s));
  CK(cudaStreamWaitEvent(d.side, d.ev_sel, 0));
  static const bool stats = env_on("AGENS_KV_STATS");
  if (stats) CK(cudaEventRecord(d.ev_t0, d.side));
  d.att_k = d.kf, d.att_v = d.vf;
  if (cache_on()) {  // hits from last step's staged blocks (device), misses from the host pool
    if (!d.slot_of) {
      if (cudaMalloc(&d.slot_of, ((size_t)c.rows + 1) * 4) || cudaMalloc(&d.misses, 4)) return (int)cudaErrorMemoryAllocation;
      cudaMemset(d.slot_of, 0xff, ((size_t)c.rows + 1) * 4), cudaMemset(d.misses, 0, 4);
      d.dev_bytes += ((size_t)c.rows + 2) * 4;
    }
    Cache& ch = d.cache[(long)c.layer * 4096 + c.kslot];
    if (!ch.k[0]) {
      for (int b = 0; b < 2; ++b) {
        ch.k[b] = dzero<__nv_bfloat16>((size_t)d.ksel * KVH * HD, d), ch.v[b] = dzero<__nv_bfloat16>((size_t)d.ksel * KVH * HD, d);
        ch.keys[b] = dzero<int>(d.ksel, d);
        if (!ch.k[b] || !ch.v[b] || !ch.keys[b]) return (int)cudaErrorMemoryAllocation;
      }
    }
    const bool warm = ch.valid && ch.last_seq + 1 == c.seq0 && ch.gen == g_gen[c.kslot];
    const int o = ch.cur, nw = 1 - ch.cur;
    if (warm) CK(ak_kv_slot_map(d.slot_of, ch.keys[o], d.ksel, 1, d.side));
    CK(ak_kv_cached_gather(L.km.rows, ch.k[o], d.slot_of, d.sel_last, d.ksel, BF_ROW, ch.k[nw], stats ? d.misses : nullptr,
                           d.side));
    CK(ak_kv_cached_gather(L.vm.rows, ch.v[o], d.slot_of, d.sel_last, d.ksel, BF_ROW, ch.v[nw], nullptr, d.side));
    if (warm) CK(ak_kv_slot_map(d.slot_of, ch.keys[o], d.ksel, 0, d.side));
    CK(cudaMemcpyAsync(ch.keys[nw], d.sel_last, (size_t)d.ksel * 4, cudaMemcpyDeviceToDevice, d.side));
    ch.cur = nw, ch.valid = true, ch.last_seq = c.seq0, ch.gen = g_gen[c.kslot];
    d.att_k = ch.k[nw], d.att_v = ch.v[nw];
  } else if (means()) {  // one mean row per selected block
    CK(ak_kv_copy_rows(L.km.rows, d.sel_last, d.kf, nullptr, d.ksel, BF_ROW, d.side));
    CK(ak_kv_copy_rows(L.vm.rows, d.sel_last, d.vf, nullptr, d.ksel, BF_ROW, d.side));
  } else {
    CK(fetch_rows(L.k, d.sel_rows, n, d.kf, d.side));
    CK(fetch_rows(L.v, d.sel_rows, n, d.vf, d.side));
  }
  CK(cudaEventRecord(d.ev_fetch, d.side));
  if (stats) {  // synchronous: timing/diagnostics only
    CK(cudaEventSynchronize(d.ev_fetch));
    float ms = 0;
    cudaEventElapsedTime(&ms, d.ev_t0, d.ev_fetch);
    d.fetch_ms += ms, ++d.fetches;
    if (cache_on()) {
      int miss = 0;
      CK(cudaMemcpy(&miss, d.misses, 4, cudaMemcpyDeviceToHost));
      CK(cudaMemset(d.misses, 0, 4));
      d.miss_rows += miss, d.req_rows += d.ksel;
    }
    std::vector<int> cur(n);
    CK(cudaMemcpy(cur.data(), d.sel_rows, (size_t)n * 4, cudaMemcpyDeviceToHost));
    std::sort(cur.begin(), cur.end());
    auto& prev = d.prev_sel[(long)c.layer * 4096 + c.kslot];
    if (!prev.empty()) {
      std::vector<int> both;
      std::set_intersection(cur.begin(), cur.end(), prev.begin(), prev.end(), std::back_inserter(both));
      d.overlap += (double)both.size() / n, ++d.overlap_n;
    }
    prev.swap(cur);
  }
  return 0;
}

int far_attend(const Ctx& c, void* qh, void* o_loc, float* lse, void* out) {
  Dev& d = g_dev[c.dev];
  CK(cudaStreamWaitEvent(c.s, d.ev_fetch, 0));
  return ak_bridge_far_attend_decode(qh, o_loc, lse, d.att_k, d.att_v, d.sel_ok, 1, d.ksel, means(), out, c.s);
}

int far_extend(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, void* idx, void* blk, void* out) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  const long seq_len = c.seq0 + c.T, m = seq_len - 1 - WIN - (C - 1);
  const long nb = m >= 0 ? m / C + 1 : 0;
  const int n = (int)(means() ? nb : C * nb);
  if ((size_t)n > d.erows) {  // grow (cudaFree synchronizes the device)
    cudaFree(d.ke), cudaFree(d.ve);
    d.dev_bytes -= d.erows * BF_ROW * 2;
    const size_t want = std::max<size_t>(n, means() ? (size_t)c.wctx / C + 1 : (size_t)c.wctx);  // means: one row per block
    d.ke = d.ve = nullptr, d.erows = 0;  // (1.1.1: the size only once both are allocated; a failure stays retryable)
    if (!(d.ke = dzero<__nv_bfloat16>(want * KVH * HD, d)) || !(d.ve = dzero<__nv_bfloat16>(want * KVH * HD, d))) {
      if (d.ke) cudaFree(d.ke), d.dev_bytes -= want * BF_ROW;
      d.ke = nullptr;
      cudaGetLastError();
      return (int)cudaErrorMemoryAllocation;
    }
    d.erows = want;
  }
  if (n > 0 && means()) {  // block means of blocks 0 .. nb-1 (pool row of each block's last token)
    CK(ak_kv_pick_rows(c.r2t, C - 1, C, n, 2, d.tmp, c.s));
    CK(ak_kv_copy_rows(L.km.rows, d.tmp, d.ke, nullptr, n, BF_ROW, c.s));
    CK(ak_kv_copy_rows(L.vm.rows, d.tmp, d.ve, nullptr, n, BF_ROW, c.s));
  } else if (n > 0) {  // rows of positions 0 .. 4nb-1, in position order (= tkc.index({r2t[0:4nb]}))
    CK(fetch_rows(L.k, c.r2t, n, d.ke, c.s));
    CK(fetch_rows(L.v, c.r2t, n, d.ve, c.s));
  }
  // AGENS_FAR_EXTEND_ENGINE=1 (budget mode, with the block means): engine kernels instead of the ATen bridge
  static const int eng = [] { const char* e = getenv("AGENS_FAR_EXTEND_ENGINE"); return e && e[0] == '1'; }();
  if (eng && means()) {
    static void* ws[2] = {};
    if (!ws[c.dev]) {
      const size_t b = ak_far_extend_engine_ws_bytes(c.wctx);
      if (cudaMalloc(&ws[c.dev], b) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
      d.dev_bytes += b;
    }
    return ak_far_extend_engine(c.T, c.seq0, qi, qh, o_loc, lse, c.r2t, c.wctx, d.ke, d.ve, idx, blk, c.rows, ws[c.dev],
                                out, c.s);
  }
  return ak_bridge_far_extend_staged(qi, qh, o_loc, lse, (int)seq_len, c.T, 0, (void*)c.r2t, 1, c.wctx, d.ke, d.ve, n,
                                     means(), idx, blk, c.rows, out, c.s);
}

int commit_chunk(const Ctx& c) {
  if (!tier() || !c.prefill) return 0;
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  const int seq_after = c.seq0 + c.T, lo = std::max(c.seq0, seq_after - RING), n = seq_after - lo;
  CK(ak_kv_window_rows(lo, n, 0, 0, chunk_base(d) - c.seq0, INT_MAX, INT_MAX, 0, d.tmp, c.s));      // chunk rows
  CK(ak_kv_window_rows(lo, n, 0, 0, ring_base(d, L.li, c.kslot), RING, INT_MAX, 0, d.tmp2, c.s));  // ring rows
  CK(move_rows(d.kw, d.tmp, d.kw, d.tmp2, n, c.s));
  return move_rows(d.vw, d.tmp, d.vw, d.tmp2, n, c.s);
}

int fork(int dev, int src_kslot, int dst_kslot, cudaStream_t s) {
  if (!tier()) return 0;
  ++g_gen[dst_kslot];
  Dev& d = g_dev[dev];
  for (auto& kv : g_layer) {
    const Layer& L = kv.second;
    if (L.dev != dev) continue;
    const size_t a = ring_base(d, L.li, src_kslot), b = ring_base(d, L.li, dst_kslot);
    for (const Pool* p : {&d.kw, &d.vw}) {
      const size_t rb = int8() ? I8_ROW : BF_ROW;
      CK(cudaMemcpyAsync((char*)p->rows + b * rb, (char*)p->rows + a * rb, (size_t)RING * rb, cudaMemcpyDeviceToDevice, s));
      if (int8())
        CK(cudaMemcpyAsync(p->scale + b * KVH, p->scale + a * KVH, (size_t)RING * SC_ROW, cudaMemcpyDeviceToDevice, s));
    }
  }
  return 0;
}

int rebuild(int dev, int kslot, const int* r2t, int seq, cudaStream_t s) {
  if (!tier()) return 0;
  ++g_gen[kslot];
  Dev& d = g_dev[dev];
  const int lo = std::max(0, seq - RING), n = seq - lo;
  for (auto& kv : g_layer) {
    const Layer& L = kv.second;
    if (L.dev != dev || n <= 0) continue;
    CK(ak_kv_window_rows(lo, n, 0, 0, ring_base(d, L.li, kslot), RING, INT_MAX, 0, d.tmp, s));
    CK(move_rows(L.k, r2t + lo, d.kw, d.tmp, n, s));
    CK(move_rows(L.v, r2t + lo, d.vw, d.tmp, n, s));
    if (means() && seq >= C) {  // block means of every closed block, from the restored pool rows
      CK(ak_kv_block_mean(L.k.rows, r2t, seq / C, KVH * HD, L.km.rows, r2t + C - 1, C, 2, s));
      CK(ak_kv_block_mean(L.v.rows, r2t, seq / C, KVH * HD, L.vm.rows, r2t + C - 1, C, 2, s));
    }
  }
  return 0;
}
size_t ring_bytes(int dev) {
  if (!tier() || dev < 0 || dev > 1) return 0;
  const Dev& d = g_dev[dev];
  return (size_t)d.nb * RING * ((int8() ? I8_ROW + SC_ROW : BF_ROW) * 2);
}

int ring_copy(int dev, int kslot, void* buf, bool to_buf, cudaStream_t s) {
  if (!tier()) return 0;
  ++g_gen[kslot];
  Dev& d = g_dev[dev];
  char* b = (char*)buf;
  for (auto& kv : g_layer) {
    const Layer& L = kv.second;
    if (L.dev != dev) continue;
    const size_t r0 = ring_base(d, L.li, kslot);
    for (const Pool* p : {&d.kw, &d.vw}) {
      const size_t rb = int8() ? I8_ROW : BF_ROW;
      char* g = (char*)p->rows + r0 * rb;
      CK(cudaMemcpyAsync(to_buf ? (void*)b : (void*)g, to_buf ? (void*)g : (void*)b, (size_t)RING * rb, cudaMemcpyDefault, s));
      b += (size_t)RING * rb;
      if (int8()) {
        char* gs = (char*)(p->scale + r0 * KVH);
        CK(cudaMemcpyAsync(to_buf ? (void*)b : (void*)gs, to_buf ? (void*)gs : (void*)b, (size_t)RING * SC_ROW,
                           cudaMemcpyDefault, s));
        b += (size_t)RING * SC_ROW;
      }
    }
  }
  return 0;
}
int copy_rows(int dev, int layer, int from, int to, int n, cudaStream_t s) {
  auto it = g_layer.find(layer);
  if (it == g_layer.end() || n <= 0) return 0;
  const Layer& L = it->second;
  for (const Pool* p : {&L.k, &L.v}) {
    const size_t rb = int8() ? I8_ROW : BF_ROW;
    CK(cudaMemcpyAsync((char*)p->rows + to * rb, (char*)p->rows + from * rb, n * rb, cudaMemcpyDefault, s));
    if (int8()) CK(cudaMemcpyAsync(p->scale + (size_t)to * KVH, p->scale + (size_t)from * KVH, (size_t)n * SC_ROW, cudaMemcpyDefault, s));
  }
  (void)dev;
  return 0;
}

// ---------------------------------------------------------------- graph-safe decode (one token, values from device)
namespace {
// rows[0] = ring row of position pos; for the block holding pos: rows[1..4] = its 4 ring rows, rows[5] = the page its
// mean goes to (the page of this step's pool row when pos closes the block, else the trash page)
__global__ void dec_rows_k(const long long* pos, const int* loc, int ring_base, int trash_page, int* rows) {
  const long long p = pos[0];
  rows[0] = ring_base + (int)(p % RING);
  const long long b0 = (p / C) * C;
  for (int c = 0; c < C; ++c) rows[1 + c] = ring_base + (int)((b0 + c) % RING);
  rows[5] = p % C == C - 1 ? loc[0] / C : trash_page;
}

// T tokens (positions pos[0..T)): ring rows rr[t], block rows br[4t..4t+3], mean page pg[t] as dec_rows_k per token
__global__ void dec_rows_multi_k(const long long* pos, const int* loc, int T, int ring_base, int trash_page, int* rows) {
  const int t = threadIdx.x;
  if (t >= T) return;
  const long long p = pos[t];
  rows[t] = ring_base + (int)(p % RING);
  const long long b0 = (p / C) * C;
  for (int c = 0; c < C; ++c) rows[T + 4 * t + c] = ring_base + (int)((b0 + c) % RING);
  rows[5 * T + t] = p % C == C - 1 ? loc[t] / C : trash_page;
}
}  // namespace

// speculative verify: the T tokens' stores of T store_dec calls in one launch per kind (pool rows, ring rows, block
// means). Same values: the ring keeps window + 8 rows, so no row of an earlier query's window is overwritten, and a
// closing token's block rows are all at or before it.
int store_dec_multi(const Ctx& c, int T, const void* k, const void* v, const int* loc, const long long* pos) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  if (T < 1 || T > 8) return (int)cudaErrorInvalidValue;
  CK(write_rows(k, T, loc, L.k, c.s));
  CK(write_rows(v, T, loc, L.v, c.s));
  if (!tier()) return 0;
  dec_rows_multi_k<<<1, 32, 0, c.s>>>(pos, loc, T, ring_base(d, L.li, c.kslot), c.rows / C, d.dtmp);
  CK(write_rows(k, T, d.dtmp, d.kw, c.s));
  CK(write_rows(v, T, d.dtmp, d.vw, c.s));
  if (means()) {
    CK(ak_kv_block_mean(d.kw.rows, d.dtmp + T, T, KVH * HD, L.km.rows, d.dtmp + 5 * T, 1, 0, c.s));
    CK(ak_kv_block_mean(d.vw.rows, d.dtmp + T, T, KVH * HD, L.vm.rows, d.dtmp + 5 * T, 1, 0, c.s));
  }
  return 0;
}

int store_dec(const Ctx& c, const void* k, const void* v, const int* loc, const long long* pos) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  CK(write_rows(k, 1, loc, L.k, c.s));
  CK(write_rows(v, 1, loc, L.v, c.s));
  if (!tier()) return 0;
  dec_rows_k<<<1, 1, 0, c.s>>>(pos, loc, ring_base(d, L.li, c.kslot), c.rows / C, d.dtmp);
  CK(write_rows(k, 1, d.dtmp, d.kw, c.s));
  CK(write_rows(v, 1, d.dtmp, d.vw, c.s));
  if (means()) {  // every step: the block mean (to its page when this position closes it, else to the trash page)
    CK(ak_kv_block_mean(d.kw.rows, d.dtmp + 1, 1, KVH * HD, L.km.rows, d.dtmp + 5, 1, 0, c.s));
    CK(ak_kv_block_mean(d.vw.rows, d.dtmp + 1, 1, KVH * HD, L.vm.rows, d.dtmp + 5, 1, 0, c.s));
  }
  return 0;
}

int local_dec(const Ctx& c, void* q, int* rng, float sm_scale, void* o, float* lse, int slot, bool dequant) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  if (!tier()) return (int)cudaErrorNotSupported;  // graph_safe() requires tiering (the ring)
  const size_t rb = int8() ? 0 : (size_t)KVH * HD;
  const int base = ring_base(d, L.li, c.kslot);
  if (int8()) {  // dequant this session's ring slot (RING rows, fixed size) into staging indexed by position % RING
    if (dequant) {
      CK(ak_kv_dequant_rows((const int8_t*)d.kw.rows + (size_t)base * I8_ROW, d.kw.scale + (size_t)base * KVH, nullptr,
                            RING, KVH, HD, d.kst, c.s));
      CK(ak_kv_dequant_rows((const int8_t*)d.vw.rows + (size_t)base * I8_ROW, d.vw.scale + (size_t)base * KVH, nullptr,
                            RING, KVH, HD, d.vst, c.s));
    }
    return ak_attn_decode(slot, q, d.kst, d.vst, d.ringidx, rng, QH, KVH, sm_scale, o, lse, c.s);
  }
  return ak_attn_decode(slot, q, (__nv_bfloat16*)d.kw.rows + base * rb, (__nv_bfloat16*)d.vw.rows + base * rb, d.ringidx,
                        rng, QH, KVH, sm_scale, o, lse, c.s);
}

int local_dec_multi(const Ctx& c, int mslot, void* q, float sm_scale, void* o, float* lse) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  if (!tier() || int8()) return (int)cudaErrorNotSupported;
  const size_t rb = (size_t)KVH * HD;
  const int base = ring_base(d, L.li, c.kslot);
  return ak_attn_decode_multi(mslot, q, (__nv_bfloat16*)d.kw.rows + base * rb, (__nv_bfloat16*)d.vw.rows + base * rb,
                              QH, KVH, sm_scale, o, lse, c.s);
}
const int* ring_index(int dev) { return g_dev[dev].ringidx; }

// verify, far regime: select on the main stream, the union fetch (PCIe for host pools) on the side stream so it
// overlaps what the caller issues next (the window attention); far_attend_multi joins it
// AGENS_SPEC_FAR_CACHE = blocks per layer kept from the previous verify step (block means; default 2048, 0 = off)
int far_cache_cap() {
  static const int cap = [] { const char* e = getenv("AGENS_SPEC_FAR_CACHE"); return e ? std::max(0, atoi(e)) : 2048; }();
  return cap;
}
std::map<int, void*> g_fcache;  // layer -> cache (allocated on first use, graph-safe afterwards)
int far_select_fetch_multi(const Ctx& c, int T, void* qi, const long long* pos, void* blk, void* wsm,
                           const long long* tag) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  CK(ak_far_decode_multi_select(T, qi, pos, c.r2t, c.wctx, blk, c.rows, wsm, c.s));
  CK(cudaEventRecord(d.ev_ms, c.s));
  CK(cudaStreamWaitEvent(d.side, d.ev_ms, 0));
  if (means() && tag && far_cache_cap() > 0) {
    void*& fc = g_fcache[c.layer];
    if (!fc) {
      if (cudaMalloc(&fc, ak_far_cache_bytes(c.wctx, far_cache_cap())) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
      d.dev_bytes += ak_far_cache_bytes(c.wctx, far_cache_cap());
      CK(ak_far_cache_init(fc, c.wctx, far_cache_cap(), d.side));
    }
    CK(ak_far_decode_multi_fetch_cached(T, c.r2t, c.wctx, L.km.rows, L.vm.rows, c.rows, wsm, fc, far_cache_cap(), tag,
                                        d.side));
    return (int)cudaEventRecord(d.ev_mf, d.side);
  }
  const int src = means() ? 1 : int8() ? 2 : 0;
  const Pool &k = means() ? L.km : L.k, &v = means() ? L.vm : L.v;
  CK(ak_far_decode_multi_fetch(T, pos, c.r2t, c.wctx, src, k.rows, v.rows, src == 2 ? k.scale : nullptr,
                               src == 2 ? v.scale : nullptr, c.rows, wsm, d.side));
  return (int)cudaEventRecord(d.ev_mf, d.side);
}
int far_attend_multi(const Ctx& c, int T, void* qh, void* o_loc, float* lse, const long long* pos, void* wsm, void* out) {
  Dev& d = g_dev[c.dev];
  CK(cudaStreamWaitEvent(c.s, d.ev_mf, 0));
  return ak_far_decode_multi_attend(T, qh, o_loc, lse, pos, c.wctx, wsm, out, c.s);
}

int far_dec(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, void* blk, void* ws,
            void* out) {
  const Layer& L = g_layer[c.layer];
  if (means())
    return ak_far_decode_src(qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 1, L.km.rows, L.vm.rows, nullptr, nullptr, blk,
                             c.rows, ws, out, c.s);
  if (int8())
    return ak_far_decode_src(qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 2, L.k.rows, L.v.rows, L.k.scale, L.v.scale, blk,
                             c.rows, ws, out, c.s);
  return ak_far_decode_src(qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 0, L.k.rows, L.v.rows, nullptr, nullptr, blk, c.rows,
                           ws, out, c.s);
}

int far_dec_multi(const Ctx& c, int T, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, void* blk,
                  void* wsm, void* out) {
  const Layer& L = g_layer[c.layer];
  if (means())
    return ak_far_decode_multi_src(T, qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 1, L.km.rows, L.vm.rows, nullptr, nullptr,
                                   blk, c.rows, wsm, out, c.s);
  if (int8())
    return ak_far_decode_multi_src(T, qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 2, L.k.rows, L.v.rows, L.k.scale,
                                   L.v.scale, blk, c.rows, wsm, out, c.s);
  return ak_far_decode_multi_src(T, qi, qh, o_loc, lse, pos, c.r2t, c.wctx, 0, L.k.rows, L.v.rows, nullptr, nullptr, blk,
                                 c.rows, wsm, out, c.s);
}

bool graph_safe() { return tier() && !cache_on(); }

// ---------------------------------------------------------------- batched graph-safe decode (rows read their KDA / ring
// slot from the device slot table; slot 0 = a padding row of a batch bucket: its ring writes go to the prefill chunk
// area, its block mean to its own (reserved) pool row's page, so it never touches a session or a trash row/page)
namespace {
__global__ void ringmap_k(const int* slots, int B, int W, int* map) {
  const long n = (long)B * W;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const int r = (int)(i / W), p = (int)(i % W), sl = slots[r];
    map[i] = (sl > 0 ? sl - 1 : 0) * RING + p % RING;
  }
}
// per row r: rows[8r] = ring row of pos (absolute in the window buffer), rows[8r+1..4] = the block's 4 ring rows,
// rows[8r+5] = the page its mean goes to (closing position -> page of loc, else the trash page)
__global__ void brows_k(const long long* pos, const int* loc, const int* slots, int B, int li, int nslots, int chunk_base,
                        int trash_page, int* rows) {
  const int r = threadIdx.x;
  if (r >= B) return;
  const long long p = pos[r];
  const int sl = slots[r];
  auto ring = [&](long long q) { return sl > 0 ? (li * nslots + sl - 1) * RING + (int)(q % RING) : chunk_base + r * C + (int)(q % C); };
  rows[8 * r] = ring(p);
  const long long b0 = (p / C) * C;
  for (int c = 0; c < C; ++c) rows[8 * r + 1 + c] = ring(b0 + c);
  rows[8 * r + 5] = p % C == C - 1 ? loc[r] / C : trash_page;
}
__global__ void dequant_ring_k(const int8_t* q, const float* sc, const int* slots, int r, int layer_base, __nv_bfloat16* dst) {
  const int sl = slots[r];
  const long base = layer_base + (long)(sl > 0 ? sl - 1 : 0) * RING;
  const long n = (long)RING * KVH * HD / 8;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long row = i / (KVH * HD / 8), c8 = i % (KVH * HD / 8), h = c8 * 8 / HD;
    const float s = sc[(base + row) * KVH + h];
    const uint2 u = *reinterpret_cast<const uint2*>(q + (base + row) * KVH * HD + c8 * 8);
    const int8_t* e = reinterpret_cast<const int8_t*>(&u);
    alignas(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16_rn(__fmul_rn((float)e[k], s));
    *reinterpret_cast<uint4*>(dst + (row * KVH * HD + c8 * 8)) = *reinterpret_cast<const uint4*>(o);
  }
}
}  // namespace

int prep_rows(int dev, const int* slots, int B, cudaStream_t s) {
  if (!tier()) return 0;
  Dev& d = g_dev[dev];
  const long n = (long)B * d.wctx;
  ringmap_k<<<(int)std::min<long>(4096, (n + 255) / 256), 256, 0, s>>>(slots, B, d.wctx, d.ringmap_b);
  return (int)cudaGetLastError();
}

int store_rows_dec(const Ctx& c, const void* k, const void* v, int B, const int* loc, const long long* pos,
                   const int* slots) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  CK(write_rows(k, B, loc, L.k, c.s));
  CK(write_rows(v, B, loc, L.v, c.s));
  if (!tier()) return 0;
  brows_k<<<1, 32, 0, c.s>>>(pos, loc, slots, B, L.li, d.nslots, chunk_base(d), c.rows / C, d.brows);
  // ring rows of the new tokens: B rows with list stride 8
  for (int r = 0; r < B; ++r) {
    CK(write_rows((const char*)k + (size_t)r * KVH * HD * 2, 1, d.brows + 8 * r, d.kw, c.s));
    CK(write_rows((const char*)v + (size_t)r * KVH * HD * 2, 1, d.brows + 8 * r, d.vw, c.s));
    if (means()) {
      CK(ak_kv_block_mean(d.kw.rows, d.brows + 8 * r + 1, 1, KVH * HD, L.km.rows, d.brows + 8 * r + 5, 1, 0, c.s));
      CK(ak_kv_block_mean(d.vw.rows, d.brows + 8 * r + 1, 1, KVH * HD, L.vm.rows, d.brows + 8 * r + 5, 1, 0, c.s));
    }
  }
  return 0;
}

int local_dec_row(const Ctx& c, int r, void* q, int* rng, float sm_scale, void* o, float* lse) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  if (!tier()) return (int)cudaErrorNotSupported;
  const int lb = L.li * d.nslots * RING;  // the layer's first ring row
  if (int8()) return (int)cudaErrorNotSupported;  // local_dec_row_i8
  const size_t rb = (size_t)KVH * HD;
  return ak_attn_decode(0, q, (__nv_bfloat16*)d.kw.rows + lb * rb, (__nv_bfloat16*)d.vw.rows + lb * rb,
                        d.ringmap_b + (size_t)r * d.wctx, rng, QH, KVH, sm_scale, o, lse, c.s);
}

int local_dec_row_i8(const Ctx& c, int r, const int* slots, void* q, int* rng, float sm_scale, void* o, float* lse) {
  const Layer& L = g_layer[c.layer];
  Dev& d = g_dev[c.dev];
  const int lb = L.li * d.nslots * RING;
  const int g = (int)std::min<long>(4096, ((long)RING * KVH * HD / 8 + 255) / 256);
  dequant_ring_k<<<g, 256, 0, c.s>>>((const int8_t*)d.kw.rows, d.kw.scale, slots, r, lb, d.kst);
  dequant_ring_k<<<g, 256, 0, c.s>>>((const int8_t*)d.vw.rows, d.vw.scale, slots, r, lb, d.vst);
  CK(cudaGetLastError());
  return ak_attn_decode(0, q, d.kst, d.vst, d.ringidx, rng, QH, KVH, sm_scale, o, lse, c.s);
}

int far_dec_row(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, const int* r2t, void* blk,
                void* ws, void* out) {
  Ctx cr = c;
  cr.r2t = r2t;
  return far_dec(cr, qi, qh, o_loc, lse, pos, blk, ws, out);
}
}  // namespace m1kv
