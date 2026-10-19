// M1: standalone Volundr forward (bf16, tp1, one request), assembled from the engine's bit-exact kernels and the M1
// parity bridges (oracle tree: flashinfer attention templates, ATen far branch). Mirrors sglv23's VolundrModel.forward
// op for op (fused mHC path, BCSA FlashInfer local branch, VOLUNDR_FUSED_BCSA far path, token mask, greedy).
// Weights are registered by checkpoint tensor name (device or host-mapped pointers) from the Rust host.
// Linears may be W4 (compressed-tensors int4 container, group 64/128, repacked to the k-tiled layout at load time):
// m = 1 -> W4 GEMV, m <= 16 -> batched W4 GEMV (rows bit-identical to m = 1), larger m -> dequantize to a bf16 scratch
// + cuBLAS (fp32 reduction). The bf16 path is unchanged (token-identical to sglv23).
// Not the M2 design: one kernel per op, plain buffers, no batching. M2a adds sessions: one shared KV row pool, per-session
// slot map (r2t) / KDA state slot / Engram prev pair / position; switching is a pointer swap, fork shares prefix rows.
#include "agens_kernels.h"
#include "agens_kv.h"
#include "m1_kv.h"
#include "m1_draft.h"
#include "m1_mtp.h"
#include "m1_dense8.h"
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
// oracle tree (libagens_oracle_attn / libagens_oracle_prefill / libagens_oracle_bridge)
int ak_oracle_bcsa_local(void* q, void* kc, void* vc, int pool, int* kv_indptr, int* kv_indices, int* last_page_len,
                         const int* kv_indptr_host, const int* qo_indptr_host, int* qo_indptr_dev, int bs, int qo_heads,
                         int kv_heads, float sm_scale, void* o, float* lse, int cuda_graph, void* stream);
int ak_oracle_dense_prefill(void* q, void* kc, void* vc, int* qo_indptr_dev, int* kv_indptr, int* kv_indices,
                            int* last_page_len, const int* qo_indptr_host, const int* kv_indptr_host, int bs, int qo_heads,
                            int kv_heads, float sm_scale, void* o, void* stream);
int ak_oracle_bcsa_prefill(void* q, void* kc, void* vc, int* qo_indptr_dev, int* kv_indptr, int* kv_indices,
                           int* last_page_len, const int* qo_indptr_host, const int* kv_indptr_host, int bs, int qo_heads,
                           int kv_heads, int window_left, float sm_scale, void* o, float* lse, void* stream);
int ak_bridge_far_decode(void* qi, void* qh, void* o_loc, void* lse, void* pos, void* reqs, void* r2t, int R, int Wctx,
                         void* kc, void* vc, int rows, void* blk, int bs, void* out, void* stream);
int ak_bridge_block_keys_decode(void* pos, void* reqs, void* out_loc, void* r2t, int R, int Wctx, void* idx, void* blk,
                                int rows, int bs, void* stream);
int ak_attn_decode_plan(int slot, const int* kv_range, int qo_heads, int kv_heads, void* stream);
int ak_attn_decode_multi_plan_mode(int mslot, int T, const int* rng, int stride, const int* indices, int max_len,
                                   int qo_heads, int kv_heads, const int* disable_slots, int mode, void* stream);
int ak_attn_decode_multi_plan(int mslot, int T, const int* rng, int stride, const int* indices, int max_len,
                              int qo_heads, int kv_heads, const int* disable_slots, void* stream);
int ak_attn_decode_multi(int mslot, void* q, void* kc, void* vc, int qo_heads, int kv_heads, float sm_scale, void* o,
                         float* lse, void* stream);
int ak_attn_decode_packed_plan(int mslot, int T, const int* rng, int stride, int qo_heads, int kv_heads,
                               const int* disable_slots, void* stream);
int ak_attn_decode_packed(int mslot, void* q, void* kc, void* vc, int* indices, int qo_heads, int kv_heads,
                          float sm_scale, void* o, float* lse, void* stream);
int ak_attn_decode(int slot, void* q, void* kc, void* vc, int* indices, int* kv_range, int qo_heads, int kv_heads,
                   float sm_scale, void* o, float* lse, void* stream);
long long ak_bridge_torch_reserved(int release);
// vision tower (oracle/bridge/vision.cpp)
int ak_vis_register(const char* name, void* host, const long long* shape, int ndim);
int ak_vis_ready();
int ak_vis_encode(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, void* out,
                  void* stream);
int ak_vis_pixels(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, float* out,
                  void* stream);
int ak_vis_encode_pixels(const float* pv, int gh, int gw, void* out, void* stream);
long long ak_vis_peak_bytes();
int ak_bridge_set_heads(int qh, int kvh);  // far bridge: this rank's q / kv heads (tensor parallel)
int ak_bridge_far_extend(void* qi_all, void* qh, void* o_loc, void* lse, int seq_len, int T, int req, void* r2t, int R,
                         int Wctx, void* kc, void* vc, void* idx, void* blk, int rows, void* out, void* stream);
}

namespace {
using bf16 = __nv_bfloat16;
constexpr int HID = 5120, NS = 4, VOCAB = 248320, VOCAB_VALID = 248077;
constexpr int HD = 256, ROT = 64, WIN = 4096, KK = 128;
constexpr int kMropeH = 11, kMropeW = 10;  // config rope_parameters.mrope_section [11, 11, 10], interleaved
// Head counts of this rank (tensor parallel, ak_m1_tp_config before ak_m1_init: each rank holds 1 / tp of the KDA,
// attention and MLP heads / rows; tp1 = the whole model). Full model: QH 24, KVH 4, KH 16, KHV 48, INTER 17408.
constexpr int QH_FULL = 24, KVH_FULL = 4, KH_FULL = 16, KHV_FULL = 48, INTER_FULL = 17408;
int QH = QH_FULL, KVH = KVH_FULL, KH = KH_FULL, KHV = KHV_FULL, INTER = INTER_FULL;
int KDIM = 2 * KH_FULL * KK + KHV_FULL * KK;
int g_tp = 1, g_rank = 0;
constexpr int IDXH = 4, IDXD = 128, ENG_DIM = 512, ENG_ROWS = 1 << 20, CHUNK = 4096;
// Prefill chunk (sglv23's chunked_prefill_size 4096 = parity with the reference). AGENS_M1_PREFILL_CHUNK (64..4096)
// trades prefill speed for the CHUNK-sized scratch (~1.7 GB activations at 4096): the 16 GB plan needs it smaller.
// Chunk boundaries change prefill numerics (KDA chunking), so the default stays 4096.
int g_chunk = CHUNK;
constexpr float EPS = 1e-6f;
// layer pattern ('k' KDA, 'b' BCSA, 'd' dense) and Engram layers come from the checkpoint config (ak_m1_init)
int NL = 72;
std::string kMixer = "kkkbkkkkbkkkbkkkkbkkkbkkkkbkkkbbkkkbkkkkbkkkbkkkkbkkkbkkkkbkkkbbkkkbkkkd";
int kEngramLayer[2] = {2, 22};

#define CK(x)                                                                         \
  do {                                                                                \
    int e_ = (int)(x);                                                                \
    if (e_) {                                                                         \
      fprintf(stderr, "m1: %s failed (%d) at %s:%d\n", #x, e_, __FILE__, __LINE__);   \
      return e_;                                                                      \
    }                                                                                 \
  } while (0)

std::map<std::string, void*> W;  // weight registry
void* w(const std::string& n) {
  auto it = W.find(n);
  if (it == W.end()) {
    fprintf(stderr, "m1: missing weight %s\n", n.c_str());
    return nullptr;
  }
  return it->second;
}
// Quantized linears (k-tiled weights + scales), keyed like the bf16 registry (name of the bf16 weight). A linear is one
// or two row parts (gate_up_proj merged from gate_proj and up_proj of different GSQ-RCO types); each part has its own
// bit width (4: W4 KT, 3: W3 KT, 2: W2 KT -- ak_wq_*) and group size, and writes rows [r0, r0 + n) of the output.
struct W4Part {
  void *wt, *st;
  int r0, n, group, bits;
};
struct W4Lin {
  int n, k;
  int np = 0;
  W4Part p[2];
};
std::map<std::string, W4Lin> W4M;
int g_w4_batch_max = 16;  // AGENS_M1_W4_BATCH_MAX (<= 128): largest m on the batched GEMV / GEMM; above: dequant + cuBLAS
int g_w4_bf16w = 0;       // AGENS_M1_W4_NUMERICS=marlin: decode/verify GEMVs use bf16(q * scale) weights like Marlin
// Batch-invariant decode (speculative decoding): bf16 linears, the mHC bmm and the Engram gate run the
// tensor-core / per-token kernels for every m <= 16, so a T-token verify gives each token the bits of a one-token
// step (W4 linears are batch-invariant on either path). AGENS_M1_BI=1 or ak_m1_spec_init turn it on.
bool g_bi = getenv("AGENS_M1_BI") && getenv("AGENS_M1_BI")[0] == '1';
constexpr int kSpecT = 8;                                  // verify block (DFlash2 block size)
// single-sequence verify staging (step_h / step_d): ints; [32..37] the sampled-accept parameter block (samp_pos: T,
// top-k, top-p, min-p, seed, step), [kStepDelta] = the session's rope delta
constexpr int kStepInts = 48, kStepDelta = 40;
// speculative verify attention in one launch per layer (ak_attn_decode_multi; AGENS_SPEC_MULTI_ATTN=0: per query)
constexpr int MSLOT_DENSE = 0, MSLOT_WIN = 1, MSLOT_PACKED = 2;
// dense layer verify as one causal request of the T queries (ak_attn_decode_packed; AGENS_SPEC_DENSE_PACKED=0: the
// T-request multi launch)
bool dense_packed() {
  static const int on = [] { const char* e = getenv("AGENS_SPEC_DENSE_PACKED"); return !(e && e[0] == '0'); }();
  return on;
}
std::vector<cudaGraph_t>* g_seg_capture = nullptr;  // set while capturing a segmented verify graph
int graph_segs() {
  static const int n = [] { const char* e = getenv("AGENS_M1_GRAPH_SEGS"); return e ? std::max(1, atoi(e)) : 4; }();
  return n;
}
bool g_multi_dense = false, g_multi_win = false;  // set per verify step (verify_body), read by attn_mixer
bool g_multi_wdev = false;  // all-GPU pools (native mode): the T BCSA windows in one launch, as the dense layer
bool multi_attn() {
  static const int on = [] { const char* e = getenv("AGENS_SPEC_MULTI_ATTN"); return !(e && e[0] == '0'); }();
  return on;
}
// AGENS_SPEC_MULTI_WIN=0: native-mode BCSA windows per query (default: one multi-request launch, bit-identical)
// AGENS_SPEC_WIN_MODE2=0: the BCSA window multi launch in mode 0 (per-query launches as the fallback)
bool win_mode2() {
  static const int on = [] { const char* e = getenv("AGENS_SPEC_WIN_MODE2"); return !(e && e[0] == '0'); }();
  return on;
}
bool g_win2 = false;  // this verify's BCSA windows run as one mode-2 multi launch
bool multi_win_dev() {
  static const int on = [] { const char* e = getenv("AGENS_SPEC_MULTI_WIN"); return !(e && e[0] == '0'); }();
  return on;
}
constexpr int kCapLayers = 5;
int g_cap_layer[kCapLayers] = {-1, -1, -1, -1, -1};        // target layers whose entry stream 0 feeds the drafter
bool g_capture = false;
float g_temp = 0.f, g_topp = 1.f, g_minp = 0.f;
int g_topk = 0;  // sampled accept: top-k (<= 0: all), top-p, min-p of the request (ak_m1_spec_sampling)
//               // speculative verify: 0 = greedy accept, > 0 = sampled accept at this temperature
int g_seed = 0, g_spec_step = 0;  // Philox key and per-request verify step counter (reset by ak_m1_spec_sampling)
// Weight pointers per (name literal, layer): looked up in the registries once, then read from a pointer table (the
// forward used to build ~1400 name strings + map lookups per decode step; a decode graph looks up nothing at all).
// Keyed by the literal's address: call sites pass string literals. Cleared when weights are (re)registered.
struct LinRef {
  bool resolved = false;
  void* bf16 = nullptr;
  const W4Lin* w4 = nullptr;
};
std::unordered_map<const char*, std::vector<LinRef>> g_lin_ref;  // [layer + 1] (-1 = model-level, e.g. lm_head)
std::unordered_map<const char*, std::vector<void*>> g_wl, g_wm;  // layer weights / mhc weights
std::unordered_map<const char*, void*> g_wg;                     // model-level weights
void clear_weight_tables() { g_lin_ref.clear(), g_wl.clear(), g_wm.clear(), g_wg.clear(); }
// host-side step statistics (AGENS_M1_STEP_STATS): enqueue / wait ms of the last forward, linear calls (all / W4)
double g_stat[4] = {};
// Batched decode (server mode): g_B rows, row r = session g_brow[r]; attention runs per row, everything else on B rows.
// g_binv (AGENS_M1_BATCH_INVARIANT=1): decode bf16 linears use the batch-invariant engine GEMV (also at bs = 1);
// otherwise a batched decode issues them per row (cuBLAS m = 1), so either way a row is bit-identical to its bs = 1 run.
constexpr int kMaxBatch = 16;
int g_B = 0, g_binv = 0;
bool g_bdev = false, g_bfar = false;
int g_pad_row = 0;  // batched attention on device tables (attn_batch); far branch in this step
bool g_dec = false;
std::vector<int> g_brow;
// batched speculative verify: g_BV sequences of kSpecT tokens (rows 8 * g_BV, sequence b's at 8 b); 0 = off
constexpr int kMaxBV = 16, kBVRes = 16;
int g_BV = 0;
// batched verify length per sequence (2..kSpecT; rows j = g_TV b + t): verifying fewer drafts per sequence at large B
// trades a little acceptance for a proportionally smaller verify forward (AGENS_SPEC_BATCH_TV: one value, or a list
// "tv1,tv2,..." for B = 1, 2, ..; the last entry applies to larger B). Default (sm90 conc_bench sweep, short prompts):
// B x TV at or under 16 rows up to B = 4 (one 16-token GEMM tile), about 30 rows to B = 8, 2 drafts past B = 10
int g_TV = kSpecT;
int batch_tv(int B) {
  static std::vector<int> tv;
  static bool init = false;
  if (!init) {
    init = true;
    const char* e = getenv("AGENS_SPEC_BATCH_TV");
    for (const char* c = e; c && *c;) {
      tv.push_back(std::max(2, std::min(kSpecT, atoi(c))));
      while (*c && *c != ',') ++c;
      if (*c == ',') ++c;
    }
    if (tv.empty()) tv = {8, 8, 5, 4, 6, 5, 4, 4, 3, 3, 2};
  }
  return tv[std::min((size_t)B, tv.size()) - 1];
}
std::vector<int> g_bv_sess;              // engine sessions of the current batched verify
const long long* g_tag_ptr = nullptr;    // far-branch cache tag the verify reads (default S.step_d + 30)
long g_lin = 0, g_lin_w4 = 0;
std::string L(int i, const char* s) { return "model.language_model.layers." + std::to_string(i) + "." + s; }
std::string M(int i, const char* s) { return "model.language_model.mhc." + std::to_string(i) + "." + s; }
void* table_get(std::unordered_map<const char*, std::vector<void*>>& t, int i, const char* s, std::string (*name)(int, const char*)) {
  std::vector<void*>& v = t[s];
  if ((int)v.size() <= i) v.resize(std::max(NL, i + 1), nullptr);
  if (!v[i]) v[i] = w(name(i, s));
  return v[i];
}
void* WL(int i, const char* s) { return table_get(g_wl, i, s, L); }  // == w(L(i, s))
void* WM(int i, const char* s) { return table_get(g_wm, i, s, M); }  // == w(M(i, s))
void* WG(const char* s) {                                             // == w(s)
  void*& p = g_wg[s];
  if (!p) p = w(s);
  return p;
}

// ---------------------------------------------------------------- small helper kernels (pure data movement)
__global__ void split_q_gate(const bf16* qg, bf16* q, bf16* g, int T, int QH) {  // [T][QH][2*256] -> q, gate [T][QH*256]
  const long n = (long)T * QH * HD;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long t = i / (QH * HD), r = i % (QH * HD), h = r / HD, d = r % HD;
    q[i] = qg[(t * QH + h) * 2 * HD + d];
    g[i] = qg[(t * QH + h) * 2 * HD + HD + d];
  }
}
__global__ void scale_f32(float* x, int n, float s) {  // lse.float() * _LN2 (torch: x * float(s))
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * s;
}
__global__ void store_kv(const bf16* k, const bf16* v, bf16* kc, bf16* vc, const int* loc, int T, int KVH) {
  const long n = (long)T * KVH * HD;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long t = i / (KVH * HD), r = i % (KVH * HD);
    kc[(long)loc[t] * KVH * HD + r] = k[i];
    vc[(long)loc[t] * KVH * HD + r] = v[i];
  }
}
__global__ void store_rows(const bf16* x, bf16* cache, const int* loc, int T, int d) {
  const long n = (long)T * d;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    cache[(long)loc[i / d] * d + i % d] = x[i];
}
// Session State rows in position order <-> pool rows (map = the session's r2t). out/in may be device or pinned host (UVA).
// shift 2: the cache is indexed by page (row / 4: block keys, one per 4-row group)
__global__ void gather_rows(const uint4* cache, const int* map, int n, int d16, uint4* out, int shift = 0) {
  const long m = (long)n * d16;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x)
    out[i] = cache[(long)(map[i / d16] >> shift) * d16 + i % d16];
}
__global__ void scatter_rows(const uint4* in, const int* map, int n, int d16, uint4* cache, int shift = 0) {
  const long m = (long)n * d16;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < m; i += (long)gridDim.x * blockDim.x)
    cache[(long)(map[i / d16] >> shift) * d16 + i % d16] = in[i];
}
__global__ void broadcast_streams(const bf16* x, bf16* H, int T) {  // H[t][s][:] = x[t][:]
  const long n = (long)T * NS * HID;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    H[i] = x[(i / (NS * HID)) * HID + i % HID];
}
// Engram rows for an extend chunk (sglv23 _engram_row_indices, extend branch, single request) + prev update.
__device__ unsigned long long eg_fold(const long long* ids, int n) {
  const unsigned long long P[5] = {1000003ull, 1000033ull, 1000037ull, 1000039ull, 1000081ull};
  unsigned long long code = 0;
  for (int j = 0; j < n; ++j) code = (code * P[j % 5] + (unsigned long long)(ids[j] + 1)) & 0x1FFFFFFFFFFFFFFFull;
  return code;
}
__global__ void engram_rows_extend(const int* ids, const long long* pos, int T, int* prev, long long* r2, long long* r3) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const long long s0 = prev[0], s1 = prev[1];
  if (t < T) {
    long long p1 = t >= 1 ? ids[t - 1] : s1;
    long long p2 = t >= 2 ? ids[t - 2] : (t == 1 ? s1 : s0);
    if (pos[t] < 1) p1 = -1;
    if (pos[t] < 2) p2 = -1;
    const long long g2[2] = {p1, ids[t]}, g3[3] = {p2, p1, ids[t]};
    r2[t] = (long long)(eg_fold(g2, 2) % ENG_ROWS);
    r3[t] = (long long)(eg_fold(g3, 3) % ENG_ROWS);
  }
}
__global__ void engram_prev_update(const int* ids, const long long* pos, int T, int* prev) {
  const long long last = T - 1;
  const int old1 = prev[1];
  const long long np2 = T >= 2 ? ids[last - 1] : (pos[last] >= 1 ? old1 : -1);
  prev[0] = (int)np2;
  prev[1] = ids[last];
}
__global__ void gather_engram(const long long* r2, const long long* r3, const uint4* t2, const uint4* t3, uint4* e,
                              int T) {
  const int t = blockIdx.x;
  const int per = ENG_DIM * 2 / 16;
  for (int i = threadIdx.x; i < 2 * per; i += blockDim.x)
    e[(size_t)t * 2 * per + i] = i < per ? t2[r2[t] * per + i] : t3[r3[t] * per + (i - per)];
}

// Decode step inputs {token, position, pool row} (staged from pinned host memory, so a captured graph reads the new
// values on every replay) -> the per-step device state the layers read; rng = decode attention KV ranges over the
// session's slot map: [0, 2) BCSA window (last min(seq, W) tokens), [2, 4) dense (all tokens).
// rope positions: rpos[a * rmax + t] for the MRoPE axes a = T, H, W; a decode / verify token is text,
// so all three are its position + the session's rope delta (staged with the step: graphs stay valid when it changes)
__device__ __forceinline__ void put_rpos(long long* rpos, int rmax, int t, long long v) {
  rpos[t] = v, rpos[rmax + t] = v, rpos[2 * rmax + t] = v;
}
__global__ void decode_step_prep(const int* in, int* ids, long long* pos, int* loc, long long* outloc, int* r2t,
                                 int* rng, long long* rpos, int rmax) {
  const int tok = in[0], p = in[1], l = in[2];
  ids[0] = tok, pos[0] = p, loc[0] = l, outloc[0] = l;
  put_rpos(rpos, rmax, 0, (long long)p + in[4]);
  r2t[p] = l;
  rng[0] = p + 1 > WIN ? p + 1 - WIN : 0, rng[1] = p + 1;
  rng[2] = 0, rng[3] = p + 1;
}
// Batched decode step (server mode): kBRec ints per row {token, position, pool row, KDA slot, r2t ptr lo/hi,
// Engram prev ptr lo/hi, rope delta, unused} staged from pinned host memory -> per-row device state: ids/pos/loc/outloc/slots, the row's
// decode ranges (rng4[r]), its r2t entry for this position, its Engram rows (r2/r3) + prev rotation, exactly the bs = 1
// decode_step_prep + engram_rows_extend(T = 1) + engram_prev_update(T = 1) per row. Session pointers are read from the
// staged table, so a captured graph serves any set of sessions.
constexpr int kBRec = 10;
__device__ __forceinline__ int* rec_ptr(const int* q) {
  return reinterpret_cast<int*>(((unsigned long long)(unsigned)q[1] << 32) | (unsigned)q[0]);
}
__global__ void batch_step_prep(const int* in, int B, int* ids, long long* pos, int* loc, long long* outloc, int* slots,
                                int* rng4, long long* r2, long long* r3, long long* rpos, int rmax) {
  const int r = threadIdx.x;
  if (r >= B) return;
  const int* q = in + r * kBRec;
  const int tok = q[0], p = q[1], l = q[2];
  put_rpos(rpos, rmax, r, (long long)p + q[8]);
  int* r2t = rec_ptr(q + 4);
  int* prev = rec_ptr(q + 6);
  ids[r] = tok, pos[r] = p, loc[r] = l, outloc[r] = l, slots[r] = q[3];
  r2t[p] = l;
  int* rng = rng4 + 4 * r;
  rng[0] = p + 1 > WIN ? p + 1 - WIN : 0, rng[1] = p + 1;
  rng[2] = 0, rng[3] = p + 1;
  const long long s0 = prev[0], s1 = prev[1];
  long long p1 = s1, p2 = s0;
  if (p < 1) p1 = -1;
  if (p < 2) p2 = -1;
  const long long g2[2] = {p1, tok}, g3[3] = {p2, p1, tok};
  r2[r] = (long long)(eg_fold(g2, 2) % ENG_ROWS);
  r3[r] = (long long)(eg_fold(g3, 3) % ENG_ROWS);
  const int old1 = prev[1];
  prev[0] = p >= 1 ? old1 : -1;
  prev[1] = tok;
}
__global__ void batch_r2t_stage(const int* in, int B, int W, int* r2t_b) {  // r2t_b[r][:] = row r's slot map
  const long n = (long)B * W;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    r2t_b[i] = rec_ptr(in + (i / W) * kBRec + 4)[i % W];
}
__global__ void decode_ranges(const long long* pos, int* rng) {  // the eager forward's decode step (T = 1)
  const int p = (int)pos[0];
  rng[0] = p + 1 > WIN ? p + 1 - WIN : 0, rng[1] = p + 1;
  rng[2] = 0, rng[3] = p + 1;
}
// sglv23 bcsa_block_keys_decode (VolundrAttention._write_block_keys, decode branch), one request: pool the C = 4
// indexer keys of the block holding this position (serial fp32 sum, x 1/C, rounded: exact for 4 bf16 values) into
// blk_cache at this step's row if the position closes the block, else into the trash row hi. 128 threads = idx_dim.
__global__ void block_keys_decode(const long long* pos, const int* loc, const int* r2t, int r2t_w, const bf16* idx,
                                  bf16* blk, int hi, int seq_rows = 0) {
  constexpr int C = 4;
  if (seq_rows) r2t += (size_t)(blockIdx.x / seq_rows) * r2t_w;  // batched verify: row's sequence map in r2t_b
  pos += blockIdx.x, loc += blockIdx.x;  // one block per token (verify: T tokens; each token's write is independent)
  const long long p = pos[0], base = (p / C) * C;
  const int d = threadIdx.x;
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const long long cc = base + c < 0 ? 0 : base + c, col = cc > r2t_w - 1 ? r2t_w - 1 : cc;
    const int sr = r2t[col], slot = sr < 0 ? 0 : (sr > hi ? hi : sr);
    acc = __fadd_rn(acc, __bfloat162float(idx[(size_t)slot * IDXD + d]));
  }
  const long long dst = (p % C == C - 1 ? loc[0] : hi) / C;  // block keys are stored per 4-row page
  blk[(size_t)dst * IDXD + d] = __float2bfloat16_rn(__fmul_rn(acc, 1.0f / C));
}

// Speculative verify step inputs, staged like decode_step_prep's: in = {T, P, tokens[8], rows[8]} -> ids/pos/loc/r2t
// for T tokens at positions P..P+T-1 and, per query t, decode attention ranges rng[4t..4t+3] (window, dense).
// hybrid drafting: the host's copy-from-context proposal (step[20] tokens at step[21..27]) replaces the drafter's
// first step[20] drafts (step[3..]) after the drafter ran (graph-safe: the switch is data)
__global__ void apply_hint(int* step) {
  const int t = threadIdx.x, nh = step[20];
  if (t < nh && t < kSpecT - 1) step[3 + t] = step[21 + t];
}
__global__ void verify_step_prep(const int* in, int* ids, long long* pos, int* loc, long long* outloc, int* r2t,
                                 int* rng, long long* rpos, int rmax) {
  const int t = threadIdx.x, T = in[0], P = in[1];
  if (t >= T) return;
  const int p = P + t, l = in[10 + t];
  ids[t] = in[2 + t], pos[t] = p, loc[t] = l, outloc[t] = l;
  put_rpos(rpos, rmax, t, (long long)p + in[kStepDelta]);
  r2t[p] = l;
  rng[4 * t] = p + 1 > WIN ? p + 1 - WIN : 0, rng[4 * t + 1] = p + 1;
  rng[4 * t + 2] = 0, rng[4 * t + 3] = p + 1;
}
// Greedy accept of a verified chain: ids = [bonus, d1..d(T-1)], tok = target argmax per position.
// a = matching prefix of the drafts; commit n = a + 1 tokens (bonus, d1..da); emitted = d1..da, tok[a].
// res = {n, emitted[0..n-1]}.
__global__ void accept_greedy(const int* ids, const int* tok, int T, int* res) {
  int a = 0;
  while (a + 1 < T && ids[a + 1] == tok[a]) ++a;
  res[0] = a + 1;
  for (int j = 0; j < a; ++j) res[1 + j] = ids[1 + j];
  res[1 + a] = tok[a];
}
// Sampled accept (temperature > 0), sglang's target-only rule for a chain from a deterministic drafter: position j's
// target distribution q_j = softmax(logits_j / temp) over the valid vocab; draft x = ids[j + 1] is accepted with
// probability q_j(x); on rejection the token is sampled from q_j without x (renormalised) and the chain stops; all
// accepted -> the bonus is sampled from the last q. Philox(seed, step, position) -> reproducible. One block.
__device__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
  __syncthreads();
  return t;
}
__device__ float block_max(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t = fmaxf(t, red[w]);
  __syncthreads();
  return t;
}
// sample index from exp((l - mx) / temp) over [0, nv), excluding `skip` (-1 = none), at uniform u (total mass z)
__device__ int block_sample(const bf16* l, int nv, float mx, float inv_t, int skip, float z, float u, float* red,
                            int* found) {
  const int per = (nv + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * per, hi = min(nv, lo + per);
  float part = 0.f;
  for (int v = lo; v < hi; ++v)
    if (v != skip) part += __expf((__bfloat162float(l[v]) - mx) * inv_t);
  // exclusive scan of the per-thread parts (sequential over warps: small block)
  __shared__ float parts[1024];
  parts[threadIdx.x] = part;
  if (threadIdx.x == 0) *found = nv - 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    float acc = 0.f, target = u * z;
    for (int t = 0; t < (int)blockDim.x; ++t) {
      if (acc + parts[t] > target) {
        const int a = t * per, b = min(nv, a + per);
        for (int v = a; v < b; ++v) {
          if (v == skip) continue;
          acc += __expf((__bfloat162float(l[v]) - mx) * inv_t);
          if (acc > target) { *found = v; break; }
        }
        break;
      }
      acc += parts[t];
    }
  }
  __syncthreads();
  (void)red;
  return *found;
}
__global__ void __launch_bounds__(1024) accept_sampled(const bf16* logits, int ld, int nv, const int* ids, int T,
                                                       float temp, const int* step, int* res) {
  __shared__ float red[32];
  __shared__ int found, nacc, fin;
  const unsigned long long seed = (unsigned)step[19], sidx = (unsigned)step[18];
  const float inv_t = 1.0f / temp;
  if (threadIdx.x == 0) nacc = 0, fin = -1;
  __syncthreads();
  for (int j = 0; j < T; ++j) {
    const bf16* l = logits + (size_t)j * ld;
    float mx = -INFINITY;
    for (int v = threadIdx.x; v < nv; v += blockDim.x) mx = fmaxf(mx, __bfloat162float(l[v]));
    mx = block_max(mx, red);
    float sm = 0.f;
    for (int v = threadIdx.x; v < nv; v += blockDim.x) sm += __expf((__bfloat162float(l[v]) - mx) * inv_t);
    const float z = block_sum(sm, red);
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, sidx * 64 + j, 0, &rng);
    const float4 u4 = curand_uniform4(&rng);
    if (j + 1 < T) {
      const int x = ids[j + 1];
      const float qx = __expf((__bfloat162float(l[x]) - mx) * inv_t) / z;
      if (u4.x < qx) {  // accept the draft (uniform: identical in every thread)
        if (threadIdx.x == 0) nacc = j + 1;
        __syncthreads();
        continue;
      }
      const int v = block_sample(l, nv, mx, inv_t, x, z - qx * z, u4.y, red, &found);
      if (threadIdx.x == 0) fin = v;
      __syncthreads();
      break;
    }
    const int v = block_sample(l, nv, mx, inv_t, -1, z, u4.y, red, &found);
    if (threadIdx.x == 0) fin = v;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int a = nacc;
    res[0] = a + 1;
    for (int j = 0; j < a; ++j) res[1 + j] = ids[1 + j];
    res[1 + a] = fin;
  }
}
// ---- speculative sampling with top-k / top-p / min-p (1.0.1): the same target-only rule as accept_sampled, over the
// filtered distribution the host sampler (crates/agens-cli/src/sampler.rs) draws from at that position:
//   q = softmax(logits / temp) restricted to the top_k logits (ties at the k-th value in index order), renormalised;
//   then the smallest prefix (logit descending, ties by index) with mass >= top_p; then p >= min_p * p_max.
// Drafts are deterministic (DFlash2 greedy drafts, context copies, MTP): point-mass proposals, so accepting draft x
// with probability q(x) and otherwise sampling from q without x (renormalised) samples q exactly (Leviathan / Chen
// with a one-hot proposal). Per position one block (samp_pos, grid T) finds the filtered set without sorting: exact
// k-th key / top-p key by two 256-bin radix passes over the order-preserving bf16 key (counts / fp64 masses), the
// boundary ties cut in index order; then accept / residual / bonus samples from Philox(seed, step, position) as
// accept_sampled; samp_chain picks the first rejection. Parameters: step[32..35] = temp, top_k (<= 0: all), top_p,
// min_p (floats as bits); scr[0..7] accept flags, scr[8..15] the token of each position on rejection / as the bonus.
__device__ __forceinline__ int okey(const bf16 x) {
  const unsigned short b = __bfloat16_as_ushort(x);
  return (b & 0x8000) ? (int)(unsigned short)~b : (int)(b | 0x8000);
}
struct SampSet {
  float mx, inv_t, minp;
  int kall, tk, kcut, pall, tp, pcut;
  __device__ __forceinline__ float mass(float lv) const { return __expf((lv - mx) * inv_t); }
  __device__ __forceinline__ bool inA(int v, int key) const { return kall || key > tk || (key == tk && v <= kcut); }
  __device__ __forceinline__ bool in(int v, const bf16 x) const {
    const int key = okey(x);
    if (!inA(v, key)) return false;
    if (!pall && !(key > tp || (key == tp && v <= pcut))) return false;
    return minp <= 0.f || mass(__bfloat162float(x)) >= minp;
  }
};
// Layout: warp w owns [w * seg, (w + 1) * seg) (seg a multiple of 32) and lane l reads base + 32 r + l in round r:
// coalesced loads, and (warp, round, lane) order is index order (tie cuts and draws follow the index order).
constexpr unsigned kFull = 0xFFFFFFFFu;
struct SampLay {
  int seg, base, rounds;
};
__device__ __forceinline__ SampLay samp_lay(int nv) {
  SampLay L;
  const int nw = blockDim.x >> 5;
  L.seg = ((nv + nw - 1) / nw + 31) & ~31, L.base = (threadIdx.x >> 5) * L.seg, L.rounds = L.seg >> 5;
  return L;
}
__device__ __forceinline__ int samp_v(const SampLay& L, int r, int nv) {
  const int v = L.base + 32 * r + (threadIdx.x & 31);
  return v < nv ? v : -1;
}
// warp-aggregated histogram adds (lanes of one bin -> one shared atomic; bin < 0: no element)
__device__ __forceinline__ void hist_add(int* hist, int bin) {
  const unsigned m = __match_any_sync(kFull, bin);
  if (bin >= 0 && (int)(threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&hist[bin], __popc(m));
}
__device__ __forceinline__ void hist_addd(double* hm, int bin, double val) {
  const unsigned m = __match_any_sync(kFull, bin);
  double sum = 0.0;
  for (int src = 0; src < 32; ++src) {
    const double o = __shfl_sync(kFull, val, src);
    if ((m >> src) & 1u) sum += o;
  }
  if (bin >= 0 && (int)(threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&hm[bin], sum);
}
__device__ double block_sumd(double v, double* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  double t = 0.0;
  for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
  __syncthreads();
  return t;
}
// the index of the n-th (1-based, index order) element with key == tau (and in A when useA); -1 if fewer exist
__device__ int samp_tie_cut(const bf16* l, int nv, const SampLay& L, int tau, int n, const SampSet& f, bool useA,
                            int* wcnt, int* sh) {
  auto pred = [&](int v) {
    if (v < 0) return false;
    const int key = okey(l[v]);
    return key == tau && (!useA || f.inA(v, key));
  };
  int c = 0;
  for (int r = 0; r < L.rounds; ++r) c += __popc(__ballot_sync(kFull, pred(samp_v(L, r, nv))));
  if ((threadIdx.x & 31) == 0) wcnt[threadIdx.x >> 5] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int nw = blockDim.x >> 5;
    int acc = 0, w = 0;
    for (; w < nw; ++w) {
      if (acc + wcnt[w] >= n) break;
      acc += wcnt[w];
    }
    sh[0] = w < nw ? w : -1, sh[1] = n - acc, sh[2] = -1;
  }
  __syncthreads();
  if (sh[0] == (int)(threadIdx.x >> 5)) {
    int rem = sh[1];
    for (int r = 0; r < L.rounds; ++r) {
      const unsigned bal = __ballot_sync(kFull, pred(samp_v(L, r, nv)));
      const int k = __popc(bal);
      if (rem <= k) {
        unsigned bb = bal;
        for (int i = 1; i < rem; ++i) bb &= bb - 1;
        if ((threadIdx.x & 31) == 0) sh[2] = L.base + 32 * r + __ffs(bb) - 1;
        break;
      }
      rem -= k;
    }
  }
  __syncthreads();
  const int out = sh[2];
  __syncthreads();
  return out;
}
// sample from the set without `skip` (total mass z) at uniform u, in index order (block-wide)
__device__ int samp_draw(const bf16* l, int nv, const SampLay& L, const SampSet& f, int skip, double z, float u,
                         double* wm, int* sh, double* shd) {
  auto mass_of = [&](int v) -> double {
    return v >= 0 && v != skip && f.in(v, l[v]) ? (double)f.mass(__bfloat162float(l[v])) : 0.0;
  };
  double part = 0.0;
  for (int r = 0; r < L.rounds; ++r) part += mass_of(samp_v(L, r, nv));
  for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(kFull, part, o);
  if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = part;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int nw = blockDim.x >> 5;
    const double target = (double)u * z;
    double acc = 0.0;
    int w = 0, last = -1;
    for (; w < nw; ++w) {
      if (wm[w] > 0.0) last = w;
      if (acc + wm[w] > target) break;
      acc += wm[w];
    }
    if (w == nw) w = last, acc -= last >= 0 ? wm[last] : 0.0;  // rounding at the top end
    sh[0] = w, shd[0] = target - acc, sh[1] = -1;
  }
  __syncthreads();
  if (sh[0] == (int)(threadIdx.x >> 5)) {
    const int lane = threadIdx.x & 31;
    double rem = shd[0];
    int lastv = -1;
    for (int r = 0; r < L.rounds; ++r) {
      const double m = mass_of(samp_v(L, r, nv));
      double sc = m;
      for (int o = 1; o < 32; o <<= 1) {
        const double t = __shfl_up_sync(kFull, sc, o);
        if (lane >= o) sc += t;
      }
      const double tot = __shfl_sync(kFull, sc, 31);
      const unsigned has = __ballot_sync(kFull, m > 0.0);
      if (has) lastv = L.base + 32 * r + (31 - __clz(has));
      if (tot > rem) {
        const unsigned hit = __ballot_sync(kFull, sc > rem && m > 0.0);
        if (lane == 0) sh[1] = hit ? L.base + 32 * r + __ffs(hit) - 1 : lastv;
        break;
      }
      rem -= tot;
    }
    if (lane == 0 && sh[1] < 0) sh[1] = lastv;
  }
  __syncthreads();
  const int out = sh[1];
  __syncthreads();
  return out;
}
// grid B x T blocks: sequence b = blockIdx.x / T, position j; its parameters prm + b * prm_stride = {temp, top_k, top_p,
// min_p, seed, step} (floats as bits); scr + 16 b; temp <= 0: the greedy accept from the target argmax tok
__global__ void __launch_bounds__(1024) samp_pos(const bf16* logits, int ld, int nv, const int* ids_all, int T,
                                                 const int* prm_all, int prm_stride, const int* tok, int* scr_all) {
  const int b = blockIdx.x / T, j = blockIdx.x % T;
  const int* ids = ids_all + (size_t)b * T;
  const int* prm = prm_all + (size_t)b * prm_stride;
  int* scr = scr_all + 16 * b;
  if (__int_as_float(prm[0]) <= 0.f) {
    if (threadIdx.x == 0) {
      const int g = tok[(size_t)b * T + j];
      scr[j] = j + 1 < T && ids[j + 1] == g, scr[8 + j] = g;
    }
    return;
  }
  const bf16* l = logits + ((size_t)b * T + j) * ld;
  __shared__ float red[32];
  __shared__ double redd[32], wm[32], hm[256], shd[2];
  __shared__ int hist[256], sh[4], wcnt[32];
  const float temp = __int_as_float(prm[0]), topp = __int_as_float(prm[2]), minp = __int_as_float(prm[3]);
  const int topk = prm[1];
  const SampLay L = samp_lay(nv);
  SampSet f;
  f.inv_t = 1.0f / temp, f.minp = minp;
  f.kall = !(topk > 0 && topk < nv), f.tk = 0, f.kcut = nv;
  f.pall = 1, f.tp = 0, f.pcut = nv;
  float mx = -INFINITY;
  for (int r = 0; r < L.rounds; ++r) {
    const int v = samp_v(L, r, nv);
    if (v >= 0) mx = fmaxf(mx, __bfloat162float(l[v]));
  }
  f.mx = block_max(mx, red);
  // top-k: the k-th largest key by two radix passes over counts; its ties kept in index order up to k
  if (!f.kall) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int r = 0; r < L.rounds; ++r) {
      const int v = samp_v(L, r, nv);
      hist_add(hist, v >= 0 ? okey(l[v]) >> 8 : -1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int cum = 0, b = 255;
      for (; b > 0 && cum + hist[b] < topk; --b) cum += hist[b];
      sh[0] = b, sh[1] = cum;
    }
    __syncthreads();
    const int b1 = sh[0], above1 = sh[1];
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int r = 0; r < L.rounds; ++r) {
      const int v = samp_v(L, r, nv);
      const int key = v >= 0 ? okey(l[v]) : -1;
      hist_add(hist, key >= 0 && (key >> 8) == b1 ? (key & 255) : -1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int cum = above1, b = 255;
      for (; b > 0 && cum + hist[b] < topk; --b) cum += hist[b];
      sh[2] = (b1 << 8) | b, sh[3] = topk - cum;
    }
    __syncthreads();
    f.tk = sh[2];
    const int need = sh[3];
    __syncthreads();
    f.kcut = samp_tie_cut(l, nv, L, f.tk, need, f, false, wcnt, sh);
  }
  // top-p: the smallest prefix of A (key descending, ties by index) with mass >= top_p * Z_A, by two radix passes over
  // fp64 masses; the boundary key's ties (equal masses) kept in index order as many as the prefix needs
  if (topp < 1.f) {
    double za = 0.0;
    for (int r = 0; r < L.rounds; ++r) {
      const int v = samp_v(L, r, nv);
      if (v >= 0 && f.inA(v, okey(l[v]))) za += (double)f.mass(__bfloat162float(l[v]));
    }
    const double target = (double)topp * block_sumd(za, redd);
    for (int b = threadIdx.x; b < 256; b += blockDim.x) hm[b] = 0.0;
    __syncthreads();
    for (int r = 0; r < L.rounds; ++r) {
      const int v = samp_v(L, r, nv);
      const int key = v >= 0 ? okey(l[v]) : -1;
      const bool in = key >= 0 && f.inA(v, key);
      hist_addd(hm, in ? key >> 8 : -1, in ? (double)f.mass(__bfloat162float(l[v])) : 0.0);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double cum = 0.0;
      int b = 255;
      for (; b > 0 && cum + hm[b] < target; --b) cum += hm[b];
      sh[0] = b, shd[1] = cum;
    }
    __syncthreads();
    const int b1 = sh[0];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) hm[b] = 0.0;
    __syncthreads();
    for (int r = 0; r < L.rounds; ++r) {
      const int v = samp_v(L, r, nv);
      const int key = v >= 0 ? okey(l[v]) : -1;
      const bool in = key >= 0 && (key >> 8) == b1 && f.inA(v, key);
      hist_addd(hm, in ? key & 255 : -1, in ? (double)f.mass(__bfloat162float(l[v])) : 0.0);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double cum = shd[1];
      int b = 255;
      for (; b > 0 && cum + hm[b] < target; --b) cum += hm[b];
      sh[2] = (b1 << 8) | b;
      const unsigned short kb = (unsigned short)sh[2];
      const unsigned short bits = (kb & 0x8000) ? (unsigned short)(kb & 0x7fff) : (unsigned short)~kb;
      const double m = (double)f.mass(__bfloat162float(__ushort_as_bfloat16(bits)));
      const double n = m > 0.0 ? ceil((target - cum) / m) : 1.0;
      sh[3] = n < 1.0 ? 1 : (n > 1e9 ? 1000000000 : (int)n);
    }
    __syncthreads();
    f.pall = 0, f.tp = sh[2];
    const int need = sh[3];
    __syncthreads();
    const int cut = samp_tie_cut(l, nv, L, f.tp, need, f, true, wcnt, sh);
    f.pcut = cut < 0 ? nv : cut;  // fewer ties than needed (fp64 rounding at the boundary): keep them all
  }
  double zf = 0.0;
  for (int r = 0; r < L.rounds; ++r) {
    const int v = samp_v(L, r, nv);
    if (v >= 0 && f.in(v, l[v])) zf += (double)f.mass(__bfloat162float(l[v]));
  }
  const double z = block_sumd(zf, redd);
  const unsigned long long seed = (unsigned)prm[4], sidx = (unsigned)prm[5];
  curandStatePhilox4_32_10_t rng;
  curand_init(seed, sidx * 64 + j, 0, &rng);
  const float4 u4 = curand_uniform4(&rng);
  if (j + 1 < T) {
    const int x = ids[j + 1];
    const bool inx = x >= 0 && x < nv && f.in(x, l[x]);
    const double px = inx ? (double)f.mass(__bfloat162float(l[x])) : 0.0;
    bool acc = inx && (double)u4.x * z < px;
    if (!acc && z - px <= 0.0) acc = true;  // x is the whole set
    int tok = x;
    if (!acc) tok = samp_draw(l, nv, L, f, x, z - px, u4.y, wm, sh, shd);
    if (threadIdx.x == 0) scr[j] = acc, scr[8 + j] = tok;
  } else {
    const int tok = samp_draw(l, nv, L, f, -1, z, u4.y, wm, sh, shd);
    if (threadIdx.x == 0) scr[j] = 0, scr[8 + j] = tok;
  }
}
// thread b: sequence b's chain (ids + b T, scr + 16 b) -> res + b * res_stride = {n, emitted[..]}
__global__ void samp_chain(const int* ids_all, int B, int T, const int* scr_all, int* res_all, int res_stride) {
  const int b = threadIdx.x;
  if (b >= B) return;
  const int *ids = ids_all + (size_t)b * T, *scr = scr_all + 16 * b;
  int* res = res_all + (size_t)b * res_stride;
  int a = 0;
  while (a + 1 < T && scr[a]) ++a;
  res[0] = a + 1;
  for (int j = 0; j < a; ++j) res[1 + j] = ids[1 + j];
  res[1 + a] = scr[8 + a];
}
// Engram prev pair after committing n of the verify tokens (engram_prev_update with T = n)
__global__ void engram_prev_commit(const int* ids, const long long* pos, const int* res, int* prev) {
  const int n = res[0];
  const int old1 = prev[1];
  const long long np2 = n >= 2 ? ids[n - 2] : (pos[n - 1] >= 1 ? old1 : -1);
  prev[0] = (int)np2;
  prev[1] = ids[n - 1];
}

// ---------------------------------------------------------------- batched speculative verify kernels
// B sequences x kSpecT tokens; row j = 8 b + t. Records (m1draft::kRecInts ints per sequence, device): [1] P, [2] bonus,
// [3..9] drafts, [10..17] pool rows, [18] KDA slot, [19] drafter ring, [20..21] r2t ptr, [22..23] Engram prev ptr,
// [24..25] far-cache tag. Each sequence gets exactly what verify_step_prep / engram_rows_extend / accept_greedy /
// engram_prev_commit give it in a single-sequence verify.
constexpr int kRI = m1draft::kRecInts;
constexpr int kRecDelta = 40;  // batched verify record: the sequence's rope delta ([34..39] sampling)
__global__ void bv_prep(const int* rec, int Wctx, int* ids, long long* pos, int* loc, long long* outloc, int* rng,
                        int* slots, int T, long long* rpos, int rmax) {
  const int b = blockIdx.x, t = threadIdx.x;
  if (t >= T) return;
  const int* r = rec + b * kRI;
  const int j = b * T + t, p = r[1] + t, l = r[10 + t];
  ids[j] = r[2 + t], pos[j] = p, loc[j] = l, outloc[j] = l;
  put_rpos(rpos, rmax, j, (long long)p + r[kRecDelta]);
  rec_ptr(r + 20)[p] = l;
  const int off = b * Wctx;  // ranges index the B x Wctx staged slot maps (r2t_b)
  rng[4 * j] = off + (p + 1 > WIN ? p + 1 - WIN : 0), rng[4 * j + 1] = off + p + 1;
  rng[4 * j + 2] = off, rng[4 * j + 3] = off + p + 1;
  if (t == 0) slots[b] = r[18];
}
__global__ void bv_apply_hint(int* rec) {  // sequence b's host proposal over its first rec[26] drafts
  int* r = rec + blockIdx.x * kRI;
  const int t = threadIdx.x;
  if (t < r[26] && t < kSpecT - 1) r[3 + t] = r[27 + t];
}
__global__ void bv_engram(const int* rec, const int* ids, const long long* pos, long long* r2, long long* r3, int T) {
  const int b = blockIdx.x, t = threadIdx.x;
  if (t >= T) return;
  const int* prev = rec_ptr(rec + b * kRI + 22);
  const int* id = ids + b * T;
  const long long* ps = pos + b * T;
  const long long s0 = prev[0], s1 = prev[1];
  long long p1 = t >= 1 ? id[t - 1] : s1;
  long long p2 = t >= 2 ? id[t - 2] : (t == 1 ? s1 : s0);
  if (ps[t] < 1) p1 = -1;
  if (ps[t] < 2) p2 = -1;
  const long long g2[2] = {p1, id[t]}, g3[3] = {p2, p1, id[t]};
  r2[b * T + t] = (long long)(eg_fold(g2, 2) % ENG_ROWS);
  r3[b * T + t] = (long long)(eg_fold(g3, 3) % ENG_ROWS);
}
__global__ void bv_r2t_stage(const int* rec, int B, int W, int* r2t_b) {  // r2t_b[b][:] = sequence b's slot map
  const long n = (long)B * W;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    r2t_b[i] = rec_ptr(rec + (i / W) * kRI + 20)[i % W];
}
__global__ void bv_accept(const int* ids, const int* tok, int B, int* res, int T) {
  const int b = threadIdx.x;
  if (b >= B) return;
  const int *id = ids + b * T, *tk = tok + b * T;
  int* r = res + b * kBVRes;
  int a = 0;
  while (a + 1 < T && id[a + 1] == tk[a]) ++a;
  r[0] = a + 1;
  for (int j = 0; j < a; ++j) r[1 + j] = id[1 + j];
  r[1 + a] = tk[a];
}
__global__ void bv_engram_commit(const int* rec, const int* ids, const long long* pos, const int* res, int B, int T) {
  const int b = threadIdx.x;
  if (b >= B) return;
  int* prev = rec_ptr(rec + b * kRI + 22);
  const int n = res[b * kBVRes];
  const int old1 = prev[1];
  const long long np2 = n >= 2 ? ids[b * T + n - 2] : (pos[b * T + n - 1] >= 1 ? old1 : -1);
  prev[0] = (int)np2;
  prev[1] = ids[b * T + n - 1];
}
// one sequence's verify ranges (the single-sequence layout of verify_step_prep) from its rows' positions
__global__ void bv_seq_ranges(const long long* pos, int* rng, int T) {
  const int t = threadIdx.x;
  if (t >= T) return;
  const int p = (int)pos[t];
  rng[4 * t] = p + 1 > WIN ? p + 1 - WIN : 0, rng[4 * t + 1] = p + 1;
  rng[4 * t + 2] = 0, rng[4 * t + 3] = p + 1;
}

// ---------------------------------------------------------------- state
struct State {
  int Wctx = 0, rows = 0;  // r2t width (sglv23 req_to_token width) and KV pool rows (+1 trash row = index rows)
  cudaStream_t s = nullptr;
  // per-layer caches
  std::vector<bf16*> kc, vc, idx, blk;  // BCSA/dense: KV [rows+1][4][256]; idx/blk [rows+1][128]
  std::vector<bf16*> conv;              // KDA: [nslots+1][3][KDIM] (slot 0 unused, as sglang's pool)
  std::vector<float*> ssm;              // KDA: [nslots+1][48][128][128]
  std::vector<float*> conv_w;           // fp32 conv weights [KDIM][4]
  int* r2t = nullptr;                   // [1][Wctx] -- the ACTIVE session's (pool rows; unused entries = trash row)
  int* prev = nullptr;                  // the active session's Engram prev pair
  int* slot = nullptr;                  // the active session's KDA state slot (device int32)
  bool* has_init = nullptr;
  void* eng_t[2][2] = {};               // host-mapped tables: [layer idx 0/1][order]
  int seq = 0;                          // tokens processed so far
  // scratch, sized for CHUNK tokens
  bf16 *H, *H2, *hmean, *hmean2, *nrm, *d, *xin, *post, *res, *y, *mixed, *t0, *t1, *t2, *t3, *gu, *act, *big0, *big1;
  bf16 *q, *gate, *k, *v, *o, *qi, *ik, *e, *eg;
  float *lse;
  long long *pos, *reqs, *r2, *r3, *outloc;
  long long* rpos = nullptr;            // rope positions [3][rmax]: MRoPE axes T, H, W of the step's tokens (rope())
  int rmax = 0;
  int *loc, *ids, *qo_dev, *kv_indptr, *kv_idx, *last1, *tok;
  void* rope = nullptr;                 // cos/sin cache of this device
  int *step_d = nullptr, *step_h = nullptr, *tok_h = nullptr;  // decode inputs (device / pinned host), next token (pinned)
  int* rng = nullptr;                   // decode attention KV ranges (decode_step_prep)
  int* slots_b = nullptr;               // batched decode: KDA state slot of each row [kMaxBatch]
  int *bstep_h = nullptr, *bstep_d = nullptr;  // batched decode inputs [kMaxBatch][kBRec] (pinned / device)
  int *r2t_b = nullptr, *rng_b = nullptr;      // batched decode: per-row staged slot map [kMaxBatch][Wctx], ranges [.][4]
  int* dummy_r2t = nullptr;             // padding rows of a batch bucket: all trash row
  int* dummy_prev = nullptr;
  void* far_ws = nullptr;               // engine far branch intermediates (ak_far_decode)
  void* far_wsm = nullptr;              // speculative verify: multi-query far branch (ak_far_decode_multi_src)
  // speculative verify (allocated by ak_m1_spec_init): per KDA layer the verify inputs the commit replays
  // [kSpecT][KDIM] pre-conv qkv, [kSpecT][KDIM] conv output, [kSpecT][KHV*KK] gate, [kSpecT][KHV] beta
  std::vector<bf16*> kst;
  int* vres = nullptr;                  // accept result {n, emitted[..]} (device), mirrored to pinned vres_h
  size_t big0_elems = 0;                // big0 capacity (bf16 elements)
  int* samp_scr = nullptr;              // sampled accept: per-position accept flags / tokens (samp_pos)
  int* vres_h = nullptr;
  bf16* cap = nullptr;                  // drafter context features: [tokens][kCapLayers][HID] (stream 0 at layer entry)
  int cap_rows = 0;
  bf16* wdq = nullptr;                  // W4 prefill: dequantized weight of the current linear (largest W4 n*k)
  // batched speculative verify (server mode, ak_m1_spec_batch): records [kMaxBV][kRecInts] (pinned / device), per-row
  // ranges (window, dense: both in the B x Wctx space of r2t_b), KDA slots, accept results [kMaxBV][16], KDA verify
  // stash for bv_max * 8 rows
  int *bv_h = nullptr, *bv_d = nullptr, *bv_rng = nullptr, *bv_slot = nullptr, *bv_res = nullptr, *bv_res_h = nullptr;
  std::vector<bf16*> kstb;
  int bv_max = 0;
  int dev = 0;
};
// Layer-split (pipeline) over up to 2 devices: SD[k] holds device k's caches + scratch; layers [0, split) on SD[0],
// [split, NL) on SD[1]. Numerics are those of tp1 (no reductions cross devices; the residual streams are copied).
State SD[2];
State* SP = &SD[0];
#define S (*SP)
int g_ndev = 1, g_split = 1 << 30;
int dev_of(int layer) { return layer < g_split ? 0 : 1; }
void use(int k) {
  SP = &SD[k];
  cudaSetDevice(SD[k].dev);
}

// Sessions. KV rows are one pool per device with identical row numbers on every device (each device holds its own
// layers' rows), allocated from one host free list; a row is shared by forks of the session that wrote it (refcount)
// and is never rewritten while referenced, so sharing is exact.
struct Session {
  bool live = false;
  int seq = 0, kslot = 0;
  std::vector<int> rows;                       // host mirror of r2t[0, seq)
  int *r2t[2] = {}, *prev[2] = {}, *slot[2] = {};
  float samp_t = 0.f, samp_p = 1.f, samp_minp = 0.f;  // batched verify sampling (ak_m1_session_sampling; 0 = greedy)
  int samp_k = 0, samp_seed = 0, samp_step = 0;
  long long cache_epoch = 0;  // owner tag of the far-branch union caches (globally unique, see g_cache_epoch)
  int cache_seq = 0;          // position at the last verify step (a step back = a new epoch)
  int rdelta = 0;             // rope position - token position of the next text token (<= 0 after images; vision.md)
};
std::vector<Session> g_sess;
long long g_cache_epoch = 0;  // tags handed out: a session's history may have changed -> a new tag
int g_cur = 0, g_nslots = 1;
// KV rows come in aligned groups of 4 ("pages"): positions 4b..4b+3 of a session (its block b) live in rows
// 4P..4P+3 of one page P, so per-block data (BCSA block keys `blk`, the tiered block-mean pool) is stored once per page,
// indexed row / 4. Pages are refcounted (forks share them); a fork inside a block copies the partial block to a page
// of its own.
std::vector<int> g_free, g_ref;                 // free page stack (top = lowest page), per-page refcount
std::vector<char> g_slot_used;
#define g_seq (g_sess[g_cur].seq)
void bind_session() {  // point every device's State at the active session's per-device buffers
  for (int d = 0; d < g_ndev; ++d) {
    SD[d].r2t = g_sess[g_cur].r2t[d], SD[d].prev = g_sess[g_cur].prev[d], SD[d].slot = g_sess[g_cur].slot[d];
  }
}
void release_rows(Session& ss) {
  for (int k = ((int)ss.rows.size() - 1) & ~3; k >= 0; k -= 4)
    if (--g_ref[ss.rows[k] >> 2] == 0) g_free.push_back(ss.rows[k] >> 2);
  ss.rows.clear();
}
int pages_needed(const Session& ss, int T) {  // new pages for T more positions
  const int have = (int)ss.rows.size() % 4 ? 4 - (int)ss.rows.size() % 4 : 0;
  return T > have ? (T - have + 3) / 4 : 0;
}
int alloc_row(Session& ss) {  // the row of position ss.rows.size(); call pages_needed first
  const int p = (int)ss.rows.size();
  int row;
  if (p % 4 == 0) {
    const int pg = g_free.back();
    g_free.pop_back();
    g_ref[pg] = 1;
    row = 4 * pg;
  } else {
    row = ss.rows[p - 1] + 1;
  }
  ss.rows.push_back(row);
  return row;
}

template <typename T>
T* dalloc(size_t n) {
  void* p = nullptr;
  cudaMalloc(&p, n * sizeof(T) > 0 ? n * sizeof(T) : 16);
  cudaMemset(p, 0, n * sizeof(T));
  return (T*)p;
}
int grid(long n) { return (int)std::min<long>(4096, (n + 255) / 256); }
size_t kda_conv_slot() { return (size_t)3 * KDIM; }        // elements per KDA slot
size_t kda_ssm_slot() { return (size_t)KHV * KK * KK; }
int reset_session_buffers(Session& ss) {  // r2t -> trash row, prev -> (-1,-1), KDA slot -> 0, on every device
  int neg[2] = {-1, -1};
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    std::vector<int> trash(S.Wctx, S.rows);
    CK(cudaMemcpy(ss.r2t[d], trash.data(), (size_t)S.Wctx * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ss.prev[d], neg, 8, cudaMemcpyHostToDevice));
    for (int i = 0; i < NL; ++i)
      if (kMixer[i] == 'k' && dev_of(i) == d) {
        CK(cudaMemset(S.conv[i] + ss.kslot * kda_conv_slot(), 0, kda_conv_slot() * 2));
        CK(cudaMemset(S.ssm[i] + ss.kslot * kda_ssm_slot(), 0, kda_ssm_slot() * 4));
      }
  }
  ss.seq = 0;
  ss.rdelta = 0;
  use(0);
  CK(m1mtp::reset_ring(ss.kslot, S.s));  // MTP context: no pending row
  return 0;
}
void drop_graphs(int id);  // a session's decode graphs captured its KDA slot / ring slot: drop them when those change
int new_session() {  // -> id, or -1 when every KDA slot is taken
  int k = 1;
  while (k <= g_nslots && g_slot_used[k]) ++k;
  if (k > g_nslots) return -1;
  int id = 0;
  while (id < (int)g_sess.size() && g_sess[id].live) ++id;
  if (id == (int)g_sess.size()) g_sess.emplace_back();
  Session& ss = g_sess[id];
  Session keep = ss;  // a reused id keeps its device buffers
  ss = Session{};
  for (int d = 0; d < 2; ++d) ss.r2t[d] = keep.r2t[d], ss.prev[d] = keep.prev[d], ss.slot[d] = keep.slot[d];
  ss.live = true, ss.kslot = k;
  ss.cache_epoch = ++g_cache_epoch;
  g_slot_used[k] = 1;
  drop_graphs(id);  // a reused id may get another slot
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    if (!ss.r2t[d]) ss.r2t[d] = dalloc<int>(S.Wctx), ss.prev[d] = dalloc<int>(2), ss.slot[d] = dalloc<int>(1);
    cudaMemcpy(ss.slot[d], &k, 4, cudaMemcpyHostToDevice);
  }
  if (reset_session_buffers(ss)) return -1;
  return id;
}
// sglv23 plans its decode attention in CUDA-graph mode (decode graphs on, the default); an eager oracle
// (--disable-cuda-graph, e.g. with --cpu-offload-gb) plans without it. AGENS_M1_DECODE_GRAPH=0 matches the latter.
int decode_graph_mode() {
  const char* e = getenv("AGENS_M1_DECODE_GRAPH");
  return e && e[0] == '0' ? 0 : 1;
}
// Decode attention planned on device (ak_attn_decode, bit-identical to the graph-mode oracle, graph-capturable);
// AGENS_M1_ORACLE_ATTN=1 keeps the host-planned oracle (also the eager-plan mode above).
bool dev_attn() {
  static const int on = decode_graph_mode() && !(getenv("AGENS_M1_ORACLE_ATTN") && getenv("AGENS_M1_ORACLE_ATTN")[0] == '1');
  return on;
}
// Far branch on engine kernels (ak_far_decode, bit-identical to the ATen bridge, capturable); AGENS_M1_FAR_BRIDGE=1
// keeps the bridge (then far-regime decode steps stay eager).
bool far_engine() {
  static const bool on = !(getenv("AGENS_M1_FAR_BRIDGE") && getenv("AGENS_M1_FAR_BRIDGE")[0] == '1');
  return on;
}
enum { SLOT_BCSA = 0, SLOT_DENSE = 1 };  // ak_attn_decode plan slots (ranges rng[0..1], rng[2..3])
int plan_decode_attention() {
  CK(ak_attn_decode_plan(SLOT_BCSA, S.rng, QH, KVH, S.s));
  return ak_attn_decode_plan(SLOT_DENSE, S.rng + 2, QH, KVH, S.s);
}

// Tiled W4 prefill: output rows per dequantized tile (AGENS_M1_W4_TILE_MB, default 64 MB of bf16 scratch; 0 = the
// whole weight at once, the A1 path). Rounded down to 256 rows.
size_t g_w4_tile_bytes = 64ull << 20;
int w4_tile_rows(int k, int n) {
  if (!g_w4_tile_bytes) return n;
  const long r = (long)(g_w4_tile_bytes / ((size_t)k * 2)) & ~255L;
  return (int)std::max(256L, std::min<long>(r, n));
}

// ---------------------------------------------------------------- layer pieces
// bf16 linear: prefill / parity decode = cuBLAS as torch issues it; decode rows in a batch see the same bits as bs = 1
int bf16_lin(const bf16* x, const void* wp, bf16* y, int m, int n, int k) {
  if (g_dec && g_binv && m <= kMaxBatch) return ak_bf16_gemv_rows(x, wp, y, m, n, k, S.s);
  if (g_dec && m > 1) {
    for (int r = 0; r < m; ++r) CK(ak_bf16_linear(x + (size_t)r * k, wp, y + (size_t)r * n, 1, n, k, S.s));
    return 0;
  }
  return ak_bf16_linear(x, wp, y, m, n, k, S.s);
}

const LinRef& lin_ref(int layer, const char* wname) {  // layer -1: a model-level linear (lm_head)
  std::vector<LinRef>& v = g_lin_ref[wname];
  if ((int)v.size() < NL + 1) v.resize(NL + 1);
  LinRef& r = v[layer + 1];
  if (!r.resolved) {
    const std::string name = layer >= 0 ? L(layer, wname) : std::string(wname);
    auto q = W4M.find(name);
    if (q != W4M.end()) r.w4 = &q->second;
    else r.bf16 = w(name);
    r.resolved = true;
  }
  return r;
}

int linear(const bf16* x, const char* wname, int layer, bf16* y, int m, int n, int k) {
  const LinRef& r = lin_ref(layer, wname);
  ++g_lin;
  if (r.w4) {
    const W4Lin& t = *r.w4;
    if (t.n != n || t.k != k) {
      fprintf(stderr, "m1: W4 %s (layer %d) is [%d, %d], called as [%d, %d]\n", wname, layer, t.n, t.k, n, k);
      return (int)cudaErrorInvalidValue;
    }
    ++g_lin_w4;
    if (m <= g_w4_batch_max || (g_dec && m <= 128)) {  // decode / verify (batched verify: up to 16 x 8 rows)
      for (int j = 0; j < t.np; ++j) {
        const W4Part& q = t.p[j];
        void* yp = y + q.r0;
        CK(ak_wq_gemv_kt_multi(q.bits, x, m, 1, &q.wt, &q.st, &yp, &q.n, &n, k, q.group, g_w4_bf16w, S.s));
      }
      return 0;
    }
    // prefill: dequantize a tile of output rows at a time into the bounded scratch S.wdq and GEMM that column tile
    // (tiles by output row whatever the parts: the same bf16 matrix and GEMM shapes as a one-part linear)
    const int tn = w4_tile_rows(k, n);
    for (int n0 = 0; n0 < n; n0 += tn) {
      const int nt = std::min(tn, n - n0);
      for (int j = 0; j < t.np; ++j) {
        const W4Part& q = t.p[j];
        const int a0 = std::max(n0, q.r0), a1 = std::min(n0 + nt, q.r0 + q.n);
        if (a0 < a1)
          CK(ak_wq_dequant_kt_rows(q.bits, q.wt, q.st, S.wdq + (size_t)(a0 - n0) * k, q.n, k, q.group, a0 - q.r0, a1 - a0,
                                   S.s));
      }
      if (tn >= n) return ak_bf16_linear_f32reduce(x, S.wdq, y, m, n, k, S.s);
      CK(ak_bf16_linear_f32reduce_ld(x, S.wdq, y + n0, m, nt, k, n, S.s));
    }
    return 0;
  }
  if (!r.bf16) return 1;
  if (g_bi && (m <= 16 || (g_dec && m <= 128))) return ak_bf16_gemv_bi(x, r.bf16, nullptr, y, m, n, k, S.s);  // speculative decoding
  return bf16_lin(x, r.bf16, y, m, n, k);
}

// Linears sharing the input x (same k). Decode with all of them W4 of one group size: one launch
// (ak_w4_gemv_kt_multi, outputs bit-identical to separate GEMVs); otherwise one linear() each.
int linear_shared(const bf16* x, int nl, const char* const* wnames, int layer, bf16* const* ys, const int* ns, int m,
                  int k) {
  if (m == 1 && nl <= 4) {
    const W4Lin* q[4];
    bool ok = true;
    for (int j = 0; j < nl && ok; ++j) {
      q[j] = lin_ref(layer, wnames[j]).w4;
      ok = q[j] && q[j]->np == 1 && q[j]->n == ns[j] && q[j]->k == k;
    }
    if (ok) {  // one launch per (bits, group) class of the linears (mixed precision: up to 3)
      bool done[4] = {false, false, false, false};
      for (int j = 0; j < nl; ++j) {
        if (done[j]) continue;
        const W4Part& pj = q[j]->p[0];
        const void *wt[4], *st[4];
        void* y[4];
        int nn[4], c = 0;
        for (int i = j; i < nl; ++i) {
          const W4Part& pi = q[i]->p[0];
          if (done[i] || pi.bits != pj.bits || pi.group != pj.group) continue;
          wt[c] = pi.wt, st[c] = pi.st, y[c] = ys[i], nn[c] = ns[i], ++c, done[i] = true;
        }
        CK(ak_wq_gemv_kt_multi(pj.bits, x, 1, c, wt, st, y, nn, nullptr, k, pj.group, g_w4_bf16w, S.s));
      }
      g_lin += nl, g_lin_w4 += nl;
      return 0;
    }
  }
  for (int j = 0; j < nl; ++j) CK(linear(x, wnames[j], layer, ys[j], m, ns[j], k));
  return 0;
}

int kda_mixer(int i, const bf16* h, bf16* out, int T, bool prefill) {
  // core [T][48][128] lives in S.o (T*6144); the gated norm writes in place like sglv23 (y = x)
  bf16 *qkv = S.big0, *ag = S.t1, *b = S.t2, *fd = S.t3, *z = S.big1, *core = S.o, *nz = S.o;
  // speculative verify (T tokens of one sequence; batched decode is g_B rows): the projections write straight into
  // the layer's stash, which the commit replays
  const bool verify = !prefill && T > 1 && !g_B;
  bf16* sc = nullptr;
  if (verify && g_BV) {  // batched verify: g_BV sequences of kSpecT rows, stash sized for bv_max sequences
    if (S.kstb.empty() || !S.kstb[i] || T != g_BV * g_TV || g_BV > S.bv_max) return (int)cudaErrorInvalidValue;
    const size_t MR = (size_t)S.bv_max * kSpecT;
    qkv = S.kstb[i], sc = qkv + MR * KDIM, ag = sc + MR * KDIM, b = ag + MR * KHV * KK;
  } else if (verify) {
    if (S.kst.empty() || !S.kst[i] || T > kSpecT) return (int)cudaErrorInvalidValue;
    qkv = S.kst[i], sc = qkv + (size_t)kSpecT * KDIM, ag = sc + (size_t)kSpecT * KDIM, b = ag + (size_t)kSpecT * KHV * KK;
  }
  {  // in_proj_qkv, in_proj_z, f_down share h (in_proj_b is bf16 in the INT4 recipe: fake-quantized)
    const char* nm[3] = {"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.f_down.weight"};
    bf16* ys[3] = {qkv, z, fd};
    const int ns[3] = {KDIM, KHV * KK, 128};
    CK(linear_shared(h, 3, nm, i, ys, ns, T, HID));
  }
  CK(linear(h, "linear_attn.in_proj_b.weight", i, b, T, KHV, HID));
  CK(linear(fd, "linear_attn.f_up.weight", i, ag, T, KHV * KK, 128));
  float* A_log = (float*)WL(i, "linear_attn.A_log.f32");
  float* dt = (float*)WL(i, "linear_attn.dt_bias.f32");
  if (getenv("AGENS_M1_TRACE")) fprintf(stderr, "kda layer %d dev %d T %d seq %d\n", i, S.dev, T, g_seq);
  if (prefill) {
    CK(ak_kda_prefill(qkv, ag, b, S.conv_w[i], A_log, dt, S.conv[i], S.ssm[i], S.slot, S.has_init, T, core, S.s));
  } else if (verify && g_BV) {  // each sequence exactly as its own verify (slots S.bv_slot)
    CK(ak_kda_conv_multi_b(qkv, KDIM, S.conv[i], S.conv_w[i], S.bv_slot, sc, KDIM, g_TV, g_BV, nullptr, 0, 0, S.s));
    CK(ak_kda_decode_multi_b(sc, KDIM, ag, KHV * KK, b, KHV, A_log, dt, S.ssm[i], S.bv_slot, core, 1.0f / sqrtf(128.0f), KH,
                             KHV, KK, KK, g_TV, g_BV, nullptr, 0, 0, S.s));
  } else if (verify) {  // T tokens, state untouched (write_state 0); inputs stashed for the commit replay
    CK(ak_kda_conv_multi(qkv, KDIM, S.conv[i], S.conv_w[i], S.slot, sc, KDIM, T, nullptr, 0, S.s));
    CK(ak_kda_decode_multi(sc, KDIM, ag, KHV * KK, b, KHV, A_log, dt, S.ssm[i], S.slot, core, 1.0f / sqrtf(128.0f), KH,
                           KHV, KK, KK, T, nullptr, 0, S.s));
  } else {
    bf16* qkvc = S.t3;  // conv output
    CK(ak_kda_conv_update(qkv, KDIM, S.conv[i], S.conv_w[i], S.slot, qkvc, T, KDIM, S.s));  // T = decode rows
    CK(ak_kda_decode(qkvc, KDIM, ag, KHV * KK, b, KHV, A_log, dt, S.ssm[i], S.slot, core, 1.0f / sqrtf(128.0f), T, KH,
                     KHV, KK, KK, S.s));
  }
  CK(ak_kda_gated_rmsnorm(core, z, WL(i, "linear_attn.norm.weight"), nz, T * KHV, KK, EPS, S.s));
  return linear(nz, "linear_attn.out_proj.weight", i, out, T, HID, KHV * KK);
}

// RoPE on S.q / S.k: from the registered cos|sin table, or (no table registered: the default) computed in the kernel,
// bit-identical (AGENS_ROPE_CACHE=1 keeps the table: 64 fp32 per position)
// The positions are the rope positions S.rpos (three MRoPE rows): equal rows = the 1-D rope exactly.
int rope(int T) {
  return ak_rope_neox_m(S.q, S.k, (const float*)S.rope, (const int64_t*)S.rpos, S.rmax, T, QH, KVH, HD, ROT, QH * HD,
                        KVH * HD, 10000000.0f, kMropeH, kMropeW, S.s);
}

// AGENS_KV_INT8_DENSE=fake (quality gate input, no memory saved): the dense layer's new K/V rows quantized to INT8 per
// (token, head) and dequantized in place with the INT8 KV kernels (ak_kv_quant_store / ak_kv_dequant_rows) before
// they are stored, so every read sees exactly what an INT8 dense KV would hold
bool dense_int8_fake() {
  static const int on = [] { const char* e = getenv("AGENS_KV_INT8_DENSE"); return e && !strcmp(e, "fake"); }();
  return on;
}
int fake_int8_rows(bf16* x, int T) {
  static int8_t* q = nullptr;
  static float* sc = nullptr;
  if (!q) q = dalloc<int8_t>((size_t)CHUNK * KVH * HD), sc = dalloc<float>((size_t)CHUNK * KVH);
  CK(ak_kv_quant_store(x, T, KVH, HD, nullptr, q, sc, S.s));
  return ak_kv_dequant_rows(q, sc, nullptr, T, KVH, HD, x, S.s);
}

int attn_mixer(int i, const bf16* h, bf16* out, int T, bool prefill, bool bcsa) {
  bf16* qg = S.big0;
  {
    const char* nm[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"};
    bf16* ys[3] = {qg, S.k, S.v};
    const int ns[3] = {QH * 2 * HD, KVH * HD, KVH * HD};
    CK(linear_shared(h, 3, nm, i, ys, ns, T, HID));
  }
  split_q_gate<<<grid((long)T * QH * HD), 256, 0, S.s>>>(qg, S.q, S.gate, T, QH);
  auto dbg_here = [&]() {
    const char* dd = getenv("AGENS_M1_DBG_DIR");
    const char* dl = getenv("AGENS_M1_DBG_LAYER");
    const char* dp = getenv("AGENS_M1_DBG_POS");
    return dd && dl && dp && atoi(dl) == i && atoi(dp) == g_seq && T == 1 ? dd : (const char*)nullptr;
  };
  auto dsave = [&](const char* dd, const char* nm, const void* p, size_t n) {
    cudaStreamSynchronize(S.s);
    std::vector<uint16_t> b(n);
    cudaMemcpy(b.data(), p, n * 2, cudaMemcpyDeviceToHost);
    FILE* f = fopen((std::string(dd) + "/" + nm + ".bin").c_str(), "wb");
    if (f) { fwrite(b.data(), 2, n, f); fclose(f); }
  };
  if (const char* dd = dbg_here()) { dsave(dd, "qg", qg, QH * 2 * HD); dsave(dd, "q_pre", S.q, QH * HD); dsave(dd, "h", h, HID); }
  CK(ak_rmsnorm(S.q, WL(i, "self_attn.q_norm.weight"), S.q, T * QH, HD, HD, HD, EPS, 1, S.s));
  if (const char* dd = dbg_here()) dsave(dd, "q_normed", S.q, QH * HD);
  CK(ak_rmsnorm(S.k, WL(i, "self_attn.k_norm.weight"), S.k, T * KVH, HD, HD, HD, EPS, 1, S.s));
  CK(rope(T));
  // KV tiering / INT8 KV (m1_kv.h): BCSA layers only, AGENS_KV_TIER / AGENS_KV_INT8; off = the code below unchanged
  const bool kvm = bcsa && m1kv::on();
  // graph-safe tiered decode: store / window / far branch read positions, rows and ranges from device memory
  const bool kvg = kvm && !prefill && dev_attn() && far_engine() && m1kv::graph_safe();
  const m1kv::Ctx kx{(int)(SP - SD), i, g_sess[g_cur].kslot, g_seq, T, prefill, S.s, S.r2t, S.Wctx, S.rows};
  if (kvg && T > 1 && multi_attn()) {  // speculative verify: the T tokens' stores batched (same rows, same values)
    CK(m1kv::store_dec_multi(kx, T, S.k, S.v, S.loc, S.pos));
  } else if (kvg) {  // speculative verify: token by token, each as its decode step stores it
    for (int t = 0; t < T; ++t)
      CK(m1kv::store_dec(kx, S.k + (size_t)t * KVH * HD, S.v + (size_t)t * KVH * HD, S.loc + t, S.pos + t));
  } else if (kvm) CK(m1kv::store(kx, S.k, S.v, S.loc));
  else if (!bcsa && m1d8::on()) CK(m1d8::store(S.k, S.v, S.loc, T, S.s));  // INT8 dense KV
  else {
    if (!bcsa && dense_int8_fake()) {  // quality probe: the dense layer's K/V as an INT8 KV would serve them
      CK(fake_int8_rows(S.k, T));
      CK(fake_int8_rows(S.v, T));
    }
    store_kv<<<grid((long)T * KVH * HD), 256, 0, S.s>>>(S.k, S.v, S.kc[i], S.vc[i], S.loc, T, KVH);
  }
  const float scale = 1.0f / 16.0f;
  if (bcsa) {
    // indexer key -> idx_cache; block keys (decode: here; extend: inside the far bridge, before the far branch)
    CK(linear(h, "self_attn.idx_k.weight", i, S.ik, T, IDXD, HID));
    CK(ak_rmsnorm(S.ik, WL(i, "self_attn.idx_norm.weight"), S.ik, T, IDXD, IDXD, IDXD, EPS, 1, S.s));
    store_rows<<<grid((long)T * IDXD), 256, 0, S.s>>>(S.ik, S.idx[i], S.loc, T, IDXD);
    if (!prefill) block_keys_decode<<<T, IDXD, 0, S.s>>>(S.pos, S.loc, S.r2t, S.Wctx, S.idx[i], S.blk[i], S.rows);
  }
  // KV list: BCSA = last min(seq, W) tokens; dense = all tokens (r2t row is the identity slot map here)
  const int seq_after = g_seq + T;
  const bool kv_far_dec = kvm && !kvg && !prefill && seq_after > WIN;
  if (kv_far_dec) {  // indexer top-k first, so the selected rows' fetch overlaps the local branch
    CK(linear(h, "self_attn.idx_q.weight", i, S.qi, T, IDXH * IDXD, HID));
    CK(ak_rmsnorm(S.qi, WL(i, "self_attn.idx_norm.weight"), S.qi, T * IDXH, IDXD, IDXD, IDXD, EPS, 1, S.s));
    CK(m1kv::far_prefetch(kx, S.qi, S.pos, S.reqs, S.blk[i]));
  }
  // verify, tiered, every query far: the index queries, top-k and the union fetch (side stream) before the window
  // attention, which then overlaps the fetch
  const bool far_multi = bcsa && kvg && T > 1 && g_seq + 1 > WIN && S.far_wsm;
  if (far_multi) {
    CK(linear(h, "self_attn.idx_q.weight", i, S.qi, T, IDXH * IDXD, HID));
    CK(ak_rmsnorm(S.qi, WL(i, "self_attn.idx_norm.weight"), S.qi, T * IDXH, IDXD, IDXD, IDXD, EPS, 1, S.s));
    CK(m1kv::far_select_fetch_multi(kx, T, S.qi, S.pos, S.blk[i], S.far_wsm,
                                    g_tag_ptr ? g_tag_ptr : reinterpret_cast<const long long*>(S.step_d + 30)));
  }
  int kv_len = bcsa && !prefill ? std::min(seq_after, WIN) : seq_after;
  int kv_start = seq_after - kv_len;
  int kv_indptr_h[2] = {0, kv_len}, qo_h[2] = {0, T};
  if (!bcsa && m1d8::on()) {  // INT8 dense KV: tiled attention (decode and verify share the per-query partition)
    if (prefill) CK(m1d8::attend_prefill(S.q, g_seq, T, S.r2t, scale, S.o, S.s));
    else CK(m1d8::attend_dec(S.q, S.pos, S.r2t, T, (g_seq + T + m1d8::TILE - 1) / m1d8::TILE, scale, S.o, S.s));
  } else if (kvg && T > 1 && g_multi_win) {  // verify, far regime: the T windows in one launch
    CK(m1kv::local_dec_multi(kx, MSLOT_WIN, S.q, scale, S.o, S.lse));
  } else if (kvg) {  // verify: query t on its own plan slot (2 + 2t), the ring dequantized once (INT8)
    for (int t = 0; t < T; ++t)
      CK(m1kv::local_dec(kx, S.q + (size_t)t * QH * HD, S.rng + 4 * t, scale, S.o + (size_t)t * QH * HD, S.lse + t * QH,
                         T > 1 ? 2 + 2 * t : 0, t == 0));
  } else if (kvm) {
    CK(m1kv::local(kx, S.q, S.qo_dev, S.kv_indptr, S.kv_idx, S.last1, scale, S.o, S.lse, decode_graph_mode()));
  } else if (!prefill && dev_attn() && T > 1) {  // verify: every query exactly as its own decode step (own plan)
    if (bcsa && g_multi_wdev)  // BCSA windows, all-GPU pools: one launch (stands down on device as the dense one)
      CK(ak_attn_decode_multi(MSLOT_WIN, S.q, S.kc[i], S.vc[i], QH, KVH, scale, S.o, S.lse, S.s));
    if (!bcsa && g_multi_dense)  // dense layer: one launch (disabled on device when the bs = 1 chunk sizes differ,
      CK(dense_packed() ? ak_attn_decode_packed(MSLOT_PACKED, S.q, S.kc[i], S.vc[i], S.r2t, QH, KVH, scale, S.o, S.lse, S.s)
                        : ak_attn_decode_multi(MSLOT_DENSE, S.q, S.kc[i], S.vc[i], QH, KVH, scale, S.o, S.lse, S.s));
    for (int t = 0; t < T && !(bcsa && g_win2); ++t)
      CK(ak_attn_decode(2 + 2 * t + (bcsa ? 0 : 1), S.q + (size_t)t * QH * HD, S.kc[i], S.vc[i], S.r2t,
                        S.rng + 4 * t + (bcsa ? 0 : 2), QH, KVH, scale, S.o + (size_t)t * QH * HD, S.lse + t * QH, S.s));
  } else if (!prefill && dev_attn()) {  // plan written once per step (plan_decode_attention), KV ranges on device
    CK(ak_attn_decode(bcsa ? SLOT_BCSA : SLOT_DENSE, S.q, S.kc[i], S.vc[i], S.r2t, S.rng + (bcsa ? 0 : 2), QH, KVH,
                      scale, S.o, S.lse, S.s));
  } else {
    cudaMemcpyAsync(S.kv_indptr, kv_indptr_h, sizeof(kv_indptr_h), cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.qo_dev, qo_h, sizeof(qo_h), cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.kv_idx, S.r2t + kv_start, (size_t)kv_len * 4, cudaMemcpyDeviceToDevice, S.s);
    if (prefill) {
      if (bcsa)
        CK(ak_oracle_bcsa_prefill(S.q, S.kc[i], S.vc[i], S.qo_dev, S.kv_indptr, S.kv_idx, S.last1, qo_h, kv_indptr_h, 1,
                                  QH, KVH, WIN - 1, scale, S.o, S.lse, S.s));
      else
        CK(ak_oracle_dense_prefill(S.q, S.kc[i], S.vc[i], S.qo_dev, S.kv_indptr, S.kv_idx, S.last1, qo_h, kv_indptr_h, 1,
                                   QH, KVH, scale, S.o, S.s));
    } else {
      CK(ak_oracle_bcsa_local(S.q, S.kc[i], S.vc[i], S.rows + 1, S.kv_indptr, S.kv_idx, S.last1, kv_indptr_h, qo_h,
                              S.qo_dev, 1, QH, KVH, scale, S.o, S.lse, decode_graph_mode(), S.s));
    }
  }
  bf16* attn_out = S.o;
  if (const char* dd = getenv("AGENS_M1_DBG_DIR")) {  // debug: q,k,v (post-RoPE), o_loc of one layer/position
    const char* dl = getenv("AGENS_M1_DBG_LAYER");
    const char* dp = getenv("AGENS_M1_DBG_POS");
    if (dl && dp && atoi(dl) == i && atoi(dp) == g_seq && T == 1) {
      cudaStreamSynchronize(S.s);
      auto save = [&](const char* nm, const void* p, size_t n) {
        std::vector<uint16_t> b(n);
        cudaMemcpy(b.data(), p, n * 2, cudaMemcpyDeviceToHost);
        FILE* f = fopen((std::string(dd) + "/" + nm + ".bin").c_str(), "wb");
        if (f) { fwrite(b.data(), 2, n, f); fclose(f); }
      };
      save("q", S.q, QH * HD); save("k", S.k, KVH * HD); save("v", S.v, KVH * HD); save("o_loc", S.o, QH * HD);
    }
  }
  if (bcsa) {
    scale_f32<<<(T * QH + 255) / 256, 256, 0, S.s>>>(S.lse, T * QH, 0.6931471805599453f);
    if (!kv_far_dec && !far_multi) {
      CK(linear(h, "self_attn.idx_q.weight", i, S.qi, T, IDXH * IDXD, HID));
      CK(ak_rmsnorm(S.qi, WL(i, "self_attn.idx_norm.weight"), S.qi, T * IDXH, IDXD, IDXD, IDXD, EPS, 1, S.s));
    }
    if (prefill && kvm) {
      CK(m1kv::far_extend(kx, S.qi, S.q, S.o, S.lse, S.idx[i], S.blk[i], S.t3));
      CK(m1kv::commit_chunk(kx));
      attn_out = S.t3;
    } else if (prefill) {
      CK(ak_bridge_far_extend(S.qi, S.q, S.o, S.lse, seq_after, T, 0, S.r2t, 1, S.Wctx, S.kc[i], S.vc[i], S.idx[i],
                              S.blk[i], S.rows, S.t3, S.s));
      attn_out = S.t3;
    } else if (kvg && seq_after > WIN && T == 1) {  // far regime: its own decode graph, like the all-GPU far branch
      CK(m1kv::far_dec(kx, S.qi, S.q, S.o, S.lse, S.pos, S.blk[i], S.far_ws, S.t3));
      attn_out = S.t3;
    } else if (far_multi) {  // verify, tiered, every query far: the union fetched above, now the attend half
      CK(m1kv::far_attend_multi(kx, T, S.q, S.o, S.lse, S.pos, S.far_wsm, S.t3));
      attn_out = S.t3;
    } else if (kvg && seq_after > WIN) {  // verify, tiered: per query, as its decode step
      for (int t = 0; t < T; ++t) {
        if (g_seq + t + 1 > WIN)
          CK(m1kv::far_dec(kx, S.qi + (size_t)t * IDXH * IDXD, S.q + (size_t)t * QH * HD, S.o + (size_t)t * QH * HD,
                           S.lse + t * QH, S.pos + t, S.blk[i], S.far_ws, S.t3 + (size_t)t * QH * HD));
        else
          CK(cudaMemcpyAsync(S.t3 + (size_t)t * QH * HD, S.o + (size_t)t * QH * HD, (size_t)QH * HD * 2,
                             cudaMemcpyDeviceToDevice, S.s));
      }
      attn_out = S.t3;
    } else if (kv_far_dec) {
      CK(m1kv::far_attend(kx, S.q, S.o, S.lse, S.t3));
      attn_out = S.t3;
    } else if (T > 1 && g_seq + 1 > WIN && far_engine() && S.far_wsm) {  // verify, all queries far, all-GPU pools
      CK(ak_far_decode_multi_src(T, S.qi, S.q, S.o, S.lse, S.pos, S.r2t, S.Wctx, 0, S.kc[i], S.vc[i], nullptr, nullptr,
                                 S.blk[i], S.rows, S.far_wsm, S.t3, S.s));
      attn_out = S.t3;
    } else if (T > 1 && g_seq + T > WIN) {  // verify: query t (position g_seq + t) takes the far branch iff its
      bool any = false;                       // decode step would (all-near verify: the local output as is)
      for (int t = 0; t < T; ++t) {
        if (g_seq + t + 1 > WIN) {
          if (!far_engine()) return (int)cudaErrorNotSupported;
          CK(ak_far_decode(S.qi + (size_t)t * IDXH * IDXD, S.q + (size_t)t * QH * HD, S.o + (size_t)t * QH * HD,
                           S.lse + t * QH, S.pos + t, S.r2t, S.Wctx, S.kc[i], S.vc[i], S.blk[i], S.rows, S.far_ws,
                           S.t3 + (size_t)t * QH * HD, S.s));
          any = true;
        } else {
          CK(cudaMemcpyAsync(S.t3 + (size_t)t * QH * HD, S.o + (size_t)t * QH * HD, (size_t)QH * HD * 2,
                             cudaMemcpyDeviceToDevice, S.s));
        }
      }
      if (any) attn_out = S.t3;
    } else if (seq_after > WIN) {  // sglv23 "far" decode graph; "near" returns the local output as is
      if (far_engine())
        CK(ak_far_decode(S.qi, S.q, S.o, S.lse, S.pos, S.r2t, S.Wctx, S.kc[i], S.vc[i], S.blk[i], S.rows, S.far_ws,
                         S.t3, S.s));
      else
        CK(ak_bridge_far_decode(S.qi, S.q, S.o, S.lse, S.pos, S.reqs, S.r2t, 1, S.Wctx, S.kc[i], S.vc[i], S.rows,
                                S.blk[i], 1, S.t3, S.s));
      attn_out = S.t3;
    }
  }
  CK(ak_bf16_binary(attn_out, S.gate, S.t2, (long)T * QH * HD, 2, S.s));  // out * sigmoid(gate)
  return linear(S.t2, "self_attn.o_proj.weight", i, out, T, HID, QH * HD);
}

int engram(int i, int which, bf16* y, int T) {
  gather_engram<<<T, 128, 0, S.s>>>(S.r2, S.r3, (const uint4*)S.eng_t[which][0], (const uint4*)S.eng_t[which][1],
                                     (uint4*)S.e, T);
  CK(ak_rmsnorm(S.e, WL(i, "engram.norm.weight"), S.e, T, 2 * ENG_DIM, 2 * ENG_DIM, 2 * ENG_DIM, EPS, 0, S.s));
  if (g_bi && (T <= 16 || (g_dec && T <= 128))) {  // speculative decoding: batch-invariant (verify rows == decode)
    CK(ak_bf16_gemv_bi(y, WL(i, "engram.gate.weight"), WL(i, "engram.gate.bias"), S.eg, T, 1, HID, S.s));
  } else if (g_dec && T > 1) {  // per row, as bs = 1 (cuBLAS addmm is not batch-invariant in general)
    for (int r = 0; r < T; ++r)
      CK(ak_bf16_linear_bias(y + (size_t)r * HID, WL(i, "engram.gate.weight"), WL(i, "engram.gate.bias"), S.eg + r, 1, 1,
                             HID, S.s));
  } else {
    CK(ak_bf16_linear_bias(y, WL(i, "engram.gate.weight"), WL(i, "engram.gate.bias"), S.eg, T, 1, HID, S.s));
  }
  CK(ak_sigmoid_to_bf16(S.eg, S.eg, T, S.s));
  CK(linear(S.e, "engram.value.weight", i, S.t2, T, HID, 2 * ENG_DIM));
  CK(ak_scale_rows(S.eg, S.t2, S.t2, T, HID, S.s));
  return ak_bf16_binary(y, S.t2, y, (long)T * HID, 0, S.s);
}

// Batched decode, attention layers: one row at a time with that row's session bound (positions / pool rows offset,
// its decode attention planned), exactly the bs = 1 call. Restores the batch's slot array afterwards.
int attn_rows(int i, const bf16* h, bf16* a, bool bcsa) {
  long long *pos0 = S.pos, *out0 = S.outloc, *rp0 = S.rpos;
  int* loc0 = S.loc;
  for (int r = 0; r < g_B; ++r) {
    g_cur = g_brow[r];
    bind_session();
    S.pos = pos0 + r, S.outloc = out0 + r, S.loc = loc0 + r, S.rpos = rp0 + r;
    int e = 0;
    if (dev_attn()) {
      decode_ranges<<<1, 1, 0, S.s>>>(S.pos, S.rng);
      e = plan_decode_attention();
    }
    if (!e) e = attn_mixer(i, h + (size_t)r * HID, a + (size_t)r * HID, 1, false, bcsa);
    S.pos = pos0, S.outloc = out0, S.loc = loc0, S.rpos = rp0;
    if (e) return e;
  }
  for (int d = 0; d < g_ndev; ++d) SD[d].slot = SD[d].slots_b;
  return 0;
}

// Batched attention layer on device session tables (one device, decode attention planned on device, engine far branch):
// projections / norms / RoPE / K,V and indexer-key stores on all B rows; per row the bs = 1 kernels with that row's
// staged slot map (S.r2t_b[r]) and ranges (S.rng_b[r]): block keys, plan + decode attention, far branch (in a step
// where any row is past the window, every row runs it: a row without far blocks gets exactly its local output).
int attn_batch(int i, const bf16* h, bf16* out, bool bcsa) {
  const int B = g_B;
  bf16* qg = S.big0;
  {
    const char* nm[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"};
    bf16* ys[3] = {qg, S.k, S.v};
    const int ns[3] = {QH * 2 * HD, KVH * HD, KVH * HD};
    CK(linear_shared(h, 3, nm, i, ys, ns, B, HID));
  }
  split_q_gate<<<grid((long)B * QH * HD), 256, 0, S.s>>>(qg, S.q, S.gate, B, QH);
  CK(ak_rmsnorm(S.q, WL(i, "self_attn.q_norm.weight"), S.q, B * QH, HD, HD, HD, EPS, 1, S.s));
  CK(ak_rmsnorm(S.k, WL(i, "self_attn.k_norm.weight"), S.k, B * KVH, HD, HD, HD, EPS, 1, S.s));
  CK(rope(B));
  const bool kvm = bcsa && m1kv::on();
  const m1kv::Ctx kx{(int)(SP - SD), i, 0, 0, B, false, S.s, S.r2t_b, S.Wctx, S.rows};
  if (kvm) CK(m1kv::store_rows_dec(kx, S.k, S.v, B, S.loc, S.pos, S.slots_b));
  else store_kv<<<grid((long)B * KVH * HD), 256, 0, S.s>>>(S.k, S.v, S.kc[i], S.vc[i], S.loc, B, KVH);
  const float scale = 1.0f / 16.0f;
  if (bcsa) {
    CK(linear(h, "self_attn.idx_k.weight", i, S.ik, B, IDXD, HID));
    CK(ak_rmsnorm(S.ik, WL(i, "self_attn.idx_norm.weight"), S.ik, B, IDXD, IDXD, IDXD, EPS, 1, S.s));
    store_rows<<<grid((long)B * IDXD), 256, 0, S.s>>>(S.ik, S.idx[i], S.loc, B, IDXD);
    for (int r = 0; r < B; ++r)
      block_keys_decode<<<1, IDXD, 0, S.s>>>(S.pos + r, S.loc + r, S.r2t_b + (size_t)r * S.Wctx, S.Wctx, S.idx[i], S.blk[i],
                                             S.rows);
  }
  const int slot = bcsa ? SLOT_BCSA : SLOT_DENSE;
  for (int r = 0; r < B; ++r) {
    int* rng = S.rng_b + 4 * r + (bcsa ? 0 : 2);
    bf16 *q = S.q + (size_t)r * QH * HD, *o = S.o + (size_t)r * QH * HD;
    float* lse = S.lse + (size_t)r * QH;
    CK(ak_attn_decode_plan(slot, rng, QH, KVH, S.s));
    if (kvm && m1kv::int8()) CK(m1kv::local_dec_row_i8(kx, r, S.slots_b, q, rng, scale, o, lse));
    else if (kvm) CK(m1kv::local_dec_row(kx, r, q, rng, scale, o, lse));
    else CK(ak_attn_decode(slot, q, S.kc[i], S.vc[i], S.r2t_b + (size_t)r * S.Wctx, rng, QH, KVH, scale, o, lse, S.s));
  }
  bf16* attn_out = S.o;
  if (bcsa) {
    scale_f32<<<(B * QH + 255) / 256, 256, 0, S.s>>>(S.lse, B * QH, 0.6931471805599453f);
    CK(linear(h, "self_attn.idx_q.weight", i, S.qi, B, IDXH * IDXD, HID));
    CK(ak_rmsnorm(S.qi, WL(i, "self_attn.idx_norm.weight"), S.qi, B * IDXH, IDXD, IDXD, IDXD, EPS, 1, S.s));
    if (g_bfar) {
      for (int r = 0; r < B; ++r) {
        bf16 *qi = S.qi + (size_t)r * IDXH * IDXD, *q = S.q + (size_t)r * QH * HD, *o = S.o + (size_t)r * QH * HD;
        bf16* y = S.t3 + (size_t)r * QH * HD;
        float* lse = S.lse + (size_t)r * QH;
        const int* r2t = S.r2t_b + (size_t)r * S.Wctx;
        if (kvm) CK(m1kv::far_dec_row(kx, qi, q, o, lse, S.pos + r, r2t, S.blk[i], S.far_ws, y));
        else CK(ak_far_decode(qi, q, o, lse, S.pos + r, r2t, S.Wctx, S.kc[i], S.vc[i], S.blk[i], S.rows, S.far_ws, y, S.s));
      }
      attn_out = S.t3;
    }
  }
  CK(ak_bf16_binary(attn_out, S.gate, S.t2, (long)B * QH * HD, 2, S.s));  // out * sigmoid(gate)
  return linear(S.t2, "self_attn.o_proj.weight", i, out, B, HID, QH * HD);
}

// Batched speculative verify, attention layers. All-GPU pools: projections / norms / RoPE / stores on all 8 B rows,
// one multi-request attention launch for the B x 8 queries (each request its own bs = 1 chunk size: bit-identical to
// its own decode call), the far branch per sequence (as its single-sequence verify). Tiered / INT8 KV: per sequence,
// the single-sequence verify layer itself (attn_mixer on its 8 rows, its plans).
constexpr int MSLOT_BVWIN = 6, MSLOT_BVDENSE = 7;
int plan_verify(int T);
std::vector<int> g_bv_reg;  // per sequence: 0 near, 1 every query far, 2 mixed
int attn_bv(int i, const bf16* h, bf16* out, bool bcsa) {
  const int B = g_BV, TV = g_TV, R = B * TV;
  if (m1kv::on() || (!bcsa && m1d8::on())) {
    long long *pos0 = S.pos, *out0 = S.outloc, *rp0 = S.rpos;
    int* loc0 = S.loc;
    const int cur0 = g_cur;
    int e = 0;
    for (int b = 0; b < B && !e; ++b) {
      g_cur = g_bv_sess[b];
      bind_session();
      S.pos = pos0 + b * TV, S.outloc = out0 + b * TV, S.loc = loc0 + b * TV, S.rpos = rp0 + b * TV;
      g_tag_ptr = reinterpret_cast<const long long*>(S.bv_d + b * kRI + 24);
      bv_seq_ranges<<<1, 32, 0, S.s>>>(S.pos, S.rng, TV);
      const int sv = g_BV;
      g_BV = 0;  // the single-sequence verify layer
      e = plan_verify(TV);
      if (!e) e = attn_mixer(i, h + (size_t)b * TV * HID, out + (size_t)b * TV * HID, TV, false, bcsa);
      g_BV = sv;
    }
    S.pos = pos0, S.outloc = out0, S.loc = loc0, S.rpos = rp0, g_tag_ptr = nullptr;
    g_cur = cur0;
    bind_session();
    return e;
  }
  bf16* qg = S.big0;
  {
    const char* nm[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"};
    bf16* ys[3] = {qg, S.k, S.v};
    const int ns[3] = {QH * 2 * HD, KVH * HD, KVH * HD};
    CK(linear_shared(h, 3, nm, i, ys, ns, R, HID));
  }
  split_q_gate<<<grid((long)R * QH * HD), 256, 0, S.s>>>(qg, S.q, S.gate, R, QH);
  CK(ak_rmsnorm(S.q, WL(i, "self_attn.q_norm.weight"), S.q, R * QH, HD, HD, HD, EPS, 1, S.s));
  CK(ak_rmsnorm(S.k, WL(i, "self_attn.k_norm.weight"), S.k, R * KVH, HD, HD, HD, EPS, 1, S.s));
  CK(rope(R));
  store_kv<<<grid((long)R * KVH * HD), 256, 0, S.s>>>(S.k, S.v, S.kc[i], S.vc[i], S.loc, R, KVH);
  const float scale = 1.0f / 16.0f;
  if (bcsa) {
    CK(linear(h, "self_attn.idx_k.weight", i, S.ik, R, IDXD, HID));
    CK(ak_rmsnorm(S.ik, WL(i, "self_attn.idx_norm.weight"), S.ik, R, IDXD, IDXD, IDXD, EPS, 1, S.s));
    store_rows<<<grid((long)R * IDXD), 256, 0, S.s>>>(S.ik, S.idx[i], S.loc, R, IDXD);
    block_keys_decode<<<R, IDXD, 0, S.s>>>(S.pos, S.loc, S.r2t_b, S.Wctx, S.idx[i], S.blk[i], S.rows, TV);
  }
  CK(ak_attn_decode_multi(bcsa ? MSLOT_BVWIN : MSLOT_BVDENSE, S.q, S.kc[i], S.vc[i], QH, KVH, scale, S.o, S.lse, S.s));
  bf16* attn_out = S.o;
  if (bcsa) {
    scale_f32<<<(R * QH + 255) / 256, 256, 0, S.s>>>(S.lse, R * QH, 0.6931471805599453f);
    bool any = false;
    for (int b = 0; b < B; ++b) any |= g_bv_reg[b] != 0;
    if (any) {
      CK(linear(h, "self_attn.idx_q.weight", i, S.qi, R, IDXH * IDXD, HID));
      CK(ak_rmsnorm(S.qi, WL(i, "self_attn.idx_norm.weight"), S.qi, R * IDXH, IDXD, IDXD, IDXD, EPS, 1, S.s));
      for (int b = 0; b < B; ++b) {
        const size_t r0 = (size_t)b * TV;
        const int* r2t = S.r2t_b + (size_t)b * S.Wctx;
        const int P = g_sess[g_bv_sess[b]].seq;
        if (g_bv_reg[b] == 1 && S.far_wsm) {  // every query far: the multi-query far branch
          CK(ak_far_decode_multi_src(TV, S.qi + r0 * IDXH * IDXD, S.q + r0 * QH * HD, S.o + r0 * QH * HD, S.lse + r0 * QH,
                                     S.pos + r0, r2t, S.Wctx, 0, S.kc[i], S.vc[i], nullptr, nullptr, S.blk[i], S.rows,
                                     S.far_wsm, S.t3 + r0 * QH * HD, S.s));
        } else {  // per query, as its decode step (near queries: their local output)
          for (int t = 0; t < TV; ++t) {
            const size_t q0 = r0 + t;
            if (g_bv_reg[b] != 0 && P + t + 1 > WIN)
              CK(ak_far_decode(S.qi + q0 * IDXH * IDXD, S.q + q0 * QH * HD, S.o + q0 * QH * HD, S.lse + q0 * QH, S.pos + q0,
                               r2t, S.Wctx, S.kc[i], S.vc[i], S.blk[i], S.rows, S.far_ws, S.t3 + q0 * QH * HD, S.s));
            else
              CK(cudaMemcpyAsync(S.t3 + q0 * QH * HD, S.o + q0 * QH * HD, (size_t)QH * HD * 2, cudaMemcpyDeviceToDevice,
                                 S.s));
          }
        }
      }
      attn_out = S.t3;
    }
  }
  CK(ak_bf16_binary(attn_out, S.gate, S.t2, (long)R * QH * HD, 2, S.s));  // out * sigmoid(gate)
  return linear(S.t2, "self_attn.o_proj.weight", i, out, R, HID, QH * HD);
}

// AGENS_M1_FUSE=0: the residual adds as separate launches (default: fused into the post-attention norm and the mHC
// combine; bit-identical)
bool fuse_on() {
  static const int on = [] { const char* e = getenv("AGENS_M1_FUSE"); return !(e && e[0] == '0'); }();
  return on;
}
const bf16* g_yb = nullptr;  // the layer's mlp output when its last residual add is left to the combine
int layer(int i, int T, bool prefill) {
  bf16 *h = S.t0, *a = S.t1;
  CK(ak_rmsnorm(S.xin, WL(i, "input_layernorm.weight"), h, T, HID, HID, HID, EPS, 1, S.s));
  if (kMixer[i] == 'k') CK(kda_mixer(i, h, a, T, prefill));
  else if (g_BV) CK(attn_bv(i, h, a, kMixer[i] == 'b'));
  else if (g_B && g_bdev) CK(attn_batch(i, h, a, kMixer[i] == 'b'));
  else if (g_B) CK(attn_rows(i, h, a, kMixer[i] == 'b'));
  else CK(attn_mixer(i, h, a, T, prefill, kMixer[i] == 'b'));
  if (g_tp > 1) CK(ak_tp_allreduce(a, (long)T * HID, S.s));  // the ranks' head shares of the mixer output
  if (const char* dd = g_B ? nullptr : getenv("AGENS_M1_DUMP_DIR")) {  // debug: attention mixer output (sglv23 dump's att{i})
    if (kMixer[i] != 'k') {
      std::vector<uint16_t> hbuf((size_t)T * HID);
      cudaMemcpyAsync(hbuf.data(), a, hbuf.size() * 2, cudaMemcpyDeviceToHost, S.s);
      cudaStreamSynchronize(S.s);
      std::string fn = std::string(dd) + "/att" + std::to_string(i) + "_seq" + std::to_string(g_seq) + ".bin";
      FILE* f = fopen(fn.c_str(), "wb");
      if (f) { fwrite(hbuf.data(), 2, hbuf.size(), f); fclose(f); }
    }
  }
  bf16* h2 = S.y;
  if (fuse_on()) {  // x + mixer and the post-attention norm in one launch (same bits)
    CK(ak_add_rmsnorm(S.xin, a, h2, WL(i, "post_attention_layernorm.weight"), h, T, HID, EPS, 1, S.s));
  } else {
    CK(ak_bf16_binary(S.xin, a, h2, (long)T * HID, 0, S.s));                         // x + mixer
    CK(ak_rmsnorm(h2, WL(i, "post_attention_layernorm.weight"), h, T, HID, HID, HID, EPS, 1, S.s));
  }
  CK(linear(h, "mlp.gate_up_proj.weight", i, S.gu, T, 2 * INTER, HID));
  CK(ak_silu_mul(S.gu, S.act, T, INTER, S.s));
  CK(linear(S.act, "mlp.down_proj.weight", i, a, T, HID, INTER));
  if (g_tp > 1) CK(ak_tp_allreduce(a, (long)T * HID, S.s));  // the ranks' INTER shares of the MLP output
  g_yb = nullptr;
  if (fuse_on() && i != kEngramLayer[0] && i != kEngramLayer[1]) {
    g_yb = a;  // + mlp: folded into the mHC combine (S.y = x + mixer, a = mlp)
    return 0;
  }
  CK(ak_bf16_binary(h2, a, S.y, (long)T * HID, 0, S.s));                             // + mlp
  if (i == kEngramLayer[0]) CK(engram(i, 0, S.y, T));
  if (i == kEngramLayer[1]) CK(engram(i, 1, S.y, T));
  return 0;
}

m1draft::Target draft_target();
int run_model(int T, bool prefill, int R = 1);
int finish_step(int* next, int T, std::chrono::steady_clock::time_point t_start);

// Multimodal prefill input (ak_m1_prefill_mm): for the n tokens of the call, rope positions [3][n]
// (MRoPE rows T, H, W; null: position + the session's rope delta) and per token the row of `emb` ([rows][HID] bf16,
// host) that replaces its token embedding (-1: the token's own). `s` = offset of the chunk being forwarded.
struct MmIn {
  const long long* rpos;
  const bf16* emb;
  const int* row;
  int n, s;
};
MmIn* g_mm = nullptr;
// the step's rope positions into S.rpos on every device (host rows rp [3][T])
void stage_rpos(const std::vector<long long>& rp, int T) {
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    for (int a = 0; a < 3; ++a)
      cudaMemcpyAsync(S.rpos + (size_t)a * S.rmax, rp.data() + (size_t)a * T, (size_t)T * 8, cudaMemcpyHostToDevice, S.s);
  }
}

int forward(const int* ids_host, int T, bool prefill, int* next) {
  const auto t_start = std::chrono::steady_clock::now();
  g_lin = g_lin_w4 = 0;
  // positions, pool rows for the new tokens (appended to the session's r2t), request row 0 -- staged on every device
  Session& ss = g_sess[g_cur];
  if (g_seq + T > SD[0].Wctx || ss.kslot <= 0) return (int)cudaErrorInvalidValue;  // kslot 0: parked
  if ((int)g_free.size() < pages_needed(ss, T)) return (int)cudaErrorMemoryAllocation;
  std::vector<long long> pos(T);
  std::vector<int> loc(T);
  for (int t = 0; t < T; ++t) pos[t] = g_seq + t, loc[t] = alloc_row(ss);
  std::vector<long long> rp((size_t)3 * T);
  for (int a = 0; a < 3; ++a)
    for (int t = 0; t < T; ++t)
      rp[(size_t)a * T + t] = g_mm && g_mm->rpos ? g_mm->rpos[(size_t)a * g_mm->n + g_mm->s + t] : pos[t] + ss.rdelta;
  if (g_mm && g_mm->rpos) ss.rdelta = (int)(rp[T - 1] - pos[T - 1]);  // the chunk's last token (text at a prompt's end)
  stage_rpos(rp, T);
  std::vector<long long> loc64(loc.begin(), loc.end());
  long long zero = 0;
  bool hi = g_seq > 0;
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    cudaMemcpyAsync(S.pos, pos.data(), T * 8, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.loc, loc.data(), T * 4, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.r2t + g_seq, loc.data(), T * 4, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.ids, ids_host, T * 4, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.reqs, &zero, 8, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.outloc, loc64.data(), (size_t)T * 8, cudaMemcpyHostToDevice, S.s);  // out_cache_loc (int64)
    cudaMemcpyAsync(S.has_init, &hi, 1, cudaMemcpyHostToDevice, S.s);
    // Engram rows (layers 2 and 22) + prev-id rotation, identically on every device (each keeps its own prev pair)
    engram_rows_extend<<<(T + 127) / 128, 128, 0, S.s>>>(S.ids, S.pos, T, S.prev, S.r2, S.r3);
    engram_prev_update<<<1, 1, 0, S.s>>>(S.ids, S.pos, T, S.prev);
  }
  if (T == 1 && !prefill && dev_attn())
    for (int d = 0; d < g_ndev; ++d) {
      use(d);
      decode_ranges<<<1, 1, 0, S.s>>>(S.pos, S.rng);
      CK(plan_decode_attention());
    }
  CK(run_model(T, prefill));
  if (prefill && g_capture && m1draft::ready() && T <= S.cap_rows && g_ndev == 1) {
    m1draft::set_ring(ss.kslot);  // the session's drafter context (one ring per KDA slot with batched speculation)
    CK(m1draft::materialize(draft_target(), S.cap, T, nullptr, S.pos));
  }
  if (m1mtp::ready() && g_ndev == 1) {  // MTP context entries of the forwarded rows (in passes of max_rows)
    m1mtp::set_ring(ss.kslot);
    CK(m1mtp::materialize(draft_target(), WG("model.language_model.norm.weight"), S.H, NS * HID, T, nullptr, S.ids, S.pos));
  }
  return finish_step(next, T, t_start);
}

bool batch_dev();
int rows_dev(const int* sess, const int* tok, int B, const std::vector<long long>& pos, const std::vector<int>& loc,
             const std::vector<int>& slots, int* next, std::chrono::steady_clock::time_point t_start);

// Batched decode: row r feeds tok[r] to session sess[r] (distinct live sessions holding a KDA slot); next[r] = its
// greedy token. Per-row setup (pool row, position, r2t entry, Engram rows + prev rotation) is the bs = 1 setup.
int forward_rows(const int* sess, const int* tok, int B, int* next) {
  if (m1d8::on()) return (int)cudaErrorNotSupported;  // INT8 dense KV: personal mode (one sequence) only
  const auto t_start = std::chrono::steady_clock::now();
  g_lin = g_lin_w4 = 0;
  if (B < 1 || B > kMaxBatch) return (int)cudaErrorInvalidValue;
  int need = 0;
  for (int r = 0; r < B; ++r) {
    const Session& ss = g_sess[sess[r]];
    if (!ss.live || ss.kslot <= 0 || ss.seq + 1 > SD[0].Wctx) return (int)cudaErrorInvalidValue;
    for (int q = 0; q < r; ++q)
      if (sess[q] == sess[r]) return (int)cudaErrorInvalidValue;
    need += pages_needed(ss, 1);
  }
  if ((int)g_free.size() < need) return (int)cudaErrorMemoryAllocation;
  const int cur0 = g_cur;
  std::vector<long long> pos(B);
  std::vector<int> loc(B), slots(B);
  for (int r = 0; r < B; ++r) {
    Session& ss = g_sess[sess[r]];
    pos[r] = ss.seq, loc[r] = alloc_row(ss), slots[r] = ss.kslot;
  }
  if (batch_dev()) return rows_dev(sess, tok, B, pos, loc, slots, next, t_start);
  {
    std::vector<long long> rp((size_t)3 * B);
    for (int a = 0; a < 3; ++a)
      for (int r = 0; r < B; ++r) rp[(size_t)a * B + r] = pos[r] + g_sess[sess[r]].rdelta;
    stage_rpos(rp, B);
  }
  std::vector<long long> loc64(loc.begin(), loc.end());
  long long zero = 0;
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    cudaMemcpyAsync(S.pos, pos.data(), B * 8, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.loc, loc.data(), B * 4, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.ids, tok, B * 4, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.reqs, &zero, 8, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.outloc, loc64.data(), (size_t)B * 8, cudaMemcpyHostToDevice, S.s);
    cudaMemcpyAsync(S.slots_b, slots.data(), B * 4, cudaMemcpyHostToDevice, S.s);
    for (int r = 0; r < B; ++r) {
      Session& ss = g_sess[sess[r]];
      cudaMemcpyAsync(ss.r2t[d] + ss.seq, &loc[r], 4, cudaMemcpyHostToDevice, S.s);
      engram_rows_extend<<<1, 128, 0, S.s>>>(S.ids + r, S.pos + r, 1, ss.prev[d], S.r2 + r, S.r3 + r);
      engram_prev_update<<<1, 1, 0, S.s>>>(S.ids + r, S.pos + r, 1, ss.prev[d]);
    }
    S.slot = S.slots_b;
  }
  g_B = B;
  g_brow.assign(sess, sess + B);
  int e = run_model(B, false, B);
  g_B = 0;
  g_cur = cur0;
  bind_session();
  if (e) return e;
  CK(cudaMemcpyAsync(next, S.tok, (size_t)B * 4, cudaMemcpyDeviceToHost, S.s));
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  for (int r = 0; r < B; ++r) ++g_sess[sess[r]].seq;
  return 0;
}

// Embeddings .. greedy token of the last R positions (R = B rows of a batched decode) into S.tok of the last device.
// Reads only device state (no host copies, no syncs unless a debug dump is on), so a decode step of it can be captured
// into a CUDA graph.
// INT8 lm_head (registered as lm_head.weight.i8 [VOCAB][HID] int8 + lm_head.weight.scale [VOCAB] fp32): the head when
// no bf16 lm_head.weight is registered; with both (comparison runs) the bf16 head decodes and ak_m1_logits_i8_row
// gives the INT8 logits of the same hidden state.
bool lmh_bf16() {  // a bf16 or W4 (AGENS_M1_LMHEAD_W4) head through linear(); else the INT8 head
  static const bool b = W.count("lm_head.weight") > 0 || W4M.count("lm_head.weight") > 0;
  return b;
}
int lm_head(const bf16* x, bf16* y, int m) {
  if (lmh_bf16()) return linear(x, "lm_head.weight", -1, y, m, VOCAB, HID);
  for (int r = 0; r < m; r += 16)
    CK(ak_i8_gemv(x + (size_t)r * HID, WG("lm_head.weight.i8"), WG("lm_head.weight.scale"), y + (size_t)r * VOCAB,
                  m - r < 16 ? m - r : 16, VOCAB, HID, VOCAB, S.s));
  return 0;
}
int run_model(int T, bool prefill, int R) {
  g_dec = !prefill;
  // embeddings -> 4 identical streams; h_mean = embedding (mean of 4 equal values is exact)
  use(dev_of(0));
  CK(ak_embed(S.ids, WG("model.language_model.embed_tokens.weight"), S.t0, T, HID, S.s));
  if (g_mm && g_mm->row && prefill) {  // image tokens: their embedding rows replaced by the tower's output (vision.md)
    const int* row = g_mm->row + g_mm->s;
    for (int t = 0; t < T;) {
      if (row[t] < 0) {
        ++t;
        continue;
      }
      int e = t + 1;
      while (e < T && row[e] == row[e - 1] + 1) ++e;
      CK(cudaMemcpyAsync(S.t0 + (size_t)t * HID, g_mm->emb + (size_t)row[t] * HID, (size_t)(e - t) * HID * 2,
                         cudaMemcpyHostToDevice, S.s));
      t = e;
    }
  }
  broadcast_streams<<<grid((long)T * NS * HID), 256, 0, S.s>>>(S.t0, S.H, T);
  cudaMemcpyAsync(S.hmean, S.t0, (size_t)T * HID * 2, cudaMemcpyDeviceToDevice, S.s);
  bf16 *H = S.H, *Hn = S.H2, *hm = S.hmean, *hmn = S.hmean2;
  for (int i = 0; i < NL; ++i) {
    if (i > 0 && dev_of(i) != dev_of(i - 1)) {  // pipeline hand-off: residual streams + their mean to the next device
      State& from = *SP;
      from.H = H, from.H2 = Hn, from.hmean = hm, from.hmean2 = hmn;
      CK(cudaStreamSynchronize(from.s));
      use(dev_of(i));
      CK(cudaMemcpyPeerAsync(S.H, S.dev, H, from.dev, (size_t)T * NS * HID * 2, S.s));
      CK(cudaMemcpyPeerAsync(S.hmean, S.dev, hm, from.dev, (size_t)T * HID * 2, S.s));
      H = S.H, Hn = S.H2, hm = S.hmean, hmn = S.hmean2;
    }
    if (g_seg_capture && i > 0 && i % std::max(1, NL / graph_segs()) == 0 && NL - i >= NL / graph_segs() / 2) {
      // segmented verify graph: close this segment and continue the capture in the next (same stream, same order)
      cudaGraph_t g = nullptr;
      CK(cudaStreamEndCapture(S.s, &g));
      g_seg_capture->push_back(g);
      CK(cudaStreamBeginCapture(S.s, cudaStreamCaptureModeThreadLocal));
    }
    if (g_capture && T <= S.cap_rows)
      for (int c = 0; c < kCapLayers; ++c)
        if (g_cap_layer[c] == i)  // drafter context feature c: residual stream 0 at this layer's entry
          CK(cudaMemcpy2DAsync(S.cap + (size_t)c * HID, (size_t)kCapLayers * HID * 2, H, (size_t)NS * HID * 2,
                               (size_t)HID * 2, T, cudaMemcpyDeviceToDevice, S.s));
    CK(ak_rmsnorm(hm, WM(i, "norm.weight"), S.nrm, T, HID, HID, HID, EPS, 0, S.s));
    if (g_bi && (T <= 16 || (g_dec && T <= 128))) CK(ak_bf16_gemv_bi(S.nrm, WM(i, "dyn.weight"), nullptr, S.d, T, 24, HID, S.s));
    else CK(bf16_lin(S.nrm, WM(i, "dyn.weight"), S.d, T, 24, HID));
    CK(ak_mhc_mix(S.d, 24, WM(i, "pre_static"), WM(i, "post_static"), WM(i, "res_static"), H, NS * HID, HID, S.xin,
                  HID, S.post, S.res, T, HID, 10, S.s));
    if (const char* dd = getenv("AGENS_M1_DUMP_DIR")) {  // debug: layer inputs (sglv23 VOLUNDR_DUMP's in{i})
      std::vector<uint16_t> hbuf((size_t)T * HID);
      cudaMemcpyAsync(hbuf.data(), S.xin, hbuf.size() * 2, cudaMemcpyDeviceToHost, S.s);
      cudaStreamSynchronize(S.s);
      std::string fn = std::string(dd) + "/in" + std::to_string(i) + "_seq" + std::to_string(g_seq) + ".bin";
      FILE* f = fopen(fn.c_str(), "wb");
      if (f) { fwrite(hbuf.data(), 2, hbuf.size(), f); fclose(f); }
    }
    CK(layer(i, T, prefill));
    if (g_bi && (T <= 16 || (g_dec && T <= 128))) CK(ak_bf16_bmm_bi(S.res, H, S.mixed, T, NS, NS, HID, S.s));
    else CK(ak_bf16_bmm(S.res, H, S.mixed, T, NS, NS, HID, S.s));
    if (g_yb)
      CK(ak_mhc_combine_add(S.mixed, NS * HID, HID, S.post, S.y, g_yb, HID, S.xin, HID, Hn, NS * HID, HID,
                            i + 1 < NL ? hmn : nullptr, HID, T, HID, S.s));
    else
      CK(ak_mhc_combine(S.mixed, NS * HID, HID, S.post, S.y, HID, S.xin, HID, Hn, NS * HID, HID, i + 1 < NL ? hmn : nullptr,
                        HID, T, HID, S.s));
    std::swap(H, Hn);
    std::swap(hm, hmn);
  }
  S.H = H, S.H2 = Hn, S.hmean = hm, S.hmean2 = hmn;
  // final norm on stream 0 of the last R tokens (speculative verify: of every token), lm_head, mask + greedy
  const int nh = !prefill && T > 1 && !g_B ? T : R;
  const bf16* last = H + (size_t)(T - nh) * NS * HID;
  CK(ak_rmsnorm(last, WG("model.language_model.norm.weight"), S.t0, nh, HID, NS * HID, HID, EPS, 1, S.s));
  CK(lm_head(S.t0, S.big0, nh));
  CK(ak_argmax(S.big0, nh, VOCAB, VOCAB_VALID, S.tok, S.s));
  if (getenv("AGENS_M1_DUMP_LOGITS")) {  // debug: top-5 logits of this step
    std::vector<uint16_t> lg(VOCAB_VALID);
    cudaMemcpyAsync(lg.data(), S.big0, (size_t)VOCAB_VALID * 2, cudaMemcpyDeviceToHost, S.s);
    cudaStreamSynchronize(S.s);
    std::vector<std::pair<float, int>> v;
    for (int j = 0; j < VOCAB_VALID; ++j) {
      uint32_t u = (uint32_t)lg[j] << 16;
      float f;
      memcpy(&f, &u, 4);
      v.push_back({f, j});
    }
    std::partial_sort(v.begin(), v.begin() + 5, v.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    fprintf(stderr, "logits pos %d:", g_seq + T - 1);
    for (int j = 0; j < 5; ++j) fprintf(stderr, " %d:%.6g", v[j].second, v[j].first);
    fprintf(stderr, "\n");
  }
  return 0;
}

int finish_step(int* next, int T, std::chrono::steady_clock::time_point t_start) {
  CK(cudaMemcpyAsync(next, S.tok, 4, cudaMemcpyDeviceToHost, S.s));
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  g_seq += T;
  return 0;
}

// ---------------------------------------------------------------- decode graphs (one device)
// A decode step reads its inputs (token, position, pool row) from a pinned staging buffer and derives everything else
// on device (decode_step_prep, device-planned attention), so the whole step -- staging copy .. greedy token copied to
// pinned host memory -- is captured once per (session, regime) and replayed: the same kernels with the same
// arguments, so replay numerics are those of the eager step. Regimes: near (seq <= W, no far branch) and far. The
// first step of a (session, regime) runs eagerly (initialises cuBLAS / plan workspaces), the second captures.
// Pointers captured per session (slot map, Engram pair, KDA slot) are why the key includes the session.
std::map<std::pair<int, int>, cudaGraphExec_t> g_graphs;
std::map<std::pair<int, int>, int> g_warm;
// speculative verify: (session, regime [+ 4 drafted]) -> the step's graph in segments (layers split into
// AGENS_M1_GRAPH_SEGS parts, default 4): launching segment s + 1 on the host overlaps the GPU running segment s,
// instead of the GPU waiting for one long launch of every node (same kernels, same stream order)
std::map<std::pair<int, int>, std::vector<cudaGraphExec_t>> g_vgraphs;

std::map<std::pair<int, int>, int> g_vwarm;
void drop_graphs(int id) {  // every graph of session id (any regime / tile count)
  for (auto it = g_graphs.begin(); it != g_graphs.end();)
    if (it->first.first == id) cudaGraphExecDestroy(it->second), it = g_graphs.erase(it);
    else ++it;
  for (auto it = g_warm.begin(); it != g_warm.end();) it = it->first.first == id ? g_warm.erase(it) : std::next(it);
  for (auto it = g_vgraphs.begin(); it != g_vgraphs.end();)
    if (it->first.first == id) {
      for (auto ex : it->second) cudaGraphExecDestroy(ex);
      it = g_vgraphs.erase(it);
    } else {
      ++it;
    }
  for (auto it = g_vwarm.begin(); it != g_vwarm.end();) it = it->first.first == id ? g_vwarm.erase(it) : std::next(it);
}
// INT8 dense KV (16gb / 12gb profiles): a single-sequence graph key holds the dense layer's tile count (one more every
// TILE = 8192 positions) above its low bits (decode: far, verify: regime + drafted). A sequence never goes back to
// fewer tiles, so the graph of a new tile count replaces that session's graphs with the same low bits (1.1.1 fix: 1.1.0
// kept every tile count's graphs, ~4 verify segments each, until the session ended; a ~100K-token generation filled the
// 16 GiB cap and cudaGraphInstantiate failed with out of memory)
template <typename G>
void drop_other_tiles(std::map<std::pair<int, int>, G>& graphs, std::map<std::pair<int, int>, int>& warm,
                      std::pair<int, int> key, int low_bits, void (*destroy)(G&)) {
  const int mask = (1 << low_bits) - 1;
  for (auto it = graphs.begin(); it != graphs.end();)
    if (it->first.first == key.first && (it->first.second & mask) == (key.second & mask) && it->first != key) {
      destroy(it->second);
      warm.erase(it->first);
      it = graphs.erase(it);
    } else {
      ++it;
    }
}
void destroy_exec(cudaGraphExec_t& ex) { cudaGraphExecDestroy(ex); }
void destroy_execs(std::vector<cudaGraphExec_t>& exs) {
  for (auto ex : exs) cudaGraphExecDestroy(ex);
}
void drop_all_graphs();  // every cached graph (single-sequence and batched), defined after the batched caches
// Instantiate captured graph segments (consumed). An instantiation needs device memory, which under the --budget-gib
// cap can run out: then every cached graph is freed (recaptured on demand) and it retries once; false when that fails
// too (error state cleared, exs empty): the caller runs the step eagerly, with the same kernels and results.
bool instantiate(std::vector<cudaGraph_t>& segs, std::vector<cudaGraphExec_t>& exs, const char* what) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    cudaError_t e = cudaSuccess;
    exs.clear();
    for (cudaGraph_t g : segs) {
      cudaGraphExec_t ex = nullptr;
      if ((e = cudaGraphInstantiate(&ex, g, 0)) != cudaSuccess) break;
      exs.push_back(ex);
    }
    if (e == cudaSuccess) break;
    destroy_execs(exs);
    exs.clear();
    cudaGetLastError();
    fprintf(stderr, "m1: %s graph instantiation failed (%s): %s\n", what, cudaGetErrorString(e),
            attempt ? "running the step eagerly" : "freeing the cached graphs and retrying");
    if (!attempt) drop_all_graphs();
  }
  for (cudaGraph_t g : segs) cudaGraphDestroy(g);
  segs.clear();
  return !exs.empty();
}
constexpr int kNoGraph = -(1 << 30);  // warm count of a key whose graph could not be instantiated: eager from then on
bool graphs_on() {
  static const bool on = [] {
    const char* e = getenv("AGENS_M1_GRAPH");
    const bool dbg = getenv("AGENS_M1_DUMP_DIR") || getenv("AGENS_M1_DBG_DIR") || getenv("AGENS_M1_DUMP_LOGITS") ||
                     getenv("AGENS_M1_TRACE");
    return !(e && e[0] == '0') && !dbg;
  }();
  // even NL: the residual ping-pong ends where it started; KV tiering / INT8 KV (m1_kv) plans on the host: eager
  return on && g_ndev == 1 && NL % 2 == 0 && dev_attn() && (!m1kv::on() || m1kv::graph_safe());
}
int decode_body() {
  CK(cudaMemcpyAsync(S.step_d, S.step_h, 5 * sizeof(int), cudaMemcpyHostToDevice, S.s));
  decode_step_prep<<<1, 1, 0, S.s>>>(S.step_d, S.ids, S.pos, S.loc, S.outloc, S.r2t, S.rng, S.rpos, S.rmax);
  engram_rows_extend<<<1, 128, 0, S.s>>>(S.ids, S.pos, 1, S.prev, S.r2, S.r3);
  engram_prev_update<<<1, 1, 0, S.s>>>(S.ids, S.pos, 1, S.prev);
  CK(plan_decode_attention());
  CK(run_model(1, false));
  return (int)cudaMemcpyAsync(S.tok_h, S.tok, sizeof(int), cudaMemcpyDeviceToHost, S.s);
}
int decode1(int token, int* next) {
  const auto t_start = std::chrono::steady_clock::now();
  Session& ss = g_sess[g_cur];
  if (g_seq + 1 > SD[0].Wctx || ss.kslot <= 0) return (int)cudaErrorInvalidValue;
  if ((int)g_free.size() < pages_needed(ss, 1)) return (int)cudaErrorMemoryAllocation;
  const int loc = alloc_row(ss);
  use(0);
  S.step_h[0] = token, S.step_h[1] = g_seq, S.step_h[2] = loc, S.step_h[3] = 0, S.step_h[4] = ss.rdelta;
  const bool far = g_seq + 1 > WIN;
  // INT8 dense KV: the step's tile count is part of the graph (a new graph every TILE positions)
  const int d8t = m1d8::on() ? (g_seq + 1 + m1d8::TILE - 1) / m1d8::TILE : 0;
  const auto key = std::make_pair(g_cur, (int)far + 2 * d8t);
  auto it = g_graphs.find(key);
  if (it != g_graphs.end()) {
    CK(cudaGraphLaunch(it->second, S.s));
  } else if (graphs_on() && g_warm[key] > 0 && (!far || far_engine())) {
    cudaGraph_t gph = nullptr;
    g_lin = g_lin_w4 = 0;
    CK(cudaStreamBeginCapture(S.s, cudaStreamCaptureModeThreadLocal));
    const int e = decode_body();
    const cudaError_t ec = cudaStreamEndCapture(S.s, &gph);
    if (e || ec) {
      fprintf(stderr, "m1: decode graph capture failed (%d, %s)\n", e, cudaGetErrorString(ec));
      return e ? e : (int)ec;
    }
    drop_other_tiles(g_graphs, g_warm, key, 1, destroy_exec);
    std::vector<cudaGraph_t> segs{gph};
    std::vector<cudaGraphExec_t> exs;
    if (instantiate(segs, exs, "decode")) {
      g_graphs[key] = exs[0];
      CK(cudaGraphLaunch(exs[0], S.s));
    } else {
      g_lin = g_lin_w4 = 0;
      CK(decode_body());
      g_warm[key] = kNoGraph;
    }
  } else {
    g_lin = g_lin_w4 = 0;
    CK(decode_body());
    ++g_warm[key];
  }
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  *next = *S.tok_h;
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  g_seq += 1;
  return 0;
}

// ---------------------------------------------------------------- speculative verify (one device)
// One verify step: tokens [bonus, d1..d(T-1)] at positions g_seq.. through the target in batch-invariant mode (each
// token gets the bits of its own decode step), greedy accept on device, then the commit -- KDA conv / recurrent state
// replayed over the n accepted tokens from the stashed verify inputs (the verify itself never writes the state),
// Engram pair -- all on device, so the step is one graph per (session, regime). The host frees the KV rows of the
// rejected tokens after reading n. Regimes: 0 near (no query takes the far branch), 1 far (all do), 2 mixed (eager).
m1draft::Target draft_target() {
  // the INT8 head when it is registered (AGENS_M1_LMHEAD_INT8=1, or =draft beside the bf16 head in native mode)
  static const bool i8_draft = [] {
    const char* e = getenv("AGENS_M1_LMHEAD_INT8");
    return W.count("lm_head.weight.i8") > 0 && !(e && !strcmp(e, "cmp"));
  }();
  if (!lmh_bf16() || i8_draft)
    return m1draft::Target{S.s, WG("model.language_model.embed_tokens.weight"), nullptr, VOCAB_VALID,
                           WG("lm_head.weight.i8"), (const float*)WG("lm_head.weight.scale")};
  m1draft::Target t{S.s, WG("model.language_model.embed_tokens.weight"), lin_ref(-1, "lm_head.weight").bf16,
                    VOCAB_VALID};
  if (auto it = W4M.find("lm_head.weight.w4draft"); it != W4M.end()) t.lm_head_w4 = it->second.p[0].wt, t.lm_head_w4s = it->second.p[0].st;
  return t;
}
int spec_commit() {
  const float scale = 1.0f / sqrtf(128.0f);
  for (int i = 0; i < NL; ++i) {
    if (kMixer[i] != 'k') continue;
    bf16* st = S.kst[i];
    bf16 *sq = st, *sc = st + (size_t)kSpecT * KDIM, *sa = sc + (size_t)kSpecT * KDIM, *sb = sa + (size_t)kSpecT * KHV * KK;
    CK(ak_kda_conv_multi(sq, KDIM, S.conv[i], S.conv_w[i], S.slot, nullptr, KDIM, kSpecT, S.vres, 1, S.s));
    CK(ak_kda_decode_multi(sc, KDIM, sa, KHV * KK, sb, KHV, (float*)WL(i, "linear_attn.A_log.f32"),
                           (float*)WL(i, "linear_attn.dt_bias.f32"), S.ssm[i], S.slot, nullptr, scale, KH, KHV, KK, KK,
                           kSpecT, S.vres, 1, S.s));
  }
  engram_prev_commit<<<1, 1, 0, S.s>>>(S.ids, S.pos, S.vres, S.prev);
  return (int)cudaGetLastError();
}
// the attention plans of one sequence's T verify queries (ranges S.rng, slot map S.r2t, position g_seq)
int plan_verify(int T) {
  // all-GPU pools: the BCSA windows of the T queries in one multi-request launch with each request's own bs = 1 chunk
  // size (mode 2: always valid, bit-identical to its decode step), so no per-query window plans / launches
  const bool win2 = !m1kv::on() && multi_win_dev() && multi_attn() && T > 1 && win_mode2();
  for (int t = 0; t < T; ++t) {
    if (!win2) CK(ak_attn_decode_plan(2 + 2 * t, S.rng + 4 * t, QH, KVH, S.s));
    CK(ak_attn_decode_plan(3 + 2 * t, S.rng + 4 * t + 2, QH, KVH, S.s));
  }
  // one launch for the T queries' attention where their bs = 1 plans allow it (bit-identical; see ak_attn_decode_multi):
  // dense layer always (falls back on device to the per-query launches), the tiered BCSA window in the far regime
  g_multi_dense = g_multi_win = g_multi_wdev = g_win2 = false;
  if (multi_attn() && T > 1) {
    static const int dslots[8] = {3, 5, 7, 9, 11, 13, 15, 17};
    if (dense_packed()) CK(ak_attn_decode_packed_plan(MSLOT_PACKED, T, S.rng + 2, 4, QH, KVH, dslots, S.s));
    else CK(ak_attn_decode_multi_plan(MSLOT_DENSE, T, S.rng + 2, 4, S.r2t, S.Wctx, QH, KVH, dslots, S.s));
    g_multi_dense = true;
    if (m1kv::on() && m1kv::graph_safe() && !m1kv::int8() && g_seq + 1 > WIN) {  // = verify regime 1
      CK(ak_attn_decode_multi_plan(MSLOT_WIN, T, S.rng, 4, m1kv::ring_index(SP - SD), WIN + 8, QH, KVH, nullptr, S.s));
      g_multi_win = true;
    } else if (win2) {
      CK(ak_attn_decode_multi_plan_mode(MSLOT_WIN, T, S.rng, 4, S.r2t, S.Wctx, QH, KVH, nullptr, 2, S.s));
      g_multi_wdev = g_win2 = true;
    } else if (!m1kv::on() && multi_win_dev()) {  // all-GPU pools: each query's window range over the row map
      static const int wslots[8] = {2, 4, 6, 8, 10, 12, 14, 16};  // the per-query BCSA slots stand down on device
      CK(ak_attn_decode_multi_plan(MSLOT_WIN, T, S.rng, 4, S.r2t, S.Wctx, QH, KVH, wslots, S.s));
      g_multi_wdev = true;
    }
  }
  return 0;
}
// MTP chain steps per drafted verify (AGENS_MTP_STEPS, <= 7): default all 7 slots alone, 4 with the DFlash2 drafter
// (whose drafts then fill slots 4..6; 1.2 sweep on a 48 GB sm89 card, 16 GiB, shortlist head: chain 4 >= chain 3 > chain 2 in
// tok/s on every class but yue, and the only one ahead of DFlash2 alone on tool calls)
int mtp_steps() {
  const char* e = getenv("AGENS_MTP_STEPS");
  const int n = e ? atoi(e) : (m1draft::ready() ? 4 : kSpecT - 1);
  return n < 1 ? 1 : (n > kSpecT - 1 ? kSpecT - 1 : n);
}
int verify_body(int T, bool draft) {
  CK(cudaMemcpyAsync(S.step_d, S.step_h, kStepInts * sizeof(int), cudaMemcpyHostToDevice, S.s));
  if (draft) {
    if (m1draft::ready()) CK(m1draft::propose(draft_target(), S.step_d));  // writes the 7 drafts into the staged tokens
    apply_hint<<<1, 32, 0, S.s>>>(S.step_d);
    // MTP chain (hybrid drafting: host proposal, then MTP through it, then the DFlash2 drafts in the slots beyond)
    if (m1mtp::ready()) CK(m1mtp::propose(draft_target(), S.step_d, mtp_steps()));
  }
  verify_step_prep<<<1, kSpecT, 0, S.s>>>(S.step_d, S.ids, S.pos, S.loc, S.outloc, S.r2t, S.rng, S.rpos, S.rmax);
  engram_rows_extend<<<1, 128, 0, S.s>>>(S.ids, S.pos, T, S.prev, S.r2, S.r3);
  CK(plan_verify(T));
  CK(run_model(T, false));
  if (g_temp > 0.f) {  // sampled accept over the request's filtered distribution (temperature, top-k / top-p / min-p)
    samp_pos<<<T, 1024, 0, S.s>>>(S.big0, VOCAB, VOCAB_VALID, S.ids, T, S.step_d + 32, 0, S.tok, S.samp_scr);
    samp_chain<<<1, 32, 0, S.s>>>(S.ids, 1, T, S.samp_scr, S.vres, 0);
  } else
    accept_greedy<<<1, 1, 0, S.s>>>(S.ids, S.tok, T, S.vres);
  CK(spec_commit());
  if (m1draft::ready() && g_capture)  // drafter context of the committed tokens
    CK(m1draft::materialize(draft_target(), S.cap, kSpecT, S.vres, S.pos));
  if (m1mtp::ready())  // MTP context of the committed tokens (+ the new pending row)
    CK(m1mtp::materialize(draft_target(), WG("model.language_model.norm.weight"), S.H, NS * HID, kSpecT, S.vres, S.ids,
                          S.pos));
  CK(cudaMemcpyAsync(S.vres_h, S.vres, (1 + kSpecT) * sizeof(int), cudaMemcpyDeviceToHost, S.s));
  CK(cudaMemcpyAsync(S.vres_h + 1 + kSpecT, S.tok, kSpecT * sizeof(int), cudaMemcpyDeviceToHost, S.s));
  return (int)cudaMemcpyAsync(S.vres_h + 1 + 2 * kSpecT, S.ids, kSpecT * sizeof(int), cudaMemcpyDeviceToHost, S.s);
}
// tokens == nullptr: the drafter proposes (T = 8, tokens[0] = bonus passed in *n_out on entry)
int g_hint_n = 0, g_hint[kSpecT];  // the next drafted verify's host proposal (ak_m1_spec_step_hint)
int verify1(const int* tokens, int T, int* emitted, int* n_out, int* target, int* drafts) {
  const auto t_start = std::chrono::steady_clock::now();
  use(0);
  const bool draft = tokens == nullptr;
  if (T < 1 || T > kSpecT || S.kst.empty() || (draft && (!(m1draft::ready() || m1mtp::ready()) || T != kSpecT)) ||
      g_seq + T > S.Wctx) {
    fprintf(stderr, "m1: verify: T %d (max %d), spec_init %s, drafter %s, seq %d + T > Wctx %d\n", T, kSpecT,
            S.kst.empty() ? "missing" : "done", m1draft::ready() ? "ready" : "not ready", g_seq, S.Wctx);
    return (int)cudaErrorInvalidValue;
  }
  if (m1kv::on() && !(m1kv::graph_safe() && dev_attn() && far_engine())) {
    fprintf(stderr, "m1: verify with tiered / INT8 KV needs the graph-safe tiered path (AGENS_KV_TIER=1|2, no "
                    "AGENS_KV_CACHE; INT8 KV only with tiering)\n");
    return (int)cudaErrorNotSupported;
  }
  Session& ss = g_sess[g_cur];
  if (ss.kslot <= 0) return (int)cudaErrorInvalidValue;  // parked
  if ((int)g_free.size() < pages_needed(ss, T)) return (int)cudaErrorMemoryAllocation;
  use(0);
  m1draft::set_ring(ss.kslot);
  m1mtp::set_ring(ss.kslot);
  int rows[kSpecT];
  for (int t = 0; t < T; ++t) rows[t] = alloc_row(ss);
  S.step_h[0] = T, S.step_h[1] = g_seq, S.step_h[18] = g_spec_step++, S.step_h[19] = g_seed;
  memcpy(&S.step_h[32], &g_temp, 4), S.step_h[33] = g_topk, memcpy(&S.step_h[34], &g_topp, 4), memcpy(&S.step_h[35], &g_minp, 4);
  S.step_h[36] = S.step_h[19], S.step_h[37] = S.step_h[18];  // samp_pos parameter block: seed, step
  S.step_h[kStepDelta] = ss.rdelta;
  S.step_h[20] = draft ? g_hint_n : 0;
  for (int t = 0; t < kSpecT - 1; ++t) S.step_h[21 + t] = t < g_hint_n ? g_hint[t] : 0;
  g_hint_n = 0;
  {  // owner tag of the far-branch union caches: (session, epoch); a new epoch when the session's history may have
     // changed under the cache (created, reset, restored, or its position went back)
    Session& se = g_sess[g_cur];
    if (g_seq < se.cache_seq) se.cache_epoch = ++g_cache_epoch;
    se.cache_seq = g_seq;
    const long long tag = se.cache_epoch;
    memcpy(&S.step_h[30], &tag, 8);
  }
  for (int t = 0; t < kSpecT; ++t)
    S.step_h[2 + t] = draft ? (t == 0 ? *n_out : 0) : (t < T ? tokens[t] : 0), S.step_h[10 + t] = t < T ? rows[t] : 0;
  const int regime = g_seq + T <= WIN ? 0 : (g_seq + 1 > WIN ? 1 : 2);
  const int d8t = m1d8::on() ? (g_seq + T + m1d8::TILE - 1) / m1d8::TILE : 0;  // INT8 dense KV: tiles in the graph
  const auto key = std::make_pair(g_cur, regime + (draft ? 4 : 0) + 8 * d8t);
  auto it = g_vgraphs.find(key);
  if (it != g_vgraphs.end() && T == kSpecT) {
    for (auto ex : it->second) CK(cudaGraphLaunch(ex, S.s));
  } else if (graphs_on() && T == kSpecT && regime != 2 && g_vwarm[key] > 0) {
    std::vector<cudaGraph_t> segs;
    g_seg_capture = &segs;
    CK(cudaStreamBeginCapture(S.s, cudaStreamCaptureModeThreadLocal));
    const int e = verify_body(T, draft);
    cudaGraph_t gph = nullptr;
    const cudaError_t ec = cudaStreamEndCapture(S.s, &gph);
    g_seg_capture = nullptr;
    segs.push_back(gph);
    if (e || ec) {
      fprintf(stderr, "m1: verify graph capture failed (%d, %s)\n", e, cudaGetErrorString(ec));
      return e ? e : (int)ec;
    }
    drop_other_tiles(g_vgraphs, g_vwarm, key, 3, destroy_execs);
    static const bool glog = getenv("AGENS_M1_GRAPH_LOG") != nullptr;
    size_t f0 = 0, f1 = 0, tot = 0;
    if (glog) cudaMemGetInfo(&f0, &tot);
    std::vector<cudaGraphExec_t> exs;
    if (instantiate(segs, exs, "verify")) {
      g_vgraphs[key] = exs;
      if (glog) {
        cudaMemGetInfo(&f1, &tot);
        fprintf(stderr, "m1: verify graph seq %d regime %d drafted %d tiles %d: %zu segments, %.1f MB, %zu cached, %.0f MB free\n",
                g_seq, regime, (int)draft, d8t, exs.size(), (f0 - f1) / 1e6, g_vgraphs.size(), f1 / 1e6);
      }
      for (auto ex : exs) CK(cudaGraphLaunch(ex, S.s));
    } else {
      g_lin = g_lin_w4 = 0;
      CK(verify_body(T, draft));
      g_vwarm[key] = kNoGraph;
    }
  } else {
    g_lin = g_lin_w4 = 0;
    CK(verify_body(T, draft));
    if (T == kSpecT) ++g_vwarm[key];
  }
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  const int n = S.vres_h[0];
  for (int j = 0; j < n; ++j) emitted[j] = S.vres_h[1 + j];
  if (target)
    for (int t = 0; t < T; ++t) target[t] = S.vres_h[1 + kSpecT + t];
  if (drafts)
    for (int t = 0; t < T; ++t) drafts[t] = S.vres_h[1 + 2 * kSpecT + t];
  *n_out = n;
  // rejected tokens' rows: truncate to the committed length; a page whose first position goes is ours alone (allocated
  // by this step, or the session's own partial page -- fork gives the child a private copy) and returns to the pool
  while ((int)ss.rows.size() > g_seq + n) {
    const int p = (int)ss.rows.size() - 1, row = ss.rows.back();
    ss.rows.pop_back();
    if (p % 4 == 0 && --g_ref[row >> 2] == 0) g_free.push_back(row >> 2);
  }
  g_seq += n;
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  return 0;
}

// ---------------------------------------------------------------- batched speculative verify (server mode)
// B sequences (distinct live sessions with a KDA slot), each: the drafter's 7 drafts after its bonus, the 8-token
// verify, greedy accept, commit -- exactly its single-sequence step (ak_m1_spec_step), with one pass over the weights
// for all 8 B rows. Session state is reached through the staged records, so one graph per (B, regimes) serves every
// set of sessions; a mixed-regime sequence (some queries past the window) makes the step eager.
int spec_commit_b(int B) {
  const float scale = 1.0f / sqrtf(128.0f);
  const size_t MR = (size_t)S.bv_max * kSpecT;  // stash capacity (rows b * g_TV + t)
  for (int i = 0; i < NL; ++i) {
    if (kMixer[i] != 'k') continue;
    bf16* st = S.kstb[i];
    bf16 *sq = st, *sc = st + MR * KDIM, *sa = sc + MR * KDIM, *sb = sa + MR * KHV * KK;
    CK(ak_kda_conv_multi_b(sq, KDIM, S.conv[i], S.conv_w[i], S.bv_slot, nullptr, KDIM, g_TV, B, S.bv_res, kBVRes, 1,
                           S.s));
    CK(ak_kda_decode_multi_b(sc, KDIM, sa, KHV * KK, sb, KHV, (float*)WL(i, "linear_attn.A_log.f32"),
                             (float*)WL(i, "linear_attn.dt_bias.f32"), S.ssm[i], S.bv_slot, nullptr, scale, KH, KHV, KK,
                             KK, g_TV, B, S.bv_res, kBVRes, 1, S.s));
  }
  bv_engram_commit<<<1, 32, 0, S.s>>>(S.bv_d, S.ids, S.pos, S.bv_res, B, g_TV);
  return (int)cudaGetLastError();
}
int verify_body_b(int B) {
  const int R = B * g_TV;
  CK(cudaMemcpyAsync(S.bv_d, S.bv_h, (size_t)B * kRI * sizeof(int), cudaMemcpyHostToDevice, S.s));
  if (m1draft::ready()) CK(m1draft::propose_b(draft_target(), B, S.bv_d));  // drafts into each record [3..9]
  bv_apply_hint<<<B, 32, 0, S.s>>>(S.bv_d);
  // MTP chains (1.2; hybrid drafting as the single-sequence verify: host proposal, MTP through it, DFlash2 beyond),
  // at most the batched verify's TV - 1 draft slots
  // (AGENS_MTP_BATCH_MAX, default 8: larger batched steps skip the chain -- 48gb profile c16 was 4% slower with it --
  // but every batch keeps the MTP context; the graph key holds B, so a graph never mixes the two)
  static const int mtp_bmax = getenv("AGENS_MTP_BATCH_MAX") ? atoi(getenv("AGENS_MTP_BATCH_MAX")) : 8;
  if (m1mtp::ready() && B <= mtp_bmax)
    CK(m1mtp::propose_b(draft_target(), B, S.bv_d, std::min(mtp_steps(), g_TV - 1)));
  bv_prep<<<B, 32, 0, S.s>>>(S.bv_d, S.Wctx, S.ids, S.pos, S.loc, S.outloc, S.bv_rng, S.bv_slot, g_TV, S.rpos, S.rmax);
  bv_engram<<<B, 32, 0, S.s>>>(S.bv_d, S.ids, S.pos, S.r2, S.r3, g_TV);
  bv_r2t_stage<<<grid((long)B * S.Wctx), 256, 0, S.s>>>(S.bv_d, B, S.Wctx, S.r2t_b);
  if (!m1kv::on()) {  // all-GPU pools: every query its own request, over the B staged slot maps
    CK(ak_attn_decode_multi_plan_mode(MSLOT_BVWIN, R, S.bv_rng, 4, S.r2t_b, WIN + 8, QH, KVH, nullptr, 2, S.s));
    CK(ak_attn_decode_multi_plan_mode(MSLOT_BVDENSE, R, S.bv_rng + 2, 4, S.r2t_b, S.Wctx, QH, KVH, nullptr, 2, S.s));
  }
  g_BV = B;
  const int e = run_model(R, false);
  g_BV = 0;
  if (e) return e;
  // per sequence: greedy accept, or the sampled accept over its filtered distribution (record [34..39], 1.0.1)
  samp_pos<<<B * g_TV, 1024, 0, S.s>>>(S.big0, VOCAB, VOCAB_VALID, S.ids, g_TV, S.bv_d + 34, kRI, S.tok, S.samp_scr);
  samp_chain<<<1, 32, 0, S.s>>>(S.ids, B, g_TV, S.samp_scr, S.bv_res, kBVRes);
  CK(spec_commit_b(B));
  if (m1draft::ready() && g_capture) CK(m1draft::materialize_b(draft_target(), S.cap, B, S.bv_res, kBVRes, S.pos, S.bv_d, g_TV));
  if (m1mtp::ready())  // MTP context of each sequence's committed tokens (+ its new pending row)
    CK(m1mtp::materialize_b(draft_target(), WG("model.language_model.norm.weight"), S.H, NS * HID, B, g_TV, S.bv_res,
                            kBVRes, S.ids, S.pos, S.bv_d));
  return (int)cudaMemcpyAsync(S.bv_res_h, S.bv_res, (size_t)B * kBVRes * sizeof(int), cudaMemcpyDeviceToHost, S.s);
}
// Graphs per (B, TV, regimes). Tiered KV runs each sequence's attention as its own verify layer, whose launches carry
// the session's KDA slot / slot map, so there the key also holds every sequence's (session, slot): a graph per set of
// sessions, the AGENS_M1_BV_TIER_GRAPHS (default 8) most recent kept.
std::map<std::vector<long long>, std::vector<cudaGraphExec_t>> g_bvgraphs;
std::map<std::vector<long long>, int> g_bvwarm;
std::deque<std::vector<long long>> g_bvtier_order;
int spec_batch(const int* sess, int B, const int* bonus, int* emitted, int* n_out, const int* hint = nullptr,
               const int* nh = nullptr) {
  const auto t_start = std::chrono::steady_clock::now();
  use(0);
  if (B < 1 || B > S.bv_max || !(m1draft::ready() || m1mtp::ready()) || g_ndev != 1 || S.kstb.empty() ||
      (m1mtp::ready() && B > m1mtp::max_seqs()))
    return (int)cudaErrorInvalidValue;
  if (m1kv::on() && !(m1kv::graph_safe() && dev_attn() && far_engine())) return (int)cudaErrorNotSupported;
  // tiered / INT8 KV: each sequence's attention runs as its own verify layer, whose single-sequence plans are sized
  // for kSpecT queries -> the full verify length there
  const int TV = g_TV = m1kv::on() || m1d8::on() ? kSpecT : batch_tv(B);
  int need = 0;
  for (int b = 0; b < B; ++b) {
    const Session& ss = g_sess[sess[b]];
    if (!ss.live || ss.kslot <= 0 || ss.seq + TV > S.Wctx) return (int)cudaErrorInvalidValue;
    need += pages_needed(ss, TV);
  }
  if ((int)g_free.size() < need) return (int)cudaErrorMemoryAllocation;
  auto put_ptr = [](int* q, const void* p) {
    const unsigned long long v = (unsigned long long)p;
    q[0] = (int)(unsigned)(v & 0xffffffffu), q[1] = (int)(unsigned)(v >> 32);
  };
  g_bv_sess.assign(sess, sess + B);
  g_bv_reg.assign(B, 0);
  long long rkey = 0;
  bool mixed = false;
  for (int b = 0; b < B; ++b) {
    Session& ss = g_sess[sess[b]];
    int* r = S.bv_h + b * kRI;
    memset(r, 0, kRI * sizeof(int));
    r[1] = ss.seq, r[2] = bonus[b];
    r[26] = nh ? std::max(0, std::min(kSpecT - 1, nh[b])) : 0;
    for (int t = 0; t < r[26]; ++t) r[27 + t] = hint[b * (kSpecT - 1) + t];
    for (int t = 0; t < TV; ++t) r[10 + t] = alloc_row(ss);
    r[18] = ss.kslot, r[19] = ss.kslot < m1draft::rings() ? ss.kslot : 0;
    memcpy(r + 34, &ss.samp_t, 4), r[35] = ss.samp_k, memcpy(r + 36, &ss.samp_p, 4), memcpy(r + 37, &ss.samp_minp, 4);
    r[38] = ss.samp_seed, r[39] = ss.samp_step++;
    r[kRecDelta] = ss.rdelta;
    put_ptr(r + 20, ss.r2t[0]), put_ptr(r + 22, ss.prev[0]);
    if (ss.seq < ss.cache_seq) ss.cache_epoch = ++g_cache_epoch;
    ss.cache_seq = ss.seq;
    memcpy(r + 24, &ss.cache_epoch, 8);
    const int reg = ss.seq + TV <= WIN ? 0 : (ss.seq + 1 > WIN ? 1 : 2);
    g_bv_reg[b] = reg, mixed |= reg == 2;
    rkey = rkey * 3 + reg;
  }
  g_lin = g_lin_w4 = 0;
  std::vector<long long> key{B, TV, rkey};
  const bool tiered = m1kv::on() || m1d8::on();
  // INT8 dense KV: the dense layer's attention bakes its tile count (ceil((seq + TV) / TILE), from the host seq) into
  // the graph, so it is part of the key as in the single-sequence verify (r101 fix: a graph captured for a session at
  // fewer tiles was replayed for a later session with the same (session, slot) and a longer context -> its last tile
  // was not attended; m1-spec-batch at batch 2, agent_20k after long_12k)
  if (tiered)
    for (int b = 0; b < B; ++b) {
      const Session& ss = g_sess[sess[b]];
      key.push_back(sess[b]), key.push_back(ss.kslot);
      key.push_back(m1d8::on() ? (ss.seq + TV + m1d8::TILE - 1) / m1d8::TILE : 0);
    }
  auto it = g_bvgraphs.find(key);
  int e = 0;
  if (it != g_bvgraphs.end()) {
    for (auto ex : it->second) CK(cudaGraphLaunch(ex, S.s));
  } else if (graphs_on() && !mixed && g_bvwarm[key] > 0) {
    std::vector<cudaGraph_t> segs;
    g_seg_capture = &segs;
    CK(cudaStreamBeginCapture(S.s, cudaStreamCaptureModeThreadLocal));
    e = verify_body_b(B);
    cudaGraph_t gph = nullptr;
    const cudaError_t ec = cudaStreamEndCapture(S.s, &gph);
    g_seg_capture = nullptr;
    segs.push_back(gph);
    if (e || ec) {
      fprintf(stderr, "m1: batched verify graph capture failed (%d, %s)\n", e, cudaGetErrorString(ec));
      return e ? e : (int)ec;
    }
    std::vector<cudaGraphExec_t> exs;
    if (instantiate(segs, exs, "batched verify")) {
      g_bvgraphs[key] = exs;
      if (tiered) {
        static const size_t keep = getenv("AGENS_M1_BV_TIER_GRAPHS") ? std::max(1, atoi(getenv("AGENS_M1_BV_TIER_GRAPHS"))) : 8;
        g_bvtier_order.push_back(key);
        while (g_bvtier_order.size() > keep) {
          auto old = g_bvgraphs.find(g_bvtier_order.front());
          if (old != g_bvgraphs.end()) {
            for (auto ex : old->second) cudaGraphExecDestroy(ex);
            g_bvgraphs.erase(old);
          }
          g_bvwarm.erase(g_bvtier_order.front());
          g_bvtier_order.pop_front();
        }
      }
      if (getenv("AGENS_M1_GRAPH_LOG"))
        fprintf(stderr, "m1: batched verify graph B %d TV %d regimes %lld captured in %.0f ms (%zu graphs)\n", B, TV, rkey,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(),
                g_bvgraphs.size());
      for (auto ex : exs) CK(cudaGraphLaunch(ex, S.s));
    } else {
      e = verify_body_b(B);
      g_bvwarm[key] = kNoGraph;
    }
  } else {
    e = verify_body_b(B);
    ++g_bvwarm[key];
  }
  if (e) return e;
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  for (int b = 0; b < B; ++b) {
    Session& ss = g_sess[sess[b]];
    const int* res = S.bv_res_h + b * kBVRes;
    const int n = res[0];
    n_out[b] = n;
    for (int j = 0; j < n; ++j) emitted[b * kSpecT + j] = res[1 + j];
    while ((int)ss.rows.size() > ss.seq + n) {  // rejected rows (as verify1)
      const int p = (int)ss.rows.size() - 1, row = ss.rows.back();
      ss.rows.pop_back();
      if (p % 4 == 0 && --g_ref[row >> 2] == 0) g_free.push_back(row >> 2);
    }
    ss.seq += n;
  }
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  return 0;
}

// ---------------------------------------------------------------- batched decode on device tables + graphs per bucket
// The batch is padded to a bucket (1, 2, 4, 8, 16); a padding row decodes token 0 at position 3 in KDA slot 0 (unused
// by sessions) into the reserved pool row g_pad_row with a slot map of trash rows, so it reads and writes nothing a
// session reads. All session state is reached through the staged table, so one graph per (bucket, far step) serves
// every set of sessions. A far step (any row past the window) runs the far branch on every row.
bool batch_dev() {
  static const bool on = !(getenv("AGENS_M1_BATCH_DEV") && getenv("AGENS_M1_BATCH_DEV")[0] == '0');
  return on && g_ndev == 1 && dev_attn() && far_engine() && (!m1kv::on() || m1kv::graph_safe());
}
std::map<std::pair<int, int>, cudaGraphExec_t> g_bgraphs;
std::map<std::pair<int, int>, int> g_bwarm;
void drop_all_graphs() {  // warm counts are kept: the next step of a key captures again
  for (auto& g : g_graphs) cudaGraphExecDestroy(g.second);
  for (auto& g : g_vgraphs) destroy_execs(g.second);
  for (auto& g : g_bvgraphs) destroy_execs(g.second);
  for (auto& g : g_bgraphs) cudaGraphExecDestroy(g.second);
  g_graphs.clear(), g_vgraphs.clear(), g_bvgraphs.clear(), g_bgraphs.clear(), g_bvtier_order.clear();
}
int batch_body(int Bp, bool far) {
  CK(cudaMemcpyAsync(S.bstep_d, S.bstep_h, (size_t)Bp * kBRec * sizeof(int), cudaMemcpyHostToDevice, S.s));
  batch_step_prep<<<1, 32, 0, S.s>>>(S.bstep_d, Bp, S.ids, S.pos, S.loc, S.outloc, S.slots_b, S.rng_b, S.r2, S.r3, S.rpos,
                                     S.rmax);
  batch_r2t_stage<<<grid((long)Bp * S.Wctx), 256, 0, S.s>>>(S.bstep_d, Bp, S.Wctx, S.r2t_b);
  CK(m1kv::prep_rows(0, S.slots_b, Bp, S.s));
  S.slot = S.slots_b;
  g_B = Bp, g_bdev = true, g_bfar = far;
  const int e = run_model(Bp, false, Bp);
  g_B = 0, g_bdev = false, g_bfar = false;
  if (e) return e;
  return (int)cudaMemcpyAsync(S.tok_h, S.tok, (size_t)Bp * sizeof(int), cudaMemcpyDeviceToHost, S.s);
}
int rows_dev(const int* sess, const int* tok, int B, const std::vector<long long>& pos, const std::vector<int>& loc,
             const std::vector<int>& slots, int* next, std::chrono::steady_clock::time_point t_start) {
  const int cur0 = g_cur;
  use(0);
  int Bp = 1;
  while (Bp < B) Bp <<= 1;
  bool far = false;
  auto put_ptr = [](int* q, const void* p) {
    const unsigned long long v = (unsigned long long)p;
    q[0] = (int)(unsigned)(v & 0xffffffffu), q[1] = (int)(unsigned)(v >> 32);
  };
  for (int r = 0; r < Bp; ++r) {
    int* q = S.bstep_h + r * kBRec;
    if (r < B) {
      const Session& ss = g_sess[sess[r]];
      q[0] = tok[r], q[1] = (int)pos[r], q[2] = loc[r], q[3] = slots[r];
      put_ptr(q + 4, ss.r2t[0]), put_ptr(q + 6, ss.prev[0]);
      q[8] = ss.rdelta, q[9] = 0;
      far |= pos[r] + 1 > WIN;
    } else {
      q[0] = 0, q[1] = 3, q[2] = g_pad_row, q[3] = -1;  // KDA slot -1: the KDA kernels skip the row
      put_ptr(q + 4, S.dummy_r2t), put_ptr(q + 6, S.dummy_prev);
      q[8] = 0, q[9] = 0;
    }
  }
  g_lin = g_lin_w4 = 0;
  const auto key = std::make_pair(Bp, (int)far);
  auto it = g_bgraphs.find(key);
  int e = 0;
  if (it != g_bgraphs.end()) {
    e = (int)cudaGraphLaunch(it->second, S.s);
  } else if (graphs_on() && g_bwarm[key] > 0) {
    cudaGraph_t gph = nullptr;
    CK(cudaStreamBeginCapture(S.s, cudaStreamCaptureModeThreadLocal));
    e = batch_body(Bp, far);
    const cudaError_t ec = cudaStreamEndCapture(S.s, &gph);
    if (!e && ec) e = (int)ec;
    if (!e) {
      std::vector<cudaGraph_t> segs{gph};
      std::vector<cudaGraphExec_t> exs;
      gph = nullptr;  // consumed by instantiate
      if (instantiate(segs, exs, "batch decode")) g_bgraphs[key] = exs[0], e = (int)cudaGraphLaunch(exs[0], S.s);
      else e = batch_body(Bp, far), g_bwarm[key] = kNoGraph;
    }
    if (gph) cudaGraphDestroy(gph);
    if (e) fprintf(stderr, "m1: batch graph capture failed (%d)\n", e);
  } else {
    e = batch_body(Bp, far);
    ++g_bwarm[key];
  }
  g_cur = cur0;
  bind_session();
  if (e) return e;
  const auto t_enq = std::chrono::steady_clock::now();
  CK(cudaStreamSynchronize(S.s));
  const auto t_end = std::chrono::steady_clock::now();
  for (int r = 0; r < B; ++r) next[r] = S.tok_h[r], ++g_sess[sess[r]].seq;
  g_stat[0] = std::chrono::duration<double, std::milli>(t_enq - t_start).count();
  g_stat[1] = std::chrono::duration<double, std::milli>(t_end - t_enq).count();
  g_stat[2] = (double)g_lin, g_stat[3] = (double)g_lin_w4;
  return 0;
}
}  // namespace

extern "C" {
int init_device(int Wctx, int rows, void* eng2_t0, void* eng2_t1, void* eng22_t0, void* eng22_t1);
int ak_m1_register(const char* name, void* ptr) {
  W[name] = ptr;
  clear_weight_tables();
  return 0;
}
// Part [r0, r0 + n) of the n_total-row linear `name` in the W{bits} KT layout (ak_wq_repack_kt); r0 = 0 starts the
// linear (replacing an earlier registration), later parts append. Parts must tile [0, n_total) in order.
int ak_m1_register_wq(const char* name, void* wt, void* st, int n_total, int k, int r0, int n, int group, int bits) {
  if (bits < 2 || bits > 4 || (group != 64 && group != 128) || n < 1 || r0 + n > n_total) return (int)cudaErrorInvalidValue;
  W4Lin& t = W4M[name];
  if (r0 == 0) t = W4Lin{n_total, k, 0, {}};
  const int end = t.np ? t.p[t.np - 1].r0 + t.p[t.np - 1].n : 0;
  if (t.np >= 2 || t.n != n_total || t.k != k || r0 != end) return (int)cudaErrorInvalidValue;
  t.p[t.np++] = W4Part{wt, st, r0, n, group, bits};
  clear_weight_tables();
  return 0;
}
int ak_m1_register_w4(const char* name, void* wt, void* st, int n, int k, int group) {
  return ak_m1_register_wq(name, wt, st, n, k, 0, n, group, 4);
}
// a W3 KT linear (ak_w3_repack_kt), registered like a W4 one
int ak_m1_register_w3(const char* name, void* wt, void* st, int n, int k, int group) {
  return ak_m1_register_wq(name, wt, st, n, k, 0, n, group, 3);
}
// [enqueue ms, wait ms, linear calls, W4 linear calls] of the last forward (prefill chunk or decode step). enqueue =
// host time to issue the step (everything before the final sync), wait = time blocked on the GPU after that.
// cudaProfilerStart/Stop for an nsys --capture-range=cudaProfilerApi window (e.g. decode steps only).
int ak_m1_profiler(int on) { return (int)(on ? cudaProfilerStart() : cudaProfilerStop()); }
int ak_m1_step_stats(double* out) {
  for (int i = 0; i < 4; ++i) out[i] = g_stat[i];
  return 0;
}

// Tensor parallel rank `rank` of `size` (1, 2 or 4: KVH = 4 heads split evenly), before ak_m1_init: this process
// holds and runs 1 / size of every KDA layer's heads (q / k heads KH, v heads KHV), every attention layer's q / kv heads
// and every MLP's intermediate rows; the mixer and MLP outputs are summed over the ranks (ak_tp_allreduce, set up by
// ak_tp_setup). Everything else (embeddings, mHC, norms, Engram, indexer, lm_head, the drafter) is replicated.
int ak_m1_tp_config(int rank, int size) {
  if (size < 1 || KVH_FULL % size || rank < 0 || rank >= size) return (int)cudaErrorInvalidValue;
  g_tp = size, g_rank = rank;
  QH = QH_FULL / size, KVH = KVH_FULL / size, KH = KH_FULL / size, KHV = KHV_FULL / size, INTER = INTER_FULL / size;
  KDIM = 2 * KH * KK + KHV * KK;
  CK(ak_far_set_heads(QH, KVH));
  CK(ak_kda_prefill_set_heads(KH, KHV));
  CK(ak_bridge_set_heads(QH, KVH));
  CK(m1d8::set_heads(QH, KVH));  // INT8 dense KV: this rank's head share
  return 0;
}
// Wctx: sglv23's req_to_token width (its far branch scores Wctx/4 blocks; must match for identical GEMM shapes) and the
// longest session. devs: CUDA device ids (1 or 2); split: first layer on devs[1] (ignored for one device).
// pool_rows: KV rows shared by all sessions (0 = Wctx); nslots: KDA state slots = max live sessions (0 = 1).
// Session 0 is created and active on return.
int ak_m1_init(int Wctx, const char* mixers, int engram_a, int engram_b, void* eng2_t0, void* eng2_t1, void* eng22_t0,
               void* eng22_t1, int ndev, const int* devs, int split, int pool_rows, int nslots) {
  g_nslots = nslots > 0 ? nslots : 1;
  if (const char* e = getenv("AGENS_M1_W4_BATCH_MAX")) g_w4_batch_max = std::max(1, std::min(128, atoi(e)));
  if (const char* e = getenv("AGENS_M1_W4_NUMERICS")) g_w4_bf16w = !strcmp(e, "marlin");
  if (const char* e = getenv("AGENS_M1_BATCH_INVARIANT")) g_binv = e[0] && e[0] != '0';
  if (const char* e = getenv("AGENS_M1_BI")) g_bi = e[0] == '1';  // also when set after load (agens serve server mode)
  if (const char* e = getenv("AGENS_M1_PREFILL_CHUNK")) g_chunk = std::max(64, std::min(CHUNK, atoi(e)));
  if (const char* e = getenv("AGENS_M1_W4_TILE_MB")) g_w4_tile_bytes = (size_t)std::max(0, atoi(e)) << 20;
  const int rows = (pool_rows > 0 ? pool_rows : Wctx) & ~3;  // whole pages
  kMixer = mixers;
  NL = (int)kMixer.size();
  kEngramLayer[0] = engram_a, kEngramLayer[1] = engram_b;
  g_ndev = ndev < 1 ? 1 : (ndev > 2 ? 2 : ndev);
  g_split = g_ndev == 2 ? split : (1 << 30);
  for (int d = 0; d < g_ndev; ++d) SD[d].dev = devs ? devs[d] : 0;
  if (g_ndev == 2) {
    cudaSetDevice(SD[0].dev);
    cudaDeviceEnablePeerAccess(SD[1].dev, 0);
    cudaSetDevice(SD[1].dev);
    cudaDeviceEnablePeerAccess(SD[0].dev, 0);
    cudaGetLastError();  // peer access may be unavailable; cudaMemcpyPeer still works (staged)
  }
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(init_device(Wctx, rows, eng2_t0, eng2_t1, eng22_t0, eng22_t1));
    CK(ak_kda_prefill_preload());  // KDA prefill modules now, not lazily under a spent budget
  }
  g_free.clear();
  for (int p = rows / 4 - 1; p >= 0; --p) g_free.push_back(p);  // pop order 0, 1, 2, ...: a lone session is the identity
  g_ref.assign(rows / 4, 0);
  g_pad_row = 4 * g_free.front();  // the highest page: padding rows of batch buckets write here, never read by sessions
  g_free.erase(g_free.begin());
  g_slot_used.assign(g_nslots + 1, 0);
  g_sess.clear();
  g_cur = new_session();
  if (g_cur != 0) return (int)cudaErrorMemoryAllocation;
  bind_session();
  use(0);
  return 0;
}

int init_device(int Wctx, int rows, void* eng2_t0, void* eng2_t1, void* eng22_t0, void* eng22_t1) {
  if (m1kv::on() && g_tp > 1) {  // m1_kv (tiered / INT8 BCSA KV) is laid out for the full model's 4 kv heads
    fprintf(stderr, "m1: AGENS_KV_TIER / AGENS_KV_INT8 (tiered or INT8 BCSA KV) are tp1 only; --tp keeps all KV on the GPUs\n");
    return (int)cudaErrorNotSupported;
  }
  const int my = (int)(SP - SD);
  S.Wctx = Wctx;
  S.rows = rows;
  cudaStreamCreateWithFlags(&S.s, cudaStreamNonBlocking);
  {  // KDA prefill takes its temporaries from the default pool (cudaMallocAsync): keep freed memory in the pool instead
     // of returning it to the driver at every sync (re-mapping ~0.5 GB per layer per chunk otherwise)
    cudaMemPool_t pool;
    uint64_t keep = UINT64_MAX;
    if (cudaDeviceGetDefaultMemPool(&pool, S.dev) == cudaSuccess)
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
  }
  S.kc.assign(NL, nullptr), S.vc.assign(NL, nullptr), S.idx.assign(NL, nullptr), S.blk.assign(NL, nullptr);
  S.conv.assign(NL, nullptr), S.ssm.assign(NL, nullptr), S.conv_w.assign(NL, nullptr);
  for (int i = 0; i < NL; ++i) {
    if (dev_of(i) != my) continue;
    if (kMixer[i] == 'k') {
      S.conv[i] = dalloc<bf16>((g_nslots + 1) * kda_conv_slot());
      S.ssm[i] = dalloc<float>((g_nslots + 1) * kda_ssm_slot());
      S.conv_w[i] = (float*)WL(i, "linear_attn.conv1d.weight.f32");
    } else if (kMixer[i] == 'b' && m1kv::on()) {  // pools owned by m1_kv (host-mapped when tiering; int8)
      CK(m1kv::alloc_layer(my, i, S.rows, (void**)&S.kc[i], (void**)&S.vc[i]));
      // AGENS_KV_IDX_HOST=1 (with tiering): the indexer keys in pinned host RAM (mapped). Only the rows of the current
      // block (decode block keys) and of a prefill chunk are ever read, so they cross PCIe once; same values.
      if (m1kv::tier() && getenv("AGENS_KV_IDX_HOST") && getenv("AGENS_KV_IDX_HOST")[0] == '1') {
        void* h = nullptr;
        const size_t b = (size_t)(S.rows + 1) * IDXD * 2;
        if (cudaHostAlloc(&h, b, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
        memset(h, 0, b);
        S.idx[i] = (bf16*)h;
      } else {
        S.idx[i] = dalloc<bf16>((size_t)(S.rows + 1) * IDXD);
      }
      S.blk[i] = dalloc<bf16>((size_t)(S.rows / 4 + 1) * IDXD);  // per page (+ trash page rows / 4)
    } else if (kMixer[i] == 'd' && m1d8::on()) {  // INT8 dense KV (m1_dense8): int8 pools + staging, no bf16 pool
      if (g_ndev != 1) return (int)cudaErrorNotSupported;
      CK(m1d8::alloc(S.rows));
    } else {
      S.kc[i] = dalloc<bf16>((size_t)(S.rows + 1) * KVH * HD);
      S.vc[i] = dalloc<bf16>((size_t)(S.rows + 1) * KVH * HD);
      if (kMixer[i] == 'b') {
        S.idx[i] = dalloc<bf16>((size_t)(S.rows + 1) * IDXD);
        S.blk[i] = dalloc<bf16>((size_t)(S.rows / 4 + 1) * IDXD);  // per page (+ trash page rows / 4)
      }
    }
  }
  if (m1kv::on()) CK(m1kv::init_device(my, g_nslots, Wctx));
  int one = 1;
  S.has_init = dalloc<bool>(1);
  S.eng_t[0][0] = eng2_t0, S.eng_t[0][1] = eng2_t1, S.eng_t[1][0] = eng22_t0, S.eng_t[1][1] = eng22_t1;
  const size_t T = g_chunk;
  S.H = dalloc<bf16>(T * NS * HID), S.H2 = dalloc<bf16>(T * NS * HID), S.mixed = dalloc<bf16>(T * NS * HID);
  S.hmean = dalloc<bf16>(T * HID), S.hmean2 = dalloc<bf16>(T * HID), S.nrm = dalloc<bf16>(T * HID);
  S.d = dalloc<bf16>(T * 24), S.xin = dalloc<bf16>(T * HID), S.post = dalloc<bf16>(T * 4), S.res = dalloc<bf16>(T * 16);
  // scratch at the full model's widths (several buffers also carry hidden-size rows; a tp rank's head share is smaller)
  S.y = dalloc<bf16>(T * HID), S.t0 = dalloc<bf16>(T * HID), S.t1 = dalloc<bf16>(T * KHV_FULL * KK);
  S.t2 = dalloc<bf16>(T * QH_FULL * HD), S.t3 = dalloc<bf16>(T * std::max(2 * KH_FULL * KK + KHV_FULL * KK, QH_FULL * HD));
  S.gu = dalloc<bf16>(T * 2 * INTER_FULL), S.act = dalloc<bf16>(T * INTER_FULL);
  S.big0_elems = std::max(T * QH_FULL * 2 * HD, (size_t)VOCAB);
  S.big0 = dalloc<bf16>(S.big0_elems), S.big1 = dalloc<bf16>(T * KHV_FULL * KK);
  S.q = dalloc<bf16>(T * QH_FULL * HD), S.gate = dalloc<bf16>(T * QH_FULL * HD), S.k = dalloc<bf16>(T * KVH_FULL * HD);
  S.v = dalloc<bf16>(T * KVH_FULL * HD), S.o = dalloc<bf16>(T * QH_FULL * HD), S.qi = dalloc<bf16>(T * IDXH * IDXD);
  S.ik = dalloc<bf16>(T * IDXD), S.e = dalloc<bf16>(T * 2 * ENG_DIM), S.eg = dalloc<bf16>(T);
  S.lse = dalloc<float>(T * QH_FULL);
  S.pos = dalloc<long long>(T), S.reqs = dalloc<long long>(1), S.r2 = dalloc<long long>(T), S.r3 = dalloc<long long>(T);
  S.rmax = (int)std::max<size_t>({T, (size_t)kMaxBatch, (size_t)kMaxBV * kSpecT, (size_t)kSpecT});
  S.rpos = dalloc<long long>((size_t)3 * S.rmax);
  S.outloc = dalloc<long long>(T);
  S.loc = dalloc<int>(T), S.ids = dalloc<int>(T), S.qo_dev = dalloc<int>(2), S.kv_indptr = dalloc<int>(2);
  S.kv_idx = dalloc<int>(Wctx), S.last1 = dalloc<int>(1), S.tok = dalloc<int>(kMaxBatch > kSpecT ? kMaxBatch : kSpecT);
  S.slots_b = dalloc<int>(kMaxBatch);
  S.bstep_d = dalloc<int>(kMaxBatch * kBRec), S.rng_b = dalloc<int>(kMaxBatch * 4);
  S.r2t_b = dalloc<int>((size_t)kMaxBatch * Wctx);
  S.dummy_prev = dalloc<int>(2);
  {
    std::vector<int> trash(Wctx, rows);
    S.dummy_r2t = dalloc<int>(Wctx);
    cudaMemcpy(S.dummy_r2t, trash.data(), (size_t)Wctx * 4, cudaMemcpyHostToDevice);
  }
  CK(cudaMallocHost(&S.bstep_h, kMaxBatch * kBRec * sizeof(int)));
  S.step_d = dalloc<int>(kStepInts), S.rng = dalloc<int>(4 * kSpecT);  // decode uses [0..3]; speculative verify the rest
  S.samp_scr = dalloc<int>(16 * kMaxBV);
  for (int i = 0; i < NL; ++i)
    if (kMixer[i] == 'b' && dev_of(i) == my && !S.far_ws) S.far_ws = dalloc<char>(ak_far_decode_ws_bytes(Wctx));
  CK(cudaMallocHost(&S.step_h, kStepInts * sizeof(int)));
  memset(S.step_h, 0, kStepInts * sizeof(int));
  CK(cudaMallocHost(&S.tok_h, kMaxBatch * sizeof(int)));
  {
    auto it = W.find("rope.cos_sin_cache." + std::to_string(S.dev));
    S.rope = it == W.end() ? nullptr : it->second;  // none registered: RoPE on the fly
  }
  size_t wdq = 0;
  for (auto& kv : W4M) wdq = std::max(wdq, (size_t)w4_tile_rows(kv.second.k, kv.second.n) * kv.second.k);
  if (wdq) S.wdq = dalloc<bf16>(wdq);
  cudaMemcpy(S.last1, &one, 4, cudaMemcpyHostToDevice);
  return (int)cudaDeviceSynchronize();
}

// Prefill a prompt (chunked at 4096 like sglv23's chunked_prefill_size) and return the first generated token.
int ak_m1_prefill(const int* ids, int n, int* next) {
  static const bool release = getenv("AGENS_M1_TORCH_RELEASE") && getenv("AGENS_M1_TORCH_RELEASE")[0] == '1';
  for (int s = 0; s < n; s += g_chunk) {
    const int T = std::min(g_chunk, n - s);
    CK(forward(ids + s, T, true, next));
    if (release) ak_bridge_torch_reserved(1);  // AGENS_M1_TORCH_RELEASE=1: the far-extend temporaries back each chunk
  }
  return 0;
}

// Prefill with image inputs: rpos3 [3][n] int64 rope positions (MRoPE rows T, H, W; null = position +
// the session's rope delta), emb [rows][HID] bf16 host (the vision tower's output), emb_row [n] (-1 = the token's own
// embedding, else its row of emb). Chunked like ak_m1_prefill; the session's rope delta afterwards = that of the last
// token (text at the end of a prompt).
int ak_m1_prefill_mm(const int* ids, int n, const long long* rpos3, const void* emb, const int* emb_row, int* next) {
  if (g_ndev != 1 && emb_row) return (int)cudaErrorNotSupported;  // the embedding rows go to device 0 only
  MmIn m{rpos3, (const bf16*)emb, emb_row, n, 0};
  g_mm = (rpos3 || emb_row) ? &m : nullptr;
  int e = 0;
  for (int s = 0; s < n && !e; s += g_chunk) {
    m.s = s;
    e = forward(ids + s, std::min(g_chunk, n - s), true, next);
  }
  g_mm = nullptr;
  return e;
}
int ak_m1_rope_delta() { return g_sess[g_cur].rdelta; }
int ak_m1_set_rope_delta(int d) {
  g_sess[g_cur].rdelta = d;
  return 0;
}
// vision tower (kernels/oracle/bridge/vision.cpp) on device 0's stream
int ak_m1_vis_register(const char* name, void* host, const long long* shape, int ndim) {
  return ak_vis_register(name, host, shape, ndim);
}
int ak_m1_vis_ready() { return ak_vis_ready(); }
int ak_m1_vis_encode(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, void* out) {
  use(0);
  return ak_vis_encode(rgb, jpeg, jlen, h, w, rh, rw, out, S.s);
}
int ak_m1_vis_pixels(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, float* out) {
  use(0);
  return ak_vis_pixels(rgb, jpeg, jlen, h, w, rh, rw, out, S.s);
}
int ak_m1_vis_encode_pixels(const float* pv, int gh, int gw, void* out) {
  use(0);
  return ak_vis_encode_pixels(pv, gh, gw, out, S.s);
}
long long ak_m1_vis_peak_bytes() {
  use(0);
  return ak_vis_peak_bytes();
}

// Speculative decoding setup (one device): batch-invariant decode on (bi = 1; decode and verify then share
// kernels, so greedy with and without speculation give the same tokens), verify buffers, and the target layers whose
// entry residual stream 0 is captured for the drafter (cap_layers[5], -1 = none; sglang's DFlash ids + 1).
int ak_m1_spec_init(int bi, const int* cap_layers, int cap_rows) {
  if (g_ndev != 1) return (int)cudaErrorNotSupported;
  use(0);
  g_bi = bi != 0;
  if (S.kst.empty()) {
    S.kst.assign(NL, nullptr);
    for (int i = 0; i < NL; ++i)
      if (kMixer[i] == 'k') S.kst[i] = dalloc<bf16>((size_t)kSpecT * (2 * KDIM + KHV * KK + KHV));
    S.vres = dalloc<int>(1 + kSpecT);
    // multi-query far branch for verify blocks entirely in the far regime (AGENS_SPEC_FAR_UNION=0: per query)
    const char* fu = getenv("AGENS_SPEC_FAR_UNION");
    if (!(fu && fu[0] == '0') && S.Wctx / 4 >= 512) {
      S.far_wsm = dalloc<char>(ak_far_decode_multi_ws_bytes(S.Wctx, kSpecT));
      if (!S.far_wsm) return (int)cudaErrorMemoryAllocation;
      CK(ak_far_decode_multi_init(S.far_wsm, S.Wctx, kSpecT, S.s));
    }
    CK(cudaMallocHost(&S.vres_h, (1 + 3 * kSpecT) * sizeof(int)));
  }
  g_capture = false;
  if (cap_layers) {
    for (int c = 0; c < kCapLayers; ++c) g_cap_layer[c] = cap_layers[c];
    const int rows = cap_rows > kSpecT ? cap_rows : kSpecT;
    if (rows > S.cap_rows) {
      if (S.cap) cudaFree(S.cap);
      S.cap = dalloc<bf16>((size_t)rows * kCapLayers * HID);
      S.cap_rows = rows;
    }
    g_capture = true;
  }
  g_graphs.clear(), g_warm.clear(), g_vgraphs.clear(), g_vwarm.clear();  // decode graphs were built for the old mode
  return (int)cudaDeviceSynchronize();
}
// Verify tokens[0..T-1] = [bonus, d1..d(T-1)] (T <= 8) at the current position; commits n (1..T) tokens and returns
// the n emitted tokens (d1..d(n-1), target's next token); target (nullable) = the target's greedy token per position.
int ak_m1_verify(const int* tokens, int T, int* emitted, int* n, int* target) {
  return verify1(tokens, T, emitted, n, target, nullptr);
}
// DFlash2 drafter: weights (bf16 device pointers, drafter checkpoint names, gate+up merged), then setup (after
// ak_m1_spec_init with the capture layers: context buffers for cap_rows tokens). -> bytes the drafter allocated.
int ak_m1_draft_register(const char* name, void* ptr) { return m1draft::register_weight(name, ptr); }
int ak_m1_draft_register_w4(const char* name, void* wt, void* st, int n, int k, int group) {
  return m1draft::register_w4(name, wt, st, n, k, group);
}
int ak_m1_draft_register_wq(const char* name, void* wt, void* st, int n, int k, int group, int bits) {
  return m1draft::register_w4(name, wt, st, n, k, group, bits);
}
// MTP head (docs/mtp.md): weights (bf16 device pointers or W4, names without "mtp.", gate+up merged), then setup
// (after ak_m1_spec_init): context rows per call (the prefill chunk), window. -> bytes the head allocated.
int ak_m1_mtp_register(const char* name, void* ptr) { return m1mtp::register_weight(name, ptr); }
int ak_m1_mtp_register_wq(const char* name, void* wt, void* st, int n, int k, int group, int bits) {
  return m1mtp::register_w4(name, wt, st, n, k, group, bits);
}
int ak_m1_mtp_shortlist(const int* ids, int n) { return m1mtp::set_shortlist(ids, n); }
long long ak_m1_mtp_setup(int max_rows, int window) {
  use(0);
  if (S.kst.empty() || g_ndev != 1) return -1;  // after ak_m1_spec_init; one device per process
  // batched speculation (ak_m1_spec_batch_init before this): chain buffers for bv_max sequences
  if (m1mtp::setup(max_rows, g_nslots + 1, window, S.bv_max > 0 ? S.bv_max : 1)) return -1;
  for (int k = 0; k <= g_nslots; ++k) m1mtp::reset_ring(k, S.s);
  g_vgraphs.clear(), g_vwarm.clear();
  return (long long)m1mtp::bytes();
}
// Sampled speculative accept: temperature (0 = greedy) and seed (Philox stream per step and position).
int ak_m1_spec_sampling(float temperature, int top_k, float top_p, float min_p, int seed) {
  const float t = temperature > 0.f ? temperature : 0.f;
  g_topk = top_k > 0 ? top_k : 0, g_topp = top_p > 0.f && top_p < 1.f ? top_p : 1.f, g_minp = min_p > 0.f ? min_p : 0.f;
  if ((t > 0.f) != (g_temp > 0.f)) {  // the accept rule (greedy / sampled) is baked into the verify graphs; its parameters are data
    for (auto& g : g_vgraphs)
      for (auto ex : g.second) cudaGraphExecDestroy(ex);
    g_vgraphs.clear(), g_vwarm.clear();
  }
  g_temp = t;
  g_spec_step = 0;  // a request's sample stream: (seed, step, position)
  g_seed = seed;
  return 0;
}
long long ak_m1_draft_setup(int mask_token_id) {
  use(0);
  if (!g_capture || S.cap_rows < 1) return -1;
  // batched speculation (ak_m1_spec_batch_init): a context ring per KDA slot, buffers for bv_max sequences
  const int rings = S.bv_max ? g_nslots + 1 : 1;
  if (m1draft::setup(S.cap_rows, mask_token_id, S.Wctx, rings, S.bv_max ? S.bv_max : 1)) return -1;
  return (long long)m1draft::bytes();
}
// One speculative step with the drafter: anchor = bonus; returns the n emitted tokens (n <= 8) and the drafts.
int ak_m1_spec_step(int bonus, int* emitted, int* n, int* drafts) {
  *n = bonus;
  return verify1(nullptr, kSpecT, emitted, n, nullptr, drafts);
}
// The same with a host proposal (hybrid drafting): hint[0..nh) replaces the drafter's first nh drafts (nh <= 7). The
// verify and accept are unchanged, so the output is the same greedy / sampled sequence either way.
int ak_m1_spec_step_hint(int bonus, const int* hint, int nh, int* emitted, int* n, int* drafts) {
  if (nh < 0 || nh > kSpecT - 1) return (int)cudaErrorInvalidValue;
  g_hint_n = nh;
  for (int t = 0; t < nh; ++t) g_hint[t] = hint[t];
  *n = bonus;
  return verify1(nullptr, kSpecT, emitted, n, nullptr, drafts);
}
int ak_m1_decode(int token, int* next) { return g_ndev == 1 ? decode1(token, next) : forward(&token, 1, false, next); }

// Batched decode step (server mode): row r feeds tok[r] to session sess[r]; next[r] = greedy token, logits of row r
// via ak_m1_logits_row. Each row is bit-identical to a bs = 1 ak_m1_decode of that session (see bf16_lin).
int ak_m1_decode_batch(const int* sess, const int* tok, int B, int* next) { return forward_rows(sess, tok, B, next); }

// Batched speculative decoding (server mode): buffers for up to max_b sequences per step; call after ak_m1_spec_init
// and before ak_m1_draft_setup (the drafter then gets one context ring per KDA slot and max_b-sequence buffers).
int ak_m1_spec_batch_init(int max_b) {
  use(0);
  if (max_b < 1 || max_b > kMaxBV || S.kst.empty()) return (int)cudaErrorInvalidValue;
  if (S.bv_max >= max_b) return 0;
  // the verify's logits land in big0 ([rows][VOCAB]): B x 8 rows exceed a 1024-token chunk's scratch from B = 7
  // (1.0.0 wrote past it at the 48gb profile's B = 16 with tiered KV -- into other per-step scratch, by luck)
  if ((size_t)max_b * kSpecT * VOCAB > S.big0_elems) {
    cudaFree(S.big0);
    S.big0_elems = (size_t)max_b * kSpecT * VOCAB;
    if (!(S.big0 = dalloc<bf16>(S.big0_elems))) return (int)cudaErrorMemoryAllocation;
  }
  for (bf16* p : S.kstb) cudaFree(p);
  S.kstb.assign(NL, nullptr);
  const size_t MR = (size_t)max_b * kSpecT;
  for (int i = 0; i < NL; ++i)
    if (kMixer[i] == 'k' && !(S.kstb[i] = dalloc<bf16>(MR * (2 * KDIM + KHV * KK + KHV))))
      return (int)cudaErrorMemoryAllocation;
  if (!S.bv_d) {
    S.bv_d = dalloc<int>((size_t)kMaxBV * kRI), S.bv_rng = dalloc<int>((size_t)kMaxBV * kSpecT * 4);
    S.bv_slot = dalloc<int>(kMaxBV), S.bv_res = dalloc<int>((size_t)kMaxBV * kBVRes);
    CK(cudaMallocHost(&S.bv_h, (size_t)kMaxBV * kRI * sizeof(int)));
    CK(cudaMallocHost(&S.bv_res_h, (size_t)kMaxBV * kBVRes * sizeof(int)));
  }
  S.bv_max = max_b;
  return (int)cudaDeviceSynchronize();
}
int ak_m1_spec_batch_max() { return SD[0].bv_max; }
// the sampling of a session's batched verify steps (its own Philox stream: seed, step from 0); temperature 0 = greedy
int ak_m1_session_sampling(int sess, float temperature, int top_k, float top_p, float min_p, int seed) {
  if (sess < 0 || sess >= (int)g_sess.size() || !g_sess[sess].live) return (int)cudaErrorInvalidValue;
  Session& ss = g_sess[sess];
  ss.samp_t = temperature > 0.f ? temperature : 0.f, ss.samp_k = top_k > 0 ? top_k : 0;
  ss.samp_p = top_p > 0.f && top_p < 1.f ? top_p : 1.f, ss.samp_minp = min_p > 0.f ? min_p : 0.f;
  ss.samp_seed = seed, ss.samp_step = 0;
  return 0;
}
// One batched step: sequence b = session sess[b] with anchor bonus[b]; emitted[b * 8 ..] = its n[b] tokens.
int ak_m1_spec_batch(const int* sess, int B, const int* bonus, int* emitted, int* n) {
  return spec_batch(sess, B, bonus, emitted, n);
}
// with host proposals (hybrid drafting): sequence b's hint[b * 7 .. + nh[b]) replaces its first nh[b] drafts
int ak_m1_spec_batch_hint(const int* sess, int B, const int* bonus, const int* hint, const int* nh, int* emitted, int* n) {
  return spec_batch(sess, B, bonus, emitted, n, hint, nh);
}


// ---------------------------------------------------------------- Session State (M2a)
// One request's complete decode state as a single blob, backend-neutral layout (little-endian, sections in order):
//   header  { u32 magic 'AGSS', u32 version 1, i32 n_layers, i32 seq, i32 engram_prev[2], i32 kda_dim, i32 reserved,
//             char mixers[128] }
//   per layer i (in order):  KDA  -> conv state [3][kda_dim] bf16, recurrent state [48][128][128] fp32
//                            BCSA -> K [seq][4][256] bf16, V [seq][4][256] bf16, idx [seq][128] bf16, blk [seq][128] bf16
//                            dense-> K, V
// Rows are in token-position order (gathered through the session's slot map), so a blob does not depend on which pool
// rows a session happened to get. dst/src: device memory or pinned host memory (UVA; kernels read/write it directly).
}  // extern "C"
namespace {
struct SsHeader {
  uint32_t magic, version;
  int32_t n_layers, seq, engram_prev[2], kda_dim, reserved;
  char mixers[128];
};
size_t state_bytes(int seq) {
  size_t b = sizeof(SsHeader);
  for (int i = 0; i < NL; ++i) {
    if (kMixer[i] == 'k') b += (size_t)3 * KDIM * 2 + (size_t)KHV * KK * KK * 4;
    else b += (size_t)seq * KVH * HD * 2 * 2 + (kMixer[i] == 'b' ? (size_t)seq * IDXD * 2 * 2 : 0);
  }
  return b;
}
template <bool kSave>
int copy_state(char* blob) {
  const int seq = g_seq, ks = g_sess[g_cur].kslot;
  size_t off = sizeof(SsHeader);
  auto cp = [&](void* engine, size_t n) {
    cudaError_t e = kSave ? cudaMemcpyAsync(blob + off, engine, n, cudaMemcpyDefault, S.s)
                          : cudaMemcpyAsync(engine, blob + off, n, cudaMemcpyDefault, S.s);
    off += n;
    return (int)e;
  };
  auto rows = [&](bf16* cache, int d, int shift = 0) {  // seq rows of d bf16 each, through r2t (>> shift: per page)
    const int d16 = d * 2 / 16;
    if (seq > 0) {
      if (kSave)
        gather_rows<<<grid((long)seq * d16), 256, 0, S.s>>>((const uint4*)cache, S.r2t, seq, d16, (uint4*)(blob + off), shift);
      else
        scatter_rows<<<grid((long)seq * d16), 256, 0, S.s>>>((const uint4*)(blob + off), S.r2t, seq, d16, (uint4*)cache, shift);
    }
    off += (size_t)seq * d * 2;
    return (int)cudaGetLastError();
  };
  for (int i = 0; i < NL; ++i) {
    use(dev_of(i));
    if (kMixer[i] == 'k') {
      CK(cp(S.conv[i] + ks * kda_conv_slot(), kda_conv_slot() * 2));
      CK(cp(S.ssm[i] + ks * kda_ssm_slot(), kda_ssm_slot() * 4));
    } else {
      CK(rows(S.kc[i], KVH * HD));
      CK(rows(S.vc[i], KVH * HD));
      if (kMixer[i] == 'b') {
        CK(rows(S.idx[i], IDXD));
        CK(rows(S.blk[i], IDXD, 2));  // the page's block key at each of its positions (equal values)
      }
    }
  }
  return 0;
}
}  // namespace

extern "C" {
size_t ak_m1_state_bytes() { return state_bytes(g_seq); }

int ak_m1_snapshot(void* dst) {
  if (m1d8::on()) return (int)cudaErrorNotSupported;
  if (m1kv::int8()) return (int)cudaErrorNotSupported;  // Session State v1 holds bf16 K/V
  SsHeader h{};
  h.magic = 0x53534741u;  // "AGSS"
  h.version = 1;
  h.n_layers = NL;
  h.seq = g_seq;
  h.kda_dim = KDIM;
  h.reserved = g_sess[g_cur].rdelta;  // the rope delta (0 for text-only sessions: the v1 blob unchanged)
  strncpy(h.mixers, kMixer.c_str(), sizeof(h.mixers) - 1);
  use(0);
  CK(cudaMemcpyAsync(h.engram_prev, S.prev, 8, cudaMemcpyDeviceToHost, S.s));
  CK(cudaStreamSynchronize(S.s));
  CK(cudaMemcpy(dst, &h, sizeof(h), cudaMemcpyDefault));
  CK(copy_state<true>((char*)dst));
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(cudaStreamSynchronize(S.s));
  }
  use(0);
  return 0;
}

int ak_m1_restore(const void* src) {
  if (m1kv::int8()) return (int)cudaErrorNotSupported;
  g_sess[g_cur].cache_epoch = ++g_cache_epoch;
  SsHeader h;
  CK(cudaMemcpy(&h, src, sizeof(h), cudaMemcpyDefault));
  if (h.magic != 0x53534741u || h.version != 1 || h.n_layers != NL || kMixer.compare(0, std::string::npos, h.mixers) != 0)
    return (int)cudaErrorInvalidValue;
  if (h.seq + g_chunk > S.Wctx) return (int)cudaErrorInvalidValue;
  // into the active session: drop its rows, take h.seq fresh ones, map them, then scatter the blob through the map
  Session& ss = g_sess[g_cur];
  release_rows(ss);
  if ((int)g_free.size() < pages_needed(ss, h.seq)) return (int)cudaErrorMemoryAllocation;
  for (int t = 0; t < h.seq; ++t) alloc_row(ss);
  g_seq = h.seq;
  ss.rdelta = h.reserved;
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(cudaMemcpy(S.prev, h.engram_prev, 8, cudaMemcpyHostToDevice));
    if (h.seq) CK(cudaMemcpy(S.r2t, ss.rows.data(), (size_t)h.seq * 4, cudaMemcpyHostToDevice));
  }
  CK(copy_state<false>((char*)const_cast<void*>(src)));
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(m1kv::rebuild(d, ss.kslot, S.r2t, h.seq, S.s));  // tiering: window ring from the restored pool rows
    CK(cudaStreamSynchronize(S.s));
  }
  use(0);
  return 0;
}

int ak_m1_seq() { return g_seq; }

int ak_m1_reset() {  // the active session back to empty (its rows return to the pool)
  g_sess[g_cur].cache_epoch = ++g_cache_epoch;
  release_rows(g_sess[g_cur]);
  CK(reset_session_buffers(g_sess[g_cur]));
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    cudaDeviceSynchronize();
  }
  use(0);
  return 0;
}

// ---------------------------------------------------------------- sessions (M2a)
int ak_m1_session_new() {  // -> id (not activated), or -1 when every KDA slot is in use
  const int cur = g_cur, id = new_session();
  g_cur = cur;
  use(0);
  return id;
}
int ak_m1_session_use(int id) {  // activate: pointer swap, no copy
  if (id < 0 || id >= (int)g_sess.size() || !g_sess[id].live) return (int)cudaErrorInvalidValue;
  g_cur = id;
  bind_session();
  use(0);
  return 0;
}
int ak_m1_session_current() { return g_cur; }
// fork: a new session with src's exact state. KV/idx/blk rows are SHARED (refcount; each branch appends its own new
// rows), the KDA slot (173 MB at full size) and Engram prev pair are copied device-to-device. -> id or -1.
int ak_m1_session_fork(int src) {
  if (m1d8::on()) return -1;  // INT8 dense KV: no fork (personal mode, one slot)
  if (src < 0 || src >= (int)g_sess.size() || !g_sess[src].live || g_sess[src].kslot <= 0) return -1;
  const int part = g_sess[src].seq % 4;
  if (part && g_free.empty()) return -1;
  const int cur = g_cur, id = new_session();
  g_cur = cur;
  if (id < 0) return -1;
  Session &a = g_sess[src], &b = g_sess[id];
  b.seq = a.seq, b.rows = a.rows, b.rdelta = a.rdelta;
  for (int k = 0; k < (int)b.rows.size(); k += 4) ++g_ref[b.rows[k] >> 2];
  int from = -1, to = -1;  // a partial last block moves to a page of the child's own (both branches append to it)
  if (part) {
    const int k0 = b.seq - part;
    from = b.rows[k0];
    --g_ref[from >> 2];
    to = 4 * g_free.back();
    g_free.pop_back();
    g_ref[to >> 2] = 1;
    for (int j = 0; j < part; ++j) b.rows[k0 + j] = to + j;
  }
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    if (cudaMemcpyAsync(b.r2t[d], a.r2t[d], (size_t)S.Wctx * 4, cudaMemcpyDeviceToDevice, S.s) ||
        cudaMemcpyAsync(b.prev[d], a.prev[d], 8, cudaMemcpyDeviceToDevice, S.s))
      return -1;
    if (part) {  // child's r2t entries + the rows' K/V/idx data on this device's attention layers
      if (cudaMemcpyAsync(b.r2t[d] + (b.seq - part), b.rows.data() + (b.seq - part), (size_t)part * 4,
                          cudaMemcpyHostToDevice, S.s))
        return -1;
      for (int i = 0; i < NL; ++i) {
        if (kMixer[i] == 'k' || dev_of(i) != d) continue;
        if (kMixer[i] == 'b' && m1kv::on()) {
          if (m1kv::copy_rows(d, i, from, to, part, S.s)) return -1;
        } else {
          const size_t kv = (size_t)KVH * HD;
          if (cudaMemcpyAsync(S.kc[i] + to * kv, S.kc[i] + from * kv, part * kv * 2, cudaMemcpyDefault, S.s) ||
              cudaMemcpyAsync(S.vc[i] + to * kv, S.vc[i] + from * kv, part * kv * 2, cudaMemcpyDefault, S.s))
            return -1;
        }
        if (kMixer[i] == 'b' && cudaMemcpyAsync(S.idx[i] + (size_t)to * IDXD, S.idx[i] + (size_t)from * IDXD,
                                                (size_t)part * IDXD * 2, cudaMemcpyDefault, S.s))
          return -1;
      }
    }
    for (int i = 0; i < NL; ++i)
      if (kMixer[i] == 'k' && dev_of(i) == d &&
          (cudaMemcpyAsync(S.conv[i] + b.kslot * kda_conv_slot(), S.conv[i] + a.kslot * kda_conv_slot(),
                           kda_conv_slot() * 2, cudaMemcpyDeviceToDevice, S.s) ||
           cudaMemcpyAsync(S.ssm[i] + b.kslot * kda_ssm_slot(), S.ssm[i] + a.kslot * kda_ssm_slot(),
                           kda_ssm_slot() * 4, cudaMemcpyDeviceToDevice, S.s)))
        return -1;
    if (m1kv::fork(d, a.kslot, b.kslot, S.s)) return -1;  // tiering: the window ring is per session
    if (d == 0 && m1draft::copy_ring(b.kslot, a.kslot, S.s)) return -1;  // the drafter context (per-slot rings)
    if (d == 0 && m1mtp::copy_ring(b.kslot, a.kslot, S.s)) return -1;    // the MTP context
    if (cudaStreamSynchronize(S.s)) return -1;
  }
  use(0);
  return id;
}
int ak_m1_session_free(int id) {  // not the active one
  if (id < 0 || id >= (int)g_sess.size() || !g_sess[id].live || id == g_cur) return (int)cudaErrorInvalidValue;
  release_rows(g_sess[id]);
  g_slot_used[g_sess[id].kslot] = 0;
  g_sess[id].live = false;  // device buffers are kept for reuse by the next session with this id
  return 0;
}
int ak_m1_pool_free_rows() { return 4 * (int)g_free.size(); }  // whole free pages
// Logits of the last forward (bf16 [vocab_valid], last token) into host memory; returns vocab_valid or < 0.
int ak_m1_logits(void* host_dst) {
  use(dev_of(NL - 1));
  const int e = (int)cudaMemcpy(host_dst, S.big0, (size_t)VOCAB_VALID * 2, cudaMemcpyDeviceToHost);
  use(0);
  return e ? -e : VOCAB_VALID;
}
// Row r of the last batched decode step (row 0 = the last token of any forward).
// INT8 lm_head logits of row r of the last head input (comparison runs: both heads registered) -> VOCAB_VALID
// Bytes torch's caching allocator holds (ATen far branch temporaries), after returning unused blocks when release.
long long ak_m1_torch_reserved(int release) { return ak_bridge_torch_reserved(release); }
int ak_m1_logits_i8_row(int r, void* host_dst) {
  if (r < 0 || r >= kMaxBatch || !W.count("lm_head.weight.i8")) return -(int)cudaErrorInvalidValue;
  use(dev_of(NL - 1));
  static bf16* lg = nullptr;
  if (!lg) lg = dalloc<bf16>(VOCAB);
  int e = ak_i8_gemv(S.t0 + (size_t)r * HID, WG("lm_head.weight.i8"), WG("lm_head.weight.scale"), lg, 1, VOCAB, HID,
                     VOCAB, S.s);
  if (!e) e = (int)cudaMemcpyAsync(host_dst, lg, (size_t)VOCAB_VALID * 2, cudaMemcpyDeviceToHost, S.s);
  if (!e) e = (int)cudaStreamSynchronize(S.s);
  use(0);
  return e ? -e : VOCAB_VALID;
}
int ak_m1_logits_row(int r, void* host_dst) {
  if (r < 0 || r >= kMaxBatch) return -(int)cudaErrorInvalidValue;
  use(dev_of(NL - 1));
  const int e = (int)cudaMemcpy(host_dst, S.big0 + (size_t)r * VOCAB, (size_t)VOCAB_VALID * 2, cudaMemcpyDeviceToHost);
  use(0);
  return e ? -e : VOCAB_VALID;
}

// ---------------------------------------------------------------- parking (Session State v2, M2a tiers for idle sessions)
// A parked session keeps its KV rows (pool / host tier), r2t and Engram pair, but gives back its KDA state slot (and
// with tiering its GPU window ring): conv + recurrent state of every KDA layer (layer order) then the ring of every
// device go to `dst` (pinned host or device memory, ak_m1_park_bytes()). Unpark takes any free slot. Exact (copies).
size_t ak_m1_park_bytes() {
  size_t b = 0;
  for (int i = 0; i < NL; ++i)
    if (kMixer[i] == 'k') b += kda_conv_slot() * 2 + kda_ssm_slot() * 4;
  for (int d = 0; d < g_ndev; ++d) b += m1kv::ring_bytes(d);
  return b;
}
namespace {
int park_copy(Session& ss, char* buf, bool to_buf) {
  size_t off = 0;
  for (int i = 0; i < NL; ++i) {
    if (kMixer[i] != 'k') continue;
    use(dev_of(i));
    char* conv = (char*)(S.conv[i] + ss.kslot * kda_conv_slot());
    char* ssm = (char*)(S.ssm[i] + ss.kslot * kda_ssm_slot());
    const size_t nc = kda_conv_slot() * 2, ns = kda_ssm_slot() * 4;
    CK(cudaMemcpyAsync(to_buf ? buf + off : conv, to_buf ? conv : buf + off, nc, cudaMemcpyDefault, S.s));
    off += nc;
    CK(cudaMemcpyAsync(to_buf ? buf + off : ssm, to_buf ? ssm : buf + off, ns, cudaMemcpyDefault, S.s));
    off += ns;
  }
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(m1kv::ring_copy(d, ss.kslot, buf + off, to_buf, S.s));
    off += m1kv::ring_bytes(d);
    CK(cudaStreamSynchronize(S.s));
  }
  use(0);
  return 0;
}
}  // namespace
int ak_m1_session_park(int id, void* dst) {
  if (id < 0 || id >= (int)g_sess.size() || !g_sess[id].live || g_sess[id].kslot <= 0) return (int)cudaErrorInvalidValue;
  Session& ss = g_sess[id];
  CK(park_copy(ss, (char*)dst, true));
  g_slot_used[ss.kslot] = 0;
  ss.kslot = 0;
  drop_graphs(id);
  return 0;
}
int ak_m1_session_unpark(int id, const void* src) {
  if (id < 0 || id >= (int)g_sess.size() || !g_sess[id].live || g_sess[id].kslot > 0) return (int)cudaErrorInvalidValue;
  int k = 1;
  while (k <= g_nslots && g_slot_used[k]) ++k;
  if (k > g_nslots) return (int)cudaErrorMemoryAllocation;
  Session& ss = g_sess[id];
  ss.kslot = k;
  g_slot_used[k] = 1;
  drop_graphs(id);
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    CK(cudaMemcpy(ss.slot[d], &k, 4, cudaMemcpyHostToDevice));
  }
  CK(park_copy(ss, (char*)const_cast<void*>(src), false));
  return 0;
}
int ak_m1_session_parked(int id) { return id >= 0 && id < (int)g_sess.size() && g_sess[id].live && g_sess[id].kslot == 0; }
int ak_m1_free_slots() {
  int n = 0;
  for (int k = 1; k <= g_nslots; ++k) n += !g_slot_used[k];
  return n;
}
// After a failed engine call (serve fails that request and keeps running; 1.1.1): end a stream capture left open, free
// every cached CUDA graph (device memory for what comes next; recaptured on demand) and clear the error state. 0: the
// context is usable; else the sticky error that makes it unusable (the caller exits).
int ak_m1_recover() {
  int e = 0;
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(S.s, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) {
      cudaGraph_t g = nullptr;
      cudaStreamEndCapture(S.s, &g);
      if (g) cudaGraphDestroy(g);
    }
  }
  g_seg_capture = nullptr;
  drop_all_graphs();
  for (int d = 0; d < g_ndev; ++d) {
    use(d);
    cudaDeviceSynchronize();
    cudaGetLastError();  // a non-sticky error is cleared here; a sticky one is returned again
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess && !e) e = (int)s;
  }
  use(0);
  if (e) fprintf(stderr, "m1: recover: the CUDA context is unusable (%s)\n", cudaGetErrorString((cudaError_t)e));
  return e;
}
}
