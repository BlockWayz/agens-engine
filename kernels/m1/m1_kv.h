// M1 engine hooks for KV tiering (AGENS_KV_TIER=1) and INT8 KV (AGENS_KV_INT8=1) of the BCSA layers; both default off,
// and with both off m1_model.cu never calls into m1_kv.cu (the bf16 all-GPU path is untouched). Design: docs/kv_tiering.md.
//   tier: each BCSA layer's bulk K/V pool [rows+1] lives in pinned host RAM (mapped); the GPU keeps a per-session ring of
//         the last 4096 positions (+ one shared prefill-chunk area) for the local branch; the far branch fetches only the
//         rows of the indexer's top-k blocks into a staging buffer. Bit-identical to the all-GPU path (data movement only).
//   int8: pool (and ring) store int8 + one fp32 scale per (token, kv head); attention reads dequantized bf16 staging.
// Dense-layer KV and the indexer side store (idx/blk) stay as they are (GPU, bf16).
#pragma once
#include <cuda_runtime.h>

namespace m1kv {
bool tier();
bool int8();
bool means();  // AGENS_KV_TIER=2 (bf16 only): the host far pool also keeps 4-row block means; the far branch fetches
               // one mean row per selected block instead of 4 raw rows (4x less PCIe), bit-identical.
               // + AGENS_KV_CACHE=1: per (layer, session) GPU cache of last step's 512 staged means; only blocks
               // that were not selected last step cross PCIe (delta fetch), bit-identical
inline bool on() { return tier() || int8(); }

// Allocation (init_device): every BCSA layer of device `dev` first, then init_device once.
// *kc/*vc = the bf16 pool (device or host-mapped; Session State snapshots read it through r2t), nullptr when int8.
int alloc_layer(int dev, int layer, int rows, void** kc, void** vc);
int init_device(int dev, int nslots, int wctx);

struct Ctx {
  int dev, layer, kslot;  // device index (0/1), layer, the session's KDA slot (1..nslots) = its ring slot
  int seq0, T;            // tokens before this call, tokens in it
  bool prefill;
  cudaStream_t s;
  const int* r2t;         // the session's position -> pool row map (device)
  int wctx, rows;
};
int store(const Ctx& c, const void* k, const void* v, const int* loc);  // replaces store_kv for the layer
// local branch (flashinfer decode or windowed prefill) over the window: same kv list positions as the all-GPU path
int local(const Ctx& c, void* q, int* qo_dev, int* kv_indptr, int* kv_idx, int* last1, float sm_scale, void* o,
          float* lse, int graph_mode);
// far decode in two halves: prefetch = indexer top-k (bridge) + async fetch of the selected rows on a side stream
// (overlaps the local branch); attend = waits for the fetch, then the far attention + merge (bridge).
int far_prefetch(const Ctx& c, void* qi, void* pos, void* reqs, void* blk);
int far_attend(const Ctx& c, void* qh, void* o_loc, float* lse, void* out);
int far_extend(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, void* idx, void* blk, void* out);
int commit_chunk(const Ctx& c);  // prefill: the chunk's last <= 4096 positions into the session's ring
// Session hooks (per device): fork copies the ring slot; rebuild refills a ring from the pool after a restore.
int fork(int dev, int src_kslot, int dst_kslot, cudaStream_t s);
int rebuild(int dev, int kslot, const int* r2t, int seq, cudaStream_t s);
// Parking: bytes of one session's rings on `dev` (0 without tiering) and their copy to / from `buf` (bumps the slot's
// block-cache generation either way).
// Graph-safe decode path (tiering without the block cache): every value comes from device memory (positions, pool
// rows, the ring row), fixed buffers, decode attention on the device-planned kernel (plan slot 0 = BCSA window, set
// by the caller), far branch on ak_far_decode_src reading the selected blocks from the (host) pools.
bool graph_safe();
bool cache_on();
int store_dec(const Ctx& c, const void* k, const void* v, const int* loc, const long long* pos);
// speculative verify: T store_dec calls in one launch per kind (same values)
int store_dec_multi(const Ctx& c, int T, const void* k, const void* v, const int* loc, const long long* pos);
// slot: the device-planned decode slot (0 decode; 2 + 2t query t of a speculative verify); dequant = false: an INT8
// ring already dequantized by an earlier query of the same verify
int local_dec(const Ctx& c, void* q, int* rng, float sm_scale, void* o, float* lse, int slot = 0, bool dequant = true);
// speculative verify in the far regime (bf16 ring): the T queries' windows in one ak_attn_decode_multi launch (planned
// by ak_attn_decode_multi_plan over ring_index); ring_index: position -> ring row of a slot (device)
int local_dec_multi(const Ctx& c, int mslot, void* q, float sm_scale, void* o, float* lse);
const int* ring_index(int dev);
// verify, far regime, split so the union fetch overlaps the window attention: select + fetch (side stream), then
// attend (joins the fetch). Outputs == far_dec_multi == T far_dec calls, bitwise.
// tag: device owner tag of the session's union caches (AGENS_SPEC_FAR_CACHE; null = no cache)
int far_select_fetch_multi(const Ctx& c, int T, void* qi, const long long* pos, void* blk, void* wsm,
                           const long long* tag = nullptr);
int far_attend_multi(const Ctx& c, int T, void* qh, void* o_loc, float* lse, const long long* pos, void* wsm, void* out);
int far_dec(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, void* blk, void* ws,
            void* out);
// speculative verify, all T queries in the far regime: ak_far_decode_multi_src (outputs == T far_dec calls, bitwise)
int far_dec_multi(const Ctx& c, int T, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, void* blk,
                  void* wsm, void* out);
// Batched graph-safe decode (rows' slots in a device table, slot 0 = padding row): per-row ring maps, stores, window,
// far branch.
int prep_rows(int dev, const int* slots, int B, cudaStream_t s);
int store_rows_dec(const Ctx& c, const void* k, const void* v, int B, const int* loc, const long long* pos,
                   const int* slots);
int local_dec_row(const Ctx& c, int r, void* q, int* rng, float sm_scale, void* o, float* lse);
int local_dec_row_i8(const Ctx& c, int r, const int* slots, void* q, int* rng, float sm_scale, void* o, float* lse);
int far_dec_row(const Ctx& c, void* qi, void* qh, void* o_loc, float* lse, const long long* pos, const int* r2t, void* blk,
                void* ws, void* out);
size_t ring_bytes(int dev);
// Fork inside a 4-row block: n pool rows of a BCSA layer from row `from` to row `to` (K/V, scales).
int copy_rows(int dev, int layer, int from, int to, int n, cudaStream_t s);
int ring_copy(int dev, int kslot, void* buf, bool to_buf, cudaStream_t s);
}  // namespace m1kv
