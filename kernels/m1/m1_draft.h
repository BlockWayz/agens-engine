// DFlash2 drafter (sglang's DFlash2DraftModel: DFlash backbone + candidate selector) for M1 speculative decoding.
// Reads only device memory once set up, so a draft step can be captured with the verify into one graph.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace m1draft {
constexpr int kBlock = 8;    // DFlash2 block: [anchor, mask x 7]
constexpr int kFeat = 5;     // target context features (layers 8, 21, 35, 53, 67 in sglang's ids)
// Batched speculative verify: one record of kRecInts ints per sequence (device), shared with the engine:
//   [1] P (anchor position)  [2] anchor (bonus) token  [3..9] the 7 drafts (written by propose_b)
//   [10..17] pool rows of the 8 verify positions  [18] KDA slot  [19] drafter ring  [20..21] r2t ptr  [22..23] Engram
//   prev ptr  [24..25] far-cache tag  [26] host proposal length nh  [27..33] host proposal (hybrid drafting: replaces
//   the first nh drafts after propose_b)
constexpr int kRecInts = 40;

int register_weight(const char* name, void* ptr);  // bf16 device pointers, checkpoint names (gate_up merged)
// a linear as W4 instead (k-tiled layout of ak_w4_repack_kt), under the name of its bf16 weight
int register_w4(const char* name, const void* wt, const void* st, int n, int k, int group, int bits = 4);
// After all weights: buffers for `max_ctx` context tokens per materialize call, the KV ring, rope cache.
int setup(int max_ctx, int mask_token_id, int max_pos, int n_rings = 1, int max_seqs = 1);  // max_pos < target Wctx
// the ring (context KV) of the single-sequence calls below (materialize / propose): one per KDA slot of the engine
void set_ring(int ring);
int rings();
// copy ring src to ring dst (session fork; no-op without per-slot rings)
int copy_ring(int dst, int src, cudaStream_t s);
bool ready();
size_t bytes();  // device memory owned by the drafter besides its weights (ring, buffers)

struct Target {      // what the drafter borrows from the target
  cudaStream_t s;
  const void* embed;    // [vocab][5120] bf16 (host-mapped is fine)
  const void* lm_head;  // [vocab][5120] bf16 (nullptr: the INT8 head below)
  int vocab_valid;
  const void* lm_head_i8 = nullptr;      // [vocab][5120] int8
  const float* lm_head_scale = nullptr;  // [vocab]
  const void* lm_head_w4 = nullptr;      // W4 g128 KT copy of the head (AGENS_M1_LMHEAD_W4=draft): weights / scales
  const void* lm_head_w4s = nullptr;
};
// Context KV of n target tokens at positions pos[0..n-1] (int64, device) from their features cap [n][5][5120]
// (n_dev, nullable: n read on device, capped at n_max).
int materialize(const Target& t, const __nv_bfloat16* cap, int n_max, const int* n_dev, const long long* pos);
// One draft block: anchor token = step[2], its position = step[1] (device ints, the verify staging buffer);
// writes the 7 drafted tokens to step[3..9].
int propose(const Target& t, int* step);
// Batched (B sequences, records rec[B][kRecInts] on device): drafts of every sequence into its record [3..9]
int propose_b(const Target& t, int B, int* rec);
// context KV of the committed tokens: cap [B * rows_per][5][5120] (sequence b's rows at b * rows_per, the first
// n_dev[b * n_stride] of them), positions pos [B * rows_per], ring of each sequence from its record
int materialize_b(const Target& t, const __nv_bfloat16* cap, int B, const int* n_dev, int n_stride, const long long* pos,
                  const int* rec, int rows_per = kBlock);
}  // namespace m1draft
