// MTP head (docs/mtp.md) for M1 speculative decoding: one decoder layer that predicts token t+2 from the
// target's final normed hidden state at t and the embedding of token t+1, chained (step s > 1 takes the head's own
// normed output as its hidden). Reads only device memory once set up, so a draft step is captured with the verify.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "m1_draft.h"

namespace m1mtp {
int register_weight(const char* name, void* ptr);  // bf16 device pointers, names without the "mtp." prefix
int register_w4(const char* name, const void* wt, const void* st, int n, int k, int group, int bits = 4);
// Before setup: the shortlist head (1.2). With the target's INT8 head, each chain step scores only lm_head rows
// ids[0..n) plus the DFlash2 drafter's kCandK candidates of its slot (when the drafter is loaded), not all 248K rows.
// n = 0: the whole head.
int set_shortlist(const int* ids, int n);
// After the weights: buffers for max_rows context rows per materialize call (the prefill chunk), one KV ring of the
// step-1 entries per ring (KDA slot), attention window `window` (the head's training window), chain buffers for
// max_seqs sequences per propose_b call (batched speculation; 1 otherwise).
int setup(int max_rows, int n_rings, int window, int max_seqs = 1);
bool ready();
size_t bytes();
int max_rows();
void set_ring(int ring);
int reset_ring(int ring, cudaStream_t s);            // an emptied session: no pending hidden, no context
int copy_ring(int dst, int src, cudaStream_t s);     // a forked session
// Context entries of committed tokens. H: the target's residual stream 0 rows (ld elements apart) of T rows at
// positions pos[0..T-1]; ids[0..T-1]: those rows' input tokens; n_dev (nullable: T): rows committed. Builds the
// step-1 entries of the pending row (position pos[0] - 1, its next token ids[0]) and of rows 0..n-2 (next tokens
// ids[1..n-1]), then keeps row n-1's normed hidden as the new pending row (its next token is the next anchor).
int materialize(const m1draft::Target& t, const void* final_norm, const __nv_bfloat16* H, int ld, int T, const int* n_dev,
                const int* ids, const long long* pos);
// The chain from the pending row: anchor token step[2] at position step[1]; step i (0-based) predicts draft slot i
// (step[3 + i]) and writes it unless the slot is a host proposal (i < step[20]); `steps` chain steps (<= 7).
int propose(const m1draft::Target& t, int* step, int steps);
// Batched (server mode): B sequences, records rec[B][m1draft::kRecInts] on device ([1] P, [2] anchor, [3..9] drafts,
// [18] KDA slot = MTP ring, [26] host proposal length). propose_b: each sequence's chain (steps <= 7) into its record's
// draft slots beyond its host proposal, all B rows per chain step in one pass over the head. materialize_b: the
// context entries of the committed rows, H [B * T rows] (sequence b's at b * T, its first n_dev[b * n_stride]
// committed), ids / pos the verify's [B * T].
int propose_b(const m1draft::Target& t, int B, int* rec, int steps);
int materialize_b(const m1draft::Target& t, const void* final_norm, const __nv_bfloat16* H, int ld, int B, int T,
                  const int* n_dev, int n_stride, const int* ids, const long long* pos, const int* rec);
int max_seqs();
}  // namespace m1mtp
