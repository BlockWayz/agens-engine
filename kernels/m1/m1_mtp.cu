// MTP head for M1 (docs/mtp.md; train_mtp.py export, mtp.* tensor names):
//   x  = fc(cat[pre_fc_norm_embedding(E[tok]), pre_fc_norm_hidden(h)])
//   x += o_proj(attn(layers.0.input_layernorm(x)) * sigmoid(gate))   24 q heads (q | gate per head) x 256, 4 kv heads,
//        Gemma q / k norm, NeoX RoPE on the first 64 dims (theta 1e7), causal, sliding window (2048)
//   x += down(silu(gate(n)) * up(n)),  n = layers.0.post_attention_layernorm(x)
//   y  = norm(x);  logits = y @ lm_head^T (the target's);  y is the next chain step's h
// All norms Gemma (x * rsqrt(mean(x^2) + eps) * (1 + w)). h of step 1 = the target's final normed hidden (lm_head input).
// Context: per committed position t a step-1 entry (k, v of the layer input built from h_t and token t+1) in a ring;
// step s of a chain anchored at t sits at position t + s - 1 and attends to the step-1 entries in its window
// (positions > t + s - 1 - W, <= t) plus the chain's own entries of steps 2..s.
// Engine flow: materialize() after every prefill chunk / verify (entries of the committed rows; the last committed
// row's hidden is kept "pending": its next token is the next anchor), propose() at the start of a drafted verify
// (step 1 = the pending row + anchor: its entry goes to the ring, its output predicts draft 1; steps 2.. chain).
// Batched speculation (server mode, 1.2): the same per sequence, B sequences per call (one row each per chain step,
// one pass over the head's weights for all of them), each on the ring of its KDA slot (record [18]).
// Like the DFlash2 drafter, the head only proposes: its numerics move acceptance, never the output tokens.
#include "m1_mtp.h"
#include "agens_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace m1mtp {
namespace {
using bf16 = __nv_bfloat16;
constexpr int HID = 5120, QH = 24, KVH = 4, HD = 256, ROT = 64, INTER = 17408, kRing = 4096, NVOC = 248320;
constexpr int kMaxSteps = 7, kNch = 16;
// context rows per materialize pass: a prefill chunk is materialized in passes of at most this many rows (1.2: the
// 1024-row buffers were 73 MB of the 16 GiB plan; 256 rows keep the dequantized-weight GEMM path, m > 128)
constexpr int kMatRows = 256;
constexpr float EPS = 1e-6f, THETA = 10000000.0f;

#define MCK(x)                                                                     \
  do {                                                                             \
    int e_ = (int)(x);                                                             \
    if (e_) {                                                                      \
      fprintf(stderr, "m1mtp: %s failed (%d) at %s:%d\n", #x, e_, __FILE__, __LINE__); \
      return e_;                                                                   \
    }                                                                              \
  } while (0)

std::map<std::string, void*> Wd;
struct W4 {
  const void *wt, *st;
  int n, k, group, bits;
};
std::map<std::string, W4> Wq;
struct Lin {
  const bf16* w = nullptr;
  const W4* q = nullptr;
  bool ok() const { return w || q; }
};
struct Mtp {
  bool ready = false;
  int max_rows = 0, n_rings = 1, ring = 0, window = 2048, max_seqs = 1;
  Lin fc, q, k, v, o, gu, down;
  const bf16 *ne = nullptr, *nh = nullptr, *in_ln = nullptr, *post_ln = nullptr, *qn = nullptr, *kn = nullptr,
             *norm = nullptr;
  bf16 *ring_k = nullptr, *ring_v = nullptr;  // [n_rings][kRing][KVH][HD]
  bf16* pend = nullptr;                       // [n_rings][HID] the pending row's normed target hidden
  int* pend_ok = nullptr;                     // [n_rings]
  // materialize rows [max_rows] (B blocks of T rows; B = 1 single-sequence): normed target rows, entry hiddens
  bf16 *hn = nullptr, *hb = nullptr, *e = nullptr, *cat = nullptr, *x = nullptr, *xn = nullptr, *kb = nullptr,
       *vb = nullptr;
  long long* rpos = nullptr;
  int *nb = nullptr, *ringb = nullptr, *tokb = nullptr;  // per sequence: committed rows, ring; chain step tokens
  // chain (one row per sequence per step; ck / cv [kMaxSteps][max_seqs][KVH][HD])
  bf16 *y = nullptr, *qg = nullptr, *qb = nullptr, *gb = nullptr, *ck = nullptr, *cv = nullptr, *ao = nullptr,
       *gu_b = nullptr, *act = nullptr, *t = nullptr, *logits = nullptr;
  float *apart = nullptr, *aml = nullptr;
  int* am = nullptr;
  long long* cpos = nullptr;
  bf16* wdq = nullptr;
  size_t wdq_bytes = 0, own = 0;
  // shortlist head: ids [nsl], per sequence its logits [max_seqs][nsl] + the drafter candidates' [max_seqs][kCandK]
  std::vector<int> sl_host;
  int nsl = 0, *sl = nullptr;
  float *sl_logit = nullptr, *cand_logit = nullptr;
} M;

template <typename T>
T* alloc(size_t n) {
  void* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T) > 0 ? n * sizeof(T) : 16) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  M.own += n * sizeof(T);
  return (T*)p;
}
const bf16* wp(const std::string& n) {
  auto it = Wd.find(n);
  if (it == Wd.end()) {
    fprintf(stderr, "m1mtp: missing weight %s\n", n.c_str());
    return nullptr;
  }
  return (const bf16*)it->second;
}
Lin lw(const std::string& n) {
  Lin l;
  auto it = Wq.find(n);
  if (it != Wq.end()) l.q = &it->second;
  else l.w = wp(n);
  return l;
}
// bf16: cuBLAS; W4: the engine's GEMV for m <= 128, else dequantized row tiles + cuBLAS (as m1_draft.cu)
int linear(const bf16* x, const Lin& w, bf16* y, int m, int n, int k, cudaStream_t s) {
  if (w.w) return ak_bf16_linear(x, w.w, y, m, n, k, s);
  const W4& q = *w.q;
  if (q.n != n || q.k != k) {
    fprintf(stderr, "m1mtp: W4 [%d, %d] called as [%d, %d]\n", q.n, q.k, n, k);
    return (int)cudaErrorInvalidValue;
  }
  if (m == 1 && q.bits == 4) return ak_w4_gemv_kt(x, q.wt, q.st, y, n, k, q.group, 0, s);
  if (m <= 128)
    return ak_wq_gemv_kt_multi(q.bits, x, m, 1, &q.wt, &q.st, reinterpret_cast<void* const*>(&y), &n, nullptr, k, q.group,
                               0, s);
  const int tn = (int)(M.wdq_bytes / 2 / k) & ~63;
  for (int n0 = 0; n0 < n; n0 += tn) {
    const int nt = n - n0 < tn ? n - n0 : tn;
    MCK(ak_wq_dequant_kt_rows(q.bits, q.wt, q.st, M.wdq, n, k, q.group, n0, nt, s));
    MCK(ak_bf16_linear_f32reduce_ld(x, M.wdq, y + n0, m, nt, k, n, s));
  }
  return 0;
}

// Records: the single-sequence verify staging buffer (step: [1] P, [2] anchor, [3..9] drafts, [20] host proposal
// length; the ring set on the host) or the batched verify records (stride m1draft::kRecInts: [26] proposal length,
// [18] the KDA slot = the ring).
struct Rec {
  int* base;
  int stride, nh_idx, ring_idx, host_ring, n_rings;
  __device__ int* at(int b) const { return base + (size_t)b * stride; }
  __device__ int ring(int b) const {
    const int r = ring_idx >= 0 ? at(b)[ring_idx] : host_ring;
    return r >= 0 && r < n_rings ? r : 0;
  }
};
Rec single_rec(int* step) { return Rec{step, 0, 20, -1, M.ring, M.n_rings}; }
Rec batch_rec(int* rec) { return Rec{rec, m1draft::kRecInts, 26, 18, 0, M.n_rings}; }

// materialize: block b's row r sits at position pos[b T] - 1 + r; nb[b] rows of it are committed
__global__ void mat_prep(Rec R, const long long* pos, int B, int T, const int* n_dev, int n_stride, long long* rpos,
                         int* nb, int* ringb) {
  const int b = blockIdx.x;
  if (b >= B) return;
  for (int r = threadIdx.x; r < T; r += blockDim.x) rpos[b * T + r] = pos[(size_t)b * T] - 1 + r;
  if (threadIdx.x == 0) {
    const int n = n_dev ? n_dev[(size_t)b * n_stride] : T;
    nb[b] = n < 0 ? 0 : (n > T ? T : n);
    ringb[b] = R.ring(b);
  }
}
// entry hidden of row (b, r): the pending row of b's ring (r = 0), else the normed target row r - 1
__global__ void gather_hidden(const bf16* __restrict__ hn, const bf16* __restrict__ pend, const int* ringb, int T,
                              bf16* __restrict__ hb) {
  const int j = blockIdx.x, b = j / T, r = j % T;
  const bf16* src = r == 0 ? pend + (size_t)ringb[b] * HID : hn + (size_t)(j - 1) * HID;
  for (int i = threadIdx.x; i < HID / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(hb + (size_t)j * HID)[i] = reinterpret_cast<const uint4*>(src)[i];
}
// entries of committed rows (row 0 only with a pending row) into each block's ring
__global__ void store_rows(const bf16* __restrict__ k, const bf16* __restrict__ v, const long long* rpos, int BT, int T,
                           const int* nb, const int* ringb, const int* pend_ok, bf16* rk, bf16* rv) {
  const long long tot = (long long)BT * KVH * HD;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
    const int j = (int)(i / (KVH * HD)), c = (int)(i % (KVH * HD)), b = j / T, r = j % T, g = ringb[b];
    if (r >= nb[b] || (r == 0 && !pend_ok[g]) || rpos[j] < 0) continue;
    const size_t o = ((size_t)g * kRing + rpos[j] % kRing) * KVH * HD + c;
    rk[o] = k[i], rv[o] = v[i];
  }
}
// the new pending row of block b: its last committed row's normed hidden (hn[b T + nb[b] - 1])
__global__ void set_pending(const bf16* __restrict__ hn, int T, const int* nb, const int* ringb, bf16* pend,
                            int* pend_ok) {
  const int b = blockIdx.x, n = nb[b], g = ringb[b];
  if (n < 1) return;
  const bf16* src = hn + ((size_t)b * T + n - 1) * HID;
  for (int i = threadIdx.x; i < HID / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(pend + (size_t)g * HID)[i] = reinterpret_cast<const uint4*>(src)[i];
  if (threadIdx.x == 0) pend_ok[g] = 1;
}
// chain step i of every sequence: its input token, position P - 1 + i, ring
__global__ void chain_prep(Rec R, int B, int i, int* tok, long long* cpos, int* ringb) {
  const int b = threadIdx.x;
  if (b >= B) return;
  const int* r = R.at(b);
  tok[b] = i == 0 ? r[2] : r[3 + i - 1];
  cpos[b] = (long long)r[1] - 1 + i;
  ringb[b] = R.ring(b);
}
__global__ void gather_pend(const bf16* __restrict__ pend, const int* ringb, bf16* __restrict__ y) {
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < HID / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(y + (size_t)b * HID)[i] = reinterpret_cast<const uint4*>(pend + (size_t)ringb[b] * HID)[i];
}
__global__ void split_gate(const bf16* __restrict__ qg, bf16* __restrict__ g) {  // block = row b * QH + h
  const int h = blockIdx.x, d = threadIdx.x;
  g[h * HD + d] = qg[h * 2 * HD + HD + d];
}
// step 0's entries (the pending rows) into the rings at position P - 1
__global__ void store_step0(const bf16* __restrict__ k, const bf16* __restrict__ v, Rec R, const int* ringb,
                            const int* pend_ok, bf16* rk, bf16* rv) {
  const int b = blockIdx.x, g = ringb[b], p = R.at(b)[1] - 1;
  if (!pend_ok[g] || p < 0) return;
  for (int c = threadIdx.x; c < KVH * HD; c += blockDim.x) {
    const size_t o = ((size_t)g * kRing + p % kRing) * KVH * HD + c;
    rk[o] = k[(size_t)b * KVH * HD + c], rv[o] = v[(size_t)b * KVH * HD + c];
  }
}
// one query row per sequence (step i): keys = its ring's entries at positions [max(0, qp - W + 1), P - 1] + its chain
// entries 1..i (ck / cv [kMaxSteps][B][KVH][HD]); grid (QH, kNch, B), 8 warps, a warp per key, lane = 8 dims
__global__ void __launch_bounds__(256) mtp_attn(const bf16* __restrict__ q, const bf16* __restrict__ rk,
                                                const bf16* __restrict__ rv, const bf16* __restrict__ ck,
                                                const bf16* __restrict__ cv, Rec R, const int* ringb, int B, int i,
                                                int W, float* part, float* ml) {
  const int h = blockIdx.x, c = blockIdx.y, b = blockIdx.z, kvh = h / (QH / KVH), w = threadIdx.x >> 5,
            lane = threadIdx.x & 31;
  const int P = R.at(b)[1], qp = P - 1 + i;
  const size_t r0 = (size_t)ringb[b] * kRing * KVH * HD;
  const int lo = qp - W + 1 > 0 ? qp - W + 1 : 0;
  const int nr = P - 1 >= lo ? P - lo : 0, tot = nr + i;
  const int per = (tot + kNch - 1) / kNch, k0 = c * per, k1 = k0 + per < tot ? k0 + per : tot;
  float qv[8], acc[8];
  const bf16* qr = q + ((size_t)b * QH + h) * HD;
  for (int j = 0; j < 8; ++j) qv[j] = __bfloat162float(qr[lane * 8 + j]), acc[j] = 0.f;
  float m = -INFINITY, l = 0.f;
  for (int kk = k0 + w; kk < k1; kk += 8) {
    const bf16 *kp, *vp;
    if (kk < nr) {
      const size_t o = r0 + (size_t)((lo + kk) % kRing) * KVH * HD + kvh * HD;
      kp = rk + o, vp = rv + o;
    } else {
      const size_t o = ((size_t)(kk - nr + 1) * B + b) * KVH * HD + kvh * HD;
      kp = ck + o, vp = cv + o;
    }
    const uint4 kr = reinterpret_cast<const uint4*>(kp)[lane], vr = reinterpret_cast<const uint4*>(vp)[lane];
    const bf16 *ke = reinterpret_cast<const bf16*>(&kr), *ve = reinterpret_cast<const bf16*>(&vr);
    float s = 0.f;
    for (int j = 0; j < 8; ++j) s += qv[j] * __bfloat162float(ke[j]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    s *= 0.0625f;  // 1 / sqrt(256)
    const float mn = fmaxf(m, s), a = __expf(m - mn), e = __expf(s - mn);
    l = l * a + e;
    for (int j = 0; j < 8; ++j) acc[j] = acc[j] * a + e * __bfloat162float(ve[j]);
    m = mn;
  }
  __shared__ float sm[8], sl[8], sa[8][HD];
  if (lane == 0) sm[w] = m, sl[w] = l;
  for (int j = 0; j < 8; ++j) sa[w][lane * 8 + j] = acc[j];
  __syncthreads();
  const int d = threadIdx.x;
  float Mx = -INFINITY;
  for (int j = 0; j < 8; ++j) Mx = fmaxf(Mx, sm[j]);
  float L = 0.f, A = 0.f;
  if (Mx > -INFINITY)
    for (int j = 0; j < 8; ++j)
      if (sm[j] > -INFINITY) {
        const float f = __expf(sm[j] - Mx);
        L += sl[j] * f, A += sa[j][d] * f;
      }
  const size_t pr = ((size_t)b * QH + h) * kNch + c;
  part[pr * HD + d] = A;
  if (d == 0) ml[pr * 2] = Mx, ml[pr * 2 + 1] = L;
}
__global__ void mtp_attn_merge(const float* __restrict__ part, const float* __restrict__ ml, bf16* __restrict__ out) {
  const int h = blockIdx.x, b = blockIdx.y, d = threadIdx.x;
  const size_t p0 = ((size_t)b * QH + h) * kNch;
  float Mx = -INFINITY;
  for (int c = 0; c < kNch; ++c) Mx = fmaxf(Mx, ml[(p0 + c) * 2]);
  float L = 0.f, A = 0.f;
  if (Mx > -INFINITY)
    for (int c = 0; c < kNch; ++c) {
      const float m = ml[(p0 + c) * 2];
      if (m == -INFINITY) continue;
      const float f = __expf(m - Mx);
      L += ml[(p0 + c) * 2 + 1] * f, A += part[(p0 + c) * HD + d] * f;
    }
  out[((size_t)b * QH + h) * HD + d] = __float2bfloat16_rn(L > 0.f ? A / L : 0.f);
}
// draft slot i <- the step's argmax, unless the slot is a host proposal or there is no pending row (no context yet)
__global__ void write_slot(Rec R, int B, int i, const int* am, const int* ringb, const int* pend_ok) {
  const int b = threadIdx.x;
  if (b >= B) return;
  int* r = R.at(b);
  if (pend_ok[ringb[b]] && i >= r[R.nh_idx] && i < kMaxSteps) r[3 + i] = am[b];
}

int lm_head(const m1draft::Target& t, const bf16* y, bf16* logits, int m, cudaStream_t s);
// Shortlist head (INT8 lm_head rows ids[o], o < n): logit[j][o] = scale[id] * sum_i y[j][i] * q[id][i], fp32; one
// warp per row as ak_i8_gemv (kernels/src/i8_gemv.cu), up to M sequences per pass over the rows
__device__ __forceinline__ float dot4(uint32_t w, uint32_t x01, uint32_t x23, float s) {
  s = fmaf(__uint_as_float(x01 << 16), (float)(int8_t)(w & 0xFF), s);
  s = fmaf(__uint_as_float(x01 & 0xFFFF0000u), (float)(int8_t)((w >> 8) & 0xFF), s);
  s = fmaf(__uint_as_float(x23 << 16), (float)(int8_t)((w >> 16) & 0xFF), s);
  s = fmaf(__uint_as_float(x23 & 0xFFFF0000u), (float)(int8_t)(w >> 24), s);
  return s;
}
__device__ __forceinline__ float row_dot(const uint4* __restrict__ wr, const bf16* __restrict__ y, int lane) {
  float acc = 0.f;
  for (int c = lane; c < HID / 16; c += 32) {
    const uint4 q = __ldg(wr + c);
    const uint4* xr = reinterpret_cast<const uint4*>(y) + 2 * c;
    const uint4 a = __ldg(xr), b = __ldg(xr + 1);
    acc = dot4(q.x, a.x, a.y, acc);
    acc = dot4(q.y, a.z, a.w, acc);
    acc = dot4(q.z, b.x, b.y, acc);
    acc = dot4(q.w, b.z, b.w, acc);
  }
  for (int d = 16; d > 0; d >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
  return acc;
}
template <int MS>
__global__ void __launch_bounds__(256) sl_gemv(const bf16* __restrict__ y, int B, const uint4* __restrict__ w,
                                               const float* __restrict__ scale, const int* __restrict__ ids, int n,
                                               float* __restrict__ out) {
  const int lane = threadIdx.x & 31, o = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (o >= n) return;
  const int id = ids[o];
  const uint4* wr = w + (size_t)id * (HID / 16);
  float acc[MS];
#pragma unroll
  for (int j = 0; j < MS; ++j) acc[j] = 0.f;
  for (int c = lane; c < HID / 16; c += 32) {
    const uint4 q = __ldg(wr + c);
#pragma unroll
    for (int j = 0; j < MS; ++j) {
      if (j >= B) break;
      const uint4* xr = reinterpret_cast<const uint4*>(y + (size_t)j * HID) + 2 * c;
      const uint4 a = __ldg(xr), b = __ldg(xr + 1);
      float s = 0.f;
      s = dot4(q.x, a.x, a.y, s);
      s = dot4(q.y, a.z, a.w, s);
      s = dot4(q.z, b.x, b.y, s);
      s = dot4(q.w, b.z, b.w, s);
      acc[j] += s;
    }
  }
  const float sc = scale[id];
#pragma unroll
  for (int j = 0; j < MS; ++j) {
    if (j >= B) break;
    float v = acc[j];
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
    if (lane == 0) out[(size_t)j * n + o] = v * sc;
  }
}
// the drafter's kCandK candidates of slot i of each sequence (grid B, a warp per candidate)
__global__ void __launch_bounds__(m1draft::kCandK * 32) cand_dot(const bf16* __restrict__ y, const uint4* __restrict__ w,
                                                                const float* __restrict__ scale,
                                                                const int* __restrict__ cid, int i,
                                                                float* __restrict__ out) {
  const int b = blockIdx.x, c = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int id = cid[((size_t)b * (m1draft::kBlock - 1) + i) * m1draft::kCandK + c];
  const float v = row_dot(w + (size_t)id * (HID / 16), y + (size_t)b * HID, lane);
  if (lane == 0) out[b * m1draft::kCandK + c] = v * scale[id];
}
// per sequence: the best of its shortlist logits and its candidates' (ties: the lower token id)
__global__ void __launch_bounds__(1024) sl_argmax(const float* __restrict__ logit, const int* __restrict__ ids, int n,
                                                  const float* __restrict__ cl, const int* __restrict__ cid, int i,
                                                  int vocab_valid, int* am) {
  const int b = blockIdx.x;
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  auto take = [&](float v, int id) {
    if (id < vocab_valid && (v > bv || (v == bv && id < bi))) bv = v, bi = id;
  };
  for (int o = threadIdx.x; o < n; o += blockDim.x) take(logit[(size_t)b * n + o], ids[o]);
  if (cid && threadIdx.x < m1draft::kCandK)
    take(cl[b * m1draft::kCandK + threadIdx.x],
         cid[((size_t)b * (m1draft::kBlock - 1) + i) * m1draft::kCandK + threadIdx.x]);
  for (int d = 16; d > 0; d >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, d);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, d);
    if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sv[w] = bv, si[w] = bi;
  __syncthreads();
  if (w == 0) {
    bv = lane < (int)(blockDim.x >> 5) ? sv[lane] : -INFINITY, bi = lane < (int)(blockDim.x >> 5) ? si[lane] : 0x7FFFFFFF;
    for (int d = 16; d > 0; d >>= 1) {
      const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, d);
      const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, d);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) am[b] = bi == 0x7FFFFFFF ? 0 : bi;
  }
}
// the chain step's argmax over the shortlist (+ the drafter's candidates of slot i), or over the whole head
int head_argmax(const m1draft::Target& t, int B, int i, cudaStream_t s) {
  if (M.nsl > 0 && t.lm_head_i8 && !t.lm_head && !t.lm_head_w4) {
    const uint4* w = (const uint4*)t.lm_head_i8;
    const dim3 g((M.nsl + 7) / 8);
    if (B == 1) sl_gemv<1><<<g, 256, 0, s>>>(M.y, B, w, t.lm_head_scale, M.sl, M.nsl, M.sl_logit);
    else if (B <= 8) sl_gemv<8><<<g, 256, 0, s>>>(M.y, B, w, t.lm_head_scale, M.sl, M.nsl, M.sl_logit);
    else sl_gemv<16><<<g, 256, 0, s>>>(M.y, B, w, t.lm_head_scale, M.sl, M.nsl, M.sl_logit);
    const int* cid = m1draft::cand_ids();
    if (cid && i < m1draft::kBlock - 1)
      cand_dot<<<B, m1draft::kCandK * 32, 0, s>>>(M.y, w, t.lm_head_scale, cid, i, M.cand_logit);
    else
      cid = nullptr;
    sl_argmax<<<B, 1024, 0, s>>>(M.sl_logit, M.sl, M.nsl, M.cand_logit, cid, i, t.vocab_valid, M.am);
    return (int)cudaGetLastError();
  }
  MCK(lm_head(t, M.y, M.logits, B, s));
  return ak_argmax(M.logits, B, NVOC, t.vocab_valid, M.am, s);
}

int lm_head(const m1draft::Target& t, const bf16* y, bf16* logits, int m, cudaStream_t s) {
  if (t.lm_head_w4) {
    const int nv = NVOC;
    void* out = logits;
    return ak_wq_gemv_kt_multi(4, y, m, 1, &t.lm_head_w4, &t.lm_head_w4s, &out, &nv, nullptr, HID, 128, 0, s);
  }
  if (t.lm_head) return ak_bf16_linear(y, t.lm_head, logits, m, NVOC, HID, s);
  return ak_i8_gemv(y, t.lm_head_i8, t.lm_head_scale, logits, m, NVOC, HID, NVOC, s);  // m <= 16
}

int materialize_impl(const m1draft::Target& t, const void* final_norm, const bf16* H, int ld, int B, int T,
                     const int* n_dev, int n_stride, const int* ids, const long long* pos, Rec R) {
  if (!M.ready) return 0;
  const int BT = B * T;
  if (B < 1 || T < 1 || BT > M.max_rows || B > M.max_seqs) return (int)cudaErrorInvalidValue;
  cudaStream_t s = t.s;
  MCK(ak_rmsnorm(H, final_norm, M.hn, BT, HID, ld, HID, EPS, 1, s));  // the target's lm_head input of each row
  mat_prep<<<B, 128, 0, s>>>(R, pos, B, T, n_dev, n_stride, M.rpos, M.nb, M.ringb);
  gather_hidden<<<BT, 256, 0, s>>>(M.hn, M.pend, M.ringb, T, M.hb);
  // entry rows: hidden hb[j], token ids[j], position rpos[j]
  MCK(ak_embed(ids, t.embed, M.e, BT, HID, s));
  MCK(ak_rmsnorm(M.e, M.ne, M.cat, BT, HID, HID, 2 * HID, EPS, 1, s));
  MCK(ak_rmsnorm(M.hb, M.nh, M.cat + HID, BT, HID, HID, 2 * HID, EPS, 1, s));
  MCK(linear(M.cat, M.fc, M.x, BT, HID, 2 * HID, s));
  MCK(ak_rmsnorm(M.x, M.in_ln, M.xn, BT, HID, HID, HID, EPS, 1, s));
  MCK(linear(M.xn, M.k, M.kb, BT, KVH * HD, HID, s));
  MCK(linear(M.xn, M.v, M.vb, BT, KVH * HD, HID, s));
  MCK(ak_rmsnorm(M.kb, M.kn, M.kb, BT * KVH, HD, HD, HD, EPS, 1, s));
  MCK(ak_rope_neox_fly(M.kb, M.kb, (const int64_t*)M.rpos, BT, 0, KVH, HD, ROT, 0, KVH * HD, THETA, s));
  const int nbk = (int)(((long)BT * KVH * HD + 255) / 256);
  store_rows<<<nbk < 1024 ? nbk : 1024, 256, 0, s>>>(M.kb, M.vb, M.rpos, BT, T, M.nb, M.ringb, M.pend_ok, M.ring_k,
                                                    M.ring_v);
  set_pending<<<B, 256, 0, s>>>(M.hn, T, M.nb, M.ringb, M.pend, M.pend_ok);
  return (int)cudaGetLastError();
}

int propose_impl(const m1draft::Target& t, int B, Rec R, int steps) {
  if (!M.ready) return 0;
  if (B < 1 || B > M.max_seqs) return (int)cudaErrorInvalidValue;
  cudaStream_t s = t.s;
  if (steps > kMaxSteps) steps = kMaxSteps;
  const size_t kv = (size_t)KVH * HD;
  for (int i = 0; i < steps; ++i) {
    chain_prep<<<1, 32, 0, s>>>(R, B, i, M.tokb, M.cpos, M.ringb);
    if (i == 0) gather_pend<<<B, 256, 0, s>>>(M.pend, M.ringb, M.y);  // step 0's hidden: the pending rows
    MCK(ak_embed(M.tokb, t.embed, M.e, B, HID, s));
    MCK(ak_rmsnorm(M.e, M.ne, M.cat, B, HID, HID, 2 * HID, EPS, 1, s));
    MCK(ak_rmsnorm(M.y, M.nh, M.cat + HID, B, HID, HID, 2 * HID, EPS, 1, s));
    MCK(linear(M.cat, M.fc, M.x, B, HID, 2 * HID, s));
    MCK(ak_rmsnorm(M.x, M.in_ln, M.xn, B, HID, HID, HID, EPS, 1, s));
    bf16 *ck = M.ck + (size_t)i * B * kv, *cv = M.cv + (size_t)i * B * kv;
    MCK(linear(M.xn, M.q, M.qg, B, 2 * QH * HD, HID, s));
    MCK(linear(M.xn, M.k, ck, B, KVH * HD, HID, s));
    MCK(linear(M.xn, M.v, cv, B, KVH * HD, HID, s));
    MCK(ak_rmsnorm(M.qg, M.qn, M.qb, B * QH, HD, 2 * HD, HD, EPS, 1, s));  // q half of each head's [q | gate]
    split_gate<<<B * QH, HD, 0, s>>>(M.qg, M.gb);
    MCK(ak_rmsnorm(ck, M.kn, ck, B * KVH, HD, HD, HD, EPS, 1, s));
    MCK(ak_rope_neox_fly(M.qb, ck, (const int64_t*)M.cpos, B, QH, KVH, HD, ROT, QH * HD, KVH * HD, THETA, s));
    if (i == 0) store_step0<<<B, 256, 0, s>>>(ck, cv, R, M.ringb, M.pend_ok, M.ring_k, M.ring_v);
    mtp_attn<<<dim3(QH, kNch, B), 256, 0, s>>>(M.qb, M.ring_k, M.ring_v, M.ck, M.cv, R, M.ringb, B, i, M.window,
                                              M.apart, M.aml);
    mtp_attn_merge<<<dim3(QH, B), HD, 0, s>>>(M.apart, M.aml, M.ao);
    MCK(ak_bf16_binary(M.ao, M.gb, M.ao, (long)B * QH * HD, 2, s));  // * sigmoid(gate)
    MCK(linear(M.ao, M.o, M.t, B, HID, QH * HD, s));
    MCK(ak_bf16_binary(M.x, M.t, M.x, (long)B * HID, 0, s));
    MCK(ak_rmsnorm(M.x, M.post_ln, M.xn, B, HID, HID, HID, EPS, 1, s));
    MCK(linear(M.xn, M.gu, M.gu_b, B, 2 * INTER, HID, s));
    MCK(ak_silu_mul(M.gu_b, M.act, B, INTER, s));
    MCK(linear(M.act, M.down, M.t, B, HID, INTER, s));
    MCK(ak_bf16_binary(M.x, M.t, M.x, (long)B * HID, 0, s));
    MCK(ak_rmsnorm(M.x, M.norm, M.y, B, HID, HID, HID, EPS, 1, s));
    MCK(head_argmax(t, B, i, s));
    write_slot<<<1, 32, 0, s>>>(R, B, i, M.am, M.ringb, M.pend_ok);
  }
  return (int)cudaGetLastError();
}
}  // namespace

int register_weight(const char* name, void* ptr) {
  Wd[name] = ptr;
  return 0;
}
int register_w4(const char* name, const void* wt, const void* st, int n, int k, int group, int bits) {
  Wq[name] = W4{wt, st, n, k, group, bits};
  return 0;
}
int set_shortlist(const int* ids, int n) {
  if (M.ready || n < 0) return (int)cudaErrorInvalidValue;
  M.sl_host.assign(ids, ids + n);
  return 0;
}
bool ready() { return M.ready; }
size_t bytes() { return M.own; }
int max_rows() { return M.max_rows; }
int max_seqs() { return M.max_seqs; }
void set_ring(int ring) { M.ring = ring >= 0 && ring < M.n_rings ? ring : 0; }
int reset_ring(int ring, cudaStream_t s) {
  if (!M.ready || ring < 0 || ring >= M.n_rings) return 0;
  return (int)cudaMemsetAsync(M.pend_ok + ring, 0, sizeof(int), s);
}
int copy_ring(int dst, int src, cudaStream_t s) {
  if (!M.ready || dst == src || dst < 0 || src < 0 || dst >= M.n_rings || src >= M.n_rings) return 0;
  const size_t n = (size_t)kRing * KVH * HD;
  MCK(cudaMemcpyAsync(M.ring_k + dst * n, M.ring_k + src * n, n * 2, cudaMemcpyDeviceToDevice, s));
  MCK(cudaMemcpyAsync(M.ring_v + dst * n, M.ring_v + src * n, n * 2, cudaMemcpyDeviceToDevice, s));
  MCK(cudaMemcpyAsync(M.pend + (size_t)dst * HID, M.pend + (size_t)src * HID, HID * 2, cudaMemcpyDeviceToDevice, s));
  return (int)cudaMemcpyAsync(M.pend_ok + dst, M.pend_ok + src, sizeof(int), cudaMemcpyDeviceToDevice, s);
}

int setup(int max_rows, int n_rings, int window, int max_seqs) {
  M.max_seqs = max_seqs < 1 ? 1 : (max_seqs > 16 ? 16 : max_seqs);  // <= 16: the INT8 head's rows per call
  M.max_rows = max_rows < 8 ? 8 : (max_rows > kMatRows ? kMatRows : max_rows);
  if (M.max_rows < M.max_seqs * 8) M.max_rows = M.max_seqs * 8;  // a batched verify's B x 8 rows
  M.n_rings = n_rings < 1 ? 1 : n_rings;
  M.window = window > 0 && window <= kRing - 2 * kMaxSteps ? window : 2048;
  M.fc = lw("fc.weight"), M.ne = wp("pre_fc_norm_embedding.weight"), M.nh = wp("pre_fc_norm_hidden.weight");
  M.norm = wp("norm.weight");
  const std::string p = "layers.0.";
  M.in_ln = wp(p + "input_layernorm.weight"), M.post_ln = wp(p + "post_attention_layernorm.weight");
  M.q = lw(p + "self_attn.q_proj.weight"), M.k = lw(p + "self_attn.k_proj.weight"), M.v = lw(p + "self_attn.v_proj.weight");
  M.o = lw(p + "self_attn.o_proj.weight"), M.qn = wp(p + "self_attn.q_norm.weight"), M.kn = wp(p + "self_attn.k_norm.weight");
  M.gu = lw(p + "mlp.gate_up_proj.weight"), M.down = lw(p + "mlp.down_proj.weight");
  for (const bf16* x : {M.ne, M.nh, M.norm, M.in_ln, M.post_ln, M.qn, M.kn})
    if (!x) return 1;
  for (const Lin* x : {&M.fc, &M.q, &M.k, &M.v, &M.o, &M.gu, &M.down})
    if (!x->ok()) return 1;
  M.ring_k = alloc<bf16>((size_t)M.n_rings * kRing * KVH * HD), M.ring_v = alloc<bf16>((size_t)M.n_rings * kRing * KVH * HD);
  M.pend = alloc<bf16>((size_t)M.n_rings * HID), M.pend_ok = alloc<int>(M.n_rings);
  const size_t R = M.max_rows, NS = M.max_seqs;
  M.hn = alloc<bf16>(R * HID), M.hb = alloc<bf16>(R * HID), M.e = alloc<bf16>(R * HID), M.cat = alloc<bf16>(R * 2 * HID);
  M.x = alloc<bf16>(R * HID), M.xn = alloc<bf16>(R * HID), M.kb = alloc<bf16>(R * KVH * HD), M.vb = alloc<bf16>(R * KVH * HD);
  M.rpos = alloc<long long>(R), M.nb = alloc<int>(NS), M.ringb = alloc<int>(NS), M.tokb = alloc<int>(NS);
  M.y = alloc<bf16>(NS * HID), M.qg = alloc<bf16>(NS * 2 * QH * HD), M.qb = alloc<bf16>(NS * QH * HD);
  M.gb = alloc<bf16>(NS * QH * HD);
  M.ck = alloc<bf16>((size_t)kMaxSteps * NS * KVH * HD), M.cv = alloc<bf16>((size_t)kMaxSteps * NS * KVH * HD);
  M.ao = alloc<bf16>(NS * QH * HD), M.gu_b = alloc<bf16>(NS * 2 * INTER), M.act = alloc<bf16>(NS * INTER);
  M.nsl = (int)M.sl_host.size();
  M.t = alloc<bf16>(NS * HID), M.logits = alloc<bf16>(NS * NVOC), M.apart = alloc<float>(NS * QH * kNch * HD);
  M.aml = alloc<float>(NS * QH * kNch * 2), M.am = alloc<int>(NS), M.cpos = alloc<long long>(NS);
  if (!Wq.empty()) M.wdq_bytes = (size_t)16 << 20, M.wdq = alloc<bf16>(M.wdq_bytes / 2);
  if (M.nsl) {
    M.sl = alloc<int>(M.nsl), M.sl_logit = alloc<float>(NS * M.nsl), M.cand_logit = alloc<float>(NS * m1draft::kCandK);
    if (!M.sl || !M.sl_logit || !M.cand_logit) return 2;
    MCK(cudaMemcpy(M.sl, M.sl_host.data(), (size_t)M.nsl * 4, cudaMemcpyHostToDevice));
  }
  if (!M.ring_k || !M.ring_v || !M.pend || !M.pend_ok || !M.hn || !M.hb || !M.e || !M.cat || !M.x || !M.xn || !M.kb ||
      !M.vb || !M.rpos || !M.nb || !M.ringb || !M.tokb || !M.y || !M.qg || !M.qb || !M.gb || !M.ck || !M.cv || !M.ao ||
      !M.gu_b || !M.act || !M.t || !M.logits || !M.apart || !M.aml || !M.am || !M.cpos || (M.wdq_bytes && !M.wdq))
    return 2;
  MCK(cudaDeviceSynchronize());
  M.ready = true;
  return 0;
}

int materialize(const m1draft::Target& t, const void* final_norm, const bf16* H, int ld, int T, const int* n_dev,
                const int* ids, const long long* pos) {
  if (n_dev || T <= M.max_rows) return materialize_impl(t, final_norm, H, ld, 1, T, n_dev, 0, ids, pos, single_rec(nullptr));
  // all T rows committed (prefill): passes of max_rows rows, each continuing from the pending row the last one left
  for (int r0 = 0; r0 < T; r0 += M.max_rows) {
    const int n = T - r0 < M.max_rows ? T - r0 : M.max_rows;
    MCK(materialize_impl(t, final_norm, H + (size_t)r0 * ld, ld, 1, n, nullptr, 0, ids + r0, pos + r0, single_rec(nullptr)));
  }
  return 0;
}
int materialize_b(const m1draft::Target& t, const void* final_norm, const bf16* H, int ld, int B, int T,
                  const int* n_dev, int n_stride, const int* ids, const long long* pos, const int* rec) {
  return materialize_impl(t, final_norm, H, ld, B, T, n_dev, n_stride, ids, pos, batch_rec(const_cast<int*>(rec)));
}
int propose(const m1draft::Target& t, int* step, int steps) { return propose_impl(t, 1, single_rec(step), steps); }
int propose_b(const m1draft::Target& t, int B, int* rec, int steps) { return propose_impl(t, B, batch_rec(rec), steps); }
}  // namespace m1mtp
