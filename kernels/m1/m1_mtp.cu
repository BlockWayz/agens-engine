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
// Like the DFlash2 drafter, the head only proposes: its numerics move acceptance, never the output tokens.
#include "m1_mtp.h"
#include "agens_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace m1mtp {
namespace {
using bf16 = __nv_bfloat16;
constexpr int HID = 5120, QH = 24, KVH = 4, HD = 256, ROT = 64, INTER = 17408, kRing = 4096, NVOC = 248320;
constexpr int kMaxSteps = 7, kNch = 16;
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
  int max_rows = 0, n_rings = 1, ring = 0, window = 2048;
  Lin fc, q, k, v, o, gu, down;
  const bf16 *ne = nullptr, *nh = nullptr, *in_ln = nullptr, *post_ln = nullptr, *qn = nullptr, *kn = nullptr,
             *norm = nullptr;
  bf16 *ring_k = nullptr, *ring_v = nullptr;  // [n_rings][kRing][KVH][HD]
  bf16* pend = nullptr;                       // [n_rings][HID] the pending row's normed target hidden
  int* pend_ok = nullptr;                     // [n_rings]
  // materialize rows [max_rows + 1]
  bf16 *hb = nullptr, *e = nullptr, *cat = nullptr, *x = nullptr, *xn = nullptr, *kb = nullptr, *vb = nullptr;
  long long* rpos = nullptr;
  // chain (one row per step)
  bf16 *y = nullptr, *qg = nullptr, *qb = nullptr, *gb = nullptr, *ck = nullptr, *cv = nullptr, *ao = nullptr,
       *gu_b = nullptr, *act = nullptr, *t = nullptr, *logits = nullptr;
  float *apart = nullptr, *aml = nullptr;
  int* am = nullptr;
  long long* cpos = nullptr;
  bf16* wdq = nullptr;
  size_t wdq_bytes = 0, own = 0;
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

// materialize: row r (0..T-1) of the entry block sits at position pos[0] - 1 + r
__global__ void rows_pos(const long long* pos, int T, long long* rpos) {
  for (int r = threadIdx.x; r < T; r += blockDim.x) rpos[r] = pos[0] - 1 + r;
}
__global__ void copy_row(const bf16* __restrict__ src, bf16* __restrict__ dst) {
  for (int i = threadIdx.x; i < HID / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
}
// entries of rows r < n (row 0 only with a pending row) into the ring
__global__ void store_rows(const bf16* __restrict__ k, const bf16* __restrict__ v, const long long* rpos, int T,
                           const int* n_dev, const int* pend_ok, bf16* rk, bf16* rv) {
  const int n = n_dev ? n_dev[0] : T;
  const long long tot = (long long)T * KVH * HD;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < tot; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / (KVH * HD)), c = (int)(i % (KVH * HD));
    if (r >= n || (r == 0 && !*pend_ok) || rpos[r] < 0) continue;
    const size_t o = (size_t)(rpos[r] % kRing) * KVH * HD + c;
    rk[o] = k[i], rv[o] = v[i];
  }
}
// the new pending row: hb[n] (= target row n - 1)
__global__ void set_pending(const bf16* __restrict__ hb, int T, const int* n_dev, bf16* pend, int* pend_ok) {
  const int n = n_dev ? n_dev[0] : T;
  const bf16* src = hb + (size_t)n * HID;
  for (int i = threadIdx.x; i < HID / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(pend)[i] = reinterpret_cast<const uint4*>(src)[i];
  if (threadIdx.x == 0) *pend_ok = 1;
}
// chain step i: position step[1] - 1 + i; step 0's entry (the pending row) goes to the ring
__global__ void chain_pos(const int* step, int i, long long* cpos) { cpos[0] = (long long)step[1] - 1 + i; }
__global__ void split_gate(const bf16* __restrict__ qg, bf16* __restrict__ g) {
  const int h = blockIdx.x, d = threadIdx.x;
  g[h * HD + d] = qg[h * 2 * HD + HD + d];
}
__global__ void store_step0(const bf16* __restrict__ k, const bf16* __restrict__ v, const int* step, const int* pend_ok,
                            bf16* rk, bf16* rv) {
  const int p = step[1] - 1;
  if (!*pend_ok || p < 0) return;
  for (int c = threadIdx.x; c < KVH * HD; c += blockDim.x) {
    const size_t o = (size_t)(p % kRing) * KVH * HD + c;
    rk[o] = k[c], rv[o] = v[c];
  }
}
// one query row (step i): keys = ring entries at positions [max(0, qp - W + 1), P - 1] + chain entries 1..i
// (ck / cv [kMaxSteps][KVH][HD]); grid (QH, kNch), 8 warps, a warp per key, lane = 8 dims
__global__ void __launch_bounds__(256) mtp_attn(const bf16* __restrict__ q, const bf16* __restrict__ rk,
                                                const bf16* __restrict__ rv, const bf16* __restrict__ ck,
                                                const bf16* __restrict__ cv, const int* step, int i, int W,
                                                float* part, float* ml) {
  const int h = blockIdx.x, c = blockIdx.y, kvh = h / (QH / KVH), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int P = step[1], qp = P - 1 + i;
  const int lo = qp - W + 1 > 0 ? qp - W + 1 : 0;
  const int nr = P - 1 >= lo ? P - lo : 0, tot = nr + i;
  const int per = (tot + kNch - 1) / kNch, k0 = c * per, k1 = k0 + per < tot ? k0 + per : tot;
  float qv[8], acc[8];
  for (int j = 0; j < 8; ++j) qv[j] = __bfloat162float(q[h * HD + lane * 8 + j]), acc[j] = 0.f;
  float m = -INFINITY, l = 0.f;
  for (int kk = k0 + w; kk < k1; kk += 8) {
    const bf16 *kp, *vp;
    if (kk < nr) {
      const size_t o = (size_t)((lo + kk) % kRing) * KVH * HD + kvh * HD;
      kp = rk + o, vp = rv + o;
    } else {
      const size_t o = (size_t)(kk - nr + 1) * KVH * HD + kvh * HD;
      kp = ck + o, vp = cv + o;
    }
    const uint4 kr = reinterpret_cast<const uint4*>(kp)[lane], vr = reinterpret_cast<const uint4*>(vp)[lane];
    const bf16 *ke = reinterpret_cast<const bf16*>(&kr), *ve = reinterpret_cast<const bf16*>(&vr);
    float s = 0.f;
    for (int j = 0; j < 8; ++j) s += qv[j] * __bfloat162float(ke[j]);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    s *= 0.0625f;  // 1 / sqrt(256)
    const float mn = fmaxf(m, s), a = __expf(m - mn), b = __expf(s - mn);
    l = l * a + b;
    for (int j = 0; j < 8; ++j) acc[j] = acc[j] * a + b * __bfloat162float(ve[j]);
    m = mn;
  }
  __shared__ float sm[8], sl[8], sa[8][HD];
  if (lane == 0) sm[w] = m, sl[w] = l;
  for (int j = 0; j < 8; ++j) sa[w][lane * 8 + j] = acc[j];
  __syncthreads();
  const int d = threadIdx.x;
  float M = -INFINITY;
  for (int j = 0; j < 8; ++j) M = fmaxf(M, sm[j]);
  float L = 0.f, A = 0.f;
  if (M > -INFINITY)
    for (int j = 0; j < 8; ++j)
      if (sm[j] > -INFINITY) {
        const float f = __expf(sm[j] - M);
        L += sl[j] * f, A += sa[j][d] * f;
      }
  part[((size_t)h * kNch + c) * HD + d] = A;
  if (d == 0) ml[(h * kNch + c) * 2] = M, ml[(h * kNch + c) * 2 + 1] = L;
}
__global__ void mtp_attn_merge(const float* __restrict__ part, const float* __restrict__ ml, bf16* __restrict__ out) {
  const int h = blockIdx.x, d = threadIdx.x;
  float M = -INFINITY;
  for (int c = 0; c < kNch; ++c) M = fmaxf(M, ml[(h * kNch + c) * 2]);
  float L = 0.f, A = 0.f;
  if (M > -INFINITY)
    for (int c = 0; c < kNch; ++c) {
      const float m = ml[(h * kNch + c) * 2];
      if (m == -INFINITY) continue;
      const float f = __expf(m - M);
      L += ml[(h * kNch + c) * 2 + 1] * f, A += part[((size_t)h * kNch + c) * HD + d] * f;
    }
  out[h * HD + d] = __float2bfloat16_rn(L > 0.f ? A / L : 0.f);
}
// draft slot i <- the step's argmax, unless the slot is a host proposal or there is no pending row (no context yet)
__global__ void write_slot(int* step, int i, const int* am, const int* pend_ok) {
  if (*pend_ok && i >= step[20] && i < kMaxSteps) step[3 + i] = am[0];
}

int lm_head(const m1draft::Target& t, const bf16* y, bf16* logits, cudaStream_t s) {
  if (t.lm_head_w4) {
    const int nv = NVOC;
    void* out = logits;
    return ak_wq_gemv_kt_multi(4, y, 1, 1, &t.lm_head_w4, &t.lm_head_w4s, &out, &nv, nullptr, HID, 128, 0, s);
  }
  if (t.lm_head) return ak_bf16_linear(y, t.lm_head, logits, 1, NVOC, HID, s);
  return ak_i8_gemv(y, t.lm_head_i8, t.lm_head_scale, logits, 1, NVOC, HID, NVOC, s);
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
bool ready() { return M.ready; }
size_t bytes() { return M.own; }
int max_rows() { return M.max_rows; }
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

int setup(int max_rows, int n_rings, int window) {
  M.max_rows = max_rows < 8 ? 8 : max_rows;
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
  const size_t R = M.max_rows + 1;
  M.hb = alloc<bf16>(R * HID), M.e = alloc<bf16>(R * HID), M.cat = alloc<bf16>(R * 2 * HID), M.x = alloc<bf16>(R * HID);
  M.xn = alloc<bf16>(R * HID), M.kb = alloc<bf16>(R * KVH * HD), M.vb = alloc<bf16>(R * KVH * HD);
  M.rpos = alloc<long long>(R);
  M.y = alloc<bf16>(HID), M.qg = alloc<bf16>(2 * QH * HD), M.qb = alloc<bf16>(QH * HD), M.gb = alloc<bf16>(QH * HD);
  M.ck = alloc<bf16>((size_t)kMaxSteps * KVH * HD), M.cv = alloc<bf16>((size_t)kMaxSteps * KVH * HD);
  M.ao = alloc<bf16>(QH * HD), M.gu_b = alloc<bf16>(2 * INTER), M.act = alloc<bf16>(INTER), M.t = alloc<bf16>(HID);
  M.logits = alloc<bf16>(NVOC), M.apart = alloc<float>((size_t)QH * kNch * HD), M.aml = alloc<float>(QH * kNch * 2);
  M.am = alloc<int>(1), M.cpos = alloc<long long>(1);
  if (!Wq.empty()) M.wdq_bytes = (size_t)16 << 20, M.wdq = alloc<bf16>(M.wdq_bytes / 2);
  if (!M.ring_k || !M.ring_v || !M.pend || !M.pend_ok || !M.hb || !M.cat || !M.xn || !M.kb || !M.vb || !M.logits ||
      !M.apart || (M.wdq_bytes && !M.wdq))
    return 2;
  MCK(cudaDeviceSynchronize());
  M.ready = true;
  return 0;
}

int materialize(const m1draft::Target& t, const void* final_norm, const bf16* H, int ld, int T, const int* n_dev,
                const int* ids, const long long* pos) {
  if (!M.ready) return 0;
  if (T > M.max_rows) return (int)cudaErrorInvalidValue;
  cudaStream_t s = t.s;
  const size_t rs = (size_t)kRing * KVH * HD, r0 = (size_t)M.ring * rs;
  bf16 *pend = M.pend + (size_t)M.ring * HID;
  int* pend_ok = M.pend_ok + M.ring;
  // hb[0] = pending row, hb[1..T] = the target's final norm of rows 0..T-1
  copy_row<<<1, 256, 0, s>>>(pend, M.hb);
  MCK(ak_rmsnorm(H, final_norm, M.hb + HID, T, HID, ld, HID, EPS, 1, s));
  rows_pos<<<1, 256, 0, s>>>(pos, T, M.rpos);
  // entry rows 0..T-1: hidden hb[r], token ids[r], position pos[0] - 1 + r
  MCK(ak_embed(ids, t.embed, M.e, T, HID, s));
  MCK(ak_rmsnorm(M.e, M.ne, M.cat, T, HID, HID, 2 * HID, EPS, 1, s));
  MCK(ak_rmsnorm(M.hb, M.nh, M.cat + HID, T, HID, HID, 2 * HID, EPS, 1, s));
  MCK(linear(M.cat, M.fc, M.x, T, HID, 2 * HID, s));
  MCK(ak_rmsnorm(M.x, M.in_ln, M.xn, T, HID, HID, HID, EPS, 1, s));
  MCK(linear(M.xn, M.k, M.kb, T, KVH * HD, HID, s));
  MCK(linear(M.xn, M.v, M.vb, T, KVH * HD, HID, s));
  MCK(ak_rmsnorm(M.kb, M.kn, M.kb, T * KVH, HD, HD, HD, EPS, 1, s));
  MCK(ak_rope_neox_fly(M.kb, M.kb, (const int64_t*)M.rpos, T, 0, KVH, HD, ROT, 0, KVH * HD, THETA, s));
  const int nb = (int)(((long)T * KVH * HD + 255) / 256);
  store_rows<<<nb < 1024 ? nb : 1024, 256, 0, s>>>(M.kb, M.vb, M.rpos, T, n_dev, pend_ok, M.ring_k + r0, M.ring_v + r0);
  set_pending<<<1, 256, 0, s>>>(M.hb, T, n_dev, pend, pend_ok);
  return (int)cudaGetLastError();
}

int propose(const m1draft::Target& t, int* step, int steps) {
  if (!M.ready) return 0;
  cudaStream_t s = t.s;
  const size_t rs = (size_t)kRing * KVH * HD, r0 = (size_t)M.ring * rs;
  const bf16* pend = M.pend + (size_t)M.ring * HID;
  const int* pend_ok = M.pend_ok + M.ring;
  if (steps > kMaxSteps) steps = kMaxSteps;
  for (int i = 0; i < steps; ++i) {
    const int* tok = i == 0 ? step + 2 : step + 3 + (i - 1);
    chain_pos<<<1, 1, 0, s>>>(step, i, M.cpos);
    MCK(ak_embed(tok, t.embed, M.e, 1, HID, s));
    MCK(ak_rmsnorm(M.e, M.ne, M.cat, 1, HID, HID, 2 * HID, EPS, 1, s));
    MCK(ak_rmsnorm(i == 0 ? pend : M.y, M.nh, M.cat + HID, 1, HID, HID, 2 * HID, EPS, 1, s));
    MCK(linear(M.cat, M.fc, M.x, 1, HID, 2 * HID, s));
    MCK(ak_rmsnorm(M.x, M.in_ln, M.xn, 1, HID, HID, HID, EPS, 1, s));
    bf16 *ck = M.ck + (size_t)i * KVH * HD, *cv = M.cv + (size_t)i * KVH * HD;
    MCK(linear(M.xn, M.q, M.qg, 1, 2 * QH * HD, HID, s));
    MCK(linear(M.xn, M.k, ck, 1, KVH * HD, HID, s));
    MCK(linear(M.xn, M.v, cv, 1, KVH * HD, HID, s));
    MCK(ak_rmsnorm(M.qg, M.qn, M.qb, QH, HD, 2 * HD, HD, EPS, 1, s));  // q half of each head's [q | gate]
    split_gate<<<QH, HD, 0, s>>>(M.qg, M.gb);
    MCK(ak_rmsnorm(ck, M.kn, ck, KVH, HD, HD, HD, EPS, 1, s));
    MCK(ak_rope_neox_fly(M.qb, ck, (const int64_t*)M.cpos, 1, QH, KVH, HD, ROT, QH * HD, KVH * HD, THETA, s));
    if (i == 0) store_step0<<<1, 256, 0, s>>>(ck, cv, step, pend_ok, M.ring_k + r0, M.ring_v + r0);
    mtp_attn<<<dim3(QH, kNch), 256, 0, s>>>(M.qb, M.ring_k + r0, M.ring_v + r0, M.ck, M.cv, step, i, M.window, M.apart,
                                           M.aml);
    mtp_attn_merge<<<QH, HD, 0, s>>>(M.apart, M.aml, M.ao);
    MCK(ak_bf16_binary(M.ao, M.gb, M.ao, QH * HD, 2, s));  // * sigmoid(gate)
    MCK(linear(M.ao, M.o, M.t, 1, HID, QH * HD, s));
    MCK(ak_bf16_binary(M.x, M.t, M.x, HID, 0, s));
    MCK(ak_rmsnorm(M.x, M.post_ln, M.xn, 1, HID, HID, HID, EPS, 1, s));
    MCK(linear(M.xn, M.gu, M.gu_b, 1, 2 * INTER, HID, s));
    MCK(ak_silu_mul(M.gu_b, M.act, 1, INTER, s));
    MCK(linear(M.act, M.down, M.t, 1, HID, INTER, s));
    MCK(ak_bf16_binary(M.x, M.t, M.x, HID, 0, s));
    MCK(ak_rmsnorm(M.x, M.norm, M.y, 1, HID, HID, HID, EPS, 1, s));
    MCK(lm_head(t, M.y, M.logits, s));
    MCK(ak_argmax(M.logits, 1, NVOC, t.vocab_valid, M.am, s));
    write_slot<<<1, 1, 0, s>>>(step, i, M.am, pend_ok);
  }
  return (int)cudaGetLastError();
}
}  // namespace m1mtp
