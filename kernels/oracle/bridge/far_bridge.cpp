// M1 PARITY BRIDGE (oracle tree; replaced by engine kernels in M2): the BCSA far branch and block-key writes of
// sglv23, transcribed op for op from refs (volundr.py VolundrAttention._far_sparse / _far_extend / _merge /
// _write_block_keys) into ATen, so torch's own einsum (cuBLAS), topk, mean and fp32 reductions run — bit-identical
// by construction with the same libtorch build. sglv23's fused Triton versions of the decode parts are bit-exact
// with these references (sglv23 changelog), so this also matches its default (VOLUNDR_FUSED_BCSA=1) path.
// Inputs are the engine's raw device buffers wrapped with from_blob; everything runs on the caller's stream.
#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {
int cur_device() {
  int d = 0;
  cudaGetDevice(&d);
  return d;
}
constexpr int C = 4, W = 4096, KSEL_MAX = 512, D = 256, G = 6, DI = 128, IDXH = 4;
int H = 24, KV = 4;  // this process's q / kv heads (tensor parallel rank share: ak_bridge_set_heads)
const double kScaling = 1.0 / 16.0;  // head_dim ** -0.5
const double kIdxScale = 0.08838834764831845;  // idx_dim ** -0.5 (python float)

// Query rows per far-extend step: about far_elems() / (4 H nb) (fp32 score elements of one step; sglv23: 2e8). A
// smaller budget (AGENS_FAR_EXTEND_ELEMS, e.g. 2.5e7 on a 16 GB card) bounds the temporaries torch's caching allocator
// keeps after a long prefill; per-row math is unchanged, only the einsum batch shapes differ.
int64_t far_elems() {
  static const int64_t e = [] {
    const char* v = getenv("AGENS_FAR_EXTEND_ELEMS");
    return v ? (int64_t)atof(v) : (int64_t)2.0e8;
  }();
  return e;
}
// Floor of the query rows per step (AGENS_FAR_EXTEND_MIN_ROWS, default 64). At long context the floor, not the element
// budget, sets the step (nb = 60K blocks: 64 rows x 4 H nb = 370M fp32 elements, ~1.5 GB of torch cache after a 240K
// prefill); a 16 GB card lowers it (budget mode: 8). Per-row math as above.
int64_t far_min_rows() {
  static const int64_t r = [] {
    const char* v = getenv("AGENS_FAR_EXTEND_MIN_ROWS");
    return v ? std::max<int64_t>(1, atoll(v)) : (int64_t)64;
  }();
  return r;
}
at::TensorOptions opt(at::ScalarType t) { return at::TensorOptions().device(at::kCUDA).dtype(t); }
at::Tensor blob(void* p, at::IntArrayRef shape, at::ScalarType t) { return at::from_blob(p, shape, opt(t)); }

struct Pools {
  at::Tensor r2t;   // [R, Wctx] int32
  at::Tensor kc;    // [rows + 1, KV, D] bf16 (row `rows` = trash / padding, like sglang's pool)
  at::Tensor vc;
  at::Tensor idx;   // [rows + 1, DI] bf16 (hi = rows)
  at::Tensor blk;
};

at::Tensor r2t_gather(const at::Tensor& r2t, const at::Tensor& req, const at::Tensor& cols) {
  auto flat = r2t.view(-1);
  return flat.index({req.unsqueeze(1) * r2t.size(1) + cols}).to(at::kLong);
}

// VolundrAttention._merge. sf [t,H,n] fp32, lse [t,H] fp32 (natural log), o_loc [t,H,D] bf16,
// vb [n,KV,D] (kv_first) or [t,n,KV,D]. Returns [t,H,D] fp32.
at::Tensor merge(const at::Tensor& sf, const at::Tensor& lse, const at::Tensor& o_loc, const at::Tensor& vb,
                 bool kv_first) {
  const int64_t t = sf.size(0), n = sf.size(2);
  auto m = at::maximum(lse, sf.amax(-1));
  auto w = at::exp(lse - m);
  auto pf = at::exp(sf - m.unsqueeze(-1));
  auto z = w + pf.sum(-1);
  pf = (pf / z.unsqueeze(-1)).to(o_loc.scalar_type()).view({t, KV, G, n});
  at::Tensor o_far = kv_first ? at::einsum("tkgn,nkd->tkgd", {pf, vb}) : at::einsum("tkgn,tnkd->tkgd", {pf, vb});
  return (w / z).unsqueeze(-1) * o_loc.to(at::kFloat) + o_far.reshape({t, H, D}).to(at::kFloat);
}
}  // namespace

extern "C" {

// Bytes torch's caching allocator holds on the current device (the ATen far branch's temporaries are kept for reuse);
// release = 1 first returns the unused blocks to the driver.
long long ak_bridge_torch_reserved(int release) {
  try {
    if (release) c10::cuda::CUDACachingAllocator::emptyCache();
    const auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(cur_device());
    return (long long)st.reserved_bytes[0].current;
  } catch (const std::exception&) {
    return -1;
  }
}

// Decode far branch (one token per request). qi [bs,4,128] bf16 = idx_norm(idx_q(h)); qh [bs,24,256] bf16 (post-RoPE
// q); o_loc [bs,24,256] bf16; lse [bs,24] fp32 natural-log (= flashinfer lse * ln2); pos/reqs [bs] int64;
// r2t [R][Wctx] int32; kc/vc [rows+1][4][256] bf16; blk [rows/4+1][128] bf16: one block key per aligned 4-row page
// (the engine allocates KV rows in pages; a block's key sits at page row/4 of its last token; trash page rows/4).
// out [bs][24*256] bf16.
int ak_bridge_far_decode(void* qi, void* qh, void* o_loc, void* lse, void* pos, void* reqs, void* r2t, int R, int Wctx,
                         void* kc, void* vc, int rows, void* blk, int bs, void* out, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    const int64_t NB = Wctx / C;
    const int64_t hi = rows;  // idx/blk rows - 1
    auto tr2t = blob(r2t, {R, Wctx}, at::kInt);
    auto tpos = blob(pos, {bs}, at::kLong), treq = blob(reqs, {bs}, at::kLong);
    auto tblk = blob(blk, {rows / C + 1, DI}, at::kBFloat16);
    auto tkc = blob(kc, {rows + 1, KV, D}, at::kBFloat16), tvc = blob(vc, {rows + 1, KV, D}, at::kBFloat16);
    auto tqi = blob(qi, {bs, IDXH, DI}, at::kBFloat16);
    auto tqh = blob(qh, {bs, H, D}, at::kBFloat16);
    auto tol = blob(o_loc, {bs, H, D}, at::kBFloat16);
    auto tlse = blob(lse, {bs, H}, at::kFloat);
    auto n = at::arange(NB, opt(at::kLong));
    auto nmax = (tpos - W - (C - 1)).div(C, "floor");
    auto valid = n.unsqueeze(0) <= nmax.unsqueeze(1);
    auto last = r2t_gather(tr2t, treq, (n * C + (C - 1)).unsqueeze(0).expand({bs, NB})).clamp_(0, hi);
    auto kbi = tblk.index({last.div(C, "floor")});  // block keys are stored per 4-row page
    auto s_idx = at::einsum("bhd,bnd->bhn", {tqi, kbi}).mean(1) * kIdxScale;
    s_idx = s_idx.masked_fill(~valid, -INFINITY);
    const int64_t ksel = std::min<int64_t>(KSEL_MAX, NB);
    auto top = std::get<1>(s_idx.topk(ksel, -1));
    auto sel_ok = valid.gather(1, top);
    auto cols = (top.unsqueeze(2) * C + at::arange(C, opt(at::kLong))).view({bs, ksel * C});
    auto loc = r2t_gather(tr2t, treq, cols.clamp_(0, Wctx - 1)).clamp_(0, hi);
    auto kb = tkc.index({loc}).view({bs, ksel, C, KV, D}).mean(2);
    auto vb = tvc.index({loc}).view({bs, ksel, C, KV, D}).mean(2);
    auto sf = at::einsum("bkgd,bnkd->bkgn", {tqh.view({bs, KV, G, D}), kb}) * kScaling;
    sf = sf.reshape({bs, H, ksel}).to(at::kFloat).masked_fill(~sel_ok.unsqueeze(1), -INFINITY);
    auto o = merge(sf, tlse, tol, vb, false);
    blob(out, {bs, H * D}, at::kBFloat16).copy_(o.reshape({bs, H * D}).to(at::kBFloat16));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_far_decode: %s\n", e.what());
    return 1;
  }
}

// Decode block-key write (sglv23 _write_block_keys, decode branch). pos/reqs/out_loc [bs] int64.
int ak_bridge_block_keys_decode(void* pos, void* reqs, void* out_loc, void* r2t, int R, int Wctx, void* idx, void* blk,
                                int rows, int bs, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    const int64_t hi = rows;
    auto tr2t = blob(r2t, {R, Wctx}, at::kInt);
    auto tpos = blob(pos, {bs}, at::kLong), treq = blob(reqs, {bs}, at::kLong), tloc = blob(out_loc, {bs}, at::kLong);
    auto tidx = blob(idx, {rows + 1, DI}, at::kBFloat16), tblk = blob(blk, {rows / C + 1, DI}, at::kBFloat16);
    auto cols = (tpos.div(C, "floor") * C).unsqueeze(1) + at::arange(C, opt(at::kLong));
    cols.clamp_(0, Wctx - 1);
    auto slots = r2t_gather(tr2t, treq, cols).clamp_(0, hi);
    auto pooled = tidx.index({slots}).view({bs, C, DI}).mean(1);
    auto closes = tpos.remainder(C) == (C - 1);
    auto dst = at::where(closes, tloc, at::full_like(tpos, hi)).div(C, "floor");
    tblk.index_copy_(0, dst, pooled.to(tblk.scalar_type()));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_block_keys_decode: %s\n", e.what());
    return 1;
  }
}

// Extend (prefill chunk) of ONE request: block keys for blocks whose last position is in this chunk, then the far
// branch for the chunk's T queries. seq_len = tokens incl. this chunk; T = chunk length; req = r2t row.
// qi_all [T,4,128] bf16; qh/o_loc [T,24,256] bf16; lse [T,24] fp32 (natural log). out [T][24*256] bf16.
int ak_bridge_far_extend(void* qi_all, void* qh, void* o_loc, void* lse, int seq_len, int T, int req, void* r2t, int R,
                         int Wctx, void* kc, void* vc, void* idx, void* blk, int rows, void* out, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    const int64_t hi = rows;
    auto tr2t = blob(r2t, {R, Wctx}, at::kInt);
    auto tidx = blob(idx, {rows + 1, DI}, at::kBFloat16), tblk = blob(blk, {rows / C + 1, DI}, at::kBFloat16);
    auto tkc = blob(kc, {rows + 1, KV, D}, at::kBFloat16), tvc = blob(vc, {rows + 1, KV, D}, at::kBFloat16);
    auto row = tr2t[req];
    // ---- _write_block_keys, extend branch
    {
      const int64_t n_lo = (int64_t)(seq_len - T) / C, n_hi = (int64_t)seq_len / C;
      if (n_hi > n_lo) {
        auto slots = row.slice(0, n_lo * C, n_hi * C).to(at::kLong).clamp_(0, hi);
        auto pooled = tidx.index({slots}).view({n_hi - n_lo, C, DI}).mean(1);
        auto dst = row.slice(0, n_lo * C + C - 1, n_hi * C, C).to(at::kLong).clamp_(0, hi).div(C, "floor");
        tblk.index_copy_(0, dst, pooled.to(tblk.scalar_type()));
      }
    }
    // ---- _far_extend
    auto tqi = blob(qi_all, {T, IDXH, DI}, at::kBFloat16);
    auto tqh = blob(qh, {T, H, D}, at::kBFloat16);
    auto tol = blob(o_loc, {T, H, D}, at::kBFloat16);
    auto tlse = blob(lse, {T, H}, at::kFloat);
    auto o = tol.to(at::kFloat);
    const int64_t nb = ((int64_t)seq_len - 1 - W - (C - 1)) >= 0 ? ((int64_t)seq_len - 1 - W - (C - 1)) / C + 1
                                                               : (((int64_t)seq_len - 1 - W - (C - 1)) - (C - 1)) / C + 1;
    if (nb > 0) {
      auto tk = row.slice(0, 0, C * nb).to(at::kLong).clamp_(0, hi);
      auto kbi = tblk.index({row.slice(0, C - 1, C * nb, C).to(at::kLong).clamp_(0, hi).div(C, "floor")});
      auto kb = tkc.index({tk}).view({nb, C, KV, D}).mean(1);
      auto vb = tvc.index({tk}).view({nb, C, KV, D}).mean(1);
      auto n = at::arange(nb, opt(at::kLong));
      const int64_t ksel = std::min<int64_t>(KSEL_MAX, nb);
      const int64_t step = std::max<int64_t>(far_min_rows(), std::min<int64_t>(T, far_elems() / std::max<int64_t>(1, 4 * H * nb)));
      for (int64_t a = 0; a < T; a += step) {
        const int64_t b = std::min<int64_t>(T, a + step), t = b - a;
        auto qpos = at::arange(seq_len - T + a, seq_len - T + b, opt(at::kLong));
        auto valid = ((n.unsqueeze(0) + 1) * C - 1) <= (qpos.unsqueeze(1) - W);
        auto s_idx = at::einsum("thd,nd->thn", {tqi.slice(0, a, b), kbi}).mean(1) * kIdxScale;
        s_idx = s_idx.masked_fill(~valid, -INFINITY);
        auto top = std::get<1>(s_idx.topk(ksel, -1));
        auto sel = at::zeros_like(valid).scatter_(-1, top, true) & valid;
        auto sf = at::einsum("tkgd,nkd->tkgn", {tqh.slice(0, a, b).view({t, KV, G, D}), kb}) * kScaling;
        sf = sf.reshape({t, H, nb}).to(at::kFloat).masked_fill(~sel.unsqueeze(1), -INFINITY);
        o.slice(0, a, b).copy_(merge(sf, tlse.slice(0, a, b), tol.slice(0, a, b), vb, true));
      }
    }
    blob(out, {T, H * D}, at::kBFloat16).copy_(o.reshape({T, H * D}).to(at::kBFloat16));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_far_extend: %s\n", e.what());
    return 1;
  }
}

// ---- KV tiering (docs/kv_tiering.md): ak_bridge_far_decode split at the point where the selected rows are known, so
// the engine can fetch them (host pool -> staging, or INT8 dequant) between the two halves. Same ops as above on the
// same values: select + gather + attend is bit-identical to ak_bridge_far_decode on the pool.
// select: sel_rows [bs][ksel*C] int32 = the pool rows ak_bridge_far_decode indexes (loc), sel_ok [bs][ksel] bool.
// ksel = min(512, Wctx/4).
int ak_bridge_far_select_decode(void* qi, void* pos, void* reqs, void* r2t, int R, int Wctx, int rows, void* blk, int bs,
                                void* sel_rows, void* sel_ok, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    const int64_t NB = Wctx / C;
    const int64_t hi = rows;
    auto tr2t = blob(r2t, {R, Wctx}, at::kInt);
    auto tpos = blob(pos, {bs}, at::kLong), treq = blob(reqs, {bs}, at::kLong);
    auto tblk = blob(blk, {rows / C + 1, DI}, at::kBFloat16);
    auto tqi = blob(qi, {bs, IDXH, DI}, at::kBFloat16);
    auto n = at::arange(NB, opt(at::kLong));
    auto nmax = (tpos - W - (C - 1)).div(C, "floor");
    auto valid = n.unsqueeze(0) <= nmax.unsqueeze(1);
    auto last = r2t_gather(tr2t, treq, (n * C + (C - 1)).unsqueeze(0).expand({bs, NB})).clamp_(0, hi);
    auto kbi = tblk.index({last.div(C, "floor")});  // block keys are stored per 4-row page
    auto s_idx = at::einsum("bhd,bnd->bhn", {tqi, kbi}).mean(1) * kIdxScale;
    s_idx = s_idx.masked_fill(~valid, -INFINITY);
    const int64_t ksel = std::min<int64_t>(KSEL_MAX, NB);
    auto top = std::get<1>(s_idx.topk(ksel, -1));
    blob(sel_ok, {bs, ksel}, at::kBool).copy_(valid.gather(1, top));
    auto cols = (top.unsqueeze(2) * C + at::arange(C, opt(at::kLong))).view({bs, ksel * C});
    auto loc = r2t_gather(tr2t, treq, cols.clamp_(0, Wctx - 1)).clamp_(0, hi);
    blob(sel_rows, {bs, ksel * C}, at::kInt).copy_(loc);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_far_select_decode: %s\n", e.what());
    return 1;
  }
}

// attend: kst/vst [bs][ksel*C][4][256] bf16 = the selected rows in sel_rows order (what tkc.index({loc}) returns),
// or with means != 0 [bs][ksel][4][256] = their 4-row block means (ak_kv_block_mean == the .mean(2) below, bitwise).
int ak_bridge_far_attend_decode(void* qh, void* o_loc, void* lse, void* kst, void* vst, void* sel_ok, int bs, int ksel,
                                int means, void* out, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    auto tqh = blob(qh, {bs, H, D}, at::kBFloat16);
    auto tol = blob(o_loc, {bs, H, D}, at::kBFloat16);
    auto tlse = blob(lse, {bs, H}, at::kFloat);
    auto sel_ok_t = blob(sel_ok, {bs, ksel}, at::kBool);
    auto kb = means ? blob(kst, {bs, ksel, KV, D}, at::kBFloat16)
                    : blob(kst, {bs, ksel * C, KV, D}, at::kBFloat16).view({bs, ksel, C, KV, D}).mean(2);
    auto vb = means ? blob(vst, {bs, ksel, KV, D}, at::kBFloat16)
                    : blob(vst, {bs, ksel * C, KV, D}, at::kBFloat16).view({bs, ksel, C, KV, D}).mean(2);
    auto sf = at::einsum("bkgd,bnkd->bkgn", {tqh.view({bs, KV, G, D}), kb}) * kScaling;
    sf = sf.reshape({bs, H, ksel}).to(at::kFloat).masked_fill(~sel_ok_t.unsqueeze(1), -INFINITY);
    auto o = merge(sf, tlse, tol, vb, false);
    blob(out, {bs, H * D}, at::kBFloat16).copy_(o.reshape({bs, H * D}).to(at::kBFloat16));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_far_attend_decode: %s\n", e.what());
    return 1;
  }
}

// ak_bridge_far_extend with the far K/V rows supplied in position order: kst/vst [n_rows][4][256] bf16 = rows of
// positions 0 .. 4*nb-1 (what tkc.index({tk}) returns), n_rows >= 4*nb; with means != 0 the nb block means instead
// (n_rows >= nb). Block keys still come from idx/blk via r2t.
int ak_bridge_far_extend_staged(void* qi_all, void* qh, void* o_loc, void* lse, int seq_len, int T, int req, void* r2t,
                                int R, int Wctx, void* kst, void* vst, int n_rows, int means, void* idx, void* blk,
                                int rows, void* out, void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    const int64_t hi = rows;
    auto tr2t = blob(r2t, {R, Wctx}, at::kInt);
    auto tidx = blob(idx, {rows + 1, DI}, at::kBFloat16), tblk = blob(blk, {rows / C + 1, DI}, at::kBFloat16);
    auto row = tr2t[req];
    {
      const int64_t n_lo = (int64_t)(seq_len - T) / C, n_hi = (int64_t)seq_len / C;
      if (n_hi > n_lo) {
        auto slots = row.slice(0, n_lo * C, n_hi * C).to(at::kLong).clamp_(0, hi);
        auto pooled = tidx.index({slots}).view({n_hi - n_lo, C, DI}).mean(1);
        auto dst = row.slice(0, n_lo * C + C - 1, n_hi * C, C).to(at::kLong).clamp_(0, hi).div(C, "floor");
        tblk.index_copy_(0, dst, pooled.to(tblk.scalar_type()));
      }
    }
    auto tqi = blob(qi_all, {T, IDXH, DI}, at::kBFloat16);
    auto tqh = blob(qh, {T, H, D}, at::kBFloat16);
    auto tol = blob(o_loc, {T, H, D}, at::kBFloat16);
    auto tlse = blob(lse, {T, H}, at::kFloat);
    auto o = tol.to(at::kFloat);
    const int64_t nb = ((int64_t)seq_len - 1 - W - (C - 1)) >= 0 ? ((int64_t)seq_len - 1 - W - (C - 1)) / C + 1
                                                               : (((int64_t)seq_len - 1 - W - (C - 1)) - (C - 1)) / C + 1;
    if (nb > 0) {
      if (n_rows < (means ? nb : C * nb)) throw std::runtime_error("too few staged rows");
      auto kbi = tblk.index({row.slice(0, C - 1, C * nb, C).to(at::kLong).clamp_(0, hi).div(C, "floor")});
      auto kb = means ? blob(kst, {nb, KV, D}, at::kBFloat16)
                      : blob(kst, {C * nb, KV, D}, at::kBFloat16).view({nb, C, KV, D}).mean(1);
      auto vb = means ? blob(vst, {nb, KV, D}, at::kBFloat16)
                      : blob(vst, {C * nb, KV, D}, at::kBFloat16).view({nb, C, KV, D}).mean(1);
      auto n = at::arange(nb, opt(at::kLong));
      const int64_t ksel = std::min<int64_t>(KSEL_MAX, nb);
      const int64_t step = std::max<int64_t>(far_min_rows(), std::min<int64_t>(T, far_elems() / std::max<int64_t>(1, 4 * H * nb)));
      for (int64_t a = 0; a < T; a += step) {
        const int64_t b = std::min<int64_t>(T, a + step), t = b - a;
        auto qpos = at::arange(seq_len - T + a, seq_len - T + b, opt(at::kLong));
        auto valid = ((n.unsqueeze(0) + 1) * C - 1) <= (qpos.unsqueeze(1) - W);
        auto s_idx = at::einsum("thd,nd->thn", {tqi.slice(0, a, b), kbi}).mean(1) * kIdxScale;
        s_idx = s_idx.masked_fill(~valid, -INFINITY);
        auto top = std::get<1>(s_idx.topk(ksel, -1));
        auto sel = at::zeros_like(valid).scatter_(-1, top, true) & valid;
        auto sf = at::einsum("tkgd,nkd->tkgn", {tqh.slice(0, a, b).view({t, KV, G, D}), kb}) * kScaling;
        sf = sf.reshape({t, H, nb}).to(at::kFloat).masked_fill(~sel.unsqueeze(1), -INFINITY);
        o.slice(0, a, b).copy_(merge(sf, tlse.slice(0, a, b), tol.slice(0, a, b), vb, true));
      }
    }
    blob(out, {T, H * D}, at::kBFloat16).copy_(o.reshape({T, H * D}).to(at::kBFloat16));
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_far_extend_staged: %s\n", e.what());
    return 1;
  }
}
}  // extern "C"
// torch's caching allocator takes `bytes` now and keeps them cached (budget mode: the plan's ATen far-branch
// temporaries, reserved before CUDA graphs can take the room). 0 or 1 (out of memory).
extern "C" int ak_bridge_torch_reserve(long long bytes) {
  try {
    auto t = at::empty({bytes}, at::TensorOptions().device(at::kCUDA, cur_device()).dtype(at::kByte));
    (void)t;
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_bridge_torch_reserve: %s\n", e.what());
    return 1;
  }
}
extern "C" int ak_bridge_set_heads(int qh, int kvh) {
  if (kvh < 1 || qh != G * kvh) return 1;
  H = qh, KV = kvh;
  return 0;
}

