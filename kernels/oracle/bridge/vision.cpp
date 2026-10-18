// Vision tower of Agens Volundr v1 (model.visual.*: a ViT, 27 blocks x 1152, merger to 5120) and the image
// preprocessing in front of it, through ATen on the GPU. The ops follow what the
// reference runs: sglang's Qwen3VLMoeVisionModel (pos-embed interpolation `fast_pos_embed_interpolate_from_list`, 2-D
// rotary from an fp32 cos|sin cache applied in fp32, pre-LN blocks, merger) and transformers' Qwen2VLImageProcessor
// (torchvision backend on the GPU: bicubic antialiased resize in fp32, clamp, round, uint8; (x - 127.5) / 127.5;
// merge-block patch order). Attention is SDPA per image (the reference uses its Triton varlen kernel on sm89: same math,
// other rounding).
// Weights stay in pinned host memory (registered by the loader); each block's weights are copied to the GPU just before
// the block runs (prefetched one block ahead on a copy stream) and dropped after, so the tower needs no resident GPU
// memory (AGENS_VISION_RESIDENT=1 keeps a device copy instead). Everything runs on the caller's stream.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr int64_t HIDV = 1152, NHV = 16, HDV = 72, DEPTH = 27, PATCH = 16, TPATCH = 2, MERGE = 2, SIDE = 48,
                  OUTD = 5120, PDIM = 3 * TPATCH * PATCH * PATCH;  // 1536 values per patch
constexpr int64_t ROWS_TILE = 1024;  // fp32 rotary / MLP temporaries in row tiles of this many patches

int cur_device() {
  int d = 0;
  cudaGetDevice(&d);
  return d;
}
at::TensorOptions dopt(at::ScalarType t) { return at::TensorOptions().device(at::kCUDA, cur_device()).dtype(t); }

std::map<std::string, at::Tensor> g_host;  // name without "model.visual." -> pinned host bf16 tensor
std::map<std::string, at::Tensor> g_res;   // resident device copies (AGENS_VISION_RESIDENT=1)
bool resident() {
  static const bool r = getenv("AGENS_VISION_RESIDENT") && getenv("AGENS_VISION_RESIDENT")[0] == '1';
  return r;
}
const at::Tensor& host_w(const std::string& n) {
  auto it = g_host.find(n);
  if (it == g_host.end()) throw std::runtime_error("vision weight missing: " + n);
  return it->second;
}

// One layer's weights on the GPU: fetched on a copy stream, ready once `ev` is reached.
struct Fetched {
  std::map<std::string, at::Tensor> t;
  at::cuda::CUDAEvent ev;
};
const char* kBlockNames[] = {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                             "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight",
                             "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"};
// the merger in two parts (fc1 and fc2 are 42 + 47 MB): fetched after the last block, one part on the GPU at a time
const char* kMerger1[] = {"merger.norm.weight", "merger.norm.bias", "merger.linear_fc1.weight", "merger.linear_fc1.bias"};
const char* kMerger2[] = {"merger.linear_fc2.weight", "merger.linear_fc2.bias"};

void fetch(Fetched& f, const std::string& prefix, const char* const* names, int n, c10::cuda::CUDAStream main,
           c10::cuda::CUDAStream copy) {
  f.t.clear();
  if (resident()) {
    for (int i = 0; i < n; ++i) {
      const std::string nm = prefix + names[i];
      auto it = g_res.find(nm);
      if (it == g_res.end()) it = g_res.emplace(nm, host_w(nm).to(dopt(at::kBFloat16))).first;
      f.t[names[i]] = it->second;
    }
    f.ev.record(main);
    return;
  }
  // allocate on the main stream (freed there after use), copy on the copy stream after the main stream's prior work
  for (int i = 0; i < n; ++i) f.t[names[i]] = at::empty(host_w(prefix + names[i]).sizes(), dopt(at::kBFloat16));
  at::cuda::CUDAEvent ready;
  ready.record(main);
  ready.block(copy);
  {
    c10::cuda::CUDAStreamGuard g(copy);
    for (int i = 0; i < n; ++i) f.t[names[i]].copy_(host_w(prefix + names[i]), /*non_blocking=*/true);
  }
  f.ev.record(copy);
}

// q / k [N, 16, 72] bf16 in place: x * cos + rotate_half(x) * sin in fp32 (sglang apply_rotary_pos_emb_native),
// cos / sin [N, 72] fp32 (= cat(c, c) of the [N, 36] gather)
void rope_(at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
  const int64_t N = x.size(0);
  for (int64_t r0 = 0; r0 < N; r0 += ROWS_TILE) {
    const int64_t r1 = std::min(N, r0 + ROWS_TILE);
    auto xs = x.slice(0, r0, r1);
    auto xf = xs.to(at::kFloat);
    auto c = cos.slice(0, r0, r1).unsqueeze(1), s = sin.slice(0, r0, r1).unsqueeze(1);
    auto x1 = xf.slice(-1, 0, HDV / 2), x2 = xf.slice(-1, HDV / 2, HDV);
    auto rot = at::cat({-x2, x1}, -1);
    xs.copy_((xf * c + rot * s).to(at::kBFloat16));
  }
}

// [N, 1536] pixel values (fp32, or already bf16 = the reference's .to(bf16) of them) -> [N / 4, 5120] bf16 image
// embeddings (grid gh x gw patches, t = 1)
at::Tensor tower(at::Tensor pv, int64_t gh, int64_t gw) {
  const int64_t N = gh * gw;
  // no cuDNN anywhere in the tower: SDPA would otherwise probe cuDNN (cudnnGetVersion), whose dlopen'ed sub-libraries
  // the slim image does not carry (its library set is the engine's ldd closure); flash / memory-efficient SDPA is used
  at::globalContext().setSDPUseCuDNN(false);
  at::globalContext().setUserEnabledCuDNN(false);
  auto main = c10::cuda::getCurrentCUDAStream();
  auto copy = c10::cuda::getStreamFromPool(false, cur_device());
  Fetched cur, nxt;
  const char* pe_names[] = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight"};
  fetch(cur, "", pe_names, 3, main, copy);
  fetch(nxt, "blocks.0.", kBlockNames, 12, main, copy);
  cur.ev.block(main);
  // patch_embed: Conv3d with kernel == stride == one linear over the 1536 values of a patch (sglang Conv3dLayer)
  auto x = at::linear(pv.to(at::kBFloat16).view({N, PDIM}), cur.t["patch_embed.proj.weight"].view({HIDV, PDIM}),
                      cur.t["patch_embed.proj.bias"]);
  pv.reset();
  {  // pos_embed: bilinear interpolation of the 48 x 48 table (align corners), weights in bf16, merge-block order
    auto hi = at::linspace(0, SIDE - 1, gh, dopt(at::kFloat)), wi = at::linspace(0, SIDE - 1, gw, dopt(at::kFloat));
    auto hf = hi.to(at::kLong), wf = wi.to(at::kLong);
    auto hc = at::clamp_max(hf + 1, SIDE - 1), wc = at::clamp_max(wf + 1, SIDE - 1);
    auto dh = hi - hf, dw = wi - wf;
    auto gdh = at::meshgrid({dh, dw}, "ij"), gf = at::meshgrid({hf, wf}, "ij"), gc = at::meshgrid({hc, wc}, "ij");
    auto w11 = gdh[0] * gdh[1];
    auto w10 = gdh[0] - w11;
    auto w01 = gdh[1] - w11;
    auto w00 = 1 - gdh[0] - w01;
    auto hg = at::stack({gf[0], gf[0], gc[0], gc[0]}), wg = at::stack({gf[1], gc[1], gf[1], gc[1]});
    auto idx = (hg * SIDE + wg).reshape({4, -1});
    auto wts = at::stack({w00, w01, w10, w11}, 0).reshape({4, -1, 1}).to(at::kBFloat16);
    auto emb = at::embedding(cur.t["pos_embed.weight"], idx);
    emb.mul_(wts);
    auto comb = emb.sum(0).reshape({gh / MERGE, MERGE, gw / MERGE, MERGE, HIDV}).permute({0, 2, 1, 3, 4}).reshape({N, HIDV});
    x.add_(comb);
  }
  at::Tensor cosf, sinf;
  {  // 2-D rotary: patch (row, col) in merge-block order, fp32 cache 1 / 10000^(2j/36) (sglang get_rope, rotary 36)
    const int64_t mx = std::max(gh, gw);
    auto inv = 1.0 / at::pow(10000.0, at::arange(0, HDV / 2, 2, dopt(at::kFloat)) / (double)(HDV / 2));
    auto t = at::arange(mx, dopt(at::kFloat));
    auto fr = at::einsum("i,j->ij", {t, inv});
    auto cs = fr.cos(), sn = fr.sin();  // [mx, 18]
    auto hp = at::arange(gh, dopt(at::kLong)).unsqueeze(1).expand({gh, gw})
                  .reshape({gh / MERGE, MERGE, gw / MERGE, MERGE}).permute({0, 2, 1, 3}).flatten();
    auto wp = at::arange(gw, dopt(at::kLong)).unsqueeze(0).expand({gh, gw})
                  .reshape({gh / MERGE, MERGE, gw / MERGE, MERGE}).permute({0, 2, 1, 3}).flatten();
    auto pid = at::stack({hp, wp}, -1);  // [N, 2]
    auto c = cs.index({pid}).flatten(1), s = sn.index({pid}).flatten(1);  // [N, 36]
    cosf = at::cat({c, c}, -1), sinf = at::cat({s, s}, -1);
  }
  for (int64_t b = 0; b < DEPTH; ++b) {
    std::swap(cur, nxt);
    if (b + 1 < DEPTH) fetch(nxt, "blocks." + std::to_string(b + 1) + ".", kBlockNames, 12, main, copy);
    else nxt.t.clear();
    cur.ev.block(main);
    auto& w = cur.t;
    {
      auto h = at::layer_norm(x, {HIDV}, w["norm1.weight"], w["norm1.bias"], 1e-6);
      const auto& qw = w["attn.qkv.weight"];
      const auto& qb = w["attn.qkv.bias"];
      auto q = at::linear(h, qw.narrow(0, 0, HIDV), qb.narrow(0, 0, HIDV)).view({N, NHV, HDV});
      auto k = at::linear(h, qw.narrow(0, HIDV, HIDV), qb.narrow(0, HIDV, HIDV)).view({N, NHV, HDV});
      auto v = at::linear(h, qw.narrow(0, 2 * HIDV, HIDV), qb.narrow(0, 2 * HIDV, HIDV)).view({N, NHV, HDV});
      h.reset();
      rope_(q, cosf, sinf);
      rope_(k, cosf, sinf);
      // flash attention directly (no SDPA backend probe, which loads cuDNN)
      auto o = std::get<0>(at::_scaled_dot_product_flash_attention(
          q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), 0.0, false,
          false, c10::nullopt));
      q.reset(), k.reset(), v.reset();
      o = o.squeeze(0).transpose(0, 1).reshape({N, HIDV});
      x.add_(at::linear(o, w["attn.proj.weight"], w["attn.proj.bias"]));
    }
    {
      auto h = at::layer_norm(x, {HIDV}, w["norm2.weight"], w["norm2.bias"], 1e-6);
      for (int64_t r0 = 0; r0 < N; r0 += ROWS_TILE) {
        const int64_t r1 = std::min(N, r0 + ROWS_TILE);
        auto a = at::gelu(at::linear(h.slice(0, r0, r1), w["mlp.linear_fc1.weight"], w["mlp.linear_fc1.bias"]), "tanh");
        x.slice(0, r0, r1).add_(at::linear(a, w["mlp.linear_fc2.weight"], w["mlp.linear_fc2.bias"]));
      }
    }
  }
  cur.t.clear();  // the last block's weights (freed on the main stream, after its kernels)
  fetch(nxt, "", kMerger1, 4, main, copy);
  nxt.ev.block(main);
  auto h = at::layer_norm(x, {HIDV}, nxt.t["merger.norm.weight"], nxt.t["merger.norm.bias"], 1e-6).view({N / 4, 4 * HIDV});
  x.reset();
  auto a = at::gelu(at::linear(h, nxt.t["merger.linear_fc1.weight"], nxt.t["merger.linear_fc1.bias"]));
  h.reset();
  nxt.t.clear();
  fetch(cur, "", kMerger2, 2, main, copy);
  cur.ev.block(main);
  return at::linear(a, cur.t["merger.linear_fc2.weight"], cur.t["merger.linear_fc2.bias"]);
}

// JPEG on the GPU with nvJPEG, as the reference loads a JPEG data URL (torchvision.io.decode_jpeg(device="cuda"),
// RGB output): [1, 3, h, w] uint8, or an undefined tensor when nvJPEG cannot decode it or its size is not h x w
// (the caller then takes the CPU-decoded pixels, as the reference falls back to PIL).
at::Tensor decode_jpeg_gpu(const uint8_t* data, size_t len, int h, int w) {
  static nvjpegHandle_t hd = nullptr;
  static nvjpegJpegState_t st = nullptr;
  static bool failed = false;
  if (failed) return {};
  if (!hd) {
    if (nvjpegCreateSimple(&hd) != NVJPEG_STATUS_SUCCESS || nvjpegJpegStateCreate(hd, &st) != NVJPEG_STATUS_SUCCESS) {
      failed = true;
      return {};
    }
  }
  int nc = 0, ws[NVJPEG_MAX_COMPONENT] = {}, hs[NVJPEG_MAX_COMPONENT] = {};
  nvjpegChromaSubsampling_t ss;
  if (nvjpegGetImageInfo(hd, data, len, &nc, &ss, ws, hs) != NVJPEG_STATUS_SUCCESS || ws[0] != w || hs[0] != h) return {};
  auto out = at::empty({3, h, w}, dopt(at::kByte));
  nvjpegImage_t img{};
  for (int c = 0; c < 3; ++c) {
    img.channel[c] = static_cast<unsigned char*>(out[c].data_ptr());
    img.pitch[c] = (size_t)w;
  }
  if (nvjpegDecode(hd, st, data, len, NVJPEG_OUTPUT_RGB, &img, c10::cuda::getCurrentCUDAStream().stream()) !=
      NVJPEG_STATUS_SUCCESS)
    return {};
  return out.unsqueeze(0);
}

// RGB8 [h][w][3] (host) -> [N, 1536] pixel values on the GPU (fp32; bf16 = their bf16 cast, half the memory), resized
// to rh x rw (multiples of 32). The resize runs one channel at a time (the antialiased bicubic filter is per channel:
// same values, a third of the fp32 temporaries).
// jpeg (nullable): the encoded image, decoded on the GPU first (the reference's path for JPEG).
at::Tensor preprocess(const uint8_t* rgb, int h, int w, int rh, int rw, bool bf16 = false, const uint8_t* jpeg = nullptr,
                      size_t jlen = 0) {
  at::Tensor img;
  if (jpeg && jlen) img = decode_jpeg_gpu(jpeg, jlen, h, w);
  if (!img.defined())
    img = at::from_blob(const_cast<uint8_t*>(rgb), {h, w, 3}, at::TensorOptions().dtype(at::kByte))
              .to(dopt(at::kByte)).permute({2, 0, 1}).unsqueeze(0);  // [1, 3, h, w], channels-last like PIL->tensor
  if (rh != h || rw != w) {
    auto out = at::empty({1, 3, rh, rw}, dopt(at::kByte));
    for (int c = 0; c < 3; ++c) {
      auto f = at::_upsample_bicubic2d_aa(img.narrow(1, c, 1).to(at::kFloat), {rh, rw}, false, c10::nullopt, c10::nullopt);
      out.narrow(1, c, 1).copy_(f.clamp_(0, 255).round_().to(at::kByte));
    }
    img = out;
  }
  auto mean = at::full({3}, 0.5, dopt(at::kFloat)) * 255.0;  // fused rescale (1/255) + normalize (0.5, 0.5)
  auto x = img.to(at::kFloat).sub_(mean.view({-1, 1, 1})).div_(mean.view({-1, 1, 1}));
  img.reset();
  if (bf16) x = x.to(at::kBFloat16);
  const int64_t gh = rh / PATCH, gw = rw / PATCH;
  x = x.reshape({1, 3, gh / MERGE, MERGE, PATCH, gw / MERGE, MERGE, PATCH}).permute({0, 2, 5, 3, 6, 1, 4, 7});
  return x.unsqueeze(6).expand({-1, -1, -1, -1, -1, -1, TPATCH, -1, -1}).reshape({gh * gw, PDIM});
}

void finish() {
  static const bool keep = getenv("AGENS_VISION_KEEP_CACHE") && getenv("AGENS_VISION_KEEP_CACHE")[0] == '1';
  if (!keep) c10::cuda::CUDACachingAllocator::emptyCache();
}
}  // namespace

extern "C" {
// A vision tensor (checkpoint name with or without "model.visual."): bf16 in pinned host memory, row-major.
int ak_vis_register(const char* name, void* host, const long long* shape, int ndim) {
  try {
    std::string n(name);
    const std::string pre = "model.visual.";
    if (n.compare(0, pre.size(), pre) == 0) n = n.substr(pre.size());
    std::vector<int64_t> sh(shape, shape + ndim);
    g_host[n] = at::from_blob(host, sh, at::TensorOptions().dtype(at::kBFloat16));
    g_res.erase(n);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_vis_register: %s\n", e.what());
    return 1;
  }
}
int ak_vis_ready() { return g_host.count("merger.linear_fc2.weight") && g_host.count("blocks.26.mlp.linear_fc2.weight") ? 1 : 0; }

// Preprocessed pixel values of one image (the reference processor's `pixel_values`): out [N][1536] fp32 host.
int ak_vis_pixels(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, float* out,
                  void* stream) {
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    auto pv = preprocess(rgb, h, w, rh, rw, false, jpeg, (size_t)jlen);
    at::from_blob(out, {pv.size(0), PDIM}, at::TensorOptions().dtype(at::kFloat)).copy_(pv);
    finish();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_vis_pixels: %s\n", e.what());
    return 1;
  }
}

// One image: RGB8 [h][w][3] host (jpeg: its encoded bytes, decoded on the GPU when possible) -> out [(rh/32)*(rw/32)][5120] bf16 host (pinned or pageable).
int ak_vis_encode(const uint8_t* rgb, const uint8_t* jpeg, long long jlen, int h, int w, int rh, int rw, void* out,
                  void* stream) {
  try {
    if (rh % (PATCH * MERGE) || rw % (PATCH * MERGE) || rh <= 0 || rw <= 0) return 2;
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    at::Tensor e;
    {
      e = tower(preprocess(rgb, h, w, rh, rw, /*bf16=*/true, jpeg, (size_t)jlen), rh / PATCH, rw / PATCH);
    }
    at::from_blob(out, {e.size(0), OUTD}, at::TensorOptions().dtype(at::kBFloat16)).copy_(e);
    e.reset();
    finish();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_vis_encode: %s\n", e.what());
    return 1;
  }
}

// The tower on given pixel values (gate: the reference processor's pixel_values): pv [gh*gw][1536] fp32 host.
int ak_vis_encode_pixels(const float* pv, int gh, int gw, void* out, void* stream) {
  try {
    if (gh % MERGE || gw % MERGE) return 2;
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), cur_device()));
    at::Tensor e;
    {
      auto t = at::from_blob(const_cast<float*>(pv), {(int64_t)gh * gw, PDIM}, at::TensorOptions().dtype(at::kFloat))
                   .to(dopt(at::kFloat));
      e = tower(t, gh, gw);
    }
    at::from_blob(out, {e.size(0), OUTD}, at::TensorOptions().dtype(at::kBFloat16)).copy_(e);
    e.reset();
    finish();
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "ak_vis_encode_pixels: %s\n", e.what());
    return 1;
  }
}

// Peak bytes torch's allocator reached on this device since the last call (and reset): the encode workspace.
long long ak_vis_peak_bytes() {
  try {
    const auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(cur_device());
    const long long p = (long long)st.reserved_bytes[0].peak;
    c10::cuda::CUDACachingAllocator::resetPeakStats(cur_device());
    return p;
  } catch (const std::exception&) {
    return -1;
  }
}
}  // extern "C"
