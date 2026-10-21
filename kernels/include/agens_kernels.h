// C ABI of libagens_kernels (consumed by crates/agens-cuda via FFI and by the benches/tests).
// All functions return 0 on success or a cudaError_t value. `stream` is a cudaStream_t (NULL = default stream).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __CUDACC__
#include <cuda_bf16.h>
#else
typedef struct { unsigned short x; } __nv_bfloat16;  // layout only (structs below)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- runtime
int ak_device_count(int* n);
int ak_set_device(int dev);
int ak_device_name(int dev, char* buf, int len);
int ak_malloc(void** ptr, size_t bytes);
int ak_free(void* ptr);
int ak_memcpy_h2d(void* dst, const void* src, size_t bytes);
int ak_memcpy_d2h(void* dst, const void* src, size_t bytes);
int ak_synchronize(void);
int ak_mem_info(size_t* free_bytes, size_t* total_bytes);
const char* ak_error_string(int code);
int ak_host_alloc(void** ptr, size_t bytes);  // pinned + mapped (device-accessible via UVA)
int ak_host_free(void* ptr);
// programmatic dependent launch in captured graphs (runtime.cu; kernels/include/agens_pdl.cuh)
int ak_pdl_on(void);
int ak_cc90(void);        // the device is compute capability 9.0 (the tuned defaults of the PDL / side-stream / split paths)
int ak_tp_push_on(void);  // tensor parallel: the push all-reduce is set up (peer GPUs; tp_ar.cu)
int ak_graph_pdl(void* graph);
// tensor-parallel greedy head (torch_ops.cu): this rank's encoded best of its vocabulary share per row, then (after the
// caller's all-reduce of hx) the argmax of the whole row -- ak_argmax's result
int ak_argmax_tp_part(const void* logits, int rows, int vocab, int base, int base_end, int rank, int nx, void* hx,
                      void* stream);
int ak_argmax_tp_combine(const void* hx, int rows, int nx, int tp, int* out, void* stream);
// two RMSNorms in one launch: of the embedding rows table[ids[b]] (w0 -> y0) and of the rows x1 (w1 -> y1) (rmsnorm.cu)
int ak_rmsnorm_pair_embed(const int32_t* ids, const void* table, const void* w0, void* y0, const void* x1, const void* w1,
                          void* y1, int B, int H, int so, float eps, int gemma, void* stream);

// ---- W4A16 GEMV, compressed-tensors pack-quantized, symmetric, group 128 (the Volundr INT4 release format).
// x [k] bf16; packed [n, k/8] int32 (nibble = q + 8, low nibble first); scales [n, k/128] bf16; y [n] bf16.
// k must be a multiple of 128. fp32 accumulation, deterministic (fixed reduction order).
int ak_w4g128_gemv_bf16(const void* x, const int32_t* packed, const void* scales, void* y, int n, int k,
                        void* stream);

// ---- bf16 linear (bit-exact reference path): y[m, n] = x[m, k] · W[n, k]^T, issued like torch F.linear.
int ak_bf16_linear(const void* x, const void* w, void* y, int m, int n, int k, void* stream);
int ak_bf16_set_workspace_bytes(size_t bytes);
// Same GEMM, fp32 reduction forced (CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION); not torch-identical by design.
int ak_bf16_linear_f32reduce(const void* x, const void* w, void* y, int m, int n, int k, void* stream);
int ak_bf16_linear_f32reduce_ld(const void* x, const void* w, void* y, int m, int n, int k, int ldy, void* stream);
int ak_bf16_linear_bias(const void* x, const void* w, const void* bias, void* y, int m, int n, int k, void* stream);
// Batch-invariant bf16 GEMV (decode, m <= 16, fp32 accumulation): row r of an m-row call == the m = 1 call on x[r].
int ak_bf16_gemv_rows(const void* x, const void* w, void* y, int m, int n, int k, void* stream);  // before first use; 0 = torch's default for the device

// ---- KDA decode (bit-exact with sglv23 bs < 8): conv1d update + SiLU, then the gated delta-rule step.
// conv_state [slots][3][dim] bf16, weight [dim][4] fp32, idx [B] int32 slot per request (-1 = skip).
int ak_kda_conv_update(const void* x, int x_stride, void* conv_state, const float* weight, const int32_t* idx, void* y,
                       int batch, int dim, void* stream);
// qkv [B][2*H*K + HV*V] bf16 (conv output), a [B][HV*K] bf16 raw gate, b [B][HV] bf16 raw beta,
// state [slots][HV][V][K] fp32 updated in place, out [B][HV][V] bf16.
int ak_kda_decode(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b, int b_stride,
                  const float* A_log, const float* dt_bias, float* state, const int32_t* idx, void* out, float scale,
                  int batch, int H, int HV, int K, int V, void* stream);

// Multi-token variants for speculative verify / commit (one sequence, slot idx[0]): T tokens, each bit-identical to
// that many sequential single steps; write_state = 0 leaves the state untouched; n_dev (nullable) = token count on
// device (capped at T). x / qkv / a / b rows t at t * stride; y / out [T][dim] / [T][HV][V] (nullable: no output).
int ak_kda_conv_multi(const void* x, int x_stride, void* conv_state, const float* weight, const int32_t* idx, void* y,
                      int dim, int T, const int* n_dev, int write_state, void* stream);
int ak_kda_decode_multi(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b, int b_stride,
                        const float* A_log, const float* dt_bias, float* state, const int32_t* idx, void* out, float scale,
                        int H, int HV, int K, int V, int T, const int* n_dev, int write_state, void* stream);

// ---- batch-invariant small-m kernels (speculative decode: decode and verify run the same per-row arithmetic)
// y[m][n] = x[m][k] . W[n][k]^T (+ bias[n]), bf16, 1 <= m <= 16, k % 32 == 0; row i independent of m.
int ak_kda_conv_multi_b(const void* x, int x_stride, void* conv_state, const float* weight, const int32_t* idx,
                        void* y, int dim, int T, int nb, const int* n_dev, int n_stride, int write_state, void* stream);
int ak_kda_decode_multi_b(const void* qkv, int qkv_stride, const void* a, int a_stride, const void* b, int b_stride,
                          const float* A_log, const float* dt_bias, float* state, const int32_t* idx, void* out,
                          float scale, int H, int HV, int K, int V, int T, int nb, const int* n_dev, int n_stride,
                          int write_state, void* stream);
int ak_bf16_gemv_bi(const void* x, const void* w, const void* bias, void* y, int m, int n, int k, void* stream);
int ak_bf16_gemv_bi_ld(const void* x, const void* w, const void* bias, void* y, int m, int n, int k, long ldy, void* stream);
// per-token C[t][m][n] = A[t][m][k] . B[t][k][n], fp32 sequential sum over k, one bf16 rounding
int ak_bf16_bmm_bi(const void* A, const void* B, void* C, int T, int m, int k, int n, void* stream);

// ---- mHC (bit-exact with sglv23's fused Triton kernels). d = dyn output [T][24] bf16; H [T][4][dim] bf16.
int ak_mhc_mix(const void* d, int d_stride, const void* pre_static, const void* post_static, const void* res_static,
               const void* H, int h_st, int h_ss, void* x, int x_st, void* post_out, void* res_out, int T, int dim,
               int iters, void* stream);
// mixed = res·H [T][4][dim]; out = mixed + post ⊗ (y - x_in); mean (nullable) = out.mean(streams) for the next mix.
int ak_mhc_combine(const void* mixed, int m_st, int m_ss, const void* post, const void* y, int y_st, const void* x_in,
                   int x_st, void* out, int o_st, int o_ss, void* mean, int mean_st, int T, int dim, void* stream);

// ---- Engram decode lookup: hash 2/3-grams (sglv23-identical), gather rows from host-mapped or device tables into
// e [B][2*dim] bf16, rotate prev [prev_rows][2] (slot -1 -> trash row prev_rows-1). rows_out [B][2] optional.
int ak_engram_decode(const int32_t* cur, int32_t* prev, int prev_rows, const int32_t* slots, const void* table2,
                     const void* table3, long long rows, void* e, int dim, long long* rows_out, int batch,
                     void* stream);
int ak_host_register(void* ptr, size_t bytes);
int ak_host_unregister(void* ptr);

// ---- SiLU(gate) * up, bit-exact with sglv23's act_and_mul_kernel: gate_up [T][2*inter] -> out [T][inter] bf16.
int ak_silu_mul(const void* gate_up, void* out, int tokens, int inter, void* stream);
// one token: SiLU(gate) * up of a tensor-parallel share ([gate share | up share], np each) into out[lo, lo + np) of the
// full row of inter, zeros elsewhere (silu_mul.cu: ak_silu_mul's bits)
int ak_silu_mul_part(const void* gate_up_part, void* out, int inter, int lo, int np, void* stream);

// ---- RoPE NeoX, partial rotary (first rot_dim of each head), in place; cache [max_pos][rot_dim] fp32 = cos|sin.
int ak_rope_neox(void* q, void* k, const float* cos_sin_cache, const int64_t* positions, int tokens, int nq, int nk,
                 int head_dim, int rot_dim, int q_stride, int k_stride, void* stream);
int ak_rope_cache(float* out, int max_pos, int rot_dim, float base, void* stream);
// interleaved MRoPE: pair j by axis T / H / W (sections sec_h, sec_w: H if j%3==1 && j<3*sec_h, W if j%3==2 && j<3*sec_w);
// positions[a * stride + t]; equal axes = ak_rope_neox exactly
int ak_rope_neox_m(void* q, void* k, const float* cos_sin_cache, const int64_t* positions, long stride, int tokens,
                   int nq, int nk, int head_dim, int rot_dim, int q_stride, int k_stride, float base, int sec_h,
                   int sec_w, void* stream);
// ak_rope_neox with cos / sin computed in the kernel (bit-identical to the ak_rope_cache table; no table)
int ak_rope_neox_fly(void* q, void* k, const int64_t* positions, int tokens, int nq, int nk, int head_dim, int rot_dim,
                     int q_stride, int k_stride, float base, void* stream);

// KDA output norm: y = rmsnorm(x) * w * g * sigmoid(g) per 128-wide head row (bit-exact with fla's Triton kernel).
int ak_kda_gated_rmsnorm(const void* x, const void* g, const void* w, void* y, int rows, int d, float eps, void* stream);

// ---- small ops as torch computes them (fp32 opmath, one bf16 rounding) + cuBLAS calls issued like torch
int ak_bf16_binary(const void* a, const void* b, void* y, long n, int op, void* stream);  // 0 add, 1 mul, 2 a*sigmoid(b)
int ak_sigmoid_to_bf16(const void* x, void* y, int n, void* stream);                      // bf16(sigmoid(float(x)))
int ak_scale_rows(const void* g, const void* v, void* y, int rows, int d, void* stream);  // y[t,:] = g[t] * v[t,:]
int ak_argmax(const void* logits, int rows, int vocab, int vocab_valid, int* out, void* stream);
int ak_embed(const int32_t* ids, const void* table, void* out, int tokens, int d, void* stream);
int ak_bf16_bmm(const void* A, const void* B, void* C, int T, int m, int k, int n, void* stream);
// the mHC shape (m = k = 4) of ak_bf16_bmm, same bits, memory bound (prefill)
int ak_bf16_bmm44(const void* A, const void* B, void* C, int T, int n, void* stream);

// ---- RMSNorm (gemma=1: x*r*(1+w), else x*r*w), bit-exact with sglv23 (flashinfer 0.6.15 CuTe-DSL RMSNormKernel)
int ak_rmsnorm(const void* x, const void* w, void* y, int rows, int d, int stride_in, int stride_out, float eps, int gemma,
               void* stream);

// ---- BCSA far branch, decode (one token; sglv23 far_sparse_fused + _merge, bit-exact intent), graph-capturable.
// qi [4][128] bf16 (idx_norm(idx_q(h))), qh [24][256] bf16 post-RoPE q, o_loc [24][256] bf16 local-window output,
// lse [24] fp32 natural log, pos [1] int64, r2t [Wctx] int32 slot map, kc/vc [rows+1][4][256] (device or host-mapped),
// blk [rows/4+1][128] (one block key per aligned 4-row page: blk[row / 4]).
// ws: ak_far_decode_ws_bytes(Wctx) bytes of device memory (intermediates). out [24][256] bf16.
size_t ak_far_decode_ws_bytes(int Wctx);
int ak_far_decode(const void* qi, const void* qh, const void* o_loc, const float* lse, const long long* pos,
                  const int* r2t, int Wctx, const void* kc, const void* vc, const void* blk, int rows, void* ws,
                  void* out, void* stream);
// KV tiering sources of the selected blocks (pools may be host-mapped): src 0 = kc/vc bf16 rows (== ak_far_decode),
// 1 = kc/vc bf16 block means per 4-row page [rows/4+1], 2 = kc/vc int8 rows + ksc/vsc fp32 scales [rows+1][4].
int ak_far_decode_src(const void* qi, const void* qh, const void* o_loc, const float* lse, const long long* pos,
                      const int* r2t, int Wctx, int src, const void* kc, const void* vc, const float* ksc,
                      const float* vsc, const void* blk, int rows, void* ws, void* out, void* stream);
// Multi-query far decode (speculative verify): T <= 8 consecutive queries (query t: qi + t*512, qh / o_loc / out +
// t*6144, lse + t*24, pos + t), each output bit-identical to its own ak_far_decode_src call; block keys gathered once
// and the union of the T top-k sets fetched once. ws: ak_far_decode_multi_ws_bytes(Wctx, T), initialised once by
// ak_far_decode_multi_init.
size_t ak_far_decode_multi_ws_bytes(int Wctx, int T);
// cross-step cache of a layer's union (block means): ak_far_cache_bytes / _init once; the cached fetch replaces the
// pool fetch (same values); tag: device long long, the owner (session + epoch) whose blocks the cache may serve
size_t ak_far_cache_bytes(int Wctx, int cap);
int ak_far_cache_init(void* cache, int Wctx, int cap, void* stream);
int ak_far_fused_on();
int ak_far_decode_multi_init(void* ws, int Wctx, int T, void* stream);
int ak_far_decode_multi_src(int T, const void* qi, const void* qh, const void* o_loc, const float* lse,
                            const long long* pos, const int* r2t, int Wctx, int src, const void* kc, const void* vc,
                            const float* ksc, const float* vsc, const void* blk, int rows, void* ws, void* out,
                            void* stream);

// ---- KDA prefill of one sequence chunk (T <= 4096), sglv23's Triton kernels AOT'd with pinned configs.
int ak_kda_prefill(const void* mixed_qkv, const void* a, const void* b, const float* conv_w, const float* A_log,
                   const float* dt_bias, void* conv_pool, float* ssm_pool, const int32_t* slot_dev, const bool* has_init_dev,
                   int T, void* o, void* stream);

// KT ("k-tiled") layout of the same weights, built once at load time (see w4_gemv.cu). k is padded to 1024.
size_t ak_w4g128_kt_weight_bytes(int n, int k);
size_t ak_w4g128_kt_scale_bytes(int n, int k);
int ak_w4g128_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, void* stream);
int ak_w4g128_gemv_kt_bf16(const void* x, const void* wt, const void* st, void* y, int n, int k, void* stream);

// KT layout with the group size as a parameter (64 or 128; k % group == 0). group = 128, bf16w = 0 gives the bits of
// the g128 entry points above. bf16w = 1: Marlin numerics -- every weight rounded to bf16(q * scale) before the fp32
// dot (what sglang's Marlin path multiplies with); bf16w = 0: exact q, fp32 scale per 32-input chunk.
// m = 1 GEMV; up to 4 linears sharing x in one launch (each y[i] bit-identical to its own GEMV); batched GEMV for
// 1 <= m <= 16 (x [m][k], y [m][n]; row i bit-identical to the m = 1 GEMV of x[i], same bf16w); dequant to dense bf16
// w [n][k] (value bf16(q * scale)) for scratch + cuBLAS.
size_t ak_w4_kt_weight_bytes(int n, int k);
size_t ak_w4_kt_scale_bytes(int n, int k, int group);
int ak_w4_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, int group, void* stream);
int ak_w4_gemv_kt(const void* x, const void* wt, const void* st, void* y, int n, int k, int group, int bf16w,
                  void* stream);
int ak_w4_gemv_kt_multi(const void* x, int nseg, const void* const* wt, const void* const* st, void* const* y,
                        const int* n, int k, int group, int bf16w, void* stream);
int ak_w4_gemv_kt_batched(const void* x, const void* wt, const void* st, void* y, int m, int n, int k, int group,
                          int bf16w, void* stream);
// Tensor-core W4 GEMV on the same k-tiled weights (g128, g64; W3 / W2 KT too): exact products, fp32 sums in another
// order than the exact path; 1 <= m <= 16, rows bit-identical for any m. The entry points above (and ak_w3_* / ak_wq_*)
// use it by default (exact path: AGENS_W4_EXACT=1, or ak_w4_set_mma(0); ak_w4_set_mma(-1) re-reads the environment).
int ak_w4_mma_gemv(const void* x, const void* wt, const void* st, void* y, int m, int n, int k, int group, void* stream);
int ak_w4_set_mma(int mode);
// sm90 tensor-core g128 W4 path with the weights streamed by cp.async.bulk into a shared-memory ring (bit-identical to
// the register MMA path): -1 auto (on for compute capability 9.x unless AGENS_W4_TMA=0), 0 off, 2 / 4 / 8 stages.
int ak_w4_set_tma(int stages);
// lean register MMA path for W4 g128 (n % 16 == 0, k % 1024 == 0; bit-identical to the MMA path): -1 auto
// (AGENS_W4_LEAN), 0 off, 1 / 2 / 4 row blocks per work item
int ak_w4_set_lean(int rb);
int ak_w4_set_kf(int on);              // W4 g128 repacked from now on in the KF layout (AGENS_W4F); previous setting
int ak_w4_is_kf(const void* wt);
int ak_w4_kf_forget(const void* wt);  // before freeing a KF weight
// W4H (sm90 / sm100; kernels/src/w4h.cu, layout spec in its header): W4 g128 in a fragment-major layout built at load
// time (n % 16 == 0, k % 128 == 0, k <= 9216; n * k / 2 weight bytes + n * k / 64 scale bytes, within the KT buffers of
// the same linear), streamed by a bulk-copy ring. ak_w4h_gemm: 1 <= m <= 128, nseg <= 4 linears sharing x, output of
// token j at y[i] + j * ldy[i] (ldy = nullptr: n[i]). Outputs bit-identical to the tensor-core KT path (ak_w4_mma_gemv & co.) for every m; rows bit-identical for any m.
size_t ak_w4h_weight_bytes(int n, int k);
size_t ak_w4h_scale_bytes(int n, int k);
int ak_w4h_ok(int n, int k);
int ak_w4h_repack(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, void* stream);
int ak_w4h_gemm(const void* x, int m, int nseg, const void* const* wt, const void* const* st, void* const* y,
                const int* n, const int* ldy, int k, void* stream);
int ak_w4h_dequant_rows(const void* wt, const void* st, void* w, int n, int k, int r0, int nr, void* stream);
// Engine use: with AGENS_W4H=1 (or ak_w4h_set_mode(1)) ak_w4_repack_kt / ak_wq_repack_kt (bits 4, group 128) write
// eligible linears in W4H instead of KT, into the same buffers, and every KT entry point (GEMV / multi / batched,
// dequant) routes those weights to the W4H kernels -- same bits. ak_w4h_forget before freeing a W4H weight.
int ak_w4h_set_mode(int mode);  // 0 off, 1 on, -1 re-read AGENS_W4H; returns the previous mode
int ak_w4h_is(const void* wt);
int ak_w4h_forget(const void* wt);
// dual layout (AGENS_W4H_MMAX = N > 0): KT kept, W4H copies for launches of m <= N; their bytes; W4H for an m-token launch
size_t ak_w4h_dual_bytes();
int ak_w4h_is_m(const void* wt, int m);
// the W4H weight / scales an m-token launch of wt uses (wt itself, or its dual-layout copy); 0: none (KT layout)
int ak_w4h_ptrs(const void* wt, const void* st, int m, const void** w, const void** s);
// W3 KT (3-bit values q in [-4, 3] in the W4 KT tiling, 3/8 byte per weight; scales as W4 KT): repack from the int4
// container (cudaErrorInvalidValue if a value needs 4 bits), and the W4 KT entry points' W3 twins -- bit-identical to
// the exact W4 KT path on the same values.
size_t ak_w3_kt_weight_bytes(int n, int k);
int ak_w3_repack_kt(const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, int group, void* stream);
int ak_w3_gemv_kt(const void* x, const void* wt, const void* st, void* y, int n, int k, int group, void* stream);
int ak_w3_gemv_kt_multi(const void* x, int nseg, const void* const* wt, const void* const* st, void* const* y,
                        const int* n, int k, int group, void* stream);
int ak_w3_gemv_kt_batched(const void* x, const void* wt, const void* st, void* y, int m, int n, int k, int group,
                          void* stream);
int ak_w3_dequant_kt_rows(const void* wt, const void* st, void* w, int n, int k, int group, int r0, int nr,
                          void* stream);
// Mixed precision (GSQ-RCO: per linear W2 / W3 / W4 x group 64 / 128). W{bits} KT: the W4 KT tiling with bits-bit
// values (W2: q in [-2, 1], 1/4 byte per weight; W3 as above), scales as W4 KT; every kernel expands a lane's chunk to the
// W4 nibble words, so results are bit-identical to W4 KT of the same group on the same values (exact and tensor-core).
// repack: from the int4 container, cudaErrorInvalidValue if a value needs more bits. gemv: 1 <= m <= 16 tokens, nseg <= 4
// linears sharing x (same bits, k, group), output of token j at y[i] + j * ldy[i] (ldy = nullptr: n[i]); bits = 4 is
// ak_w4_gemv_kt / _multi / _batched.
size_t ak_wq_kt_weight_bytes(int bits, int n, int k);
int ak_wq_repack_kt(int bits, const int32_t* packed, const void* scales, void* wt, void* st, int n, int k, int group,
                    void* stream);
int ak_wq_gemv_kt_multi(int bits, const void* x, int m, int nseg, const void* const* wt, const void* const* st,
                        void* const* y, const int* n, const int* ldy, int k, int group, int bf16w, void* stream);
int ak_wq_dequant_kt_rows(int bits, const void* wt, const void* st, void* w, int n, int k, int group, int r0, int nr,
                          void* stream);
// INT8 weight GEMV (lm_head): y[j][o] (row stride ldy) = scale[o] * sum_i x[j][i] * q[o][i]; q int8 [n][k], scale fp32
// [n]; 1 <= m <= 16, k % 16 == 0; row j bit-identical for any m.
int ak_i8_gemv(const void* x, const void* w, const void* scale, void* y, int m, int n, int k, int ldy, void* stream);
int ak_w4_dequant_kt(const void* wt, const void* st, void* w, int n, int k, int group, void* stream);
// rows [r0, r0 + nr) of it into w [nr][k] (tiled prefill: a bounded scratch instead of the whole weight)
int ak_w4_dequant_kt_rows(const void* wt, const void* st, void* w, int n, int k, int group, int r0, int nr,
                          void* stream);

// Tuning entry points (bench only): variant = 10 * rows_per_warp + unroll.
int ak_w4g128_gemv_bf16_variant(int variant, const void* x, const int32_t* packed, const void* scales, void* y, int n,
                                int k, void* stream);
int ak_w4g128_gemv_kt_variant(int variant, const void* x, const void* wt, const void* st, void* y, int n, int k,
                              void* stream);
// rows/warp r in {1,2,4}, unroll u in {1,2,4}, warps/block in {4,8,16}, blocks-per-SM cap (0 = occupancy); same bits
int ak_w4g128_gemv_kt_tune(int r, int u, int warps, int bps, const void* x, const void* wt, const void* st, void* y,
                           int n, int k, void* stream);

// fused residual add + RMSNorm (s = bf16(a + b) written to s_out, y = RMSNorm(s)); d in (3072, 6144]
int ak_add_rmsnorm(const void* a, const void* b, void* s_out, const void* w, void* y, int rows, int d, float eps,
                   int gemma, void* stream);
// Fused mHC layer boundary for decode / verify rows (hidden 5120): layer i's res.H bmm + combine (+ its last residual add
// yb) -> layer i + 1's mean RMSNorm, dyn GEMV (batch-invariant), mix + Sinkhorn and input RMSNorm; bit-identical to
// ak_bf16_bmm_bi, ak_mhc_combine(_add), ak_rmsnorm (gemma 0), ak_bf16_gemv_bi (n 24), ak_mhc_mix, ak_rmsnorm (gemma 1).
typedef struct MhcFusedArgs {
  const __nv_bfloat16 *dcur, *res_s_cur;  // layer i: dyn rows [T][24] and res_static (its res is recomputed from them)
  const __nv_bfloat16 *Hold, *post, *Y, *Yb, *Xi;  // layer i: H [T][4][5120], post [T][4], rows [T][5120] (Yb nullable)
  __nv_bfloat16 *Hn, *hmn, *cap;                   // out: H' [T][4][5120], mean [T][5120], stream 0 copy (nullable)
  long cap_ld;
  const __nv_bfloat16 *wn, *wdyn, *pre_s, *post_s, *res_s, *wln;  // layer i + 1: norm.weight, dyn [24][5120], statics, input_layernorm
  __nv_bfloat16 *d, *xin, *post_o, *res_o, *h;     // out: dyn [T][24] (may be dcur), x_in (may be Xi), post, res, normalized x_in
  float eps;
  int iters, write_res;                            // write_res: also layer i + 1's res (else the next boundary computes it)
  const __nv_bfloat16* wb;                         // nullable: layer i + 1's in_proj_b [nb_out][5120] (KDA), bout = h . wb^T
  int nb_out;                                      // <= 48
  __nv_bfloat16* bout;                             // [T][nb_out]
  const __nv_bfloat16* res_cur;                    // nullable: layer i's res [T][16], written by the previous boundary's
                                                   // write_res (then dcur / res_s_cur are not read)
} MhcFusedArgs;
int ak_mhc_fused_ok(void);  // the device can launch it (thread-block clusters: sm90+)
int ak_mhc_fused(const MhcFusedArgs* args, int T, void* stream);
// the same with layer i's MLP tensor-parallel all-reduce folded in (Yb = this rank's partial; push path only);
// ak_mhc_fused_ar_ok(T): whether it can take T rows (else all-reduce Yb and call ak_mhc_fused)
int ak_mhc_fused_ar_ok(int T);
int ak_mhc_fused_ar(const MhcFusedArgs* args, int T, void* stream);
// Speculative commit of every KDA layer in one launch (K = V = 128, T <= 8): per layer the conv-state replay of the stashed
// pre-conv rows sq and the recurrent-state replay of the stashed conv outputs sc / gates sa / betas sb over the n_dev
// accepted tokens; bit-identical to ak_kda_conv_multi(_b) + ak_kda_decode_multi(_b) per layer (write_state 1, no outputs).
#define kKdaCommitMaxLayers 64
typedef struct KdaCommitLayer {
  const __nv_bfloat16 *sq, *sc, *sa, *sb;
  void* conv;
  const float *conv_w, *A_log, *dt_bias;
  float* ssm;
} KdaCommitLayer;
typedef struct KdaCommitArgs {
  KdaCommitLayer L[kKdaCommitMaxLayers];
  int nl, conv_dim, a_stride, b_stride, T, n_stride, H, HV;  // conv_dim = the stash row stride of sq / sc too
  const int32_t* slot;                                    // [nb] state slots
  const int* n_dev;                                       // accepted count of sequence z at n_dev[z * n_stride]
  float scale;
} KdaCommitArgs;
int ak_kda_commit_all(const KdaCommitArgs* args, int nb, void* stream);
// One KDA layer's speculative verify in one launch (K = V = 128, T <= 8 tokens per sequence, nb sequences): conv over the
// stashed pre-conv rows x (state untouched) -> y (the stash sc), recurrence (state untouched), gated RMSNorm with gate g
// -> out; bit-identical to ak_kda_conv_multi(_b) + ak_kda_decode_multi(_b) + ak_kda_gated_rmsnorm.
typedef struct KdaVerifyArgs {
  const __nv_bfloat16* x;      // [nb * T][x_stride] pre-conv q|k|v rows
  __nv_bfloat16* y;            // [nb * T][dim] conv outputs (the commit's stash)
  const __nv_bfloat16* conv;   // conv state [slots][3][dim]
  const float* conv_w;         // [dim][4]
  const __nv_bfloat16 *ag, *b; // gates [rows][a_stride], betas [rows][b_stride]
  const float *A_log, *dt_bias;
  const float* ssm;            // [slots][HV][V][K]
  const int32_t* slot;         // [nb]
  const __nv_bfloat16 *g, *w;  // the norm's gate z [rows][HV][128] and weight [128]
  __nv_bfloat16* out;          // [rows][HV][128]
  int x_stride, dim, a_stride, b_stride, H, HV, T;
  float scale, eps;
} KdaVerifyArgs;
int ak_kda_verify_fused(const KdaVerifyArgs* args, int nb, void* stream);
// Attention q / k prep for decode / verify rows (head_dim 256, rot 64): split_q_gate + q / k RMSNorm (gemma) + MRoPE +
// store_kv in one launch, bit-identical to those kernels in that order.
typedef struct QkPrepArgs {
  const __nv_bfloat16* qg;            // [T][nq][2][256] (q | gate per head)
  __nv_bfloat16 *q, *gate, *k;        // out q, gate [T][nq][256]; k [T][nk][256] in place
  const __nv_bfloat16* v;             // [T][nk][256]
  __nv_bfloat16 *kc, *vc;             // caches [rows][nk][256], row loc[t] (kc null: no store)
  const int* loc;
  const __nv_bfloat16 *wq, *wk;       // q_norm / k_norm weights [256]
  const float* cache;                 // nullable cos|sin [pos][64]
  const int64_t* pos;                 // [3][pstride] MRoPE positions (pstride 0: one row for every axis = plain RoPE)
  long pstride;
  int T, nq, nk, sec_h, sec_w;
  float eps, theta;
} QkPrepArgs;
int ak_qk_prep(const QkPrepArgs* args, void* stream);
// ak_mhc_combine with y = bf16(y + yb) folded in
int ak_mhc_combine_add(const void* mixed, int m_st, int m_ss, const void* post, const void* y, const void* yb, int y_st,
                       const void* x_in, int x_st, void* out, int o_st, int o_ss, void* mean, int mean_st, int T, int dim,
                       void* stream);

// tensor parallel (kernels/src/tp_ar.cu): exchange regions, setup, the rank-order all-reduce of bf16 x[n]
int ak_tp_region_alloc(size_t bytes, void** dev_ptr, void* ipc_handle);
int ak_tp_region_open(const void* ipc_handle, void** dev_ptr);
int ak_tp_region_shm(const char* name, size_t bytes, int create, void** dev_ptr);
int ak_tp_can_peer(int dev, int peer);
int ak_tp_setup(int rank, int size, void* const* regions, long slot_elems);
size_t ak_tp_region_bytes(long slot_elems);
int ak_tp_allreduce(void* x, long n, void* stream);
// the mixer all-reduce fused with its residual add + RMSNorm (bits of ak_tp_allreduce + ak_add_rmsnorm; mo unchanged);
// cudaErrorNotSupported when the push path cannot take it (the caller runs the two calls)
int ak_tp_allreduce_add_rmsnorm(const void* mo, const void* res, void* s_out, const void* w, void* y, int rows, int d,
                                float eps, int gemma, void* stream);
long ak_tp_push_selftest(int iters);
double ak_tp_bench2(int iters, long rows, int which, int graph);
// this process's head share for the far branch (q / kv heads, 6 q heads per kv head) and the KDA prefill (q / k heads)
int ak_far_set_heads(int qh, int kvh);
int ak_kda_prefill_set_heads(int h, int hv);
int ak_kda_prefill_preload();  // load every module of the device's table now (init)
// budget mode: the KDA prefill temporaries of calls up to T tokens from an arena reserved now (0: the stream-ordered pool)
size_t ak_kda_prefill_temp_bytes(int T);
int ak_kda_prefill_set_arena(int T);

#ifdef __cplusplus
}
#endif
