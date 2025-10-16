/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

void set_params_fprop(
    Flash_fwd_params &params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_mask,
    const size_t h_bias,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor mask,
    const at::Tensor bias,
    at::Tensor out,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *seqused_k,
    void *p_d,
    void *softmax_lse_d,
    float softmax_scale,
    bool is_causal,
    const float softcap,
    bool has_mask,
    bool has_bias,
    bool seqlenq_ngroups_swapped=false,
    const bool unpadded_lse=false
) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.mask_ptr = has_mask ? mask.data_ptr() : nullptr;
    params.bias_ptr = has_bias ? bias.data_ptr() : nullptr;
    params.o_ptr = out.data_ptr();
    
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_row_stride = k.stride(-3);
    params.k_head_stride = k.stride(-2);
    params.v_row_stride = v.stride(-3);
    params.v_head_stride = v.stride(-2);
    params.mask_head_stride = has_mask ? (mask.size(-3) == 1 ? 0 : mask.stride(-3)) : 0;
    params.mask_row_stride = has_mask ? (mask.size(-2) == 1 ? 0 : mask.stride(-2)) : 0;
    params.bias_head_stride = has_bias ? (bias.size(-3) == 1 ? 0 : bias.stride(-3)) : 0;
    params.bias_row_stride = has_bias ? (bias.size(-2) == 1 ? 0 : bias.stride(-2)) : 0;
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.mask_batch_stride = has_mask ? (mask.size(0) == 1 ? 0 : mask.stride(0)) : 0;
        params.bias_batch_stride = has_bias ? (bias.size(0) == 1 ? 0 : bias.stride(0)) : 0;
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.mask_batch_stride *= seqlen_q;
            params.bias_batch_stride *= seqlen_q;
            params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_mask = h_mask;
    params.h_bias = h_bias;
    params.h_h_k_ratio = h / h_k;
    params.h_h_mask_ratio = h / h_mask;
    params.h_h_bias_ratio = h / h_bias;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash dynamic mask attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.is_causal = is_causal;
    params.has_mask = has_mask;
    params.has_bias = has_bias;
    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash dynamic mask attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(
    Flash_bwd_params &params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_mask,
    const size_t h_bias,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor mask,
    const at::Tensor bias,
    const at::Tensor out,
    const at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor dbias,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *dq_accum_d,
    void *dk_accum_d,
    void *dv_accum_d,
    void *softmax_lse_d,
    void *dsoftmax_sum_d,
    float softmax_scale,
    bool is_causal,
    const float softcap,
    bool has_mask,
    bool has_bias,
    bool accum_dbias,
    bool deterministic,
    const bool unpadded_lse
) {
    set_params_fprop(
        params,
        b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, h_mask, h_bias, d, d_rounded,
        q, k, v, mask, bias, out,
        cu_seqlens_q_d,
        cu_seqlens_k_d,
        nullptr,
        nullptr,
        softmax_lse_d,
        softmax_scale,
        is_causal,
        softcap,
        has_mask,
        has_bias,
        false,  // seqlenq_ngroups_swapped
        unpadded_lse
    );

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dbias_ptr = has_bias ? dbias.data_ptr() : nullptr;

    // All stride are in elements, not bytes.
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_row_stride = dq.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_row_stride = dk.stride(-3);
    params.dk_head_stride = dk.stride(-2);
    params.dv_row_stride = dv.stride(-3);
    params.dv_head_stride = dv.stride(-2);
    params.dbias_head_stride = has_bias ? (dbias.size(-3) == 1 ? 0 : dbias.stride(-3)) : 0;
    params.dbias_row_stride = has_bias ? (dbias.size(-2) == 1 ? 0 : dbias.stride(-2)) : 0;

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
        params.dbias_batch_stride = has_bias ? (dbias.size(0) == 1 ? 0 : dbias.stride(0)) : 0;
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.accum_dbias = accum_dbias;

    params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(params.has_mask, Has_mask, [&] {
                    BOOL_SWITCH(params.has_bias, Has_bias, [&] {
                        if (params.num_splits <= 1 && !force_split_kernel) {    // If we don't set it num_splits == 0
                            run_mha_fwd_<elem_type, kHeadDim, Is_causal, Has_mask, Has_bias>(params, stream);
                        } else {
                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal, Has_mask, Has_bias>(params, stream);
                        }
                    });
                });
            });
        });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(
    Flash_fwd_params &params,
    const int batch_size,
    const int num_heads,
    const int head_size,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int head_size_rounded,
    const int num_splits,
    const int num_sm,
    struct c10::TensorOptions opts
) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = params.has_mask || params.has_bias
        ? 64
        : head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (num_splits < 1) {
        // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
    }
    if (params.num_splits > 1) {
        softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
        out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_ptr = out_accum.data_ptr();
    }
    TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");

    return std::make_tuple(softmax_lse_accum, out_accum);
}

std::vector<at::Tensor>
mha_fwd(
    at::Tensor &q,                                  // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,                            // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,                            // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    std::optional<at::Tensor> &mask_,               // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x round_multiple(seqlen_k, 128)
    std::optional<at::Tensor> &bias_,               // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x round_multiple(seqlen_k, 128)
    std::optional<at::Tensor> &out_,                // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    const bool return_softmax
) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashDynamicMaskAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "FlashDynamicMaskAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    auto opts = q.options();

    bool has_mask = mask_.has_value();
    at::Tensor mask;
    if (has_mask) {
        mask = mask_.value();
        TORCH_CHECK(mask.dtype() == torch::kBool, "mask must have dtype bool");
        CHECK_DEVICE(mask);
        TORCH_CHECK(mask.dim() == 4, "mask must have 4 dimensions with shape (batch_size, nheads, seqlen_q, seqlen_k_rounded)");
        TORCH_CHECK(mask.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    } else {
        mask = torch::empty({0}, opts);
    }
    bool has_bias = bias_.has_value();
    at::Tensor bias;
    if (has_bias) {
        bias = bias_.value();
        TORCH_CHECK(bias.dtype() == q_dtype, "bias must have the same dtype as inputs");
        CHECK_DEVICE(bias);
        TORCH_CHECK(bias.dim() == 4, "bias must have 4 dimensions with shape (batch_size, nheads, seqlen_q, seqlen_k_rounded)");
        TORCH_CHECK(bias.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    } else {
        bias = torch::empty({0}, opts);
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    int num_heads_mask = has_mask ? mask.size(1) : 1;
    int num_heads_bias = has_bias ? bias.size(1) : 1;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashDynamicMaskAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (has_mask) {
        TORCH_CHECK(mask.size(0) == batch_size || mask.size(0) == 1, "Batch dimension in mask must be 1 or equal to batch size");
        TORCH_CHECK(num_heads_mask == 1 || num_heads_mask == num_heads_k || num_heads_mask == num_heads, "Number of heads in mask must be 1, h_k or h");
        TORCH_CHECK(mask.size(2) == 1 || mask.size(2) == seqlen_q, "Query length dimension in mask must be 1 or equal to seqlen_q");
        TORCH_CHECK(mask.size(3) == seqlen_k_rounded, "Key length dimension in mask must be seqlen_k_rounded");
    }
    if (has_bias) {
        TORCH_CHECK(bias.size(0) == batch_size || bias.size(0) == 1, "Batch dimension in bias must be 1 or equal to batch size");
        TORCH_CHECK(num_heads_bias == 1 || num_heads_bias == num_heads_k || num_heads_bias == num_heads, "Number of heads in bias must be 1, h_k or h");
        TORCH_CHECK(bias.size(2) == 1 || bias.size(2) == seqlen_q, "Query length dimension in bias must be 1 or equal to seqlen_q");
        TORCH_CHECK(bias.size(3) == seqlen_k_rounded, "Key length dimension in bias must be seqlen_k_rounded");
    }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1) { is_causal = false; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0;
    const int ngroups = num_heads / num_heads_k;
    const int batch_size_mask_og = has_mask ? mask.size(0) : batch_size;
    const int batch_size_bias_og = has_bias ? bias.size(0) : batch_size;
    const int num_heads_mask_og = num_heads_mask;
    const int num_heads_bias_og = num_heads_bias;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        if (has_mask) {
            if (num_heads_mask == 1) {
                mask = mask.reshape({batch_size_mask_og, 1, 1, seqlen_k_rounded});
            } else if (num_heads_mask == num_heads_k) {
                mask = mask.reshape({batch_size_mask_og, num_heads_k, 1, seqlen_k_rounded});
            } else if (num_heads_mask == num_heads) {
                mask = mask.reshape({batch_size_mask_og, num_heads_k, ngroups, seqlen_k_rounded});
            }
        }
        if (has_bias) {
            if (num_heads_bias == 1) {
                bias = bias.reshape({batch_size_bias_og, 1, 1, seqlen_k_rounded});
            } else if (num_heads_bias == num_heads_k) {
                bias = bias.reshape({batch_size_bias_og, num_heads_k, 1, seqlen_k_rounded});
            } else if (num_heads_bias == num_heads) {
                bias = bias.reshape({batch_size_bias_og, num_heads_k, ngroups, seqlen_k_rounded});
            }
        }
        num_heads_mask = has_mask ? ((num_heads_mask == num_heads) ? num_heads_k : num_heads_mask) : 1;
        num_heads_bias = has_bias ? ((num_heads_bias == num_heads) ? num_heads_k : num_heads_bias) : 1;
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    } else {
        out = torch::empty_like(q);
    }

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    if (return_softmax) {
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    } else {
        p = torch::empty({0}, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        seqlen_q, seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k, num_heads_mask, num_heads_bias,
        head_size, head_size_rounded,
        q, k, v, mask, bias, out,
        /*cu_seqlens_q_d=*/nullptr,
        /*cu_seqlens_k_d=*/nullptr,
        /*seqused_k=*/nullptr,
        return_softmax ? p.data_ptr() : nullptr,
        softmax_lse.data_ptr(),
        softmax_scale,
        is_causal,
        softcap,
        has_mask,
        has_bias
    );

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, /*num_splits*/ 0, get_num_sm(get_current_device()), opts
    );

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        if (has_mask) {
            mask = mask.reshape({batch_size_mask_og, num_heads_mask_og, 1, seqlen_k_rounded});
        }
        if (has_bias) {
            bias = bias.reshape({batch_size_bias_og, num_heads_bias_og, 1, seqlen_k_rounded});
        }
    }

    return {out, softmax_lse, p};
}

std::vector<at::Tensor>
mha_varlen_fwd(
    at::Tensor &q,                                  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,                            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor &v,                            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<at::Tensor> &out_,                // total_q x num_heads x head_size
    const at::Tensor &cu_seqlens_q,                 // b+1
    const at::Tensor &cu_seqlens_k,                 // b+1
    std::optional<at::Tensor> &seqused_k,           // b. If given, only this many elements of each batch element's keys are used.
    std::optional<const at::Tensor> &leftpad_k_,    // batch_size
    std::optional<at::Tensor> &block_table_,        // batch_size x max_num_blocks_per_seq
    int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    const float softcap,
    const bool return_softmax
) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashDynamicMaskAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "FlashDynamicMaskAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);

    auto opts = q.options();

    bool has_mask = false;
    at::Tensor mask;
    mask = torch::empty({0}, opts);
    bool has_bias = false;
    at::Tensor bias;
    bias = torch::empty({0}, opts);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);
    int num_heads_mask = has_mask ? mask.size(1) : 1;
    int num_heads_bias = has_bias ? bias.size(1) : 1;

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

    if (max_seqlen_q == 1) { is_causal = false; }   // is_causal=true is the same as is_causal=false in this case

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashDynamicMaskAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    if (return_softmax) {
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    } else {
        p = torch::empty({0}, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) { p.zero_(); }
    }

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q, max_seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k, num_heads_mask, num_heads_bias,
        head_size, head_size_rounded,
        q, k, v, mask, bias, out,
        cu_seqlens_q_d,
        cu_seqlens_k.data_ptr(),
        seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
        return_softmax ? p.data_ptr() : nullptr,
        softmax_lse.data_ptr(),
        softmax_scale,
        is_causal,
        softcap,
        has_mask,
        has_bias,
        seqlenq_ngroups_swapped,
        /*unpadded_lse*/true
    );
    params.total_q = total_q;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }
    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        std::tie(softmax_lse_accum, out_accum) =
            set_params_splitkv(
                params, batch_size, num_heads, head_size,
                max_seqlen_k, max_seqlen_q, head_size_rounded,
                /*num_splits*/ 0, get_num_sm(get_current_device()), opts
            );
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream, paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
    }

    return { out, softmax_lse, p };
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(params.has_mask, Has_mask, [&] {
                    BOOL_SWITCH(params.has_bias, Has_bias, [&] {
                        run_mha_bwd_<elem_type, kHeadDim, Is_causal, Has_mask, Has_bias>(params, stream);
                    });
                });
            });
        });
    });
}

std::vector<at::Tensor>
mha_bwd(
    const at::Tensor &dout,                         // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
    const at::Tensor &q,                            // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &k,                            // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &v,                            // batch_size x seqlen_k x num_heads_k x head_size
    const std::optional<at::Tensor> &mask_,         // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x {seqlen_k|1}
    const std::optional<at::Tensor> &bias_,         // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x {seqlen_k|1}
    const at::Tensor &out,                          // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &softmax_lse,                  // b x h x seqlen_q
    std::optional<at::Tensor> &dq_,                 // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor> &dk_,                 // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor> &dv_,                 // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor> &dbias_,              // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x {seqlen_k|1}
    const float softmax_scale,
    const bool is_causal,
    const float softcap,
    const bool deterministic
) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash dynamic mask attention build does not support backward.");
    #endif

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashDynamicMaskAttention only supports Ampere GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "FlashDynamicMaskAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    auto opts = q.options();

    bool has_mask = mask_.has_value();
    at::Tensor mask;
    if (has_mask) {
        mask = mask_.value();
        TORCH_CHECK(mask.dtype() == torch::kBool, "mask must have dtype bool");
        CHECK_DEVICE(mask);
        TORCH_CHECK(mask.dim() == 4, "mask must have 4 dimensions with shape (batch_size, nheads, seqlen_q, seqlen_k)");
        TORCH_CHECK(mask.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    } else {
        mask = torch::empty({0}, opts);
    }
    bool has_bias = bias_.has_value();
    at::Tensor bias;
    if (has_bias) {
        bias = bias_.value();
        TORCH_CHECK(bias.dtype() == q_dtype, "bias must have the same dtype as inputs");
        CHECK_DEVICE(bias);
        TORCH_CHECK(bias.dim() == 4, "bias must have 4 dimensions with shape (batch_size, nheads, seqlen_q, seqlen_k)");
        TORCH_CHECK(bias.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    } else {
        bias = torch::empty({0}, opts);
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    int num_heads_mask = has_mask ? mask.size(1) : 1;
    int num_heads_bias = has_bias ? bias.size(1) : 1;
    int batch_size_mask = has_mask ? mask.size(0) : batch_size;
    int batch_size_bias = has_bias ? bias.size(0) : batch_size;
    int seqlen_q_mask = has_mask ? mask.size(2) : seqlen_q;
    int seqlen_q_bias = has_bias ? bias.size(2) : seqlen_q;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashDynamicMaskAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (has_mask) {
        TORCH_CHECK(mask.size(0) == batch_size || mask.size(0) == 1, "Batch dimension in mask must be 1 or equal to batch size");
        TORCH_CHECK(num_heads_mask == 1 || num_heads_mask == num_heads_k || num_heads_mask == num_heads, "Number of heads in mask must be 1, h_k or h");
        TORCH_CHECK(mask.size(2) == 1 || mask.size(2) == seqlen_q, "Query length dimension in mask must be 1 or equal to seqlen_q");
        TORCH_CHECK(mask.size(3) == seqlen_k_rounded, "Key length dimension in mask must be seqlen_k_rounded");
    }
    if (has_bias) {
        TORCH_CHECK(bias.size(0) == batch_size || bias.size(0) == 1, "Batch dimension in bias must be 1 or equal to batch size");
        TORCH_CHECK(num_heads_bias == 1 || num_heads_bias == num_heads_k || num_heads_bias == num_heads, "Number of heads in bias must be 1, h_k or h");
        TORCH_CHECK(bias.size(2) == 1 || bias.size(2) == seqlen_q, "Query length dimension in bias must be 1 or equal to seqlen_q");
        TORCH_CHECK(bias.size(3) == seqlen_k_rounded, "Key length dimension in bias must be seqlen_k_rounded");
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor dq, dk, dv, dbias;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }
    if (has_bias) {
        if (dbias_.has_value()) {
            dbias = dbias_.value();
            TORCH_CHECK(dbias.dtype() == q_dtype, "dbias must have the same dtype as q");
            CHECK_DEVICE(dbias);
            TORCH_CHECK(dbias.dim() == 4, "dbias must have 4 dimensions with shape (batch_size, nheads, seqlen_q, seqlen_k)");
            TORCH_CHECK(dbias.stride(-1) == 1, "dbias must have contiguous last dimension");
            TORCH_CHECK(dbias.size(0) == batch_size || dbias.size(0) == 1, "Batch dimension in dbias must be 1 or equal to batch size");
            TORCH_CHECK(dbias.size(1) == num_heads || dbias.size(1) == num_heads_k || dbias.size(1) == 1, "Number of heads in dbias must be 1, h_k or h");
            TORCH_CHECK(dbias.size(2) == seqlen_q || dbias.size(2) == 1, "Query length dimension in dbias must be 1 or equal to seqlen_q");
            TORCH_CHECK(dbias.size(3) == seqlen_k_rounded, "Key length dimension in dbias must be seqlen_k_rounded");
        } else {
            dbias = torch::empty({batch_size_bias, num_heads_bias, seqlen_q_bias, seqlen_k_rounded}, opts);
        }
    } else {
        dbias = torch::empty({0}, opts);
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded, dbias_expanded;
    dk_expanded = num_heads_k != num_heads  // MQA / GQA
        ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts)
        : dk;
    dv_expanded = num_heads_k != num_heads  // MQA / GQA
        ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts)
        : dv;
    dbias_expanded = has_bias
        ? (num_heads_bias != num_heads || batch_size_bias == 1 || seqlen_q_bias == 1)     // MQA / GQA or dbias has different batch size or seqlen_q
            ? (seqlen_q_bias == 1)
                ? torch::zeros({batch_size, num_heads, 1, seqlen_k_rounded}, opts.dtype(at::kFloat))
                : torch::zeros({batch_size, num_heads, seqlen_q, seqlen_k_rounded}, opts)
            : dbias
        : torch::empty({0}, opts);
    bool accum_dbias = has_bias && (seqlen_q_bias == 1 && seqlen_q != 1);

    Flash_bwd_params params;

    set_params_dgrad(
        params,
        batch_size,
        seqlen_q, seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k, num_heads_mask, num_heads_bias,
        head_size, head_size_rounded,
        q, k, v, mask, bias, out,
        dout, dq, dk_expanded, dv_expanded, dbias_expanded,
        nullptr,
        nullptr,
        loop ? dq_accum.data_ptr() : nullptr,
        // loop ? dk_accum.data_ptr() : nullptr,
        // loop ? dv_accum.data_ptr() : nullptr,
        nullptr,
        nullptr,
        softmax_lse.data_ptr(),
        softmax_d.data_ptr(),
        softmax_scale,
        is_causal,
        softcap,
        has_mask,
        has_bias,
        accum_dbias,
        deterministic,
        /*unpadded_lse*/false
    );
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    auto launch = &run_mha_bwd;

    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        dbias_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }
    // For MQA/GQA or dbias has different batch size or seqlen_q, we need to sum dbias across the groups, batch and seqlen_q
    if (has_bias) {
        if (num_heads_bias != num_heads && batch_size_bias == batch_size && seqlen_q_bias == seqlen_q) {
            at::sum_out(dbias, at::reshape(dbias_expanded, {batch_size, num_heads_bias, num_heads / num_heads_bias, seqlen_q, seqlen_k_rounded}), {2});
        } else {
            if (accum_dbias) {
                dbias_expanded = at::sum(at::reshape(dbias_expanded, {batch_size, num_heads_bias, num_heads / num_heads_bias, 1, seqlen_k_rounded}), {2});
            } else {
                dbias_expanded = at::sum(at::reshape(dbias_expanded, {batch_size, num_heads_bias, num_heads / num_heads_bias, seqlen_q, seqlen_k_rounded}), {2});
            }
            if (batch_size_bias == 1) {
                dbias_expanded = at::sum(dbias_expanded, {0}, true);
            }
            dbias.copy_(dbias_expanded);
        }
    }

    return {dq, dk, dv, dbias, softmax_d};
}

std::vector<at::Tensor>
mha_varlen_bwd(
    const at::Tensor &dout,                         // total_q x num_heads, x head_size
    const at::Tensor &q,                            // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,                            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &v,                            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &out,                          // total_q x num_heads x head_size
    const at::Tensor &softmax_lse,                  // h x total_q, softmax logsumexp
    std::optional<at::Tensor> &dq_,                 // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor> &dk_,                 // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor> &dv_,                 // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,                 // b+1
    const at::Tensor &cu_seqlens_k,                 // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,                         // max sequence length to choose the kernel
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    const float softcap,
    const bool deterministic
) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash dynamic mask attention build does not support backward.");
    #endif

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashDynamicMaskAttention only supports Ampere GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "FlashDynamicMaskAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);

    auto opts = q.options();

    bool has_mask = false;
    at::Tensor mask;
    mask = torch::empty({0}, opts);
    bool has_bias = false;
    at::Tensor bias;
    bias = torch::empty({0}, opts);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    const int num_heads_mask = has_mask ? mask.size(1) : 1;
    const int num_heads_bias = has_bias ? bias.size(1) : 1;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashDynamicMaskAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv, dbias;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }
    dbias = torch::empty({0}, opts);

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
        // because that would be too large if there is a very long sequence and the rest of the sequences are short.
        // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
        // Note that 128 is the max block size on the seqlen_q dimension.
        // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
        // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
        // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
        // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
        // Same holds for softmax_d, since LSE is stored in unpadded format.
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded, dbias_expanded;
    if (num_heads_k != num_heads) {     // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        // dbias_expanded = torch::empty({total_q, num_heads, max_seqlen_k}, opts);
        dbias_expanded = torch::empty({0}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
        dbias_expanded = dbias;
    }

    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        // dbias_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params params;

    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q, max_seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        num_heads, num_heads_k, num_heads_mask, num_heads_bias,
        head_size, head_size_rounded,
        q, k, v, mask, bias, out,
        dout, dq, dk_expanded, dv_expanded, dbias_expanded,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        loop ? dq_accum.data_ptr() : nullptr,
        nullptr,
        nullptr,
        softmax_lse.data_ptr(),
        softmax_d.data_ptr(),
        softmax_scale,
        is_causal,
        softcap,
        has_mask,
        has_bias,
        /*accum_dbias*/false,
        deterministic,
        /*unpadded_lse*/true
    );
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total_q;

    auto launch = &run_mha_bwd;

    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        // dbias_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        // at::sum_out(dbias, at::reshape(dbias_expanded, {total_q, num_heads_k, num_heads / num_heads_k, max_seqlen_k}), {2});
    }

    return { dq, dk, dv, softmax_d };
}

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDynamicMaskAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass with variable length");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass with variable length");
}
