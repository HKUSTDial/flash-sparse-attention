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
#define CHECK_SHAPE(x, ...)                                                            \
    TORCH_CHECK(                                                                       \
        x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                                \
        #x " must have shape (" #__VA_ARGS__ ")"                                       \
    )
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

void set_params_fprop(
    Flash_fwd_params &params,
    // sizes
    const size_t batch_size,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_b,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor b,
    at::Tensor out,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *seqused_k,
    void *p_d,
    void *softmax_lse_d,
    float softmax_scale,
    bool is_causal,
    const float softcap,
    bool seqlenq_ngroups_swapped = false,
    const bool unpadded_lse = false
) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.b_ptr = b.data_ptr();
    params.o_ptr = out.data_ptr();

    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_row_stride = k.stride(-3);
    params.k_head_stride = k.stride(-2);
    params.v_row_stride = v.stride(-3);
    params.v_head_stride = v.stride(-2);
    params.b_head_stride = b.size(-3) == 1 ? 0 : b.stride(-3);
    params.b_row_stride = b.size(-2) == 1 ? 0 : b.stride(-2);
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.b_batch_stride = b.size(0) == 1 ? 0 : b.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
            params.q_batch_stride *= seqlen_q;
            params.b_batch_stride *= seqlen_q;
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
    params.b = batch_size;
    params.h = h;
    params.h_k = h_k;
    params.h_b = h_b;
    params.h_h_k_ratio = h / h_k;
    params.h_h_b_ratio = h / h_b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(
        softcap <= 0.0,
        "This flash sparse attention build does not support softcap."
    );
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
    TORCH_CHECK(
        d == d_rounded,
        "This flash sparse attention build does not support headdim not being a "
        "multiple of 32."
    );
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(
    Flash_bwd_params &params,
    // sizes
    const size_t batch_size,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_b,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor b,
    const at::Tensor out,
    const at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor db,
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
    bool deterministic,
    const bool unpadded_lse
) {
    set_params_fprop(
        params,
        batch_size,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        h,
        h_k,
        h_b,
        d,
        d_rounded,
        q,
        k,
        v,
        b,
        out,
        cu_seqlens_q_d,
        cu_seqlens_k_d,
        nullptr,
        nullptr,
        softmax_lse_d,
        softmax_scale,
        is_causal,
        softcap,
        false, // seqlenq_ngroups_swapped
        unpadded_lse
    );

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.db_ptr = db.data_ptr();

    // All stride are in elements, not bytes.
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_row_stride = dq.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_row_stride = dk.stride(-3);
    params.dk_head_stride = dk.stride(-2);
    params.dv_row_stride = dv.stride(-3);
    params.dv_head_stride = dv.stride(-2);
    params.db_head_stride = db.size(-3) == 1 ? 0 : db.stride(-3);
    params.db_row_stride = db.size(-2) == 1 ? 0 : db.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
        params.db_batch_stride = db.size(0) == 1 ? 0 : db.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_mha_fwd(
    Flash_fwd_params &params,
    cudaStream_t stream,
    bool force_split_kernel = false
) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 &&
                    !force_split_kernel) { // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(
                        params,
                        stream
                    );
                }
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
inline int num_splits_heuristic(
    int batch_nheads_mblocks,
    int num_SMs,
    int num_n_blocks,
    int max_splits
) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) {
        return 1;
    }
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
        return num_splits == 1 ||
            ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
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
    // const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int block_n = 64;
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (num_splits < 1) {
        // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
        params.num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks,
            num_sm * 2,
            num_n_blocks,
            128
        );
    }
    if (params.num_splits > 1) {
        softmax_lse_accum = torch::empty(
            {params.num_splits, batch_size, num_heads, max_seqlen_q},
            opts.dtype(at::kFloat)
        );
        out_accum = torch::empty(
            {params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded},
            opts.dtype(at::kFloat)
        );
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_ptr = out_accum.data_ptr();
    }
    TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");

    return std::make_tuple(softmax_lse_accum, out_accum);
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,                   // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,             // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,             // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    at::Tensor &b,                   // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x round_multiple(seqlen_k, 128)
    std::optional<at::Tensor> &out_, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    const bool return_softmax
) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(
        is_sm8x_min,
        "FlashSparseAttention only supports Ampere GPUs or newer."
    );

    auto q_dtype = q.dtype();
    TORCH_CHECK(
        q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
        "FlashSparseAttention only support fp16 and bf16 data type"
    );
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(b.dtype() == q_dtype, "query and bias must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(b);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(b.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    auto opts = q.options();
    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    int num_heads_b = b.size(1);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(
        head_size <= 256,
        "FlashSparseAttention forward only supports head dimension at most 256"
    );
    TORCH_CHECK(
        head_size % 8 == 0,
        "query, key, value, and out_ must have a head_size that is a multiple of 8"
    );
    TORCH_CHECK(
        num_heads % num_heads_k == 0,
        "Number of heads in key/value must divide number of heads in query"
    );

    TORCH_CHECK(
        b.size(0) == batch_size || b.size(0) == 1,
        "Batch dimension in bias must be 1 or equal to batch size"
    );
    TORCH_CHECK(
        num_heads_b == 1 || num_heads_b == num_heads_k || num_heads_b == num_heads,
        "Number of heads in bias must be 1, h_k or h"
    );
    TORCH_CHECK(
        b.size(2) == 1 || b.size(2) == seqlen_q,
        "Query length dimension in bias must be 1 or equal to seqlen_q"
    );
    TORCH_CHECK(
        b.size(3) == seqlen_k_rounded,
        "Key length dimension in bias must be seqlen_k_rounded"
    );

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1) {
        is_causal = false;
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && head_size % 8 == 0;
    const int ngroups = num_heads / num_heads_k;
    const int batch_size_b_og = b.size(0);
    const int num_heads_b_og = num_heads_b;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        if (num_heads_b == 1) {
            b = b.reshape({batch_size_b_og, 1, 1, seqlen_k_rounded});
        } else if (num_heads_b == num_heads_k) {
            b = b.reshape({batch_size_b_og, num_heads_k, 1, seqlen_k_rounded});
        } else if (num_heads_b == num_heads) {
            b = b.reshape({batch_size_b_og, num_heads_k, ngroups, seqlen_k_rounded});
        }
        num_heads_b = (num_heads_b == num_heads) ? num_heads_k : num_heads_b;
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(
            out.dtype() == q_dtype,
            "Output must have the same dtype as inputs"
        );
        CHECK_DEVICE(out);
        TORCH_CHECK(
            out.stride(-1) == 1,
            "Output tensor must have contiguous last dimension"
        );
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
        p = torch::empty(
            {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded},
            opts
        );
    } else {
        p = torch::empty({0}, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        num_heads,
        num_heads_k,
        num_heads_b,
        head_size,
        head_size_rounded,
        q,
        k,
        v,
        b,
        out,
        /*cu_seqlens_q_d=*/nullptr,
        /*cu_seqlens_k_d=*/nullptr,
        /*seqused_k=*/nullptr,
        return_softmax ? p.data_ptr() : nullptr,
        softmax_lse.data_ptr(),
        softmax_scale,
        is_causal,
        softcap
    );

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params,
        batch_size,
        num_heads,
        head_size,
        seqlen_k,
        seqlen_q,
        head_size_rounded,
        /*num_splits*/ 0,
        get_num_sm(get_current_device()),
        opts
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
        out = out.transpose(1, 2).reshape(
            {batch_size, 1, num_heads_k * seqlen_q, head_size}
        );
        q = q.transpose(1, 2).reshape(
            {batch_size, 1, num_heads_k * seqlen_q, head_size}
        );
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        b = b.reshape({batch_size_b_og, num_heads_b_og, 1, seqlen_k_rounded});
    }

    return {out, softmax_lse, p};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

std::vector<at::Tensor> mha_bwd(
    const at::Tensor &dout,         // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
    const at::Tensor &q,            // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &k,            // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &v,            // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &b,            // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x {seqlen_k|1}
    const at::Tensor &out,          // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &softmax_lse,  // b x h x seqlen_q
    std::optional<at::Tensor> &dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor> &dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor> &dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor> &db_, // {batch_size|1} x {num_heads|num_heads_k|1} x {seqlen_q|1} x {seqlen_k|1}
    const float softmax_scale,
    const bool is_causal,
    const float softcap,
    const bool deterministic
) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
    TORCH_CHECK(false, "This flash sparse attention build does not support backward.");
    #endif

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(
        is_sm8x_min,
        "FlashSparseAttention only supports Ampere GPUs or newer."
    );

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(
        q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
        "FlashSparseAttention only support fp16 and bf16 data type"
    );
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(b.dtype() == q_dtype, "query and bias must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(b);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);
    CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(b.stride(-1) == 1, "bias tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    auto opts = q.options();
    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    int num_heads_b = b.size(1);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    int batch_size_db = b.size(0);
    int seqlen_q_db = b.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(
        head_size <= 256,
        "FlashSparseAttention backward only supports head dimension at most 256"
    );
    TORCH_CHECK(
        num_heads % num_heads_k == 0,
        "Number of heads in key/value must divide number of heads in query"
    );

    TORCH_CHECK(
        b.size(0) == batch_size || b.size(0) == 1,
        "Batch dimension in bias must be 1 or equal to batch size"
    );
    TORCH_CHECK(
        num_heads_b == 1 || num_heads_b == num_heads_k || num_heads_b == num_heads,
        "Number of heads in bias must be 1, h_k or h"
    );
    TORCH_CHECK(
        b.size(2) == 1 || b.size(2) == seqlen_q,
        "Query length dimension in bias must be 1 or equal to seqlen_q"
    );
    TORCH_CHECK(
        b.size(3) == seqlen_k_rounded,
        "Key length dimension in bias must be seqlen_k_rounded"
    );

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor dq, dk, dv, db;
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
    if (db_.has_value()) {
        db = db_.value();
        TORCH_CHECK(db.dtype() == q_dtype, "db must have the same dtype as q");
        CHECK_DEVICE(db);
        TORCH_CHECK(
            db.dim() == 4,
            "db must have 4 dimensions with shape (batch_size, nheads, seqlen_q, "
            "seqlen_k)"
        );
        TORCH_CHECK(db.stride(-1) == 1, "db must have contiguous last dimension");
        TORCH_CHECK(
            db.size(0) == batch_size || db.size(0) == 1,
            "Batch dimension in db must be 1 or equal to batch size"
        );
        TORCH_CHECK(
            db.size(1) == num_heads || db.size(1) == num_heads_k || db.size(1) == 1,
            "Number of heads in db must be 1, h_k or h"
        );
        TORCH_CHECK(
            db.size(2) == seqlen_q || db.size(2) == 1,
            "Query length dimension in db must be 1 or equal to seqlen_q"
        );
        TORCH_CHECK(
            db.size(3) == seqlen_k_rounded,
            "Key length dimension in db must be seqlen_k_rounded"
        );
    } else {
        db = torch::empty(
            {batch_size_db, num_heads_b, seqlen_q_db, seqlen_k_rounded},
            opts
        );
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));

    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty(
                {batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
                opts.dtype(at::kFloat)
            );
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros(
                {nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
                opts.dtype(at::kFloat)
            );
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded},
        // opts.dtype(at::kFloat)); dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
        // head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded, db_expanded;
    dk_expanded = num_heads_k != num_heads // MQA / GQA
        ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts)
        : dk;
    dv_expanded = num_heads_k != num_heads // MQA / GQA
        ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts)
        : dv;
    db_expanded = (num_heads_b != num_heads || batch_size_db == 1 || seqlen_q_db == 1)
        ? (seqlen_q_db == 1)
            ? torch::zeros({batch_size, num_heads, 1, seqlen_k_rounded}, opts)
            : torch::zeros({batch_size, num_heads, seqlen_q, seqlen_k_rounded}, opts)
        : db;

    Flash_bwd_params params;

    set_params_dgrad(
        params,
        batch_size,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        num_heads,
        num_heads_k,
        num_heads_b,
        head_size,
        head_size_rounded,
        q,
        k,
        v,
        b,
        out,
        dout,
        dq,
        dk_expanded,
        dv_expanded,
        db_expanded,
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
        deterministic,
        /*unpadded_lse*/ false
    );
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    auto launch = &run_mha_bwd;

    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        db_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(
            dk,
            at::reshape(
                dk_expanded,
                {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}
            ),
            {3}
        );
        at::sum_out(
            dv,
            at::reshape(
                dv_expanded,
                {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}
            ),
            {3}
        );
    }
    // For MQA/GQA or dbias has different batch size or seqlen_q, we need to sum dbias across the groups, batch and
    // seqlen_q
    if (num_heads_b != num_heads && batch_size_db == batch_size &&
        seqlen_q_db == seqlen_q) {
        at::sum_out(
            db,
            at::reshape(
                db_expanded,
                {batch_size,
                 num_heads_b,
                 num_heads / num_heads_b,
                 seqlen_q,
                 seqlen_k_rounded}
            ),
            {2}
        );
    } else {
        if (seqlen_q_db == 1) {
            db_expanded = at::sum(
                at::reshape(
                    db_expanded,
                    {batch_size,
                     num_heads_b,
                     num_heads / num_heads_b,
                     1,
                     seqlen_k_rounded}
                ),
                {2}
            );
        } else {
            db_expanded = at::sum(
                at::reshape(
                    db_expanded,
                    {batch_size,
                     num_heads_b,
                     num_heads / num_heads_b,
                     seqlen_q,
                     seqlen_k_rounded}
                ),
                {2}
            );
        }
        if (batch_size_db == 1) {
            db_expanded = at::sum(db_expanded, {0}, true);
        }
        db.copy_(db_expanded);
    }

    return {dq, dk, dv, db, softmax_d};
}

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashSparseAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
}
