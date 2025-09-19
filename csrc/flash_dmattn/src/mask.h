/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Yifan Wu and Bingheng Wu and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {

using namespace cute;

template <bool Causal_mask=false, bool Has_mask=true, bool Has_bias=true, typename TensorType, typename MaskType, typename BiasType>
__forceinline__ __device__ void apply_mask(
    TensorType &tensor,
    const MaskType &mask,
    const BiasType &bias,
    const float scale_softmax,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset,
    const int max_seqlen_q,
    const int warp_row_stride
) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(TensorType::rank == 2, "Only support 2D Tensor");
    if constexpr (Has_mask)
        static_assert(MaskType::rank == 2, "Only support 2D Mask");
    if constexpr (Has_bias)
        static_assert(BiasType::rank == 2, "Only support 2D Bias");

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

    if constexpr (Has_mask && Has_bias) {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        // Without the "make_coord" we get wrong results
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // Apply scaling and bias or masking
                        tensor(coord) = (col_idx >= col_idx_limit) || (!mask(coord))
                            ? -INFINITY
                            : tensor(coord) * scale_softmax + bias(coord);
                    }
                }
            }
        }
    } else if constexpr (Has_mask && !Has_bias) {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        // Without the "make_coord" we get wrong results
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // Apply scaling or masking
                        tensor(coord) = (col_idx >= col_idx_limit) || (!mask(coord))
                            ? -INFINITY
                            : tensor(coord) * scale_softmax;
                    }
                }
            }
        }
    } else if constexpr (!Has_mask && Has_bias) {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        // Without the "make_coord" we get wrong results
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // Apply scaling and bias
                        tensor(coord) = (col_idx >= col_idx_limit)
                            ? -INFINITY
                            : tensor(coord) * scale_softmax + bias(coord);
                    }
                }
            }
        }
    } else {    // !Has_mask && !Has_bias
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        // Without the "make_coord" we get wrong results
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // Apply scaling
                        tensor(coord) = (col_idx >= col_idx_limit)
                            ? -INFINITY
                            : tensor(coord) * scale_softmax;
                    }
                }
            }
        }
    }
}

struct Mask {
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Mask(
        const int max_seqlen_k,
        const int max_seqlen_q
    )  // Constructor
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    template <bool Causal_mask=false, bool Has_mask=true, bool Has_bias=true, typename TensorType, typename MaskType, typename BiasType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        const MaskType &mask_,                      // Attention Mask (MMA=4, MMA_M, MMA_N)
        const BiasType &bias_,                      // Attention Bias (MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(TensorType::rank == 3, "tensor_ must be 3D Tensor");
        if constexpr (Has_mask)
            static_assert(MaskType::rank == 3, "mask_ must be 3D Tensor");
        if constexpr (Has_bias)
            static_assert(BiasType::rank == 3, "Bias must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");

        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

        if constexpr (Has_mask && Has_bias) {
            Tensor mask = make_tensor(mask_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(mask_.layout()));
            Tensor bias = make_tensor(bias_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(bias_.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            // Apply scaling and bias or masking
                            tensor(coord) = (col_idx >= col_idx_limit) || (!mask(coord))
                                ? -INFINITY
                                : tensor(coord) * scale_softmax + bias(coord);
                        }
                    }
                }
            }
        } else if constexpr (Has_mask && !Has_bias) {
            Tensor mask = make_tensor(mask_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(mask_.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            // Apply scaling or masking
                            tensor(coord) = (col_idx >= col_idx_limit) || (!mask(coord))
                                ? -INFINITY
                                : tensor(coord) * scale_softmax;
                        }
                    }
                }
            }
        } else if constexpr (!Has_mask && Has_bias) {
            Tensor bias = make_tensor(bias_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(bias_.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            // Apply scaling and bias
                            tensor(coord) = (col_idx >= col_idx_limit)
                                ? -INFINITY
                                : tensor(coord) * scale_softmax + bias(coord);
                        }
                    }
                }
            }
        } else {    // !Has_mask && !Has_bias
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            // Apply scaling
                            tensor(coord) = (col_idx >= col_idx_limit)
                                ? -INFINITY
                                : tensor(coord) * scale_softmax;
                        }
                    }
                }
            }
        }   
    }
};

} // namespace FLASH_NAMESPACE
