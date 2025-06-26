/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Yifan Wu and Bingheng Wu and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/fast_math.h>
#include <cub/block/block_merge_sort.cuh>

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 32
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

template <bool Is_causal>
struct DynamicMask {
    const int max_seqlen_k, max_seqlen_q;
    const int keep_window_size;

    __forceinline__ __device__ DynamicMask(
        const int max_seqlen_k,
        const int max_seqlen_q,
        const int keep_window_size
    )  // Constructor
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , keep_window_size(keep_window_size) {
    };

    template <bool Causal_mask=false, bool Is_even_MN=true, typename TensorType, typename ZOHType, typename ActiveMaskType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        ZOHType &tSrZOH,                            // ZOH states (MMA=4, MMA_M, MMA_N)
        ActiveMaskType &tSrAM,                      // Active Mask (MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(TensorType::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(ZOHType::rank == 3, "tZOH must be 3D Tensor");
        static_assert(ActiveMaskType::rank == 3, "tActiveMask must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        const bool Need_masking = Causal_mask || !Is_even_MN || (keep_window_size < max_seqlen_k);
        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        Tensor zoh = make_tensor(tSrZOH.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrZOH.layout()));
        Tensor active_mask = make_tensor(tSrAM.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrAM.layout()));

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        if (Need_masking) {
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
                            bool inactive = (col_idx >= col_idx_limit) || (active_mask(coord) <= 0.0f);
                            if (inactive) {
                                tensor(coord) = -INFINITY;
                            } else {
                                // Apply scaling and zoh
                                tensor(coord) = tensor(coord) * scale_softmax + zoh(coord);
                            }
                        }
                    }
                }
            }
        } else {
            // If no masking is needed, just scale the tensor and add zoh
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                // const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    // const int row_idx = row_idx_base + i * 8;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        // const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            // const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            tensor(coord) = tensor(coord) * scale_softmax + zoh(coord);
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
