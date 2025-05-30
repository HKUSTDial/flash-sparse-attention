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
#define ITEMS_PER_THREAD 16
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

// Value-Index pair for top-k selection
template<typename ValueType>
struct TopKPair {
    ValueType value;
    int col_index;
    
    __device__ __forceinline__ TopKPair() : value(ValueType(-INFINITY)), col_index(-1) {}
    __device__ __forceinline__ TopKPair(ValueType v, int idx) : value(v), col_index(idx) {}
    
    __device__ __forceinline__ bool is_valid() const {
        return col_index >= 0 && isfinite(value);
    }
};

// Comparison functor for descending sort (greater values first)
template<typename ValueType>
struct DescendingComparator {
    __device__ __forceinline__ bool operator()(const TopKPair<ValueType>& a, const TopKPair<ValueType>& b) const {
        if (isfinite(a.value) && isfinite(b.value)) {
            return a.value > b.value;           // Descending order
        } else if (isfinite(a.value)) {
            return true;                        // a is valid, b is not
        } else if (isfinite(b.value)) {
            return false;                       // b is valid, a is not
        } else {
            return a.col_index < b.col_index;   // Compare indices if both are invalid
        }
    }
};

template <bool Is_causal, int kNThreads>
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

    template <typename TensorZeroHold, typename TensorActiveIndices>
    __forceinline__ __device__ void get_active_zerohold(
        TensorZeroHold &gZeroHold,                  // Zero-hold states tensor (actual_seqlen_k)
        TensorActiveIndices &gActiveIndices,        // Active indices tensor (keep_window_size)
        const int row_idx                           // Row index offset
    ) {
        static_assert(TensorZeroHold::rank == 1, "gZeroHold must be 1D Tensor (actual_seqlen_k)");
        static_assert(TensorActiveIndices::rank == 1, "gActiveIndices must be 1D Tensor (keep_window_size)");
        // Skip if out of bounds
        if (row_idx >= max_seqlen_q) return;
        using ElementZeroHold = typename TensorZeroHold::value_type;

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        const int col_idx_limit = Is_causal ? min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;

        // Initialize all active indices as invalid
        for (int k_idx = tid; k_idx < keep_window_size; k_idx += num_threads) {
            gActiveIndices(k_idx) = -1;
        }
        __syncthreads();

        // If no valid elements, return early
        if (keep_window_size <= 0) {
            return;
        }

        // if keep_window_size >= col_idx_limit, use all indices
        if (keep_window_size >= col_idx_limit) {
            for (int k_idx = tid; k_idx < col_idx_limit && k_idx < keep_window_size; k_idx += num_threads) {
                gActiveIndices(k_idx) = k_idx;
            }
            __syncthreads();
            return;
        }
        
        // Initialize all elements as invalid
        constexpr int max_items_per_thread = ITEMS_PER_THREAD;
        TopKPair<ElementZeroHold> thread_data[max_items_per_thread];

        #pragma unroll
        for (int item = 0; item < max_items_per_thread; ++item) {
            thread_data[item] = TopKPair<ElementZeroHold>();
        }

        // Collect valid elements from current row
        #pragma unroll
        for (int item = 0; item < max_items_per_thread; ++item) {
            int global_k_idx = tid + item * num_threads;
            if (global_k_idx < col_idx_limit) {
                // Get the value from the zero-hold tensor
                ElementZeroHold val = gZeroHold(global_k_idx);
                thread_data[item] = TopKPair<ElementZeroHold>(val, global_k_idx);
            }
        }

        // Declare shared memory for BlockMergeSort at block scope
        using BlockMergeSortT = cub::BlockMergeSort<TopKPair<ElementZeroHold>, kNThreads, max_items_per_thread>;
        __shared__ typename BlockMergeSortT::TempStorage temp_storage;
        // Block-wide collaborative sorting with explicit comparator
        DescendingComparator<ElementZeroHold> comp;
        BlockMergeSortT(temp_storage).Sort(thread_data, comp);
        __syncthreads();

        // Mark top-k elements as active
        #pragma unroll
        for (int item = 0; item < max_items_per_thread; ++item) {
            int global_pos = tid * max_items_per_thread + item;
            if (global_pos < keep_window_size && thread_data[item].is_valid()) {
                gActiveIndices(global_pos) = thread_data[item].col_index;
            }
        }
        __syncthreads();
    }

    template <bool Causal_mask=false, bool Is_even_MN=true, typename TensorType, typename ZeroHoldType, typename ActiveIndicesType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, 3D)
        ZeroHoldType &tZeroHold,                    // Zero-hold states (3D)
        ActiveIndicesType &tActiveIndices,          // Active indices (3D)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(TensorType::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(ZeroHoldType::rank == 3, "tZeroHold must be 3D Tensor");
        static_assert(ActiveIndicesType::rank == 3, "tActiveIndices must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        if constexpr (Need_masking) {
            // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            Tensor zero_hold = make_tensor(tZeroHold.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tZeroHold.layout()));
            Tensor active_indices = make_tensor(tActiveIndices.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tActiveIndices.layout()));

            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit = Causal_mask ? min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            int active_col_idx = active_indices(coord);
                            bool is_boundary_valid = Is_even_MN ? true : (col_idx < max_seqlen_k);
                            bool is_active = (col_idx < col_idx_limit) && (active_col_idx >= 0) && is_boundary_valid;
                            if (is_active) {
                                // Apply scaling and zero-hold
                                tensor(coord) = tensor(coord) * scale_softmax + zero_hold(coord);
                            } else {
                                // Non-active positions or out-of-bounds set to -INFINITY
                                tensor(coord) = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
