/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Bingheng Wu and Tri Dao.
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

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 128  // Common CUDA thread block size (multiple of 32)
#endif

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 4
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
        // if (isfinite(a.value) && isfinite(b.value)) {
        //     return a.value > b.value;
        // } else if (isfinite(a.value)) {
        //     return true;  // a is valid, b is not
        // } else if (isfinite(b.value)) {
        //     return false; // b is valid, a is not
        // } else {
        //     return a.col_index < b.col_index; // Compare indices if both are invalid
        // }
        return a.value > b.value;  // Descending order
    }
};

template <bool Is_causal, int BlockThreads>
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
        TensorZeroHold &tZeroHold,                  // Zero-hold states tensor (3D)
        TensorActiveIndices &tActiveIndices,        // Active indices tensor (3D)   
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(TensorZeroHold::rank == 3, "tZeroHold must be 3D Tensor");
        static_assert(TensorActiveIndices::rank == 3, "tActiveIndices must be 3D Tensor");
        static_assert(decltype(size<0>(tZeroHold))::value == 4, "First dimension must be 4");

        using ElementZeroHold = typename TensorZeroHold::value_type;
        using ElementActiveIndices = typename TensorActiveIndices::value_type;
        
        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor zero_hold = make_tensor(tZeroHold.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tZeroHold.layout()));
        Tensor active_indices = make_tensor(tActiveIndices.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tActiveIndices.layout()));

        const int tid = threadIdx.x;
        const int lane_id = tid % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

        // Initialize active indices based on validity and causal mask
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(zero_hold); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(zero_hold); ++i) {
                const int row_idx = row_idx_base + i * 8;
                // Skip if out of bounds
                if (row_idx >= max_seqlen_q) continue;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        
                        bool valid = (col_idx < max_seqlen_k);
                        bool causal_masked = Is_causal && (col_idx > row_idx);
                        
                        // Mark as active if valid and not causally masked
                        active_indices(coord) = valid && !causal_masked;
                        
                        // Clear zero_hold values for invalid or causally masked positions
                        if (!valid || causal_masked) {
                            zero_hold(coord) = ElementZeroHold(-INFINITY);
                        }
                    }
                }
            }
        }
        __syncthreads();

        // if keep_window_size >= max_seqlen_k, skip top-k
        if (keep_window_size <= 0 || keep_window_size >= max_seqlen_k) {
            return;
        }

        // Declare shared memory for BlockMergeSort at block scope
        using BlockMergeSortT = cub::BlockMergeSort<TopKPair<ElementZeroHold>, BlockThreads, ITEMS_PER_THREAD>;
        __shared__ typename BlockMergeSortT::TempStorage temp_storage;
        // Process each row with TopK sorting
        for (int mi = 0; mi < size<0, 1>(zero_hold); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            for (int i = 0; i < size<0, 0>(zero_hold); ++i) {
                const int row_idx = row_idx_base + i * 8;
                if (row_idx >= max_seqlen_q) continue;
                
                // Step 1: Thread-local storage for collecting current row elements
                TopKPair<ElementZeroHold> thread_data[ITEMS_PER_THREAD];
                
                // Initialize all elements as invalid
                for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                    thread_data[item] = TopKPair<ElementZeroHold>();
                }
                
                // Collect data from current row
                for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                    int global_idx = tid * ITEMS_PER_THREAD + item;
                    
                    if (global_idx < max_seqlen_k) {
                        // Find element with column index = global_idx in current row
                        for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                                const int col_idx = col_idx_base + j;
                                if (col_idx == global_idx) {
                                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                                    
                                    // If active, collect its value and index
                                    if (active_indices(coord)) {
                                        ElementZeroHold val = zero_hold(coord);
                                        thread_data[item] = TopKPair<ElementZeroHold>(val, col_idx);
                                    }
                                    break;  // Found the element, no need to continue
                                }
                            }
                        }
                    }
                }
                
                // Step 2: Block-wide collaborative sorting with explicit comparator
                DescendingComparator<ElementZeroHold> comp;
                BlockMergeSortT(temp_storage).Sort(thread_data, comp);
                __syncthreads();  // Ensure sorting is complete
                
                // Step 3: Update active_indices - keep only topk
                // Traverse each coordinate and check if its col_idx is in topk
                for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        
                        // If current position is active, check if it's in topk
                        if (active_indices(coord)) {
                            // Check if this element is in thread's own topk data
                            bool is_in_topk = false;
                            
                            for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                                // Global position in sorted order
                                int global_pos = tid * ITEMS_PER_THREAD + item;
                                
                                // Only elements with global_pos < keep_window_size are topk
                                if (global_pos < keep_window_size && 
                                    thread_data[item].col_index == col_idx) {
                                    is_in_topk = true;
                                    break;
                                }
                            }
                            
                            // If not in topk, set as inactive
                            if (!is_in_topk) {
                                active_indices(coord) = false;
                            }
                        }
                    }
                }
                __syncthreads();  // Ensure row processing is complete
            }
        }
    }

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
    __forceinline__ __device__ void apply_mask(
        Tensor<Engine0, Layout0> &tensor_,          // acc_s (attention scores, 3D)
        Tensor<Engine1, Layout1> &tZeroHold,        // Zero-hold states (3D)
        Tensor<Engine2, Layout2> &tActiveIndices,   // Active indices (3D)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(Layout0::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(Layout1::rank == 3, "tZeroHold must be 3D Tensor");
        static_assert(Layout2::rank == 3, "tActiveIndices must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        
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
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // bounds checking for row_idx and col_idx
                        bool valid = (row_idx < max_seqlen_q) && (col_idx < max_seqlen_k);
                        bool is_active = valid && active_indices(coord);
                        if (is_active) {
                            // Apply scaling and zero-hold
                            auto zero_hold_val = zero_hold(coord);
                            tensor(coord) = tensor(coord) * scale_softmax + zero_hold_val;
                        } else {
                            // Non-active positions or out-of-bounds set to -INFINITY
                            tensor(coord) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
