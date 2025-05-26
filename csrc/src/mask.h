/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/fast_math.h>

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 128  // Common CUDA thread block size (multiple of 32)
#endif

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 4
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

// Apply causal masking for dynamic mask with 1 row block
template <typename TensorZeroHold, typename TensorActiveIndices, bool Is_causal>
__forceinline__ __device__ void apply_causal_mask_1rowblock(
    TensorZeroHold zero_hold_states,    // Zero-hold states tensor view for one row
    TensorActiveIndices active_indices,      // Active indices tensor view for one row
    int query_idx,                              // Current query position (row index)
    int key_len                                 // Key length (sequence length for keys)
) {
    static_assert(TensorZeroHold::rank == 1, "Tensors must be 1D");
    static_assert(TensorActiveIndices::rank == 1, "Tensors must be 1D");
    using Element = typename TensorZeroHold::value_type;

    if constexpr (Is_causal) {
        const int tid = threadIdx.x;
        const int stride = blockDim.x;
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
            const bool is_masked = k_idx > query_idx;
            if (is_masked) {
                zero_hold_states(k_idx) = Element(0.0f);
                active_indices(k_idx) = false;
            }
        }
    }
}

// Apply top-k selection for dynamic mask with 1 row block
template <typename TensorZeroHold, typename TensorActiveIndices, int BlockThreads>
__forceinline__ __device__ void apply_topk_window_selection_1rowblock(
    TensorZeroHold zero_hold_states,              // Zero-hold states for in-place modification [key_len]
    TensorActiveIndices active_indices,                   // Active indices to mark selected positions [key_len]
    const int key_len,                                      // Key length
    const int keep_window_size                              // Maximum window size to keep
) {
    static_assert(TensorZeroHold::rank == 1, "Tensors must be 1D");
    static_assert(TensorActiveIndices::rank == 1, "Tensors must be 1D");
    using Element = typename TensorZeroHold::value_type;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Skip if no reduction needed
    if (key_len <= keep_window_size) {
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
            active_indices(k_idx) = (static_cast<float>(zero_hold_states(k_idx)) != 0.0f);
        }
        return;
    }
    
    // Shared memory for reduction
    __shared__ float s_max_vals[BlockThreads];
    __shared__ int s_max_indices[BlockThreads];
    
    // Initialize active_indices to false
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
        active_indices(k_idx) = false;
    }
    __syncthreads();
    
    // Iteratively select top-k elements
    for (int selected_count = 0; selected_count < keep_window_size && selected_count < key_len; ++selected_count) {
        float thread_max = -FLT_MAX;
        int thread_max_idx = -1;
        
        // Find maximum among non-selected elements
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
            if (!active_indices(k_idx)) {  // Not yet selected
                float current_val = static_cast<float>(zero_hold_states(k_idx));
                if (current_val > thread_max) {
                    thread_max = current_val;
                    thread_max_idx = k_idx;
                }
            }
        }

        // Store thread-local maximum
        s_max_vals[tid] = thread_max;
        s_max_indices[tid] = thread_max_idx;
        __syncthreads();
        
        // Parallel reduction to find global maximum
        #pragma unroll
        for (int reduction_stride = BlockThreads / 2; reduction_stride > 0; reduction_stride >>= 1) {
            if (tid < reduction_stride && s_max_vals[tid] < s_max_vals[tid + reduction_stride]) {
                s_max_vals[tid] = s_max_vals[tid + reduction_stride];
                s_max_indices[tid] = s_max_indices[tid + reduction_stride];
            }
            __syncthreads();
        }
        
        // Mark the selected index as active
        if (tid == 0 && s_max_indices[0] >= 0) {
            active_indices[s_max_indices[0]] = true;
        }
        __syncthreads();
    }

    // Clear non-selected values
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
        if (!active_indices(k_idx)) {
            zero_hold_states(k_idx) = Element(0.0f);
        }
    }
}

// Apply dynamic mask with 1 row block
template <typename TensorZeroHold, typename TensorActiveIndices, bool Is_causal, int BlockThreads>
__forceinline__ __device__ void apply_dynamic_mask_1rowblock(
    TensorZeroHold zero_hold_states,            // Zero-hold states tensor view
    TensorActiveIndices active_indices,         // Active indices tensor view
    int query_idx,                              // Current query position (row index)
    const int key_len,                          // Sequence length for keys
    const int keep_window_size                  // Maximum window size to keep
) {
    static_assert(TensorZeroHold::rank == 1, "Tensors must be 1D");
    static_assert(TensorActiveIndices::rank == 1, "Tensors must be 1D");

    // Initialize all indices as active
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += stride) {
        active_indices(k_idx) = true;
    }
    __syncthreads();

    // Apply causal mask across the row
    apply_causal_mask_1rowblock<TensorZeroHold, TensorActiveIndices, Is_causal>(
        zero_hold_states,
        active_indices,
        query_idx, 
        key_len
    );
    __syncthreads();

    // Top-k window selection
    apply_topk_window_selection_1rowblock<TensorZeroHold, TensorActiveIndices, BlockThreads>(
        zero_hold_states,
        active_indices,
        key_len, 
        keep_window_size
    );
    __syncthreads();
}

// Struct wrapper for dynamic mask application
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

        // Apply top-k selection per row if needed
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(zero_hold); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(zero_hold); ++i) {
                const int row_idx = row_idx_base + i * 8;
                // Skip if out of bounds
                if (row_idx >= max_seqlen_q) continue;
            
                // Temporarily mark all active elements as inactive for selection
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        if (active_indices(coord)) {
                            active_indices(coord) = false;
                        }
                    }
                }
                __syncthreads();
                
                // Shared memory for reduction
                __shared__ float s_max_vals[BlockThreads];
                __shared__ int s_max_indices_nj[BlockThreads];
                __shared__ int s_max_indices_j[BlockThreads];
                
                // Iteratively select top-k elements
                for (int k = 0; k < keep_window_size; ++k) {
                    float thread_max = -FLT_MAX;
                    int thread_max_nj = -1;
                    int thread_max_j = -1;
                    
                    // Each thread finds its local maximum using the same loop structure
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                            const int col_idx = col_idx_base + j;
                            auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                            
                            bool valid = (col_idx < max_seqlen_k) && !(Is_causal && col_idx > row_idx);
                            float val = static_cast<float>(zero_hold(coord));
                            if (valid && !active_indices(coord) && !isinf(val) && val > thread_max) {
                                thread_max = val;
                                thread_max_nj = nj;
                                thread_max_j = j;
                            }
                        }
                    }
                    
                    // Store thread-local maximum
                    s_max_vals[tid] = thread_max;
                    s_max_indices_nj[tid] = thread_max_nj;
                    s_max_indices_j[tid] = thread_max_j;
                    __syncthreads();
                    
                    // Parallel reduction to find global maximum
                    for (int stride = BlockThreads / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) {
                            if (s_max_vals[tid] < s_max_vals[tid + stride]) {
                                s_max_vals[tid] = s_max_vals[tid + stride];
                                s_max_indices_nj[tid] = s_max_indices_nj[tid + stride];
                                s_max_indices_j[tid] = s_max_indices_j[tid + stride];
                            }
                        }
                        __syncthreads();
                    }
                    
                    // Mark the selected index as active
                    if (tid == 0 && s_max_indices_nj[0] >= 0 && s_max_indices_j[0] >= 0) {
                        auto coord = make_coord(make_coord(i, mi), make_coord(s_max_indices_j[0], s_max_indices_nj[0]));
                        active_indices(coord) = true;
                    }
                    __syncthreads();
                    
                    // Early exit if no more valid elements
                    if (s_max_vals[0] == -FLT_MAX) {
                        break;
                    }
                }
                
                // Clear non-selected values using the same loop structure
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(zero_hold); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(zero_hold); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        if (!active_indices(coord)) {
                            zero_hold(coord) = ElementZeroHold(-INFINITY);
                        }
                    }
                }
                __syncthreads();
                
            }
        }
    }


    template <
        bool Causal_mask=false, bool Is_even_MN=true,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2
    >
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

                        // 边界检查
                        bool valid = (row_idx < max_seqlen_q) && (col_idx < max_seqlen_k);
                        bool is_active = valid && active_indices(coord);
                        
                        if (is_active) {
                            // 应用 scaling 和 zero_hold mask
                            auto zero_hold_val = zero_hold(coord);
                            tensor(coord) = tensor(coord) * scale_softmax + zero_hold_val;
                        } else {
                            // 非活跃位置或超出边界设为 -INFINITY
                            tensor(coord) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
