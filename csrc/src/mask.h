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
#define BLOCK_THREADS 256  // Common CUDA thread block size (multiple of 32)
#endif

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 32
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

// Apply causal masking for dynamic mask with 1 row block
template <typename Element, bool Is_causal>
__forceinline__ __device__ void apply_causal_mask_1rowblock(
    Element* zero_hold_states,          // Zero-hold states for one query row [key_len]
    bool* active_indices,               // Active indices to mark masked positions [key_len]
    int query_idx,                      // Current query position (row index)
    int key_len                         // Key length (sequence length for keys)
) {
    if constexpr (Is_causal) {
        int tid = threadIdx.x;
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
            const bool is_masked = k_idx > query_idx;
            if (is_masked) {
                zero_hold_states[k_idx] = Element(0.0f);
                active_indices[k_idx] = false;
            }
        }
    }
}

// Apply top-k selection for dynamic mask with 1 row block
template <typename Element>
__forceinline__ __device__ void apply_topk_window_selection_1rowblock(
    Element* zero_hold_states,              // Zero-hold states for in-place modification [key_len]
    bool* active_indices,                   // Active indices to mark selected positions [key_len]
    const int key_len,                      // Key length
    const int keep_window_size              // Maximum window size to keep
) {
    using namespace cute;
    using namespace cutlass;
    
    // Skip if no reduction needed
    if (key_len <= keep_window_size) {
        // Mark all as active
        int tid = threadIdx.x;
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
            active_indices[k_idx] = (static_cast<float>(zero_hold_states[k_idx]) != Element(0.0f));
        }
        return;
    }

    int tid = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float s_max_vals[BLOCK_THREADS];
    __shared__ int s_max_indices[BLOCK_THREADS];
    
    // Temporary array to track selected indices
    __shared__ int selected_indices[ITEMS_PER_THREAD * BLOCK_THREADS];
    
    // Initialize active_indices to false
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        active_indices[k_idx] = false;
    }
    __syncthreads();
    
    // Iteratively select top-k elements
    for (int selected_count = 0; selected_count < keep_window_size && selected_count < key_len; ++selected_count) {
        float thread_max = -FLT_MAX;
        int thread_max_idx = -1;
        
        // Find maximum among non-selected elements
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
            if (!active_indices[k_idx]) {  // Not yet selected
                float current_val = static_cast<float>(zero_hold_states[k_idx]);
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
        for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_max_vals[tid] < s_max_vals[tid + stride]) {
                    s_max_vals[tid] = s_max_vals[tid + stride];
                    s_max_indices[tid] = s_max_indices[tid + stride];
                }
            }
            __syncthreads();
        }
        
        // Mark the selected index as active
        if (tid == 0 && s_max_indices[0] >= 0) {
            selected_indices[selected_count] = s_max_indices[0];
            active_indices[s_max_indices[0]] = true;
        }
        __syncthreads();
    }

    // Clear non-selected values
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        if (!active_indices[k_idx]) {
            zero_hold_states[k_idx] = Element(0.0f);
        }
    }
    __syncthreads();
}

// Apply dynamic mask with 1 row block
template <
    typename Engine, typename Layout,               // tensor (in-place)
    typename Element, bool Is_causal
>
__forceinline__ __device__ void apply_dynamic_mask_1rowblock(
    Tensor<Engine, Layout> &zero_hold_states,       // In-place tensor [key_len]
    bool* active_indices,                           // Active indices [key_len]
    int query_idx,                                  // Current query position (row index)
    const int key_len,                              // Sequence length for keys
    const int keep_window_size                      // Maximum window size to keep
) {
    static_assert(Layout::rank == 1, "Tensor must be 1D");

    // Initialize all indices as active
    int tid = threadIdx.x;
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        active_indices[k_idx] = true;
    }
    __syncthreads();

    // Apply causal mask across the row
    apply_causal_mask_1rowblock<Element, Is_causal>(
        zero_hold_states.data().get(),
        active_indices,
        query_idx, key_len
    );
    __syncthreads();

    // Top-k window selection
    apply_topk_window_selection_1rowblock<Element>(
        zero_hold_states.data().get(),
        active_indices,
        key_len, keep_window_size
    );
    __syncthreads();
}

// Struct wrapper for dynamic mask application
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

    template <
        typename Engine, typename Layout,
        typename Element, bool Is_causal
    >
    __forceinline__ __device__ void apply_mask_1rowblock(
        Tensor<Engine, Layout> &zero_hold_states,   // In-place tensor
        bool* active_indices,                       // Active indices (bool)
        int query_idx,                              // Query index
        int key_len                                 // Key length
    ) {
        apply_dynamic_mask_1rowblock<
            Engine, Layout,
            Element, Is_causal
        >(
            zero_hold_states, active_indices, query_idx, key_len, keep_window_size
        );
    }
};

} // namespace FLASH_NAMESPACE
