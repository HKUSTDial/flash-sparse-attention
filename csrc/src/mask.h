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
    float* zero_hold_states,            // Zero-hold states for one query row [key_len]
    int query_idx,                      // Current query position (row index)
    int key_len                         // Key length (sequence length for keys)
) {
    if constexpr (Is_causal) {
        int tid = threadIdx.x;
        #pragma unroll
        for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
            const bool is_masked = k_idx > query_idx;
            if (is_masked) {
                zero_hold_states[k_idx] = 0.0f;
            }
        }
    }
}

// Apply top-k selection for dynamic mask with 1 row block
template <typename Element>
__forceinline__ __device__ void apply_topk_window_selection_1rowblock(
    float* zero_hold_states,                // Zero-hold states for in-place modification [key_len] 
    int* sort_indices,                      // Shared memory for sorting indices [key_len]
    const int key_len,                      // Key length
    const int keep_window_size              // Maximum window size to keep
) {
    using namespace cute;
    using namespace cutlass;
    
    // Skip if no reduction needed
    if (key_len <= keep_window_size) {
        return;
    }

    int tid = threadIdx.x;
    
    // Initialize indices
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        sort_indices[k_idx] = k_idx;
    }
    __syncthreads();
    
    // Partial selection sort to find top elements
    for (int window_idx = 0; window_idx < keep_window_size && window_idx < key_len; ++window_idx) {
        float thread_max = -FLT_MAX;
        int thread_max_idx = -1;
        
        // Find maximum in remaining elements
        #pragma unroll
        for (int k_idx = window_idx + tid; k_idx < key_len; k_idx += blockDim.x) {
            int actual_idx = sort_indices[k_idx];
            if (zero_hold_states[actual_idx] > thread_max) {
                thread_max = zero_hold_states[actual_idx];
                thread_max_idx = k_idx;
            }
        }
        
        // Reduction to find global maximum
        __shared__ float s_max_vals[BLOCK_THREADS];
        __shared__ int s_max_indices[BLOCK_THREADS];
        s_max_vals[tid] = thread_max;
        s_max_indices[tid] = thread_max_idx;
        __syncthreads();
        
        for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_max_vals[tid] < s_max_vals[tid + stride]) {
                    s_max_vals[tid] = s_max_vals[tid + stride];
                    s_max_indices[tid] = s_max_indices[tid + stride];
                }
            }
            __syncthreads();
        }
        
        // Swap to bring maximum to current window position
        if (tid == 0 && s_max_indices[0] >= 0) {
            int swap_idx = s_max_indices[0];
            int tmp_idx = sort_indices[window_idx];
            sort_indices[window_idx] = sort_indices[swap_idx];
            sort_indices[swap_idx] = tmp_idx;
        }
        __syncthreads();
    }
    
    // Create temporary buffer to store top-k values
    __shared__ float s_topk_vals[BLOCK_THREADS * ITEMS_PER_THREAD];
    float* topk_vals = s_topk_vals;
    
    // Store top-k values
    #pragma unroll
    for (int window_idx = tid; window_idx < keep_window_size && window_idx < key_len; window_idx += blockDim.x) {
        int idx = sort_indices[window_idx];
        if (idx >= 0 && idx < key_len) {
            topk_vals[window_idx] = zero_hold_states[idx];
        }
    }
    __syncthreads();
    
    // Reset tensor to zero
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        zero_hold_states[k_idx] = 0.0f;
    }
    __syncthreads();
    
    // Scatter top-k values back to original positions
    #pragma unroll
    for (int window_idx = tid; window_idx < keep_window_size && window_idx < key_len; window_idx += blockDim.x) {
        int idx = sort_indices[window_idx];
        if (idx >= 0 && idx < key_len) {
            zero_hold_states[idx] = topk_vals[window_idx];
        }
    }
    __syncthreads();
}

// Apply dynamic mask with 1 row block
template <
    typename Engine, typename Layout,               // float tensor (in-place)
    typename Element, bool Is_causal
>
__forceinline__ __device__ void apply_dynamic_mask_1rowblock(
    Tensor<Engine, Layout> &zero_hold_states,       // In-place tensor [key_len]
    int query_idx,                                  // Current query position (row index)
    const int key_len,                              // Sequence length for keys
    const int keep_window_size,                     // Maximum window size to keep
    int* sort_indices                               // for sorting indices [key_len]
) {
    static_assert(Layout::rank == 1, "Tensor must be 1D");

    // Apply causal mask across the row
    apply_causal_mask_1rowblock<Element, Is_causal>(
        zero_hold_states.data().get(),
        query_idx, key_len
    );
    __syncthreads();

    // Top-k window selection
    apply_topk_window_selection_1rowblock<Element>(
        zero_hold_states.data().get(),
        sort_indices,
        key_len, keep_window_size
    );
    __syncthreads();
}

// Struct wrapper for dynamic mask application
struct DynamicMask {
    const int keep_window_size;

    __forceinline__ __device__ DynamicMask(const int keep_window_size = 2048)
        : keep_window_size(keep_window_size) {}

    template <
        typename Engine, typename Layout,
        typename Element, bool Is_causal
    >
    __forceinline__ __device__ void apply_mask_1rowblock(
        Tensor<Engine, Layout> &zero_hold_states,   // In-place tensor (float)
        int query_idx,                              // Query index
        int key_len,                                // Key length
        int* sort_indices                           // Sort indices buffer (int only)
    ) {
        apply_dynamic_mask_1rowblock<
            Engine, Layout,
            Element, Is_causal
        >(
            zero_hold_states, query_idx, key_len, keep_window_size,
            sort_indices
        );
    }
};

} // namespace FLASH_NAMESPACE
