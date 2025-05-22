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
        #pragma unroll
        for (int k_idx = 0; k_idx < key_len; ++k_idx) {
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
    float* shared_key_mask_values,          // Shared memory for mask values for one query [key_len]
    float* sort_keys,                       // Shared memory for sorting keys [key_len]
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
    
    // Copy values to sorting buffers
    #pragma unroll
    for (int k_idx = threadIdx.x; k_idx < key_len; k_idx += blockDim.x) {
        sort_keys[k_idx] = shared_key_mask_values[k_idx];
        sort_indices[k_idx] = k_idx;
    }
    __syncthreads();
    
    // Partial selection sort to find top elements
    for (int window_idx = 0; window_idx < keep_window_size && window_idx < key_len; ++window_idx) {
        float thread_max = -FLT_MAX;
        int thread_max_idx = -1;
        #pragma unroll
        for (int k_idx = window_idx + threadIdx.x; k_idx < key_len; k_idx += blockDim.x) {
            if (sort_keys[k_idx] > thread_max) {
                thread_max = sort_keys[k_idx];
                thread_max_idx = k_idx;
            }
        }
        __shared__ float s_max_vals[BLOCK_THREADS];
        __shared__ int s_max_indices[BLOCK_THREADS];
        s_max_vals[threadIdx.x] = thread_max;
        s_max_indices[threadIdx.x] = thread_max_idx;
        __syncthreads();
        for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                if (s_max_vals[threadIdx.x] < s_max_vals[threadIdx.x + stride]) {
                    s_max_vals[threadIdx.x] = s_max_vals[threadIdx.x + stride];
                    s_max_indices[threadIdx.x] = s_max_indices[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && s_max_indices[0] >= 0) {
            float tmp_key = sort_keys[window_idx];
            int tmp_idx = sort_indices[window_idx];
            sort_keys[window_idx] = sort_keys[s_max_indices[0]];
            sort_indices[window_idx] = sort_indices[s_max_indices[0]];
            sort_keys[s_max_indices[0]] = tmp_key;
            sort_indices[s_max_indices[0]] = tmp_idx;
        }
        __syncthreads();
    }
    
    // Reset and scatter top values
    #pragma unroll
    for (int k_idx = threadIdx.x; k_idx < key_len; k_idx += blockDim.x) {
        shared_key_mask_values[k_idx] = 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int window_idx = threadIdx.x; window_idx < keep_window_size && window_idx < key_len; window_idx += blockDim.x) {
        int idx = sort_indices[window_idx];
        if (idx >= 0 && idx < key_len) {
            shared_key_mask_values[idx] = sort_keys[window_idx];
        }
    }
    __syncthreads();
}

// Apply dynamic mask with 1 row block
template <
    typename EngineDst, typename LayoutDst,                     // float
    typename EngineSrc, typename LayoutSrc,                     // half/bfloat16
    typename EngineSortKey, typename LayoutSortKey,             // float
    typename EngineSortIdx, typename LayoutSortIdx,             // int
    typename Element, bool Is_causal
>
__forceinline__ __device__ void apply_dynamic_mask_1rowblock(
    Tensor<EngineDst, LayoutDst> &tensor,                       // Output 1D tensor [key_len]
    Tensor<EngineSrc, LayoutSrc> const &zero_hold_states,       // Pre-calculated zero_hold states [key_len]
    int query_idx,                                              // Current query position (row index)
    const int key_len,                                          // Sequence length for keys
    const int keep_window_size,                                 // Maximum window size to keep
    Tensor<EngineDst, LayoutDst> &row_vals,                     // Shared memory buffer for mask values [key_len]
    Tensor<EngineSortKey, LayoutSortKey> &sort_keys,           // Shared memory buffer for sorting keys [key_len]
    Tensor<EngineSortIdx, LayoutSortIdx> &sort_indices         // Shared memory buffer for sorting indices [key_len]
) {
    static_assert(LayoutDst::rank == 1, "Tensor must be 1D");
    int tid = threadIdx.x;

    // Load zero_hold and initialize row values
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        row_vals[k_idx] = static_cast<float>(zero_hold_states[k_idx]);
    }
    __syncthreads();

    // Apply causal mask across the row
    apply_causal_mask_1rowblock<Element, Is_causal>(
        row_vals.data().get(),
        query_idx, key_len
    );
    __syncthreads();

    // Top-k window selection
    apply_topk_window_selection_1rowblock<Element>(
        row_vals.data().get(),
        sort_keys.data().get(),
        sort_indices.data().get(),
        key_len, keep_window_size
    );
    __syncthreads();

    // Write back to tensor
    #pragma unroll
    for (int k_idx = tid; k_idx < key_len; k_idx += blockDim.x) {
        tensor(k_idx) = row_vals[k_idx];
    }
}

// Struct wrapper for dynamic mask application
struct DynamicMask {
    const int keep_window_size;

    __forceinline__ __device__ DynamicMask(const int keep_window_size = 2048)
        : keep_window_size(keep_window_size) {}

    template <
        typename EngineDst, typename LayoutDst,
        typename EngineSrc, typename LayoutSrc,
        typename EngineSortKey, typename LayoutSortKey,
        typename EngineSortIdx, typename LayoutSortIdx,
        typename Element, bool Is_causal
    >
    __forceinline__ __device__ void apply_mask_1rowblock(
        Tensor<EngineDst, LayoutDst> &tensor,                   // float
        Tensor<EngineSrc, LayoutSrc> const &zero_hold_states,   // half/bfloat16
        int query_idx,
        const int key_len,
        Tensor<EngineDst, LayoutDst> &row_vals,
        Tensor<EngineSortKey, LayoutSortKey> &sort_keys,
        Tensor<EngineSortIdx, LayoutSortIdx> &sort_indices
    ) {
        apply_dynamic_mask_1rowblock<
            EngineDst, LayoutDst,
            EngineSrc, LayoutSrc,
            EngineSortKey, LayoutSortKey,
            EngineSortIdx, LayoutSortIdx,
            Element, Is_causal
        >(
            tensor, zero_hold_states, query_idx, key_len, keep_window_size,
            row_vals, sort_keys, sort_indices
        );
    }
};

} // namespace FLASH_NAMESPACE
