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

// Value-Index pair for top-k selection
template<typename ValueType>
struct TopKPair {
    ValueType value;
    int index;
    
    __device__ __forceinline__ TopKPair()
        : value(ValueType(-INFINITY)), index(-1) {}
    __device__ __forceinline__ TopKPair(ValueType v, int idx)
        : value(v), index(idx) {}

    __device__ __forceinline__ bool is_valid() const {
        return index >= 0 && isfinite(value);
    }
};

// Comparison functor for descending sort (greater values first)
template<typename ValueType>
struct DescendingComparator {
    __device__ __forceinline__ bool operator()(const TopKPair<ValueType>& a, const TopKPair<ValueType>& b) const {
        if (isfinite(a.value) && isfinite(b.value)) {
            return a.value > b.value;               // Descending order
        } else if (isfinite(a.value)) {
            return true;                            // a is valid, b is not
        } else if (isfinite(b.value)) {
            return false;                           // b is valid, a is not
        } else {
            return a.index < b.index;               // Compare indices if both are invalid
        }
    }
};

// Comparison functor for ascending index sort (lower indices first)
template<typename ValueType>
struct AscendingIndexComparator {
    __device__ __forceinline__ bool operator()(const TopKPair<ValueType>& a, const TopKPair<ValueType>& b) const {
        if (a.index >= 0 && b.index >= 0) {
            return a.index < b.index;               // Ascending order by index
        } else if (a.index >= 0) {
            return true;                            // a is valid, b is not
        } else if (b.index >= 0) {
            return false;                           // b is valid, a is not
        } else {
            return false;                           // Both are invalid, keep original order
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

    template <typename TensorZOH, typename TensorActiveIndices>
    __forceinline__ __device__ void get_active_zoh(
        TensorZOH &gZOH,                            // ZOH states tensor (kBlockM, actual_seqlen_k)
        TensorActiveIndices &gActiveIndices,        // Active indices tensor (kBlockM, keep_window_size)
        const int m_block,                          // Block index for rows
        const int kBlockM                           // Number of rows per block
    ) {
        static_assert(TensorZOH::rank == 2, "gZOH must be 2D Tensor (kBlockM, actual_seqlen_k)");
        static_assert(TensorActiveIndices::rank == 2, "gActiveIndices must be 2D Tensor (kBlockM, keep_window_size)");

        using ElementZOH = typename TensorZOH::value_type;

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;
        constexpr int max_items_per_thread = ITEMS_PER_THREAD;
        
        // Declare shared memory outside the row loop - reused for all rows
        using BlockMergeSortT = cub::BlockMergeSort<TopKPair<ElementZOH>, kNThreads, max_items_per_thread>;
        __shared__ union {
            typename BlockMergeSortT::TempStorage sort_storage;
        } temp_storage;
        
        // Process one row at a time to ensure all threads work on the same row
        #pragma unroll
        for (int row_offset = 0; row_offset < kBlockM; row_offset++) {
            int row_idx = m_block * kBlockM + row_offset;
            
            // Skip if beyond query sequence length
            if (row_idx >= max_seqlen_q) continue;
            
            const int col_idx_limit = Is_causal ? 
                min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : 
                max_seqlen_k;
            
            // Initialize all active indices as invalid
            for (int k_idx = tid; k_idx < keep_window_size; k_idx += num_threads) {
                gActiveIndices(row_offset, k_idx) = -1;
            }
            __syncthreads();
            
            // If no valid elements, skip to next row
            if (keep_window_size <= 0) {
                continue;
            }
            
            // If keep_window_size >= col_idx_limit, use all indices
            if (keep_window_size >= col_idx_limit) {
                for (int k_idx = tid; k_idx < col_idx_limit && k_idx < keep_window_size; k_idx += num_threads) {
                    gActiveIndices(row_offset, k_idx) = k_idx;
                }
                __syncthreads();
                continue;
            }
            
            // Initialize thread data
            TopKPair<ElementZOH> value_data[max_items_per_thread];
            #pragma unroll
            for (int item = 0; item < max_items_per_thread; ++item) {
                value_data[item] = TopKPair<ElementZOH>();
            }
            
            // Collect valid elements from current row
            #pragma unroll
            for (int item = 0; item < max_items_per_thread; ++item) {
                int global_k_idx = tid + item * num_threads;
                if (global_k_idx < col_idx_limit) {
                    // Get the value from the zoh tensor
                    ElementZOH val = gZOH(row_offset, global_k_idx);
                    value_data[item] = TopKPair<ElementZOH>(val, global_k_idx);
                }
            }
            
            // Block-wide collaborative sorting by value (descending)
            DescendingComparator<ElementZOH> comp;
            BlockMergeSortT(temp_storage.sort_storage).Sort(value_data, comp);
            __syncthreads();
            
            // Store the top-k elements temporarily
            TopKPair<ElementZOH> index_data[max_items_per_thread];
            #pragma unroll
            for (int item = 0; item < max_items_per_thread; ++item) {
                int global_pos = tid * max_items_per_thread + item;
                if (global_pos < keep_window_size && value_data[item].is_valid()) {
                    index_data[item] = value_data[item];
                } else {
                    index_data[item] = TopKPair<ElementZOH>(); // Invalid
                }
            }

            // Block-wide collaborative sorting by index (ascending)
            AscendingIndexComparator<ElementZOH> comp_idx;
            BlockMergeSortT(temp_storage.sort_storage).Sort(index_data, comp_idx);
            __syncthreads();
            
            // Store sorted indices back to global memory
            #pragma unroll
            for (int item = 0; item < max_items_per_thread; ++item) {
                int global_pos = tid * max_items_per_thread + item;
                if (global_pos < keep_window_size) {
                    gActiveIndices(row_offset, global_pos) = index_data[item].index;
                }
            }
            __syncthreads();
        }
    }

    template <bool Causal_mask=false, bool Is_even_MN=true, typename TensorType, typename ZOHType, typename ActiveIndicesType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        ZOHType &tZOH,                              // ZOH states (MMA=4, MMA_M, MMA_N)
        ActiveIndicesType &tActiveIndices,          // Active indices (MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(TensorType::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(ZOHType::rank == 3, "tZOH must be 3D Tensor");
        static_assert(ActiveIndicesType::rank == 3, "tActiveIndices must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        const bool Need_masking = Causal_mask || !Is_even_MN || (keep_window_size < max_seqlen_k);
        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        Tensor zoh = make_tensor(tZOH.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tZOH.layout()));
        Tensor active_indices = make_tensor(tActiveIndices.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tActiveIndices.layout()));

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
                            bool inactive = (col_idx >= col_idx_limit) || (active_indices(coord) < 0);
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
                            tensor(coord) = tensor(coord) * scale_softmax + zoh(coord);
                        }
                    }
                }
            }
        }
    }

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_causal_mask(
        Tensor<Engine, Layout> &tensor_,            // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        static_assert(Tensor<Engine, Layout>::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        if constexpr (Need_masking) {
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
                            bool inactive = (col_idx >= col_idx_limit);
                            if (inactive) {
                                tensor(coord) = -INFINITY;
                            } else {
                                // Apply scaling
                                tensor(coord) = tensor(coord) * scale_softmax;
                            }
                        }
                    }
                }
            }
        } else {
            // If no masking is needed, just scale the tensor
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
                            tensor(coord) = tensor(coord) * scale_softmax;
                        }
                    }
                }
            }
        }
    }
};

} // namespace FLASH_NAMESPACE
