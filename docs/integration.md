# Flash Dynamic Mask Attention Integration Guide

## Overview

This document describes the integration of Dynamic Mask Attention into the Flash Attention framework. The integration enables efficient sparse attention computation by combining Flash Attention's memory-efficient approach with dynamic masking capabilities for handling extremely long sequences.

The integration implements a unified sparse computation approach with block-level skip logic: Python frontend pre-computes Attention Mask and Attention Bias tensors, while the CUDA backend performs block-level skip decisions and sparse attention computation for both forward and backward passes.

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [Core Modifications](#core-modifications)
3. [Implementation Details](#implementation-details)
4. [Sparse Computation Strategy](#sparse-computation-strategy)
5. [Memory Layout](#memory-layout)
6. [Performance Considerations](#performance-considerations)
7. [API Changes](#api-changes)

## Integration Architecture

### High-Level Design

The Dynamic Mask Attention integration implements a unified sparse computation approach with block-level skip logic for both forward and backward passes:

1. **Dynamic Mask Computation**: Python frontend pre-computes Attention Mask and Attention Bias tensors
2. **Unified Sparse Execution**: CUDA backend performs block-level skip decisions for both forward and backward passes
3. **Memory Optimization**: Smart shared memory aliasing and barrier synchronization


### Key Components

- **Attention Mask**: Binary mask `(batch, num_kv_heads, query_len, key_len)` indicating which positions should be computed (1.0) or skipped (0.0)
- **Attention Bias**: Dynamic attention bias values `(batch, num_kv_heads, query_len, key_len)` applied to attention scores before softmax
- **Block-level Skip Logic**: Unified OR-reduction over (BlockM × BlockN) tiles to determine if computation should be performed
- **LSE Caching**: Log-sum-exp values cached during forward pass for numerically stable backward recomputation
- **Shared Memory Aliasing**: Smart memory reuse with explicit barrier synchronization
- **Complete Gradient Chain**: Full gradient computation pipeline with sparse skip capability
- **Memory Optimization**: Reduced shared memory footprint enabling larger tile sizes and higher occupancy

## Core Modifications

### 1. Parameter Structure Extensions (`flash.h`)

**Purpose**: Extended parameter structures to support dynamic masking tensors with proper memory layout information.

**Changes Made**:
```cpp
struct QKV_params {
    // The QKV matrices.
    void *__restrict__ q_ptr;   // Query tensor [batch_size, num_heads, query_len, head_dim]
    void *__restrict__ k_ptr;   // Key tensor [batch_size, num_kv_heads, key_len, head_dim]
    void *__restrict__ v_ptr;   // Value tensor [batch_size, num_kv_heads, key_len, head_dim]

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride, k_batch_stride, v_batch_stride;
    index_t q_row_stride, k_row_stride, v_row_stride;
    index_t q_head_stride, k_head_stride, v_head_stride;

    // The number of heads.
    int h, h_k;
    int h_h_k_ratio; // precompute h / h_k
};

struct Mask_params {
    void * __restrict__ mask_ptr;       // Attention mask tensor [batch_size, num_kv_heads, query_len, key_len]

    // The stride of the attention mask tensors.
    index_t mask_batch_stride;          // Stride between batches of attention mask
    index_t mask_head_stride;           // Stride between heads of attention mask
    index_t mask_row_stride;            // Stride between rows of attention mask
};

struct Bias_params {
    void *__restrict__ bias_ptr;        // Attention bias tensor [batch_size, num_kv_heads, query_len, key_len]

    // The stride of the attention bias tensor.
    index_t bias_batch_stride;          // Stride between batches of attention bias
    index_t bias_head_stride;           // Stride between heads of attention bias
    index_t bias_row_stride;            // Stride between rows of attention bias
};

struct Flash_fwd_params : public QKV_params, public Mask_params, public Bias_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
    float softcap;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the K_new and V_new matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    int num_splits;  // For split-KV version

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};
```

**Rationale**: 
- **Multiple Inheritance Design**: Cleanly separates QKV parameters from Mask/Bias parameters while maintaining unified access
- **Comprehensive Stride Information**: Provides all necessary stride information for efficient tensor indexing in CUDA kernels
- **Memory Layout Optimization**: Enables optimal memory access patterns for both regular and sparse tensors

### 2. Kernel Traits and Memory Layout (`kernel_traits.h`)

**Purpose**: Define kernel characteristics and memory layouts optimized for dynamic masking operations, supporting both SM75 and SM80+ architectures.

**Changes Made**:
```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kNWarps = kNWarps_;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    static constexpr bool Has_cp_async = true;
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    static constexpr bool Has_cp_async = false;
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
    
    // Specialized traits for mask and bias operations
    using SmemCopyAtomMask = SmemCopyAtom;
    using SmemCopyAtomBias = SmemCopyAtom;
};
```

**Rationale**:
- **Architecture Adaptation**: Automatically selects optimal MMA atoms and copy operations based on GPU architecture
- **Type Safety**: Template-based design ensures type consistency across mask, bias, and attention operations
- **Performance Optimization**: Leverages specialized load/store instructions (LDSM) for maximum memory bandwidth

### 3. Block Information Extension (`block_info.h`)

**Purpose**: Calculate memory offsets for attention bias and attention masks within thread blocks, enabling efficient global memory access.

**Changes Made**:
```cpp
template<bool Varlen=true>
struct BlockInfo {
    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : 
            (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : 
            seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t mask_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        index_t offset = sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
        sum_s_k == -1 ? offset += leftpad_k : offset += uint32_t(sum_s_k + leftpad_k);
        return offset;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t bias_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        index_t offset = sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
        sum_s_k == -1 ? offset += leftpad_k : offset += uint32_t(sum_s_k + leftpad_k);
        return offset;
    }

    const int sum_s_q, sum_s_k;
    const int actual_seqlen_q;
    const int leftpad_k;
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};
```

**Rationale**:
- **Unified Offset Calculation**: Provides dedicated methods for calculating mask and bias tensor offsets
- **Variable Length Support**: Handles both fixed and variable length sequences through template specialization
- **Memory Access Optimization**: Encapsulates complex address arithmetic for efficient global memory access

### 4. Memory Copy Operations (`utils.h`)

**Purpose**: Implement efficient tensor operations and layout conversions optimized for Flash Attention's memory hierarchy.

**Changes Made**:
```cpp
namespace FLASH_NAMESPACE {

// Convert accumulator layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

// Type conversion utilities for different precisions
template<typename T>
__forceinline__ __device__ T convert_type(float x) {
    return T(x);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ cutlass::bfloat16_t convert_type<cutlass::bfloat16_t>(float x) {
    return cutlass::bfloat16_t(x);
}
#endif

// Warp-level reduction operations
template<int THREADS>
__forceinline__ __device__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = THREADS / 2; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

// GEMM operations with register and shared memory variants
template <
    bool A_in_regs=false, bool B_in_regs=false,
    typename Tensor0, typename Tensor1, typename Tensor2, 
    typename Tensor3, typename Tensor4,
    typename TiledMma, typename TiledCopyA, typename TiledCopyB,
    typename ThrCopyA, typename ThrCopyB
>
__forceinline__ __device__ void gemm(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 &tCsA, Tensor4 &tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B
) {
    if constexpr (!A_in_regs) {
        copy(smem_tiled_copy_A, tCsA, tCrA);
    }
    if constexpr (!B_in_regs) {
        copy(smem_tiled_copy_B, tCsB, tCrB);
    }
    
    // Perform matrix multiplication
    gemm(tiled_mma, acc, tCrA, tCrB, acc);
}

}  // namespace FLASH_NAMESPACE
```

**Rationale**:
- **Layout Conversion**: Efficient transformation between MMA and row-column layouts for easier tensor manipulation
- **Multi-Precision Support**: Proper type conversion utilities for FP16 and BF16 operations
- **Memory Hierarchy Management**: Flexible GEMM operations supporting different data residency patterns
- **Performance Optimization**: Warp-level reductions and vectorized operations for maximum throughput

### 5. Dynamic Masking Logic (`mask.h`)

**Purpose**: Implement the core dynamic masking functionality that applies attention bias and attention masks during attention computation.

**Changes Made**:
```cpp
template <bool Causal_mask=false, typename TensorType, typename MaskType, typename BiasType>
__forceinline__ __device__ void apply_mask(
    TensorType &tensor,
    MaskType &mask,
    BiasType &bias,
    const float scale_softmax,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset,
    const int max_seqlen_q,
    const int warp_row_stride
) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(TensorType::rank == 2, "Only support 2D Tensor");
    static_assert(MaskType::rank == 2, "Only support 2D Mask");
    static_assert(BiasType::rank == 2, "Only support 2D Bias");
    
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit = Causal_mask ? 
                std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : 
                max_seqlen_k;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                    // Apply scaling and bias or masking
                    tensor(coord) = (col_idx >= col_idx_limit) || (mask(coord) == 0.0f)
                        ? -INFINITY
                        : tensor(coord) * scale_softmax + bias(coord);
                }
            }
        }
    }
}

template <bool Is_causal>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Mask(
        const int max_seqlen_k,
        const int max_seqlen_q
    )  // Constructor
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    template <bool Causal_mask=false, bool Is_even_MN=true, typename TensorType, typename MaskType, typename BiasType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        MaskType &tSrMask,                          // Attention Mask (MMA=4, MMA_M, MMA_N)
        BiasType &tSrBias,                          // Attention Bias (MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride                   // Warp row stride
    ) {
        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        Tensor mask = make_tensor(tSrMask.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrMask.layout()));
        Tensor bias = make_tensor(tSrBias.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrBias.layout()));

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? 
                    std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : 
                    max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        // Apply scaling and bias or masking
                        tensor(coord) = (col_idx >= col_idx_limit) || (mask(coord) == 0.0f)
                            ? -INFINITY
                            : tensor(coord) * scale_softmax + bias(coord);
                    }
                }
            }
        }
    }
};
```

**Rationale**:
- **Register-Level Operations**: All masking operations performed in registers for maximum efficiency
- **Unified Masking Logic**: Combines causal masking, boundary checking, and dynamic masking in a single pass
- **Layout Conversion**: Properly handles MMA tensor layout conversion for efficient indexing
- **Numerical Stability**: Proper handling of infinity values for masked positions ensures stable softmax computation

### 6. Backward Pass Integration (`flash_bwd_kernel.h`)

**Purpose**: Extend backward pass computation to support dynamic masking with proper gradient computation for masked positions.

**Changes Made**:
```cpp
struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV and dBias matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ dbias_ptr;

    // To accumulate dQ, dK, dV
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
    index_t dbias_batch_stride;
    index_t dbias_head_stride;
    index_t dbias_row_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, 
         bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {
    // Backward pass computation with dynamic masking support
    // Includes proper gradient computation through masked attention scores
    // Maintains numerical stability for masked positions
    
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    
    // Initialize block information and tensor views
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    
    // Set up gradient computation with masking awareness
    // Load bias and mask gradients when computing dBias
    // Apply masking logic consistently with forward pass
}
```

**Rationale**:
- **Gradient Consistency**: Ensures gradients are computed consistently with forward pass masking logic
- **Memory Layout Preservation**: Maintains the same memory layout and stride patterns as forward pass
- **Numerical Stability**: Proper handling of gradients at masked positions to prevent NaN propagation

### 7. Attention Kernel Modifications (`flash_fwd_kernel.h`)

**Purpose**: Integrate dynamic masking into the core attention computation kernels while maintaining Flash Attention's memory efficiency and optimization strategies.

**Changes Made**:
```cpp
template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, 
         bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    
    // Initialize block information
    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    
    // Set up tensor views for Q, K, V matrices
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)), 
                           make_shape(binfo.actual_seqlen_q, Int<Kernel_traits::kHeadDim>{}),
                           make_stride(params.q_row_stride, _1{}));
    
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                           make_shape(binfo.actual_seqlen_k, Int<Kernel_traits::kHeadDim>{}),
                           make_stride(params.k_row_stride, _1{}));
    
    // Set up mask and bias tensor views if available
    Tensor mMask, mBias;
    if (params.mask_ptr != nullptr) {
        mMask = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.mask_ptr) + binfo.mask_offset(params.mask_batch_stride, params.mask_row_stride, bidb)),
                           make_shape(binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                           make_stride(params.mask_row_stride, _1{}));
    }
    
    if (params.bias_ptr != nullptr) {
        mBias = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.bias_ptr) + binfo.bias_offset(params.bias_batch_stride, params.bias_row_stride, bidb)),
                           make_shape(binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                           make_stride(params.bias_row_stride, _1{}));
    }
    
    // Main computation loop with dynamic masking integration
    for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
        // Standard Flash Attention computation: Q*K^T
        gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, 
             smem_tiled_copy_Q, smem_tiled_copy_K,
             smem_thr_copy_Q, smem_thr_copy_K);
        
        // Apply dynamic masking if mask/bias tensors are provided
        if (params.mask_ptr != nullptr || params.bias_ptr != nullptr) {
            Mask<Is_causal> mask(params.seqlen_k, params.seqlen_q);
            mask.apply_mask<Is_causal, Is_even_MN>(acc_s, tSrMask, tSrBias, params.scale_softmax,
                                                   n_block * Kernel_traits::kBlockN, m_block * Kernel_traits::kBlockM, 
                                                   Kernel_traits::kBlockM);
        }
        
        // Continue with softmax computation
        softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || !Is_even_MN>(
            acc_s, acc_o, params.scale_softmax_log2
        );
        
        // Attention * V computation
        gemm</*A_in_regs=*/true>(acc_o, acc_s, tSrV, acc_s, tSsV, tiled_mma,
                                 smem_tiled_copy_S, smem_tiled_copy_V, 
                                 smem_thr_copy_S, smem_thr_copy_V);
    }
}
```

**Rationale**:
- **Seamless Integration**: Dynamic masking logic integrated into existing Flash Attention computation flow without affecting core performance
- **Memory Efficiency Preservation**: Maintains Flash Attention's tiling and shared memory optimization strategies
- **Conditional Execution**: Only applies masking operations when mask/bias tensors are actually provided
- **Template Specialization**: Compile-time optimization eliminates runtime branching for better performance

### 8. Launch Template Updates (`flash_fwd_launch_template.h`)

**Purpose**: Update kernel launch templates to support dynamic masking functionality with proper template instantiation and dispatch logic.

**Changes Made**:
```cpp
// Determine if the architecture supports FLASH and define parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define unsupported architecture error handling
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashDynamicMaskAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Kernel definition macro for cleaner code
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Split>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    // Handle different precision types and head dimensions
    BOOL_SWITCH(params.is_bf16, Is_Bf16, [&] {
        using elem_type = std::conditional_t<Is_Bf16, cutlass::bfloat16_t, cutlass::half_t>;
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.seqlen_k % Kernel_traits::kBlockN == 0, Is_even_N, [&] {
                BOOL_SWITCH(params.d == kHeadDim, Is_even_K, [&] {
                    SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>;
                        // Launch kernel with appropriate grid and block dimensions
                        kernel<<<grid, Kernel_traits::kNWarps * 32, smem_size, stream>>>(params);
                    });
                });
            });
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Template instantiations for different configurations
template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream);
```

**Rationale**:
- **Template Dispatch**: Efficient compile-time branching based on runtime parameters for optimal performance
- **Architecture Support**: Proper handling of different GPU architectures with appropriate error messages
- **Memory Management**: Correct shared memory allocation based on kernel requirements
- **Type Safety**: Strong typing through template parameters ensures correctness across different precisions

**Purpose**: Update kernel launch functions to properly configure and validate dynamic masking parameters, ensuring correct shared memory allocation and kernel selection.

**Changes Made**:
```cpp
template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // Calculate shared memory requirements
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    // Set up grid dimensions
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    
    // Determine kernel variant based on sequence lengths and alignment
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && 
                           params.seqlen_k % Kernel_traits::kBlockN == 0 && 
                           params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    
    // Launch appropriate kernel variant with dynamic masking support
    BOOL_SWITCH(is_even_MN, IsEvenMN, [&] {
        BOOL_SWITCH(is_even_K, IsEvenK, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmax, [&] {
                auto kernel = &flash_fwd_kernel<Kernel_traits, Is_causal, 
                                              IsEvenMN, IsEvenK, /*Is_softcap=*/false, ReturnSoftmax>;
                
                // Configure dynamic shared memory if needed
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                
                // Launch kernel with extended parameter set
                kernel<<<grid, Kernel_traits::kNWarps * 32, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // Split-K variant launch with dynamic masking support
    // Handles cases where sequence length exceeds single kernel capacity
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    
    // Configure split parameters based on sequence length and hardware capabilities
    const int num_splits = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    // ... split-K launch logic with dynamic masking support
}
```

**Rationale**:
- **Resource Management**: Proper shared memory allocation and validation for extended tensor requirements
- **Kernel Selection**: Intelligent kernel variant selection based on problem size and hardware capabilities
- **Error Handling**: Comprehensive validation of parameters and device limits
- **Performance Optimization**: Compile-time optimizations through template specialization

### 9. API Interface Extensions (`flash_api.cpp`)

**Purpose**: Extend the Python-facing API to support dynamic masking tensors with comprehensive validation and backward compatibility.

**Changes Made**:
```cpp
void set_params_fprop(
    Flash_fwd_params &params,
    // ... existing parameters ...
    const at::Tensor mask,                        // Attention mask tensor
    const at::Tensor bias,                        // Attention bias tensor
    // ... other parameters ...
) {
    // Reset parameters and set basic properties
    params = {};
    params.is_bf16 = q.dtype() == torch::kBFloat16;
    
    // Set attention mask pointers and strides
    params.mask_ptr = mask.data_ptr();
    params.mask_batch_stride = mask.stride(-4);
    params.mask_head_stride = mask.stride(-3);
    params.mask_row_stride = mask.stride(-2);
    
    // Set attention bias pointers and strides  
    params.bias_ptr = bias.data_ptr();
    params.bias_batch_stride = bias.stride(-4);
    params.bias_head_stride = bias.stride(-3);
    params.bias_row_stride = bias.stride(-2);
    
    // ... existing parameter setup ...
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,                              // Query tensor
    const at::Tensor &k,                        // Key tensor
    const at::Tensor &v,                        // Value tensor
    const at::Tensor &mask,                     // Attention mask tensor
    const at::Tensor &bias,                     // Attention bias tensor
    std::optional<at::Tensor> &out_,            // Optional output tensor
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    const bool return_softmax
) {
    // Comprehensive input validation
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(mask); CHECK_DEVICE(bias);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(mask); CHECK_CONTIGUOUS(bias);
    
    // Validate tensor shapes
    auto batch_size = q.size(0);
    auto seqlen_q = q.size(1); 
    auto num_heads = q.size(2);
    auto head_dim = q.size(3);
    auto seqlen_k = k.size(1);
    auto num_heads_k = k.size(2);
    
    CHECK_SHAPE(mask, batch_size, num_heads_k, seqlen_q, seqlen_k);
    CHECK_SHAPE(bias, batch_size, num_heads_k, seqlen_q, seqlen_k);
    
    // Validate data types consistency
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), 
                "All QKV tensors must have the same dtype");
    TORCH_CHECK(mask.dtype() == q.dtype(), 
                "Attention mask must have the same dtype as QKV tensors");
    TORCH_CHECK(bias.dtype() == q.dtype(), 
                "Attention bias must have the same dtype as QKV tensors");
    
    // Set up parameters and launch computation
    Flash_fwd_params params;
    set_params_fprop(params, batch_size, seqlen_q, seqlen_k, /* ... */, 
                     q, k, v, mask, bias, /* ... */);
    
    // Launch kernel with appropriate configuration
    run_mha_fwd(params, at::cuda::getCurrentCUDAStream());
    
    // Return results
    return {out, softmax_lse};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDynamicMaskAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass with dynamic masking",
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("mask"), py::arg("bias"),                    // Updated arguments
          py::arg("out") = py::none(),
          py::arg("softmax_scale") = 0.0f, 
          py::arg("is_causal") = false,
          py::arg("softcap") = 0.0f,
          py::arg("return_softmax") = false);
}
```

**Rationale**:
- **Comprehensive Validation**: Thorough validation of all input tensors for shape, type, and device consistency
- **Backward Compatibility**: Maintains existing parameter order while adding new functionality
- **Error Handling**: Clear error messages for common usage mistakes
- **Type Safety**: Strict type checking to prevent runtime errors
- **Documentation**: Clear parameter documentation for Python users

## Implementation Details

### C++ API Interface (`flash_api.cpp`)

The core C++ API provides the following main functions for Dynamic Mask Attention:

```cpp
namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,                      // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,                // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,                // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &mask,             // batch_size x num_heads_k x seqlen_q x seqlen_k
    const at::Tensor &bias,             // batch_size x num_heads_k x seqlen_q x seqlen_k
    std::optional<at::Tensor> &out_,    // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    const bool return_softmax
);

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,                      // total_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,                // total_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,                // total_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &mask,             // total_q x num_heads_k x max_seqlen_k
    const at::Tensor &bias,             // total_q x num_heads_k x max_seqlen_k
    std::optional<at::Tensor> &out_,    // total_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &cu_seqlens_q,     // batch_size + 1
    const at::Tensor &cu_seqlens_k,     // batch_size + 1
    std::optional<at::Tensor> &seqused_k,
    std::optional<at::Tensor> &leftpad_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    const bool return_softmax
);

std::vector<at::Tensor> mha_bwd(
    const at::Tensor &dout,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &q,                // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,                // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,                // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &mask,             // batch_size x num_heads_k x seqlen_q x seqlen_k
    const at::Tensor &bias,             // batch_size x num_heads_k x seqlen_q x seqlen_k
    const at::Tensor &out,              // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &softmax_lse,      // batch_size x num_heads x seqlen_q
    std::optional<at::Tensor> &dq_,
    std::optional<at::Tensor> &dk_,
    std::optional<at::Tensor> &dv_,
    std::optional<at::Tensor> &dbias_,
    const float softmax_scale,
    bool is_causal,
    const float softcap,
    bool deterministic,
    std::optional<at::Generator> gen_
);

}  // namespace FLASH_NAMESPACE
```

### Parameter Setup and Validation

The implementation includes comprehensive parameter validation and setup:

```cpp
void set_params_fprop(
    Flash_fwd_params &params,
    const size_t b, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
    const at::Tensor q, const at::Tensor k, const at::Tensor v,
    const at::Tensor mask, const at::Tensor bias, at::Tensor out,
    void *cu_seqlens_q_d, void *cu_seqlens_k_d, void *seqused_k,
    void *p_d, void *softmax_lse_d, float softmax_scale, bool is_causal,
    const float softcap, bool seqlenq_ngroups_swapped=false,
    const bool unpadded_lse=false
) {
    // Reset parameters
    params = {};
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set tensor pointers
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.mask_ptr = mask.data_ptr();
    params.bias_ptr = bias.data_ptr();
    params.o_ptr = out.data_ptr();

    // Set stride information (all strides are in elements, not bytes)
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.mask_row_stride = mask.stride(-2);
    params.bias_row_stride = bias.stride(-2);
    params.o_row_stride = out.stride(-3);
    
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.mask_head_stride = mask.stride(-3);
    params.bias_head_stride = bias.stride(-3);
    params.o_head_stride = out.stride(-2);

    // Set batch stride information
    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.mask_batch_stride = mask.stride(0);
        params.bias_batch_stride = bias.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    // Set sequence length and dimension parameters
    params.b = b; params.h = h; params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d; params.d_rounded = d_rounded;
    
    // Set scaling and control parameters
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = softcap;
    params.is_causal = is_causal;
    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}
```

### Python Binding and Interface

The C++ functions are exposed to Python through PyBind11:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDynamicMaskAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass with variable length");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass with variable length");
}
```

### Python Frontend Integration Example

Dynamic Mask Attention can be integrated into transformer models as follows:

```python
import torch
import torch.nn as nn
import flash_dmattn_cuda as flash_dmattn

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Prepare mask and bias tensors with proper shapes
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, self.num_kv_heads, seq_len, seq_len), 
                                      dtype=query_states.dtype, device=query_states.device)
        
        if attention_bias is None:
            attention_bias = torch.zeros((batch_size, self.num_kv_heads, seq_len, seq_len),
                                       dtype=query_states.dtype, device=query_states.device)
        
        # Call Flash Dynamic Mask Attention
        output, _ = flash_dmattn.fwd(
            query_states, key_states, value_states,
            attention_mask, attention_bias,
            None,  # out
            self.scaling,  # softmax_scale
            False,  # is_causal
            0.0,   # softcap
            False  # return_softmax
        )
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)
```
        
        # Call attention implementation
        attn_output, attn_weights = flash_dynamic_mask_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attn_bias,
            scaling=self.scaling,
        )
        
        return attn_output, attn_weights
```

The attention bias generation process:

1. **Value-based Dynamic States**: 
   ```python
   dt_states = self.dt_proj(value_states_flattened)
   dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
   ```

2. **Bias Expansion**: 
   ```python
   attn_bias = dt_states[:, :, None, :].expand(-1, -1, query_len, -1)
   ```

3. **Mask Processing**: Done internally in `_flash_dynamic_mask_attention_forward`


### CUDA Backend: Sparse Attention Computation

The CUDA backend implements the sparse attention computation through `_flash_dynamic_mask_attention_forward`:

```python
def _flash_dynamic_mask_attention_forward(
    query_states, key_states, value_states,
    attention_mask, attention_bias,
    query_length, key_length,
    is_causal, softmax_scale=None, softcap=None,
    target_dtype=None, implementation=None, **kwargs
):
    dtype = query_states.dtype
    min_dtype = torch.finfo(dtype).min
    batch_size, _, num_kv_heads, _ = key_states.shape

    # Initialize attention bias if not provided
    if attention_bias is None:
        attention_bias = torch.zeros(
            (batch_size, num_kv_heads, query_length, key_length), 
            dtype=dtype, device=query_states.device
        )

    # Apply attention mask to bias
    if attention_mask is not None:
        attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype)
        attention_mask = attention_mask.to(dtype)

    # Call Flash Attention with dynamic masking
    out = flash_dmattn_func(
        query_states, key_states, value_states, 
        attn_mask=attention_mask, attn_bias=attention_bias, 
        scale=softmax_scale, is_causal=is_causal
    )

    return out[0] if isinstance(out, tuple) else out
```

The backend processing stages:

1. **Bias Initialization**: Create zero bias tensor if not provided
2. **Mask Application**: Apply boolean attention mask to bias tensor
3. **Flash Attention Call**: Execute optimized CUDA kernels with sparse patterns

#### Updated Forward Algorithm

The implementation introduces unified block-level skip logic that optimizes computation by skipping entire tiles when they are fully masked:

```cpp
// Forward pass with unified skip logic
for m_block in M_tiles:
    load Q_tile
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)          // Block-level skip decision
        if !any_active:
            advance_pointers()               // Skip computation, advance to next tile
            continue
        
        // Only execute for active tiles
        load K_tile, V_tile                  // Load data only when needed
        S = Q_tile @ K_tile^T + bias_block   // Sparse Q*K^T GEMM
        S_masked = apply_mask(S, mask_block) // Apply dynamic masking
        P = softmax(S_masked, LSE_cache)     // Softmax with LSE caching
        O_partial += P @ V_tile              // Sparse Score*V GEMM
write O
```

Key improvements:
- **Block-level Skip Logic**: OR-reduction over entire (BlockM × BlockN) tile determines if computation is needed
- **Early Skip Decision**: Mask evaluation happens before expensive K/V loading and computation
- **Pointer Management**: Safe pointer advancement ensures correct memory layout for subsequent tiles

#### Updated Backward Algorithm

The backward pass also benefits from the unified skip logic, maintaining numerical correctness while significantly reducing computation for sparse patterns:

```cpp
// Backward pass with unified skip logic
for m_block in reversed(M_tiles):
    load Q_tile, dO_tile
    init accum_dQ
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)              // Same skip decision as forward
        if !any_active:
            advance_pointers_zero_side_outputs() // Skip computation, zero side outputs
            continue
            
        // Only execute for active tiles
        load K_tile, V_tile
        
        # Recompute (identical to forward for active tiles)
        S = Q_tile @ K_tile^T + bias_block
        P = softmax(S, LSE_cache)                // Use cached LSE for stability
        
        # Gradient computation chain (5 GEMMs)
        dV += P^T @ dO_tile                      // Accumulate dV
        dP = dO_tile @ V_tile^T                  // Compute dP
        dS = g(P, dP)                            // dS = (dP - (P ⊙ dP).sum(axis)) * P
        dQ += dS @ K_tile                        // Accumulate dQ
        dK += dS^T @ Q_tile                      // Accumulate dK
    write dQ, accumulate dK, dV
```

Key features:
- **Recomputation Strategy**: Forward computation is recomputed only for active tiles to maintain numerical precision
- **LSE Caching**: Uses cached log-sum-exp values from forward pass for stable softmax recomputation
- **Gradient Chain**: All five gradient GEMMs are skipped for fully masked tiles, maintaining mathematical correctness
- **Zero Handling**: Properly handles zero contributions from skipped tiles in accumulation

#### Skip Logic Correctness

The mathematical correctness of the skip logic relies on the following principles:

1. **Forward Skip**: If a tile is entirely masked (active_mask = 0), its contribution to the output is exactly zero:
   ```
   O_contribution = P @ V = 0 @ V = 0
   ```

2. **Backward Skip**: For fully masked tiles, all intermediate gradients are zero:
   ```
   P = 0  ⟹  dS = 0  ⟹  dQ = dK = dV = 0 (from this tile)
   ```

3. **LSE Preservation**: Skipped tiles don't contribute to the log-sum-exp, maintaining numerical stability.

### Sparse Computation Strategy

### Block-level Skip Logic

The implementation introduces unified block-level skip logic that operates at the tile granularity rather than individual elements:

1. **Tile-level Active Detection**: 
   ```cpp
   any_active = OR_reduce(mask_block)  // Single bit indicating if any position in tile is active
   ```

2. **Skip Decision**: Binary branch based on tile activity:
   ```cpp
   if (!any_active) {
       advance_pointers();              // Forward: skip all computation
       advance_pointers_zero_outputs(); // Backward: skip computation, zero side outputs
       continue;
   }
   ```

3. **Computational Benefits**: 
   - Skip entire K/V loads for inactive tiles
   - Eliminate all 5 GEMMs in backward pass for inactive tiles
   - Reduce memory bandwidth and arithmetic operations proportional to sparsity

### Sparsity Pattern Recognition

The Dynamic Mask Attention implements structured sparsity based on learned importance scores:

1. **Attention Bias Computation**: Attention bias values are computed based on dynamic states derived from value tensors
   - Learned projection matrices map value features to importance scores
   - Coefficient parameters control the dynamic range of importance values
   - Activation functions ensure appropriate bias magnitude

2. **Binary Attention Mask**: 
   - 1.0 for positions that should be computed
   - 0.0 for positions that should be skipped

### Performance Model (Updated)

For block-level sparsity with active tile fraction $p$, skip overhead ratio $\varepsilon$, and early-exit efficiency $\eta$:

$$
\text{Speedup} \approx \frac{1}{p + (1-p)(\varepsilon + \eta \cdot \text{LoadOverhead})}
$$

Where:
- $p$: fraction of active tiles
- $\varepsilon$: skip branching overhead
- $\eta$: efficiency of early memory load exit
- $\text{LoadOverhead}$: relative cost of K/V loading vs computation

Upper bound as $\varepsilon, \eta \to 0$: $1/p$

### Shared Memory Aliasing

The implementation introduces smart shared memory aliasing to reduce footprint and enable larger tile sizes:

1. **sMask ↔ sP Aliasing**: Mask shared memory region is reused for storing softmax probabilities P after mask consumption
2. **sBias ↔ sdS Aliasing**: Bias shared memory region is reused for gradient computations dS
3. **Barrier Synchronization**: Explicit `__syncthreads()` calls ensure safe transitions between aliased usage

```cpp
// Example aliasing pattern
load mask -> sMask
any_active = or_reduce(sMask)
if any_active:
    compute S
    __syncthreads()  // ensure mask fully consumed
    softmax -> write P into aliased region (sP)  // reuse sMask region as sP
    ...
__syncthreads()  // ensure dS consumed
// reuse sBias region as sdS in next iteration
```

### Memory Efficiency Optimizations

1. **Shared Memory Aliasing**: Smart reuse of memory regions (sMask ↔ sP, sBias ↔ sdS) with explicit barrier synchronization
2. **Block-level Skip**: Early exit from computation and memory loading for inactive tiles
3. **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
4. **Register-Optimized Operations**: Critical masking and gradient operations performed in register memory
5. **Coalesced Memory Access**: Optimized access patterns for GPU memory hierarchy
6. **Template Specialization**: Compile-time optimization eliminates runtime branching overhead

## Memory Layout

### Tensor Memory Organization

The Dynamic Mask Attention extends Flash Attention's memory layout to include attention masks and attention bias:

```
Global Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Q:         [batch, seqlen_q, num_heads, head_dim]               │
│ K:         [batch, seqlen_k, num_heads_k, head_dim]             │  
│ V:         [batch, seqlen_k, num_heads_k, head_dim]             │
│ Mask:      [batch, num_heads_k, seqlen_q, seqlen_k]             │
│ Bias:      [batch, num_heads_k, seqlen_q, seqlen_k]             │
│ Output:    [batch, seqlen_q, num_heads, head_dim]               │
└─────────────────────────────────────────────────────────────────┘

Shared Memory Layout (per thread block):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Tile:    [kBlockM, head_dim]     │ K Tile:    [kBlockN, head_dim] │
│ V Tile:    [kBlockN, head_dim]     │ S Tile:    [kBlockM, kBlockN]  │
│ AM Tile:   [kBlockM, kBlockN]      │ Bias Tile: [kBlockM, kBlockN]  │
└─────────────────────────────────────────────────────────────────────┘

Register Memory (per thread):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Frag:    [MMA_M, head_dim/N]     │ K Frag:    [MMA_N, head_dim/N] │
│ V Frag:    [MMA_N, head_dim/N]     │ S Frag:    [MMA_M, MMA_N]      │
│ AM Frag:   [MMA_M, MMA_N]          │ Bias Frag: [MMA_M, MMA_N]      │
│ Acc Frag:  [MMA_M, head_dim/N]     │                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

#### Attention Mask and Attention Bias Loading
```cpp
// Global to Shared Memory (coalesced access)
Tensor tSgBias = local_partition(mBias, smem_tiled_copy_Bias, thread_idx);
Tensor tSsBias = local_partition(sBias, smem_tiled_copy_Bias, thread_idx);

// Each thread loads a contiguous chunk to maximize memory bandwidth
copy(smem_tiled_copy_Bias, tSgBias, tSsBias);

// Shared to Register Memory (bank-conflict-free)
Tensor tSrBias = local_partition(sBias, smem_thr_copy_Bias, thread_idx);
copy(smem_thr_copy_Bias, tSsBias, tSrBias);
```

#### Memory Layout Transformations
```cpp
// Convert MMA accumulator layout to row-column layout for masking
// From: (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
auto convert_layout_acc_rowcol = [](auto layout) {
    return make_layout(
        make_layout(make_shape(Int<2>{}, get<1>(layout.shape())), 
                   make_stride(Int<get<1>(layout.stride())* 2>{}, get<1>(layout.stride()))),
        make_layout(make_shape(Int<2>{}, get<2>(layout.shape())),
                   make_stride(Int<1>{}, Int<2>{}))
    );
};
```

### Shared Memory Optimization

#### Bank Conflict Avoidance
- Attention bias and attention masks use the same copy patterns as Q/K/V to avoid bank conflicts
- Padding added when necessary to ensure 128-bit aligned access
- Thread block size chosen to maximize occupancy while maintaining memory efficiency

#### Memory Coalescing
```cpp
// Example: Loading 128-bit aligned chunks for optimal bandwidth
using SmemCopyAtomBias = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;  // 128-bit loads
using SmemCopyAtomAttnMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
```

## Performance Considerations

### Memory Efficiency
- **Shared Memory Aliasing**: Smart memory reuse (sMask ↔ sP, sBias ↔ sdS) reduces footprint by ~30%
- **Block-level Skip**: Early exit eliminates unnecessary memory loads for inactive tiles
- **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
- **Coalesced Access**: Optimized tensor layouts for GPU memory hierarchy

### Computational Efficiency  
- **Unified Skip Logic**: Both forward and backward passes benefit from block-level computation skipping
- **5-GEMM Chain Skip**: Complete gradient computation chain skipped for inactive tiles
- **Early Branch Decision**: Mask OR-reduction happens before expensive K/V loads
- **Warp-Level Optimization**: Operations optimized for GPU warp execution model

### Scalability
- **Block-level Granularity**: Tile-level sparsity more efficient than element-level for long sequences
- **Multi-Head Support**: Efficient handling of multiple attention heads with per-head sparsity patterns
- **Barrier Optimization**: Minimal synchronization overhead through smart aliasing strategies

### Performance Model

Expected speedup for various sparsity levels:
- **50% sparsity**: ~1.8x speedup
- **75% sparsity**: ~3.2x speedup  
- **90% sparsity**: ~6.5x speedup

Performance factors:
- Skip overhead typically <5% of dense computation time
- Memory bandwidth reduction scales linearly with sparsity
- Shared memory aliasing enables 20-30% larger tile sizes

## API Changes

### New Required Parameters

The Dynamic Mask Attention integration introduces new required parameters to the forward pass:

- **`attn_mask`** (`torch.Tensor`): Attention mask tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Binary mask (1.0 = compute, 0.0 = skip) indicating which positions should be processed
  - Determines the sparsity pattern for computational efficiency

- **`attn_bias`** (`torch.Tensor`): Attention bias tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Contains dynamic attention bias values applied to attention scores before softmax
  - Must have the same dtype and device as Q/K/V tensors

### Updated Function Signature

```python
def fwd(
    q: torch.Tensor,                              # Query tensor
    k: torch.Tensor,                              # Key tensor  
    v: torch.Tensor,                              # Value tensor
    attn_mask: torch.Tensor,                      # Attention mask (REQUIRED)
    attn_bias: torch.Tensor,                      # Attention bias (REQUIRED)
    out: Optional[torch.Tensor] = None,           # Pre-allocated output
    softmax_scale: float = None,                  # Attention scaling
    is_causal: bool = False,                      # Causal masking
    softcap: float = 0.0,                         # Soft capping
    return_softmax: bool = False,                 # Return attention weights
) -> List[torch.Tensor]
```

### Backward Compatibility

**Breaking Change Notice**: The integration requires attention bias and attention mask tensors as mandatory parameters. This is a breaking change from the original Flash Attention API.

**Migration Path**: Users need to:
1. Add attention mask and bias generation logic to attention modules
2. Implement appropriate mask and bias computation within the attention forward pass
3. Ensure proper tensor shapes and dtypes for mask and bias tensors

### Complete Usage Example

```python
import torch
import torch.nn as nn
import flash_dmattn_cuda as flash_dmattn

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Prepare mask and bias tensors with proper shapes
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, self.num_kv_heads, seq_len, seq_len), 
                                      dtype=query_states.dtype, device=query_states.device)
        
        if attention_bias is None:
            attention_bias = torch.zeros((batch_size, self.num_kv_heads, seq_len, seq_len),
                                       dtype=query_states.dtype, device=query_states.device)
        
        # Call Flash Dynamic Mask Attention
        output, _ = flash_dmattn.fwd(
            query_states, key_states, value_states,
            attention_mask, attention_bias,
            None,  # out
            self.scaling,  # softmax_scale
            False,  # is_causal
            0.0,   # softcap
            False  # return_softmax
        )
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)
```
        
        # Call attention implementation
        attn_output, attn_weights = flash_dynamic_mask_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attn_bias,
            scaling=self.scaling,
        )
        
        return attn_output, attn_weights
```

The attention bias generation process:

1. **Value-based Dynamic States**: 
   ```python
   dt_states = self.dt_proj(value_states_flattened)
   dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
   ```

2. **Bias Expansion**: 
   ```python
   attn_bias = dt_states[:, :, None, :].expand(-1, -1, query_len, -1)
   ```

3. **Mask Processing**: Done internally in `_flash_dynamic_mask_attention_forward`


### CUDA Backend: Sparse Attention Computation

The CUDA backend implements the sparse attention computation through `_flash_dynamic_mask_attention_forward`:

```python
def _flash_dynamic_mask_attention_forward(
    query_states, key_states, value_states,
    attention_mask, attention_bias,
    query_length, key_length,
    is_causal, softmax_scale=None, softcap=None,
    target_dtype=None, implementation=None, **kwargs
):
    dtype = query_states.dtype
    min_dtype = torch.finfo(dtype).min
    batch_size, _, num_kv_heads, _ = key_states.shape

    # Initialize attention bias if not provided
    if attention_bias is None:
        attention_bias = torch.zeros(
            (batch_size, num_kv_heads, query_length, key_length), 
            dtype=dtype, device=query_states.device
        )

    # Apply attention mask to bias
    if attention_mask is not None:
        attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype)
        attention_mask = attention_mask.to(dtype)

    # Call Flash Attention with dynamic masking
    out = flash_dmattn_func(
        query_states, key_states, value_states, 
        attn_mask=attention_mask, attn_bias=attention_bias, 
        scale=softmax_scale, is_causal=is_causal
    )

    return out[0] if isinstance(out, tuple) else out
```

The backend processing stages:

1. **Bias Initialization**: Create zero bias tensor if not provided
2. **Mask Application**: Apply boolean attention mask to bias tensor
3. **Flash Attention Call**: Execute optimized CUDA kernels with sparse patterns

#### Updated Forward Algorithm

The implementation introduces unified block-level skip logic that optimizes computation by skipping entire tiles when they are fully masked:

```cpp
// Forward pass with unified skip logic
for m_block in M_tiles:
    load Q_tile
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)          // Block-level skip decision
        if !any_active:
            advance_pointers()               // Skip computation, advance to next tile
            continue
        
        // Only execute for active tiles
        load K_tile, V_tile                  // Load data only when needed
        S = Q_tile @ K_tile^T + bias_block   // Sparse Q*K^T GEMM
        S_masked = apply_mask(S, mask_block) // Apply dynamic masking
        P = softmax(S_masked, LSE_cache)     // Softmax with LSE caching
        O_partial += P @ V_tile              // Sparse Score*V GEMM
write O
```

Key improvements:
- **Block-level Skip Logic**: OR-reduction over entire (BlockM × BlockN) tile determines if computation is needed
- **Early Skip Decision**: Mask evaluation happens before expensive K/V loading and computation
- **Pointer Management**: Safe pointer advancement ensures correct memory layout for subsequent tiles

#### Updated Backward Algorithm

The backward pass also benefits from the unified skip logic, maintaining numerical correctness while significantly reducing computation for sparse patterns:

```cpp
// Backward pass with unified skip logic
for m_block in reversed(M_tiles):
    load Q_tile, dO_tile
    init accum_dQ
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)              // Same skip decision as forward
        if !any_active:
            advance_pointers_zero_side_outputs() // Skip computation, zero side outputs
            continue
            
        // Only execute for active tiles
        load K_tile, V_tile
        
        # Recompute (identical to forward for active tiles)
        S = Q_tile @ K_tile^T + bias_block
        P = softmax(S, LSE_cache)                // Use cached LSE for stability
        
        # Gradient computation chain (5 GEMMs)
        dV += P^T @ dO_tile                      // Accumulate dV
        dP = dO_tile @ V_tile^T                  // Compute dP
        dS = g(P, dP)                            // dS = (dP - (P ⊙ dP).sum(axis)) * P
        dQ += dS @ K_tile                        // Accumulate dQ
        dK += dS^T @ Q_tile                      // Accumulate dK
    write dQ, accumulate dK, dV
```

Key features:
- **Recomputation Strategy**: Forward computation is recomputed only for active tiles to maintain numerical precision
- **LSE Caching**: Uses cached log-sum-exp values from forward pass for stable softmax recomputation
- **Gradient Chain**: All five gradient GEMMs are skipped for fully masked tiles, maintaining mathematical correctness
- **Zero Handling**: Properly handles zero contributions from skipped tiles in accumulation

#### Skip Logic Correctness

The mathematical correctness of the skip logic relies on the following principles:

1. **Forward Skip**: If a tile is entirely masked (active_mask = 0), its contribution to the output is exactly zero:
   ```
   O_contribution = P @ V = 0 @ V = 0
   ```

2. **Backward Skip**: For fully masked tiles, all intermediate gradients are zero:
   ```
   P = 0  ⟹  dS = 0  ⟹  dQ = dK = dV = 0 (from this tile)
   ```

3. **LSE Preservation**: Skipped tiles don't contribute to the log-sum-exp, maintaining numerical stability.

### Sparse Computation Strategy

### Block-level Skip Logic

The implementation introduces unified block-level skip logic that operates at the tile granularity rather than individual elements:

1. **Tile-level Active Detection**: 
   ```cpp
   any_active = OR_reduce(mask_block)  // Single bit indicating if any position in tile is active
   ```

2. **Skip Decision**: Binary branch based on tile activity:
   ```cpp
   if (!any_active) {
       advance_pointers();              // Forward: skip all computation
       advance_pointers_zero_outputs(); // Backward: skip computation, zero side outputs
       continue;
   }
   ```

3. **Computational Benefits**: 
   - Skip entire K/V loads for inactive tiles
   - Eliminate all 5 GEMMs in backward pass for inactive tiles
   - Reduce memory bandwidth and arithmetic operations proportional to sparsity

### Sparsity Pattern Recognition

The Dynamic Mask Attention implements structured sparsity based on learned importance scores:

1. **Attention Bias Computation**: Attention bias values are computed based on dynamic states derived from value tensors
   - Learned projection matrices map value features to importance scores
   - Coefficient parameters control the dynamic range of importance values
   - Activation functions ensure appropriate bias magnitude

2. **Binary Attention Mask**: 
   - 1.0 for positions that should be computed
   - 0.0 for positions that should be skipped

### Performance Model (Updated)

For block-level sparsity with active tile fraction $p$, skip overhead ratio $\varepsilon$, and early-exit efficiency $\eta$:

$$
\text{Speedup} \approx \frac{1}{p + (1-p)(\varepsilon + \eta \cdot \text{LoadOverhead})}
$$

Where:
- $p$: fraction of active tiles
- $\varepsilon$: skip branching overhead
- $\eta$: efficiency of early memory load exit
- $\text{LoadOverhead}$: relative cost of K/V loading vs computation

Upper bound as $\varepsilon, \eta \to 0$: $1/p$

### Shared Memory Aliasing

The implementation introduces smart shared memory aliasing to reduce footprint and enable larger tile sizes:

1. **sMask ↔ sP Aliasing**: Mask shared memory region is reused for storing softmax probabilities P after mask consumption
2. **sBias ↔ sdS Aliasing**: Bias shared memory region is reused for gradient computations dS
3. **Barrier Synchronization**: Explicit `__syncthreads()` calls ensure safe transitions between aliased usage

```cpp
// Example aliasing pattern
load mask -> sMask
any_active = or_reduce(sMask)
if any_active:
    compute S
    __syncthreads()  // ensure mask fully consumed
    softmax -> write P into aliased region (sP)  // reuse sMask region as sP
    ...
__syncthreads()  // ensure dS consumed
// reuse sBias region as sdS in next iteration
```

### Memory Efficiency Optimizations

1. **Shared Memory Aliasing**: Smart reuse of memory regions (sMask ↔ sP, sBias ↔ sdS) with explicit barrier synchronization
2. **Block-level Skip**: Early exit from computation and memory loading for inactive tiles
3. **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
4. **Register-Optimized Operations**: Critical masking and gradient operations performed in register memory
5. **Coalesced Memory Access**: Optimized access patterns for GPU memory hierarchy
6. **Template Specialization**: Compile-time optimization eliminates runtime branching overhead

## Memory Layout

### Tensor Memory Organization

The Dynamic Mask Attention extends Flash Attention's memory layout to include attention masks and attention bias:

```
Global Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Q:         [batch, seqlen_q, num_heads, head_dim]               │
│ K:         [batch, seqlen_k, num_heads_k, head_dim]            │  
│ V:         [batch, seqlen_k, num_heads_k, head_dim]            │
│ AttnMask:  [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Bias:      [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Output:    [batch, seqlen_q, num_heads, head_dim]               │
└─────────────────────────────────────────────────────────────────┘

Shared Memory Layout (per thread block):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Tile:    [kBlockM, head_dim]     │ K Tile:    [kBlockN, head_dim] │
│ V Tile:    [kBlockN, head_dim]     │ S Tile:    [kBlockM, kBlockN]  │
│ AM Tile:   [kBlockM, kBlockN]      │ Bias Tile: [kBlockM, kBlockN]  │
└─────────────────────────────────────────────────────────────────────┘

Register Memory (per thread):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Frag:    [MMA_M, head_dim/N]     │ K Frag:    [MMA_N, head_dim/N] │
│ V Frag:    [MMA_N, head_dim/N]     │ S Frag:    [MMA_M, MMA_N]      │
│ AM Frag:   [MMA_M, MMA_N]          │ Bias Frag: [MMA_M, MMA_N]      │
│ Acc Frag:  [MMA_M, head_dim/N]     │                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

#### Attention Mask and Attention Bias Loading
```cpp
// Global to Shared Memory (coalesced access)
Tensor tSgBias = local_partition(mBias, smem_tiled_copy_Bias, thread_idx);
Tensor tSsBias = local_partition(sBias, smem_tiled_copy_Bias, thread_idx);

// Each thread loads a contiguous chunk to maximize memory bandwidth
copy(smem_tiled_copy_Bias, tSgBias, tSsBias);

// Shared to Register Memory (bank-conflict-free)
Tensor tSrBias = local_partition(sBias, smem_thr_copy_Bias, thread_idx);
copy(smem_thr_copy_Bias, tSsBias, tSrBias);
```

#### Memory Layout Transformations
```cpp
// Convert MMA accumulator layout to row-column layout for masking
// From: (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
auto convert_layout_acc_rowcol = [](auto layout) {
    return make_layout(
        make_layout(make_shape(Int<2>{}, get<1>(layout.shape())), 
                   make_stride(Int<get<1>(layout.stride())* 2>{}, get<1>(layout.stride()))),
        make_layout(make_shape(Int<2>{}, get<2>(layout.shape())),
                   make_stride(Int<1>{}, Int<2>{}))
    );
};
```

### Shared Memory Optimization

#### Bank Conflict Avoidance
- Attention bias and attention masks use the same copy patterns as Q/K/V to avoid bank conflicts
- Padding added when necessary to ensure 128-bit aligned access
- Thread block size chosen to maximize occupancy while maintaining memory efficiency

#### Memory Coalescing
```cpp
// Example: Loading 128-bit aligned chunks for optimal bandwidth
using SmemCopyAtomBias = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;  // 128-bit loads
using SmemCopyAtomAttnMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
```

## Performance Considerations

### Memory Efficiency
- **Shared Memory Aliasing**: Smart memory reuse (sMask ↔ sP, sBias ↔ sdS) reduces footprint by ~30%
- **Block-level Skip**: Early exit eliminates unnecessary memory loads for inactive tiles
- **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
- **Coalesced Access**: Optimized tensor layouts for GPU memory hierarchy

### Computational Efficiency  
- **Unified Skip Logic**: Both forward and backward passes benefit from block-level computation skipping
- **5-GEMM Chain Skip**: Complete gradient computation chain skipped for inactive tiles
- **Early Branch Decision**: Mask OR-reduction happens before expensive K/V loads
- **Warp-Level Optimization**: Operations optimized for GPU warp execution model

### Scalability
- **Block-level Granularity**: Tile-level sparsity more efficient than element-level for long sequences
- **Multi-Head Support**: Efficient handling of multiple attention heads with per-head sparsity patterns
- **Barrier Optimization**: Minimal synchronization overhead through smart aliasing strategies

### Performance Model

Expected speedup for various sparsity levels:
- **50% sparsity**: ~1.8x speedup
- **75% sparsity**: ~3.2x speedup  
- **90% sparsity**: ~6.5x speedup

Performance factors:
- Skip overhead typically <5% of dense computation time
- Memory bandwidth reduction scales linearly with sparsity
- Shared memory aliasing enables 20-30% larger tile sizes

## API Changes

### New Required Parameters

The Dynamic Mask Attention integration introduces new required parameters to the forward pass:

- **`attn_mask`** (`torch.Tensor`): Attention mask tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Binary mask (1.0 = compute, 0.0 = skip) indicating which positions should be processed
  - Determines the sparsity pattern for computational efficiency

- **`attn_bias`** (`torch.Tensor`): Attention bias tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Contains dynamic attention bias values applied to attention scores before softmax
  - Must have the same dtype and device as Q/K/V tensors

### Updated Function Signature

```python
def fwd(
    q: torch.Tensor,                              # Query tensor
    k: torch.Tensor,                              # Key tensor  
    v: torch.Tensor,                              # Value tensor
    attn_mask: torch.Tensor,                      # Attention mask (REQUIRED)
    attn_bias: torch.Tensor,                      # Attention bias (REQUIRED)
    out: Optional[torch.Tensor] = None,           # Pre-allocated output
    softmax_scale: float = None,                  # Attention scaling
    is_causal: bool = False,                      # Causal masking
    softcap: float = 0.0,                         # Soft capping
    return_softmax: bool = False,                 # Return attention weights
) -> List[torch.Tensor]
```

### Backward Compatibility

**Breaking Change Notice**: The integration requires attention bias and attention mask tensors as mandatory parameters. This is a breaking change from the original Flash Attention API.

**Migration Path**: Users need to:
1. Add attention mask and bias generation logic to attention modules
2. Implement appropriate mask and bias computation within the attention forward pass
3. Ensure proper tensor shapes and dtypes for mask and bias tensors

### Complete Usage Example

```python
import torch
import torch.nn as nn
import flash_dmattn_cuda as flash_dmattn

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Prepare mask and bias tensors with proper shapes
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, self.num_kv_heads, seq_len, seq_len), 
                                      dtype=query_states.dtype, device=query_states.device)
        
        if attention_bias is None:
            attention_bias = torch.zeros((batch_size, self.num_kv_heads, seq_len, seq_len),
                                       dtype=query_states.dtype, device=query_states.device)
        
        # Call Flash Dynamic Mask Attention
        output, _ = flash_dmattn.fwd(
            query_states, key_states, value_states,
            attention_mask, attention_bias,
            None,  # out
            self.scaling,  # softmax_scale
            False,  # is_causal
            0.0,   # softcap
            False  # return_softmax
        )
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)
```
        
        # Call attention implementation
        attn_output, attn_weights = flash_dynamic_mask_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attn_bias,
            scaling=self.scaling,
        )
        
        return attn_output, attn_weights
```

The attention bias generation process:

1. **Value-based Dynamic States**: 
   ```python
   dt_states = self.dt_proj(value_states_flattened)
   dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
   ```

2. **Bias Expansion**: 
   ```python
   attn_bias = dt_states[:, :, None, :].expand(-1, -1, query_len, -1)
   ```

3. **Mask Processing**: Done internally in `_flash_dynamic_mask_attention_forward`


### CUDA Backend: Sparse Attention Computation

The CUDA backend implements the sparse attention computation through `_flash_dynamic_mask_attention_forward`:

```python
def _flash_dynamic_mask_attention_forward(
    query_states, key_states, value_states,
    attention_mask, attention_bias,
    query_length, key_length,
    is_causal, softmax_scale=None, softcap=None,
    target_dtype=None, implementation=None, **kwargs
):
    dtype = query_states.dtype
    min_dtype = torch.finfo(dtype).min
    batch_size, _, num_kv_heads, _ = key_states.shape

    # Initialize attention bias if not provided
    if attention_bias is None:
        attention_bias = torch.zeros(
            (batch_size, num_kv_heads, query_length, key_length), 
            dtype=dtype, device=query_states.device
        )

    # Apply attention mask to bias
    if attention_mask is not None:
        attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype)
        attention_mask = attention_mask.to(dtype)

    # Call Flash Attention with dynamic masking
    out = flash_dmattn_func(
        query_states, key_states, value_states, 
        attn_mask=attention_mask, attn_bias=attention_bias, 
        scale=softmax_scale, is_causal=is_causal
    )

    return out[0] if isinstance(out, tuple) else out
```

The backend processing stages:

1. **Bias Initialization**: Create zero bias tensor if not provided
2. **Mask Application**: Apply boolean attention mask to bias tensor
3. **Flash Attention Call**: Execute optimized CUDA kernels with sparse patterns

#### Updated Forward Algorithm

The implementation introduces unified block-level skip logic that optimizes computation by skipping entire tiles when they are fully masked:

```cpp
// Forward pass with unified skip logic
for m_block in M_tiles:
    load Q_tile
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)          // Block-level skip decision
        if !any_active:
            advance_pointers()               // Skip computation, advance to next tile
            continue
        
        // Only execute for active tiles
        load K_tile, V_tile                  // Load data only when needed
        S = Q_tile @ K_tile^T + bias_block   // Sparse Q*K^T GEMM
        S_masked = apply_mask(S, mask_block) // Apply dynamic masking
        P = softmax(S_masked, LSE_cache)     // Softmax with LSE caching
        O_partial += P @ V_tile              // Sparse Score*V GEMM
write O
```

Key improvements:
- **Block-level Skip Logic**: OR-reduction over entire (BlockM × BlockN) tile determines if computation is needed
- **Early Skip Decision**: Mask evaluation happens before expensive K/V loading and computation
- **Pointer Management**: Safe pointer advancement ensures correct memory layout for subsequent tiles

#### Updated Backward Algorithm

The backward pass also benefits from the unified skip logic, maintaining numerical correctness while significantly reducing computation for sparse patterns:

```cpp
// Backward pass with unified skip logic
for m_block in reversed(M_tiles):
    load Q_tile, dO_tile
    init accum_dQ
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)              // Same skip decision as forward
        if !any_active:
            advance_pointers_zero_side_outputs() // Skip computation, zero side outputs
            continue
            
        // Only execute for active tiles
        load K_tile, V_tile
        
        # Recompute (identical to forward for active tiles)
        S = Q_tile @ K_tile^T + bias_block
        P = softmax(S, LSE_cache)                // Use cached LSE for stability
        
        # Gradient computation chain (5 GEMMs)
        dV += P^T @ dO_tile                      // Accumulate dV
        dP = dO_tile @ V_tile^T                  // Compute dP
        dS = g(P, dP)                            // dS = (dP - (P ⊙ dP).sum(axis)) * P
        dQ += dS @ K_tile                        // Accumulate dQ
        dK += dS^T @ Q_tile                      // Accumulate dK
    write dQ, accumulate dK, dV
```

Key features:
- **Recomputation Strategy**: Forward computation is recomputed only for active tiles to maintain numerical precision
- **LSE Caching**: Uses cached log-sum-exp values from forward pass for stable softmax recomputation
- **Gradient Chain**: All five gradient GEMMs are skipped for fully masked tiles, maintaining mathematical correctness
- **Zero Handling**: Properly handles zero contributions from skipped tiles in accumulation

#### Skip Logic Correctness

The mathematical correctness of the skip logic relies on the following principles:

1. **Forward Skip**: If a tile is entirely masked (active_mask = 0), its contribution to the output is exactly zero:
   ```
   O_contribution = P @ V = 0 @ V = 0
   ```

2. **Backward Skip**: For fully masked tiles, all intermediate gradients are zero:
   ```
   P = 0  ⟹  dS = 0  ⟹  dQ = dK = dV = 0 (from this tile)
   ```

3. **LSE Preservation**: Skipped tiles don't contribute to the log-sum-exp, maintaining numerical stability.

### Sparse Computation Strategy

### Block-level Skip Logic

The implementation introduces unified block-level skip logic that operates at the tile granularity rather than individual elements:

1. **Tile-level Active Detection**: 
   ```cpp
   any_active = OR_reduce(mask_block)  // Single bit indicating if any position in tile is active
   ```

2. **Skip Decision**: Binary branch based on tile activity:
   ```cpp
   if (!any_active) {
       advance_pointers();              // Forward: skip all computation
       advance_pointers_zero_outputs(); // Backward: skip computation, zero side outputs
       continue;
   }
   ```

3. **Computational Benefits**: 
   - Skip entire K/V loads for inactive tiles
   - Eliminate all 5 GEMMs in backward pass for inactive tiles
   - Reduce memory bandwidth and arithmetic operations proportional to sparsity

### Sparsity Pattern Recognition

The Dynamic Mask Attention implements structured sparsity based on learned importance scores:

1. **Attention Bias Computation**: Attention bias values are computed based on dynamic states derived from value tensors
   - Learned projection matrices map value features to importance scores
   - Coefficient parameters control the dynamic range of importance values
   - Activation functions ensure appropriate bias magnitude

2. **Binary Attention Mask**: 
   - 1.0 for positions that should be computed
   - 0.0 for positions that should be skipped

### Performance Model (Updated)

For block-level sparsity with active tile fraction $p$, skip overhead ratio $\varepsilon$, and early-exit efficiency $\eta$:

$$
\text{Speedup} \approx \frac{1}{p + (1-p)(\varepsilon + \eta \cdot \text{LoadOverhead})}
$$

Where:
- $p$: fraction of active tiles
- $\varepsilon$: skip branching overhead
- $\eta$: efficiency of early memory load exit
- $\text{LoadOverhead}$: relative cost of K/V loading vs computation

Upper bound as $\varepsilon, \eta \to 0$: $1/p$

### Shared Memory Aliasing

The implementation introduces smart shared memory aliasing to reduce footprint and enable larger tile sizes:

1. **sMask ↔ sP Aliasing**: Mask shared memory region is reused for storing softmax probabilities P after mask consumption
2. **sBias ↔ sdS Aliasing**: Bias shared memory region is reused for gradient computations dS
3. **Barrier Synchronization**: Explicit `__syncthreads()` calls ensure safe transitions between aliased usage

```cpp
// Example aliasing pattern
load mask -> sMask
any_active = or_reduce(sMask)
if any_active:
    compute S
    __syncthreads()  // ensure mask fully consumed
    softmax -> write P into aliased region (sP)  // reuse sMask region as sP
    ...
__syncthreads()  // ensure dS consumed
// reuse sBias region as sdS in next iteration
```

### Memory Efficiency Optimizations

1. **Shared Memory Aliasing**: Smart reuse of memory regions (sMask ↔ sP, sBias ↔ sdS) with explicit barrier synchronization
2. **Block-level Skip**: Early exit from computation and memory loading for inactive tiles
3. **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
4. **Register-Optimized Operations**: Critical masking and gradient operations performed in register memory
5. **Coalesced Memory Access**: Optimized access patterns for GPU memory hierarchy
6. **Template Specialization**: Compile-time optimization eliminates runtime branching overhead

## Memory Layout

### Tensor Memory Organization

The Dynamic Mask Attention extends Flash Attention's memory layout to include attention masks and attention bias:

```
Global Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Q:         [batch, seqlen_q, num_heads, head_dim]               │
│ K:         [batch, seqlen_k, num_heads_k, head_dim]            │  
│ V:         [batch, seqlen_k, num_heads_k, head_dim]            │
│ AttnMask:  [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Bias:      [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Output:    [batch, seqlen_q, num_heads, head_dim]               │
└─────────────────────────────────────────────────────────────────┘

Shared Memory Layout (per thread block):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Tile:    [kBlockM, head_dim]     │ K Tile:    [kBlockN, head_dim] │
│ V Tile:    [kBlockN, head_dim]     │ S Tile:    [kBlockM, kBlockN]  │
│ AM Tile:   [kBlockM, kBlockN]      │ Bias Tile: [kBlockM, kBlockN]  │
└─────────────────────────────────────────────────────────────────────┘

Register Memory (per thread):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Frag:    [MMA_M, head_dim/N]     │ K Frag:    [MMA_N, head_dim/N] │
│ V Frag:    [MMA_N, head_dim/N]     │ S Frag:    [MMA_M, MMA_N]      │
│ AM Frag:   [MMA_M, MMA_N]          │ Bias Frag: [MMA_M, MMA_N]      │
│ Acc Frag:  [MMA_M, head_dim/N]     │                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

#### Attention Mask and Attention Bias Loading
```cpp
// Global to Shared Memory (coalesced access)
Tensor tSgBias = local_partition(mBias, smem_tiled_copy_Bias, thread_idx);
Tensor tSsBias = local_partition(sBias, smem_tiled_copy_Bias, thread_idx);

// Each thread loads a contiguous chunk to maximize memory bandwidth
copy(smem_tiled_copy_Bias, tSgBias, tSsBias);

// Shared to Register Memory (bank-conflict-free)
Tensor tSrBias = local_partition(sBias, smem_thr_copy_Bias, thread_idx);
copy(smem_thr_copy_Bias, tSsBias, tSrBias);
```

#### Memory Layout Transformations
```cpp
// Convert MMA accumulator layout to row-column layout for masking
// From: (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
auto convert_layout_acc_rowcol = [](auto layout) {
    return make_layout(
        make_layout(make_shape(Int<2>{}, get<1>(layout.shape())), 
                   make_stride(Int<get<1>(layout.stride())* 2>{}, get<1>(layout.stride()))),
        make_layout(make_shape(Int<2>{}, get<2>(layout.shape())),
                   make_stride(Int<1>{}, Int<2>{}))
    );
};
```

### Shared Memory Optimization

#### Bank Conflict Avoidance
- Attention bias and attention masks use the same copy patterns as Q/K/V to avoid bank conflicts
- Padding added when necessary to ensure 128-bit aligned access
- Thread block size chosen to maximize occupancy while maintaining memory efficiency

#### Memory Coalescing
```cpp
// Example: Loading 128-bit aligned chunks for optimal bandwidth
using SmemCopyAtomBias = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;  // 128-bit loads
using SmemCopyAtomAttnMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
```

## Performance Considerations

### Memory Efficiency
- **Shared Memory Aliasing**: Smart memory reuse (sMask ↔ sP, sBias ↔ sdS) reduces footprint by ~30%
- **Block-level Skip**: Early exit eliminates unnecessary memory loads for inactive tiles
- **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
- **Coalesced Access**: Optimized tensor layouts for GPU memory hierarchy

### Computational Efficiency  
- **Unified Skip Logic**: Both forward and backward passes benefit from block-level computation skipping
- **5-GEMM Chain Skip**: Complete gradient computation chain skipped for inactive tiles
- **Early Branch Decision**: Mask OR-reduction happens before expensive K/V loads
- **Warp-Level Optimization**: Operations optimized for GPU warp execution model

### Scalability
- **Block-level Granularity**: Tile-level sparsity more efficient than element-level for long sequences
- **Multi-Head Support**: Efficient handling of multiple attention heads with per-head sparsity patterns
- **Barrier Optimization**: Minimal synchronization overhead through smart aliasing strategies

### Performance Model

Expected speedup for various sparsity levels:
- **50% sparsity**: ~1.8x speedup
- **75% sparsity**: ~3.2x speedup  
- **90% sparsity**: ~6.5x speedup

Performance factors:
- Skip overhead typically <5% of dense computation time
- Memory bandwidth reduction scales linearly with sparsity
- Shared memory aliasing enables 20-30% larger tile sizes

## API Changes

### New Required Parameters

The Dynamic Mask Attention integration introduces new required parameters to the forward pass:

- **`attn_mask`** (`torch.Tensor`): Attention mask tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Binary mask (1.0 = compute, 0.0 = skip) indicating which positions should be processed
  - Determines the sparsity pattern for computational efficiency

- **`attn_bias`** (`torch.Tensor`): Attention bias tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Contains dynamic attention bias values applied to attention scores before softmax
  - Must have the same dtype and device as Q/K/V tensors

### Updated Function Signature

```python
def fwd(
    q: torch.Tensor,                              # Query tensor
    k: torch.Tensor,                              # Key tensor  
    v: torch.Tensor,                              # Value tensor
    attn_mask: torch.Tensor,                      # Attention mask (REQUIRED)
    attn_bias: torch.Tensor,                      # Attention bias (REQUIRED)
    out: Optional[torch.Tensor] = None,           # Pre-allocated output
    softmax_scale: float = None,                  # Attention scaling
    is_causal: bool = False,                      # Causal masking
    softcap: float = 0.0,                         # Soft capping
    return_softmax: bool = False,                 # Return attention weights
) -> List[torch.Tensor]
```

### Backward Compatibility

**Breaking Change Notice**: The integration requires attention bias and attention mask tensors as mandatory parameters. This is a breaking change from the original Flash Attention API.

**Migration Path**: Users need to:
1. Add attention mask and bias generation logic to attention modules
2. Implement appropriate mask and bias computation within the attention forward pass
3. Ensure proper tensor shapes and dtypes for mask and bias tensors

### Complete Usage Example

```python
import torch
import torch.nn as nn
from flash_dmattn.integration.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Prepare mask and bias tensors with proper shapes
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, self.num_kv_heads, seq_len, seq_len), 
                                      dtype=query_states.dtype, device=query_states.device)
        
        if attention_bias is None:
            attention_bias = torch.zeros((batch_size, self.num_kv_heads, seq_len, seq_len),
                                       dtype=query_states.dtype, device=query_states.device)
        
        # Call attention implementation
        attn_output, attn_weights = flash_dynamic_mask_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            scaling=self.scaling,
        )
        
        return attn_output, attn_weights
```

The attention bias generation process:

1. **Value-based Dynamic States**:
   ```python
   dt_states = self.dt_proj(value_states_flattened)
   dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
   ```

2. **Bias Expansion**: 
   ```python
   attn_bias = dt_states[:, :, None, :].expand(-1, -1, query_len, -1)
   ```

3. **Mask Processing**: Done internally in `_flash_dynamic_mask_attention_forward`


### CUDA Backend: Sparse Attention Computation

The CUDA backend implements the sparse attention computation through `_flash_dynamic_mask_attention_forward`:

```python
def _flash_dynamic_mask_attention_forward(
    query_states, key_states, value_states,
    attention_mask, attention_bias,
    query_length, key_length,
    is_causal, softmax_scale=None, softcap=None,
    target_dtype=None, implementation=None, **kwargs
):
    dtype = query_states.dtype
    min_dtype = torch.finfo(dtype).min
    batch_size, _, num_kv_heads, _ = key_states.shape

    # Initialize attention bias if not provided
    if attention_bias is None:
        attention_bias = torch.zeros(
            (batch_size, num_kv_heads, query_length, key_length), 
            dtype=dtype, device=query_states.device
        )

    # Apply attention mask to bias
    if attention_mask is not None:
        attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype)
        attention_mask = attention_mask.to(dtype)

    # Call Flash Attention with dynamic masking
    out = flash_dmattn_func(
        query_states, key_states, value_states, 
        attn_mask=attention_mask, attn_bias=attention_bias, 
        scale=softmax_scale, is_causal=is_causal
    )

    return out[0] if isinstance(out, tuple) else out
```

The backend processing stages:

1. **Bias Initialization**: Create zero bias tensor if not provided
2. **Mask Application**: Apply boolean attention mask to bias tensor
3. **Flash Attention Call**: Execute optimized CUDA kernels with sparse patterns

#### Forward Algorithm

The implementation introduces unified block-level skip logic that optimizes computation by skipping entire tiles when they are fully masked:

```cpp
// Forward pass with unified skip logic
for m_block in M_tiles:
    load Q_tile
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)          // Block-level skip decision
        if !any_active:
            advance_pointers()               // Skip computation, advance to next tile
            continue
        
        // Only execute for active tiles
        load K_tile, V_tile                  // Load data only when needed
        S = Q_tile @ K_tile^T + bias_block   // Sparse Q*K^T GEMM
        S_masked = apply_mask(S, mask_block) // Apply dynamic masking
        P = softmax(S_masked, LSE_cache)     // Softmax with LSE caching
        O_partial += P @ V_tile              // Sparse Score*V GEMM
write O
```

Key improvements:
- **Block-level Skip Logic**: OR-reduction over entire (BlockM × BlockN) tile determines if computation is needed
- **Early Skip Decision**: Mask evaluation happens before expensive K/V loading and computation
- **Pointer Management**: Safe pointer advancement ensures correct memory layout for subsequent tiles

#### Backward Algorithm

The backward pass also benefits from the unified skip logic, maintaining numerical correctness while significantly reducing computation for sparse patterns:

```cpp
// Backward pass with unified skip logic
for m_block in reversed(M_tiles):
    load Q_tile, dO_tile
    init accum_dQ
    for n_block in N_tiles_stream:
        load mask_block
        any_active = OR(mask_block)              // Same skip decision as forward
        if !any_active:
            advance_pointers_zero_side_outputs() // Skip computation, zero side outputs
            continue
            
        // Only execute for active tiles
        load K_tile, V_tile
        
        // Recompute (identical to forward for active tiles)
        S = Q_tile @ K_tile^T + bias_block
        P = softmax(S, LSE_cache)                // Use cached LSE for stability

        // Gradient computation chain (5 GEMMs)
        dV += P^T @ dO_tile                      // Accumulate dV
        dP = dO_tile @ V_tile^T                  // Compute dP
        dS = g(P, dP)                            // dS = (dP - (P ⊙ dP).sum(axis)) * P
        dQ += dS @ K_tile                        // Accumulate dQ
        dK += dS^T @ Q_tile                      // Accumulate dK
    write dQ, accumulate dK, dV
```

Key features:
- **Recomputation Strategy**: Forward computation is recomputed only for active tiles to maintain numerical precision
- **LSE Caching**: Uses cached log-sum-exp values from forward pass for stable softmax recomputation
- **Gradient Chain**: All five gradient GEMMs are skipped for fully masked tiles, maintaining mathematical correctness
- **Zero Handling**: Properly handles zero contributions from skipped tiles in accumulation

#### Skip Logic Correctness

The mathematical correctness of the skip logic relies on the following principles:

1. **Forward Skip**: If a tile is entirely masked (active_mask = 0), its contribution to the output is exactly zero:
   ```
   O_contribution = P @ V = 0 @ V = 0
   ```

2. **Backward Skip**: For fully masked tiles, all intermediate gradients are zero:
   ```
   P = 0  ⟹  dS = 0  ⟹  dQ = dK = dV = 0 (from this tile)
   ```

3. **LSE Preservation**: Skipped tiles don't contribute to the log-sum-exp, maintaining numerical stability.

### Sparse Computation Strategy

### Block-level Skip Logic

The implementation introduces unified block-level skip logic that operates at the tile granularity rather than individual elements:

1. **Tile-level Active Detection**: 
   ```cpp
   any_active = OR_reduce(mask_block)  // Single bit indicating if any position in tile is active
   ```

2. **Skip Decision**: Binary branch based on tile activity:
   ```cpp
   if (!any_active) {
       advance_pointers();              // Forward: skip all computation
       advance_pointers_zero_outputs(); // Backward: skip computation, zero side outputs
       continue;
   }
   ```

3. **Computational Benefits**: 
   - Skip entire K/V loads for inactive tiles
   - Eliminate all 5 GEMMs in backward pass for inactive tiles
   - Reduce memory bandwidth and arithmetic operations proportional to sparsity

### Sparsity Pattern Recognition

The Dynamic Mask Attention implements structured sparsity based on learned importance scores:

1. **Attention Bias Computation**: Attention bias values are computed based on dynamic states derived from value tensors
   - Learned projection matrices map value features to importance scores
   - Coefficient parameters control the dynamic range of importance values
   - Activation functions ensure appropriate bias magnitude

2. **Binary Attention Mask**: 
   - 1.0 for positions that should be computed
   - 0.0 for positions that should be skipped

### Performance Model

For block-level sparsity with active tile fraction $p$, skip overhead ratio $\varepsilon$, and early-exit efficiency $\eta$:

$$
\text{Speedup} \approx \frac{1}{p + (1-p)(\varepsilon + \eta \cdot \text{LoadOverhead})}
$$

Where:
- $p$: fraction of active tiles
- $\varepsilon$: skip branching overhead
- $\eta$: efficiency of early memory load exit
- $\text{LoadOverhead}$: relative cost of K/V loading vs computation

Upper bound as $\varepsilon, \eta \to 0$: $1/p$

### Shared Memory Aliasing

The implementation introduces smart shared memory aliasing to reduce footprint and enable larger tile sizes:

1. **sMask ↔ sP Aliasing**: Mask shared memory region is reused for storing softmax probabilities P after mask consumption
2. **sBias ↔ sdS Aliasing**: Bias shared memory region is reused for gradient computations dS
3. **Barrier Synchronization**: Explicit `__syncthreads()` calls ensure safe transitions between aliased usage

```cpp
// Example aliasing pattern
load mask -> sMask
any_active = or_reduce(sMask)
if any_active:
    compute S
    __syncthreads()  // ensure mask fully consumed
    softmax -> write P into aliased region (sP)  // reuse sMask region as sP
    ...
__syncthreads()  // ensure dS consumed
// reuse sBias region as sdS in next iteration
```

### Memory Efficiency Optimizations

1. **Shared Memory Aliasing**: Smart reuse of memory regions (sMask ↔ sP, sBias ↔ sdS) with explicit barrier synchronization
2. **Block-level Skip**: Early exit from computation and memory loading for inactive tiles
3. **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
4. **Register-Optimized Operations**: Critical masking and gradient operations performed in register memory
5. **Coalesced Memory Access**: Optimized access patterns for GPU memory hierarchy
6. **Template Specialization**: Compile-time optimization eliminates runtime branching overhead

## Memory Layout

### Tensor Memory Organization

The Dynamic Mask Attention extends Flash Attention's memory layout to include attention masks and attention bias:

```
Global Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Q:         [batch, seqlen_q, num_heads, head_dim]               │
│ K:         [batch, seqlen_k, num_heads_k, head_dim]             │  
│ V:         [batch, seqlen_k, num_heads_k, head_dim]             │
│ AttnMask:  [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Bias:      [batch, num_kv_heads, seqlen_q, seqlen_k]            │
│ Output:    [batch, seqlen_q, num_heads, head_dim]               │
└─────────────────────────────────────────────────────────────────┘

Shared Memory Layout (per thread block):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Tile:    [kBlockM, head_dim]     │ K Tile:    [kBlockN, head_dim] │
│ V Tile:    [kBlockN, head_dim]     │ S Tile:    [kBlockM, kBlockN]  │
│ AM Tile:   [kBlockM, kBlockN]      │ Bias Tile: [kBlockM, kBlockN]  │
└─────────────────────────────────────────────────────────────────────┘

Register Memory (per thread):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Frag:    [MMA_M, head_dim/N]     │ K Frag:    [MMA_N, head_dim/N] │
│ V Frag:    [MMA_N, head_dim/N]     │ S Frag:    [MMA_M, MMA_N]      │
│ AM Frag:   [MMA_M, MMA_N]          │ Bias Frag: [MMA_M, MMA_N]      │
│ Acc Frag:  [MMA_M, head_dim/N]     │                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

#### Attention Mask and Attention Bias Loading
```cpp
// Global to Shared Memory (coalesced access)
Tensor tSgBias = local_partition(mBias, smem_tiled_copy_Bias, thread_idx);
Tensor tSsBias = local_partition(sBias, smem_tiled_copy_Bias, thread_idx);

// Each thread loads a contiguous chunk to maximize memory bandwidth
copy(smem_tiled_copy_Bias, tSgBias, tSsBias);

// Shared to Register Memory (bank-conflict-free)
Tensor tSrBias = local_partition(sBias, smem_thr_copy_Bias, thread_idx);
copy(smem_thr_copy_Bias, tSsBias, tSrBias);
```

#### Memory Layout Transformations
```cpp
// Convert MMA accumulator layout to row-column layout for masking
// From: (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
auto convert_layout_acc_rowcol = [](auto layout) {
    return make_layout(
        make_layout(make_shape(Int<2>{}, get<1>(layout.shape())), 
                   make_stride(Int<get<1>(layout.stride())* 2>{}, get<1>(layout.stride()))),
        make_layout(make_shape(Int<2>{}, get<2>(layout.shape())),
                   make_stride(Int<1>{}, Int<2>{}))
    );
};
```

### Shared Memory Optimization

#### Bank Conflict Avoidance
- Attention bias and attention masks use the same copy patterns as Q/K/V to avoid bank conflicts
- Padding added when necessary to ensure 128-bit aligned access
- Thread block size chosen to maximize occupancy while maintaining memory efficiency

#### Memory Coalescing
```cpp
// Example: Loading 128-bit aligned chunks for optimal bandwidth
using SmemCopyAtomBias = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;  // 128-bit loads
using SmemCopyAtomAttnMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
```

## Performance Considerations

### Memory Efficiency
- **Shared Memory Aliasing**: Smart memory reuse (sMask ↔ sP, sBias ↔ sdS) reduces footprint by ~30%
- **Block-level Skip**: Early exit eliminates unnecessary memory loads for inactive tiles
- **LSE Caching**: Forward pass LSE values cached and reused in backward pass for numerical stability
- **Coalesced Access**: Optimized tensor layouts for GPU memory hierarchy

### Computational Efficiency  
- **Unified Skip Logic**: Both forward and backward passes benefit from block-level computation skipping
- **5-GEMM Chain Skip**: Complete gradient computation chain skipped for inactive tiles
- **Early Branch Decision**: Mask OR-reduction happens before expensive K/V loads
- **Warp-Level Optimization**: Operations optimized for GPU warp execution model

### Scalability
- **Block-level Granularity**: Tile-level sparsity more efficient than element-level for long sequences
- **Multi-Head Support**: Efficient handling of multiple attention heads with per-head sparsity patterns
- **Barrier Optimization**: Minimal synchronization overhead through smart aliasing strategies

### Performance Model

Expected speedup for various sparsity levels:
- **50% sparsity**: ~1.8x speedup
- **75% sparsity**: ~3.2x speedup  
- **90% sparsity**: ~6.5x speedup

Performance factors:
- Skip overhead typically <5% of dense computation time
- Memory bandwidth reduction scales linearly with sparsity
- Shared memory aliasing enables 20-30% larger tile sizes

## API Changes

### New Required Parameters

The Dynamic Mask Attention integration introduces new required parameters to the forward pass:

- **`attn_mask`** (`torch.Tensor`): Attention mask tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Binary mask (1.0 = compute, 0.0 = skip) indicating which positions should be processed
  - Determines the sparsity pattern for computational efficiency

- **`attn_bias`** (`torch.Tensor`): Attention bias tensor of shape `(batch, num_kv_heads, seqlen_q, seqlen_k)`
  - Contains dynamic attention bias values applied to attention scores before softmax
  - Must have the same dtype and device as Q/K/V tensors

### Updated Function Signature

```python
def fwd(
    q: torch.Tensor,                              # Query tensor
    k: torch.Tensor,                              # Key tensor  
    v: torch.Tensor,                              # Value tensor
    attn_mask: torch.Tensor,                      # Attention mask (REQUIRED)
    attn_bias: torch.Tensor,                      # Attention bias (REQUIRED)
    out: Optional[torch.Tensor] = None,           # Pre-allocated output
    softmax_scale: float = None,                  # Attention scaling
    is_causal: bool = False,                      # Causal masking
    softcap: float = 0.0,                         # Soft capping
    return_softmax: bool = False,                 # Return attention weights
) -> List[torch.Tensor]
```

### Backward Compatibility

**Breaking Change Notice**: The integration requires attention bias and attention mask tensors as mandatory parameters. This is a breaking change from the original Flash Attention API.

**Migration Path**: Users need to:
1. Add attention mask and bias generation logic to attention modules
2. Implement appropriate mask and bias computation within the attention forward pass
3. Ensure proper tensor shapes and dtypes for mask and bias tensors

### Complete Usage Example

```python
import torch
import torch.nn as nn
import flash_dmattn_cuda as flash_dmattn

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Generate attention bias from value states
        dt_states = self.dt_proj(
            value_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        )
        dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
        attention_bias = dt_states[:, :, None, :].expand(-1, -1, seq_len, -1).to(hidden_states.dtype)

        # Prepare attention mask for multi-head
        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1, self.num_kv_heads, -1, -1)

        # Flash Dynamic Mask Attention
        attn_output, _ = flash_dynamic_mask_attention_forward(
            self,
            query_states,
            key_states, 
            value_states,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            scaling=self.scaling,
        )

        # Output projection
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

# Usage example
config = type('Config', (), {
    'hidden_size': 768,
    'num_attention_heads': 12,
    'num_key_value_heads': 12,
})()

attention = DynamicMaskAttention(config)
hidden_states = torch.randn(2, 4096, 768, device='cuda', dtype=torch.bfloat16)
output = attention(hidden_states)
print(f"Output shape: {output.shape}")  # [2, 4096, 768]
```

### Integration with Existing Codebases

For users migrating from Flash Attention, the typical changes required are:

```python
# Before (Flash Attention)
class StandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)

    def forward(self, hidden_states):
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        output = flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=self.scaling, causal=True)
        return self.o_proj(output)

# After (Dynamic Mask Attention)
class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Same standard projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim) 
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)
        
        # Add dynamic mask parameters
        self.A = nn.Parameter(torch.zeros(config.num_key_value_heads))
        self.dt_proj = nn.Linear(config.num_key_value_heads * self.head_dim, config.num_key_value_heads)
        self.keep_window_size = config.keep_window_size

    def forward(self, hidden_states):
        # Standard Q, K, V projections
        query_states = self.q_proj(hidden_states).view(...).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(...).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(...).transpose(1, 2)
        
        # Generate attention bias from value states  
        dt_states = self.dt_proj(value_states.transpose(1, 2).reshape(...))
        dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
        attention_bias = dt_states[:, :, None, :].expand(-1, -1, seq_len, -1)
        
        # Use Flash Dynamic Mask Attention
        attn_output, _ = flash_dynamic_mask_attention_forward(
            self, query_states, key_states, value_states,
            attention_mask=attention_mask, attention_bias=attention_bias,
            scaling=self.scaling
        )
        
        return self.o_proj(attn_output.reshape(...))
```