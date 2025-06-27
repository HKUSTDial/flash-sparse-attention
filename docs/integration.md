# Flash Dynamic Mask Attention Integration Guide

## Overview

This document describes the integration of Dynamic Mask Attention into the Flash Attention framework. The integration enables efficient sparse attention computation by combining Flash Attention's memory-efficient approach with dynamic masking capabilities for handling extremely long sequences.

The integration implements a two-stage approach: Python frontend pre-computes Zero-Order Hold states and Active Mask tensors, while the CUDA backend performs sparse attention computation using these pre-computed masks.

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

The Dynamic Mask Attention integration follows a two-phase approach:

1. **Dynamic Mask Computation**: Python frontend pre-computes ZOH states and Active Mask tensors
2. **Sparse Attention Execution**: CUDA backend performs sparse attention computation using the pre-computed masks

```
Python Frontend                    CUDA Backend
┌─────────────────────────────┐    ┌──────────────────────────────┐
│ dt_states = exp(A * softplus│    │ Global Memory Loading        │
│ (V @ dt_proj^T))            │────│ ├─ ZOH States                │
│                             │    │ ├─ Active Mask               │
│ prepare_dynamic_mask()      │    │ └─ Q, K, V Tensors           │
│ ├─ ZOH States Generation    │    │                              │
│ ├─ Active Mask via TopK     │    │ Sparse Attention Computation │
│ └─ Dynamic Bias Calculation │    │ ├─ Sparse Q*K^T GEMM         │
└─────────────────────────────┘    │ ├─ Masked Softmax with ZOH   │
                                   │ └─ Sparse Score*V GEMM       │
                                   └──────────────────────────────┘
```

### Key Components

- **ZOH States**: Dynamic attention bias values `(batch, num_heads, query_len, key_len)` derived from value states and learned projections
- **Active Mask**: Binary mask `(batch, num_heads, query_len, key_len)` indicating which positions should be computed (1.0) or skipped (0.0)
- **Sparse GEMM**: Optimized matrix multiplication that only computes non-masked regions
- **Dynamic Masking**: Integration of ZOH bias and active mask into attention score computation

## Core Modifications

### 1. Parameter Structure Extensions (`flash.h`)

**Purpose**: Extended parameter structures to support dynamic masking tensors with proper memory layout information.

**Changes Made**:
```cpp
struct ZOH_params {
    void *__restrict__ zoh_ptr;                    // ZOH states pointer
    void *__restrict__ active_mask_ptr;            // Active mask pointer
    index_t zoh_batch_stride;                      // Batch stride for ZOH states
    index_t active_mask_batch_stride;              // Batch stride for active mask
    index_t zoh_head_stride;                       // Head stride for ZOH states
    index_t active_mask_head_stride;               // Head stride for active mask
    index_t zoh_row_stride;                        // Row stride for ZOH states
    index_t active_mask_row_stride;                // Row stride for active mask
    int keep_window_size;                          // Sparsity control parameter
};

struct Flash_fwd_params : public QKV_params, public ZOH_params {
    // Inherits both QKV and ZOH parameters through multiple inheritance
    // Enables unified parameter passing to CUDA kernels
};
```

**Rationale**: 
- **Multiple Inheritance Design**: Cleanly separates QKV parameters from ZOH parameters while maintaining unified access
- **Comprehensive Stride Information**: Provides all necessary stride information for efficient tensor indexing in CUDA kernels
- **Memory Layout Optimization**: Enables optimal memory access patterns for both regular and sparse tensors

### 2. Kernel Traits and Memory Layout (`kernel_traits.h`)

**Purpose**: Define shared memory layouts and copy operations optimized for dynamic masking tensors.

**Changes Made**:
```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    // ...existing Flash Attention traits...
    
    // ZOH States shared memory layout - matches attention score layout
    using SmemLayoutZOH = decltype(make_layout(
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
        make_stride(Int<kBlockN>{}, _1{})
    ));
    
    // Active Mask shared memory layout - row-major for efficient indexing
    using SmemLayoutActiveMask = decltype(make_layout(
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
        make_stride(Int<kBlockN>{}, _1{})
    ));
    
    // Optimized copy atoms for ZOH and Active Mask data movement
    using SmemCopyAtomZOH = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomActiveMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    
    // Shared memory size calculations including masking tensors
    static constexpr int kSmemSizeZOH = kBlockM * kBlockN * sizeof(elem_type);
    static constexpr int kSmemSizeActiveMask = kBlockM * kBlockN * sizeof(elem_type);
};
```

**Rationale**:
- **Layout Consistency**: ZOH states use the same layout as attention scores for efficient fusion
- **Memory Access Optimization**: Copy atoms leverage GPU's specialized load/store units for maximum bandwidth
- **Shared Memory Management**: Explicit size calculations ensure proper memory allocation

### 3. Block Information Extension (`block_info.h`)

**Purpose**: Calculate memory offsets for ZOH states and active masks within thread blocks, enabling efficient global memory access.

**Changes Made**:
```cpp
template<bool Is_even_MN=true>
struct BlockInfo {
    // ...existing Flash Attention block info...
    
    index_t zoh_offset;                           // Global memory offset for ZOH states
    index_t active_mask_offset;                   // Global memory offset for active mask
    
    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb, const int bidh, const int m_block) {
        // ...existing initialization...
        
        // Calculate ZOH states offset: [batch][head][query_start_row][0]
        zoh_offset = bidb * params.zoh_batch_stride + 
                     bidh * params.zoh_head_stride + 
                     m_block * kBlockM * params.zoh_row_stride;
                     
        // Calculate Active Mask offset: [batch][head][query_start_row][0] 
        active_mask_offset = bidb * params.active_mask_batch_stride + 
                            bidh * params.active_mask_head_stride + 
                            m_block * kBlockM * params.active_mask_row_stride;
    }
};
```

**Rationale**:
- **Unified Offset Calculation**: Encapsulates complex address arithmetic in a single location
- **Block-Aware Indexing**: Accounts for thread block positioning within the global attention matrix
- **Type Safety**: Template-based design ensures compile-time optimization and type checking

### 4. Memory Copy Operations (`utils.h`)

**Purpose**: Implement efficient memory copy operations for loading ZOH states and active masks from global to shared memory.

**Changes Made**:
```cpp
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void copy_ZOH(
    Tensor0 &tSgZOH,                              // Global ZOH tensor view
    Tensor1 &tSsZOH,                              // Shared ZOH tensor view  
    Tensor2 &tSrZOH,                              // Register ZOH tensor view
    Tensor3 &tSgAM,                               // Global Active Mask tensor view
    Tensor4 &tSsAM,                               // Shared Active Mask tensor view
    TiledMma tiled_mma,                           // MMA tile configuration
    TiledCopy smem_tiled_copy_ZOH,                // Tiled copy for ZOH
    ThrCopy smem_thr_copy_ZOH                     // Thread copy for ZOH
) {
    // Copy ZOH states: Global Memory -> Shared Memory
    copy(smem_tiled_copy_ZOH, tSgZOH, tSsZOH);
    
    // Copy Active Mask: Global Memory -> Shared Memory  
    copy(smem_tiled_copy_ZOH, tSgAM, tSsAM);
    
    // Synchronize to ensure all data is loaded before computation
    __syncthreads();
    
    // Copy to registers for computation: Shared Memory -> Registers
    copy(smem_thr_copy_ZOH, tSsZOH, tSrZOH);
    copy(smem_thr_copy_ZOH, tSsAM, tSrAM);
}
```

**Rationale**:
- **Multi-Level Memory Hierarchy**: Efficiently manages data movement through global -> shared -> register memory levels
- **Coalesced Access Patterns**: Leverages CUTLASS copy operations for optimal memory bandwidth utilization
- **Synchronization Management**: Proper thread synchronization ensures data consistency across the thread block

### 5. Dynamic Masking Logic (`mask.h`)

**Purpose**: Implement the core dynamic masking functionality that applies ZOH states and active masks during attention computation.

**Changes Made**:
```cpp
template <bool Is_causal>
struct DynamicMask {
    const int max_seqlen_k, max_seqlen_q;
    const int keep_window_size;

    template <bool Causal_mask=false, bool Is_even_MN=true, 
              typename TensorType, typename ZOHType, typename ActiveMaskType>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // Attention scores (MMA=4, MMA_M, MMA_N)
        ZOHType &tSrZOH,                            // ZOH states in registers
        ActiveMaskType &tSrAM,                      // Active mask in registers
        const float scale_softmax,                 // Attention scaling factor
        const int col_idx_offset_,                 // Column index offset for this thread block
        const int row_idx_offset,                  // Row index offset for this thread block
        const int warp_row_stride                  // Row stride within warp
    ) {
        // Convert MMA layout to row-column layout for easier indexing
        Tensor tensor = make_tensor(tensor_.data(), convert_layout_acc_rowcol(tensor_.layout()));
        Tensor zoh = make_tensor(tSrZOH.data(), convert_layout_acc_rowcol(tSrZOH.layout()));
        Tensor active_mask = make_tensor(tSrAM.data(), convert_layout_acc_rowcol(tSrAM.layout()));

        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                // Apply causal masking if enabled
                const int col_idx_limit = Causal_mask ? 
                    std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : 
                    max_seqlen_k;
                    
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j * 2;
                        
                        if (col_idx < col_idx_limit && row_idx < max_seqlen_q && col_idx < max_seqlen_k) {
                            // Check if this position should be computed (active mask = 1.0)
                            if (active_mask(i, mi, j, nj) == 0.0f) {
                                // Masked position: set to -infinity
                                tensor(i, mi, j, nj) = -INFINITY;
                            } else {
                                // Active position: apply scaling and add ZOH bias
                                tensor(i, mi, j, nj) = tensor(i, mi, j, nj) * scale_softmax + zoh(i, mi, j, nj);
                            }
                        } else {
                            // Out of bounds: always mask
                            tensor(i, mi, j, nj) = -INFINITY;
                        }
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
- **Numerical Stability**: Proper handling of infinity values for masked positions ensures stable softmax computation

### 6. Sparse Matrix Operations (`utils.h`)

**Purpose**: Implement sparse GEMM operations that utilize active masks to skip computation for masked regions, significantly reducing computational overhead.

**Changes Made**:
```cpp
template <bool A_in_regs=false, bool B_in_regs=false,
          typename Tensor0, typename Tensor1, typename Tensor2, 
          typename Tensor3, typename Tensor4, typename Tensor5,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void sparse_gemm(
    Tensor0 &acc,                               // Output accumulator tensor
    Tensor1 &tCrA,                              // A matrix in registers (Query)
    Tensor2 &tCrB,                              // B matrix in registers (Key/Value)
    Tensor3 &tCsA,                              // A matrix in shared memory
    Tensor4 &tCsB,                              // B matrix in shared memory  
    Tensor5 &active_mask,                       // Sparsity mask in registers
    TiledMma tiled_mma,                         // MMA tile configuration
    TiledCopyA smem_tiled_copy_A,               // Copy configuration for A
    TiledCopyB smem_tiled_copy_B,               // Copy configuration for B
    ThrCopyA smem_thr_copy_A,                   // Thread copy for A
    ThrCopyB smem_thr_copy_B                    // Thread copy for B
) {
    // Load data based on sparsity pattern
    if constexpr (!A_in_regs) {
        copy(smem_tiled_copy_A, tCsA, tCrA);
    }
    if constexpr (!B_in_regs) {
        copy(smem_tiled_copy_B, tCsB, tCrB);
    }
    
    // Perform sparse matrix multiplication
    // Only compute where active_mask indicates active positions
    sparse_gemm_impl(tiled_mma, acc, tCrA, tCrB, active_mask);
}

template <bool A_in_regs=false, bool B_in_regs=false,
          typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename Tensor5,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void sparse_gemm_rs(
    Tensor0 &acc,                                // Accumulator (attention scores)
    Tensor1 &tCrA,                              // Query in registers
    Tensor2 &tCrB,                              // Key in registers  
    Tensor3 &tCsA,                              // Query in shared memory
    Tensor4 &tCsB,                              // Key in shared memory
    Tensor5 &active_mask,                       // Active mask for sparsity
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B
) {
    // Row-major sparse GEMM variant optimized for Q*K^T computation
    // Utilizes active mask to determine which K vectors to process
}
```

**Rationale**:
- **Computational Efficiency**: Skips matrix multiplication for masked regions, reducing FLOPs proportional to sparsity
- **Memory Bandwidth Optimization**: Avoids loading unnecessary data for masked positions
- **Flexible Sparsity Support**: Supports different sparsity patterns through the active mask tensor
- **Register/Shared Memory Optimization**: Provides variants for different data residency scenarios

### 7. Attention Kernel Modifications (`flash_fwd_kernel.h`)

**Purpose**: Integrate dynamic masking into the core attention computation kernels while maintaining Flash Attention's memory efficiency and optimization strategies.

**Changes Made**:
```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, 
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, 
         bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params &params, 
    const int bidb, 
    const int bidh, 
    const int m_block
) {
    // Initialize block information with ZOH and active mask offsets
    const BlockInfo binfo(params, bidb, bidh, m_block);
    
    // Set up tensor views for ZOH states and active masks
    Tensor mZOH = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.zoh_ptr) + binfo.zoh_offset),
                              make_shape(binfo.actual_seqlen_q, params.seqlen_k),
                              make_stride(params.zoh_row_stride, _1{}));
    
    Tensor mActiveMask = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.active_mask_ptr) + binfo.active_mask_offset),
                                     make_shape(binfo.actual_seqlen_q, params.seqlen_k),
                                     make_stride(params.active_mask_row_stride, _1{}));

    // Main computation loop over key/value blocks
    for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
        // Load ZOH states and active masks for this block
        copy_ZOH(tSgZOH, tSsZOH, tSrZOH, tSgActiveMask, tSsActiveMask, 
                 tiled_mma, smem_tiled_copy_ZOH, smem_thr_copy_ZOH);
        
        // Perform sparse Q*K^T computation
        sparse_gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tSrActiveMask,
                       tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                       smem_thr_copy_Q, smem_thr_copy_K);
        
        // Apply dynamic masking (ZOH bias + active mask)
        DynamicMask<Is_causal> dynamic_mask(params.seqlen_k, params.seqlen_q, params.keep_window_size);
        dynamic_mask.apply_mask(acc_s, tSrZOH, tSrActiveMask, params.scale_softmax,
                               n_block * kBlockN, m_block * kBlockM, kBlockM);
        
        // Continue with softmax and attention*V computation
        softmax.template softmax</*Is_first=*/true>(acc_s);
        
        // Sparse attention*V computation
        sparse_gemm_rs(acc_o, acc_s, tSrV, tSsS, tSsV, tSrActiveMask,
                    tiled_mma, smem_tiled_copy_S, smem_tiled_copy_V,
                    smem_thr_copy_S, smem_thr_copy_V);
    }
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, 
         bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(
    const Params &params, 
    const int bidb, 
    const int bidh, 
    const int m_block, 
    const int n_split_idx, 
    const int num_n_splits
) {
    // Split-K variant with dynamic masking support
    // Handles distributed computation across multiple thread blocks
    // Maintains sparsity patterns across splits
}
```

**Rationale**:
- **Seamless Integration**: Dynamic masking logic integrated into existing Flash Attention computation flow
- **Memory Efficiency Preservation**: Maintains Flash Attention's tiling and shared memory optimization strategies
- **Split-K Support**: Extends dynamic masking to split-K attention variants for very long sequences
- **Template Specialization**: Compile-time optimization through template parameters

### 8. Launch Template Updates (`flash_fwd_launch_template.h`)

**Purpose**: Update kernel launch functions to properly configure and validate dynamic masking parameters, ensuring correct shared memory allocation and kernel selection.

**Changes Made**:
```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // Calculate shared memory requirements including ZOH and active mask tensors
    const size_t smem_size = Kernel_traits::kSmemSize + 
                            Kernel_traits::kSmemSizeZOH + 
                            Kernel_traits::kSmemSizeActiveMask;
    
    // Validate that shared memory requirements don't exceed device limits
    TORCH_CHECK(smem_size <= 48 * 1024, "Shared memory requirement exceeds device limit");
    
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
                auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout, Is_causal, 
                                              IsEvenMN, IsEvenK, /*Is_softcap=*/false, ReturnSoftmax>;
                
                // Configure dynamic shared memory
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
    const at::Tensor zoh,                         // ZOH states tensor
    const at::Tensor active_mask,                 // Active mask tensor
    const size_t keep_window_size,                // Sparsity control parameter
    // ... other parameters ...
) {
    // Reset parameters and set basic properties
    params = {};
    params.is_bf16 = q.dtype() == torch::kBFloat16;
    
    // Set ZOH states pointers and strides
    params.zoh_ptr = zoh.data_ptr();
    params.zoh_batch_stride = zoh.stride(-4);          // [batch, head, query, key]
    params.zoh_head_stride = zoh.stride(-3);
    params.zoh_row_stride = zoh.stride(-2);
    
    // Set Active Mask pointers and strides  
    params.active_mask_ptr = active_mask.data_ptr();
    params.active_mask_batch_stride = active_mask.stride(-4);
    params.active_mask_head_stride = active_mask.stride(-3);
    params.active_mask_row_stride = active_mask.stride(-2);
    
    // Set sparsity control parameter
    params.keep_window_size = keep_window_size;
    
    // ... existing parameter setup ...
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,                              // Query tensor
    const at::Tensor &k,                        // Key tensor
    const at::Tensor &v,                        // Value tensor
    const at::Tensor &zoh,                      // ZOH states tensor
    const at::Tensor &active_mask,              // Active mask tensor
    std::optional<at::Tensor> &out_,            // Optional output tensor
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    const int keep_window_size,                 // Sparsity control
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
) {
    // Comprehensive input validation
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(zoh); CHECK_DEVICE(active_mask);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(zoh); CHECK_CONTIGUOUS(active_mask);
    
    // Validate tensor shapes
    auto batch_size = q.size(0);
    auto seqlen_q = q.size(1); 
    auto num_heads = q.size(2);
    auto head_dim = q.size(3);
    auto seqlen_k = k.size(1);
    auto num_heads_k = k.size(2);
    
    CHECK_SHAPE(zoh, batch_size, num_heads_k, seqlen_q, seqlen_k);
    CHECK_SHAPE(active_mask, batch_size, num_heads_k, seqlen_q, seqlen_k);
    
    // Validate data types consistency
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), 
                "All QKV tensors must have the same dtype");
    TORCH_CHECK(zoh.dtype() == q.dtype(), 
                "ZOH states must have the same dtype as QKV tensors");
    TORCH_CHECK(active_mask.dtype() == q.dtype(), 
                "Active mask must have the same dtype as QKV tensors");
    
    // Validate sparsity parameter
    TORCH_CHECK(keep_window_size > 0 && keep_window_size <= seqlen_k,
                "keep_window_size must be positive and <= seqlen_k");
    
    // Set up parameters and launch computation
    Flash_fwd_params params;
    set_params_fprop(params, batch_size, seqlen_q, seqlen_k, /* ... */, 
                     q, k, v, zoh, active_mask, /* ... */, keep_window_size, /* ... */);
    
    // Launch kernel with appropriate configuration
    run_mha_fwd(params, at::cuda::getCurrentCUDAStream());
    
    // Return results
    return {out, softmax_lse, /* ... */};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDynamicMaskAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass with dynamic masking",
          py::arg("q"), py::arg("k"), py::arg("v"), 
          py::arg("zoh"), py::arg("active_mask"),                    // New required arguments
          py::arg("out") = py::none(),
          py::arg("p_dropout") = 0.0f,
          py::arg("softmax_scale") = 0.0f, 
          py::arg("is_causal") = false,
          py::arg("keep_window_size") = 2048,                       // New sparsity control
          py::arg("softcap") = 0.0f,
          py::arg("return_softmax") = false,
          py::arg("gen") = py::none());
}
```

**Rationale**:
- **Comprehensive Validation**: Thorough validation of all input tensors for shape, type, and device consistency
- **Backward Compatibility**: Maintains existing parameter order while adding new functionality
- **Error Handling**: Clear error messages for common usage mistakes
- **Type Safety**: Strict type checking to prevent runtime errors
- **Documentation**: Clear parameter documentation for Python users

## Implementation Details

### Python Frontend: Dynamic Mask Generation

The Python frontend is responsible for computing the ZOH states and active masks before passing them to the CUDA backend:

```python
def prepare_dynamic_mask(
    hidden_states: torch.Tensor,
    dt_states: torch.Tensor,
    keep_window_size: int = 2048,
    attention_mask: torch.Tensor = None,
):
    """
    Core DMA function that generates dynamic attention masks for sparse computation.
    
    Process:
    1. Expand dt_states to match attention matrix dimensions
    2. Apply optional causal/padding masks  
    3. Use TopK selection to identify most important positions
    4. Generate binary active mask for CUDA computation
    """
    min_dtype = torch.finfo(hidden_states.dtype).min
    dtype = hidden_states.dtype
    
    # Expand dt_states: [batch, num_heads, key_len] -> [batch, num_heads, query_len, key_len]
    attn_mask = dt_states[:, :, None, :].expand(-1, -1, hidden_states.shape[2], -1)

    # Apply causal/padding masks by setting masked positions to -inf
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(attention_mask, 0.0, min_dtype)
        attn_mask = attn_mask.masked_fill(attention_mask != 0, min_dtype)
    
    # Only apply when sequence length exceeds window size
    if attn_mask.shape[-1] > keep_window_size:
        active_mask = torch.zeros_like(attn_mask, dtype=dtype, device=attn_mask.device)
        # TopK selection identifies most important positions for each query
        topk_indices = torch.topk(attn_mask, keep_window_size, dim=-1, 
                                 largest=True, sorted=False).indices
        # Create binary mask: 1.0 for active positions, 0.0 for masked
        active_mask = active_mask.scatter(-1, topk_indices, 1.0)
        # Set non-selected positions to -inf in attention mask
        attn_mask = attn_mask.masked_fill(active_mask == 0.0, min_dtype)
    else:
        # If sequence length is within window size, all positions are active
        active_mask = torch.ones_like(attn_mask, dtype=dtype, device=attn_mask.device)
    return attn_mask, active_mask
```

### CUDA Backend: Sparse Attention Computation

The CUDA backend implements three key stages of sparse attention:

#### Stage 1: Memory Loading and Tensor Setup
```cpp
// Set up tensor views for ZOH states and active masks
Tensor mZOH = make_tensor(
    make_gmem_ptr(reinterpret_cast<Element *>(params.zoh_ptr) + binfo.zoh_offset),
    make_shape(binfo.actual_seqlen_q, params.seqlen_k),
    make_stride(params.zoh_row_stride, _1{})
);

Tensor mActiveMask = make_tensor(
    make_gmem_ptr(reinterpret_cast<Element *>(params.active_mask_ptr) + binfo.active_mask_offset),
    make_shape(binfo.actual_seqlen_q, params.seqlen_k), 
    make_stride(params.active_mask_row_stride, _1{})
);

// Load data through memory hierarchy: Global -> Shared -> Registers
copy_ZOH(tSgZOH, tSsZOH, tSrZOH, tSgActiveMask, tSsActiveMask,
         tiled_mma, smem_tiled_copy_ZOH, smem_thr_copy_ZOH);
```

#### Stage 2: Sparse Q*K^T Computation
```cpp
// Sparse GEMM that skips computation for masked positions
sparse_gemm_rs(acc_s, tSrQ, tSrK, tSsQ, tSsK, tSrActiveMask,
               tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
               smem_thr_copy_Q, smem_thr_copy_K);

// Apply dynamic masking: scaling + ZOH bias + masking
DynamicMask<Is_causal> dynamic_mask(params.seqlen_k, params.seqlen_q, params.keep_window_size);
dynamic_mask.apply_mask(acc_s, tSrZOH, tSrActiveMask, params.scale_softmax,
                       n_block * kBlockN, m_block * kBlockM, kBlockM);
```

#### Stage 3: Softmax and Sparse Attention*V
```cpp
// Online softmax computation (unchanged from Flash Attention)
softmax.template online_softmax</*Is_first=*/true>(acc_s);

// Sparse attention*V computation
sparse_gemm(acc_o, acc_s, tSrV, tSsS, tSsV, tSrActiveMask,
            tiled_mma, smem_tiled_copy_S, smem_tiled_copy_V,
            smem_thr_copy_S, smem_thr_copy_V);
```

## Sparse Computation Strategy

### Sparsity Pattern Recognition

The Dynamic Mask Attention implements structured sparsity based on learned importance scores:

1. **ZOH State Computation**: `dt_states = exp(A * softplus(V @ dt_proj^T))`
   - Learned projection matrix `dt_proj` maps value features to importance scores
   - Coefficient `A` controls the dynamic range of importance values
   - Exponential activation ensures positive importance scores

2. **TopK Selection**: For sequences longer than `keep_window_size`:
   - Select top-K most important positions per query token
   - K = `keep_window_size` (typically 512-2048)
   - Maintains fixed computational complexity regardless of sequence length

3. **Binary Active Mask**: 
   - 1.0 for positions selected by TopK (compute)
   - 0.0 for positions not selected (skip computation)

### Sparse GEMM Implementation

The sparse GEMM operations leverage the active mask to skip computation:

```cpp
template<typename TiledMma, typename AccType, typename AType, typename BType, typename MaskType>
__forceinline__ __device__ void sparse_gemm_impl(
    TiledMma tiled_mma,
    AccType &acc,
    AType &tCrA, 
    BType &tCrB,
    MaskType &active_mask
) {
    // Convert layouts for efficient indexing
    auto acc_rowcol = make_tensor(acc.data(), convert_layout_acc_rowcol(acc.layout()));
    auto mask_rowcol = make_tensor(active_mask.data(), convert_layout_acc_rowcol(active_mask.layout()));
    
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(acc_rowcol); ++mi) {
        #pragma unroll  
        for (int ni = 0; ni < size<1, 1>(acc_rowcol); ++ni) {
            // Check if this position should be computed
            if (mask_rowcol(0, mi, 0, ni) != 0.0f) {
                // Perform computation only for active positions
                gemm(tiled_mma, acc(_, mi, _, ni), tCrA(_, mi, _), tCrB(_, _, ni));
            }
            // Skip computation for masked positions (acc remains unchanged)
        }
    }
}
```

### Memory Efficiency Optimizations

1. **Shared Memory Reuse**: ZOH states and active masks share copy infrastructure with Q/K/V tensors
2. **Register Allocation**: Critical masking operations performed in registers to minimize memory traffic
3. **Coalesced Access**: Memory access patterns optimized for GPU memory hierarchy
4. **Template Specialization**: Compile-time optimization eliminates runtime branching

## Memory Layout

### Tensor Memory Organization

The Dynamic Mask Attention extends Flash Attention's memory layout to include ZOH states and active masks:

```
Global Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ Q:         [batch, seqlen_q, num_heads, head_dim]               │
│ K:         [batch, seqlen_k, num_heads_k, head_dim]             │  
│ V:         [batch, seqlen_k, num_heads_k, head_dim]             │
│ ZOH:       [batch, num_heads_k, seqlen_q, seqlen_k]             │
│ AM:        [batch, num_heads_k, seqlen_q, seqlen_k]             │
│ Output:    [batch, seqlen_q, num_heads, head_dim]               │
└─────────────────────────────────────────────────────────────────┘

Shared Memory Layout (per thread block):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Tile:    [kBlockM, head_dim]     │ K Tile:    [kBlockN, head_dim] │
│ V Tile:    [kBlockN, head_dim]     │ S Tile:    [kBlockM, kBlockN]  │
│ ZOH Tile:  [kBlockM, kBlockN]      │ AM Tile:   [kBlockM, kBlockN]  │
└─────────────────────────────────────────────────────────────────────┘

Register Memory (per thread):
┌─────────────────────────────────────────────────────────────────────┐
│ Q Frag:    [MMA_M, head_dim/N]     │ K Frag:    [MMA_N, head_dim/N] │
│ V Frag:    [MMA_N, head_dim/N]     │ S Frag:    [MMA_M, MMA_N]      │
│ ZOH Frag:  [MMA_M, MMA_N]          │ AM Frag:   [MMA_M, MMA_N]      │
│ Acc Frag:  [MMA_M, head_dim/N]     │                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

#### ZOH States and Active Mask Loading
```cpp
// Global to Shared Memory (coalesced access)
Tensor tSgZOH = local_partition(mZOH, smem_tiled_copy_ZOH, thread_idx);
Tensor tSsZOH = local_partition(sZOH, smem_tiled_copy_ZOH, thread_idx);

// Each thread loads a contiguous chunk to maximize memory bandwidth
copy(smem_tiled_copy_ZOH, tSgZOH, tSsZOH);

// Shared to Register Memory (bank-conflict-free)
Tensor tSrZOH = local_partition(sZOH, smem_thr_copy_ZOH, thread_idx);
copy(smem_thr_copy_ZOH, tSsZOH, tSrZOH);
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
- ZOH states and active masks use the same copy patterns as Q/K/V to avoid bank conflicts
- Padding added when necessary to ensure 128-bit aligned access
- Thread block size chosen to maximize occupancy while maintaining memory efficiency

#### Memory Coalescing
```cpp
// Example: Loading 128-bit aligned chunks for optimal bandwidth
using SmemCopyAtomZOH = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;  // 128-bit loads
using SmemCopyAtomActiveMask = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
```

## Performance Considerations

### Memory Efficiency
- **Reduced Memory Bandwidth**: Sparse computation reduces memory traffic
- **Optimized Layouts**: Tensor layouts optimized for GPU memory hierarchy
- **Shared Memory Reuse**: Efficient use of limited shared memory resources

### Computational Efficiency  
- **Sparse GEMM**: Skips computation for masked regions
- **Fused Operations**: Masking integrated into existing computation kernels
- **Warp-Level Optimization**: Optimized for GPU warp execution model

### Scalability
- **Long Sequence Support**: Efficient handling of sequences > 32K tokens
- **Configurable Sparsity**: `keep_window_size` parameter controls sparsity level
- **Multi-Head Support**: Efficient handling of multiple attention heads

## API Changes

### New Required Parameters

The Dynamic Mask Attention integration introduces new required parameters to the forward pass:

- **`zoh`** (`torch.Tensor`): ZOH states tensor of shape `(batch, num_heads_k, seqlen_q, seqlen_k)`
  - Contains dynamic attention bias values derived from value states
  - Must have the same dtype and device as Q/K/V tensors

- **`active_mask`** (`torch.Tensor`): Active mask tensor of shape `(batch, num_heads_k, seqlen_q, seqlen_k)`
  - Binary mask (1.0 = compute, 0.0 = skip) indicating which positions should be processed
  - Determines the sparsity pattern for computational efficiency

- **`keep_window_size`** (`int`): Sparsity control parameter
  - Maximum number of key positions to attend to per query token
  - Controls the computational complexity and memory usage
  - Typical values: 512-2048 for long sequences

### Updated Function Signature

```python
def fwd(
    q: torch.Tensor,                              # Query tensor
    k: torch.Tensor,                              # Key tensor  
    v: torch.Tensor,                              # Value tensor
    zoh: torch.Tensor,                            # ZOH states (NEW)
    active_mask: torch.Tensor,                    # Active mask (NEW)
    out: Optional[torch.Tensor] = None,           # Pre-allocated output
    p_dropout: float = 0.0,                       # Dropout probability
    softmax_scale: float = None,                  # Attention scaling
    is_causal: bool = False,                      # Causal masking
    keep_window_size: int = 2048,                 # Sparsity control (NEW)
    softcap: float = 0.0,                         # Soft capping
    return_softmax: bool = False,                 # Return attention weights
    gen: Optional[torch.Generator] = None         # Random generator
) -> List[torch.Tensor]
```

### Backward Compatibility

**Breaking Change Notice**: The integration requires ZOH states and active mask tensors as mandatory parameters. This is a breaking change from the original Flash Attention API.

**Migration Path**: Users need to:
1. Implement ZOH state computation using the `prepare_dynamic_mask` function
2. Update function calls to include the new required parameters
3. Choose appropriate `keep_window_size` values based on their use case

### Complete Usage Example

```python
import torch
import torch.nn.functional as F
import flash_dma

# Setup
batch_size, seqlen_q, seqlen_k = 2, 4096, 4096
num_heads, head_dim = 12, 128
device, dtype = 'cuda', torch.bfloat16

# Input tensors
q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype)

# Dynamic Mask Attention requires additional parameters
dt_proj = torch.randn(num_heads, num_heads * head_dim, device=device, dtype=dtype)
A = torch.randn(num_heads, device=device, dtype=dtype)

# Step 1: Compute ZOH states
dt_states = torch.matmul(
    v.transpose(-2, -3).reshape(batch_size, seqlen_k, -1), 
    dt_proj.T
)
dt_states = torch.exp(A * F.softplus(dt_states)).transpose(-1, -2)

# Step 2: Generate dynamic masks
zoh_states, active_mask = flash_dma.prepare_dynamic_mask(
    q, dt_states, keep_window_size=2048, attention_mask=None
)

# Step 3: Run Dynamic Mask Attention
output = flash_dma.fwd(
    q, k, v, zoh_states, active_mask,
    keep_window_size=2048,
    softmax_scale=1.0 / (head_dim ** 0.5),
    is_causal=False
)

print(f"Output shape: {output[0].shape}")  # [batch_size, seqlen_q, num_heads, head_dim]
```

### Integration with Existing Codebases

For users migrating from Flash Attention, the typical changes required are:

```python
# Before (Flash Attention)
output = flash_attn.flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=scale, causal=True)

# After (Dynamic Mask Attention)
# 1. Add ZOH computation
dt_states = compute_dt_states(v, dt_proj, A)
zoh_states, active_mask = prepare_dynamic_mask(q, dt_states, keep_window_size=2048)

# 2. Update function call
output = flash_dma.fwd(q, k, v, zoh_states, active_mask, 
                      p_dropout=0.1, softmax_scale=scale, is_causal=True,
                      keep_window_size=2048)
```

## Future Enhancements

### Planned Improvements

1. **Backward Pass Integration**: Complete gradient computation support for training Dynamic Mask Attention models
   - Sparse gradient computation for ZOH states
   - Efficient gradient propagation through active masks
   - Memory-optimized backward kernels

2. **Adaptive Sparsity Patterns**: Dynamic adjustment of attention patterns based on input characteristics
   - Learned sparsity controllers
   - Content-aware mask generation
   - Adaptive `keep_window_size` selection

3. **Multi-GPU Distributed Support**: Optimizations for large-scale distributed training
   - Efficient tensor parallelism for long sequences
   - Communication-optimal attention computation
   - Memory-balanced workload distribution

4. **Advanced Memory Optimizations**: Further reduce memory footprint for extremely long sequences
   - Progressive attention computation
   - Hierarchical sparsity patterns
   - Memory-efficient checkpoint/recomputation strategies

5. **Hardware-Specific Optimizations**: Leverage newer GPU architectures
   - Hopper architecture optimizations
   - Sparse Tensor Core utilization
   - Advanced memory hierarchy exploitation

### Performance Targets

- **Sequence Length**: Support up to 1M+ tokens efficiently
- **Memory Reduction**: 50-80% memory savings compared to dense attention
- **Speed**: Maintain or improve upon Flash Attention performance for long sequences
- **Sparsity**: Flexible sparsity ratios from 10% to 90% depending on use case

## Conclusion

The Dynamic Mask Attention integration successfully combines Flash Attention's memory efficiency with structured sparsity to enable efficient processing of extremely long sequences. The implementation maintains the core optimization principles of Flash Attention while adding the capability to skip computation for less important token interactions.

Key achievements of this integration:

1. **Seamless Integration**: All dynamic masking functionality integrated into Flash Attention's kernel architecture without compromising existing optimizations

2. **Comprehensive Implementation**: Complete pipeline from Python preprocessing to optimized CUDA kernels with proper memory management

3. **Flexible Sparsity Control**: Configurable sparsity levels through the `keep_window_size` parameter to balance quality and efficiency

4. **Robust Validation**: Extensive testing infrastructure ensures numerical equivalence with reference implementations

5. **Performance Optimization**: Sparse computation patterns reduce both memory usage and computational overhead for long sequences

This integration enables practitioners to efficiently handle very long sequences in transformer models while maintaining the numerical stability and optimization benefits that have made Flash Attention the standard for efficient attention computation.