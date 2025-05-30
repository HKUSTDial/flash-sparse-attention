#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// 包含CUTE库相关头文件
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

// 包含CUTLASS库相关头文件
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

// 项目相关头文件
#include "src/flash.h" // flash.h 包含了 namespace_config.h
#include "src/kernel_traits.h"
#include "src/flash_attention_fwd_kernel.h"
#include "src/utils.h"

// 确保使用正确的命名空间
using namespace cute;

namespace FLASH_NAMESPACE {

template <typename scalar_t, int HeadDim, bool IsCausal>
__global__ void run_attention_fwd_kernel(Flash_fwd_params params) {
    // 修改块大小选择逻辑，确保与HeadDim兼容
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kNWarps = 4;
    
    using Kernel_traits = Flash_fwd_kernel_traits<HeadDim, kBlockM, kBlockN, kNWarps, false, false, scalar_t>;
    
    constexpr bool kIsEvenMN = true; 
    constexpr bool kIsEvenK = true;  
    constexpr bool kReturnSoftmax = false; 
    
    compute_attn<Kernel_traits, false, IsCausal, kIsEvenMN, kIsEvenK, false, kReturnSoftmax>(params);
}

// 修改host-side启动函数
template <typename scalar_t, int HeadDim, bool IsCausal>
void launch_attention_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kNWarps = 4;
    using Kernel_traits = Flash_fwd_kernel_traits<HeadDim, kBlockM, kBlockN, kNWarps, false, false, scalar_t>;

    dim3 grid_dim(
        cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM),
        params.b,
        params.h
    );
    dim3 block_dim(Kernel_traits::kNThreads);
    size_t smem_size = Kernel_traits::kSmemSize;

    // 检查共享内存限制
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (smem_size > prop.sharedMemPerBlock) {
        printf("Warning: Shared memory size (%zu) exceeds device limit (%zu)\n", 
               smem_size, prop.sharedMemPerBlock);
        return;
    }

    // 如果需要，设置动态共享内存
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            run_attention_fwd_kernel<scalar_t, HeadDim, IsCausal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    run_attention_fwd_kernel<scalar_t, HeadDim, IsCausal><<<grid_dim, block_dim, smem_size, stream>>>(params);
    AT_CUDA_CHECK(cudaGetLastError());
}

// 动态掩码注意力调度函数
template <typename scalar_t>
std::vector<torch::Tensor> dynamic_mask_attention_dispatch(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& zero_hold_states,
    torch::Tensor& active_indices,
    torch::Tensor& output,
    torch::Tensor& softmax_lse,
    float scale,
    int keep_window_size,
    bool is_causal,
    bool return_softmax
) {
    const int batch_size = query_states.size(0);
    const int seq_len_q = query_states.size(1);
    const int num_heads = query_states.size(2);
    const int head_dim = query_states.size(3);
    const int seq_len_k = key_states.size(1);
    const int num_kv_heads = key_states.size(2);

    // 确保对齐
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_dim_rounded = round_multiple(head_dim, 32);
    const int seq_len_q_rounded = round_multiple(seq_len_q, 128);
    const int seq_len_k_rounded = round_multiple(seq_len_k, 128);
    
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));
    
    // 设置参数
    params.is_bf16 = query_states.scalar_type() == torch::kBFloat16;
    params.q_ptr = query_states.data_ptr();
    params.k_ptr = key_states.data_ptr();
    params.v_ptr = value_states.data_ptr();
    params.o_ptr = output.data_ptr();
    params.zero_hold_ptr = zero_hold_states.data_ptr();
    params.active_indices_ptr = active_indices.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    
    // 基本维度参数
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = num_heads / num_kv_heads;
    params.seqlen_q = seq_len_q;
    params.seqlen_k = seq_len_k;
    params.seqlen_q_rounded = seq_len_q_rounded;
    params.seqlen_k_rounded = seq_len_k_rounded;
    params.d = head_dim;
    params.d_rounded = head_dim_rounded;
    params.total_q = seq_len_q * batch_size;

    // 步长参数 - 确保与PyTorch tensor的内存布局匹配
    params.q_batch_stride = query_states.stride(0);
    params.k_batch_stride = key_states.stride(0);
    params.v_batch_stride = value_states.stride(0);
    params.o_batch_stride = output.stride(0);
    params.zero_hold_batch_stride = zero_hold_states.stride(0);
    params.active_indices_batch_stride = active_indices.stride(0);

    params.q_row_stride = query_states.stride(1);
    params.k_row_stride = key_states.stride(1);
    params.v_row_stride = value_states.stride(1);
    params.o_row_stride = output.stride(1);
    params.zero_hold_row_stride = zero_hold_states.stride(2);
    params.active_indices_row_stride = active_indices.stride(2);

    params.q_head_stride = query_states.stride(2);
    params.k_head_stride = key_states.stride(2);
    params.v_head_stride = value_states.stride(2);
    params.o_head_stride = output.stride(2);
    params.zero_hold_head_stride = zero_hold_states.stride(1);
    params.active_indices_head_stride = active_indices.stride(1);

    /// 缩放和掩码参数
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * M_LOG2E;
    params.softcap = 0.0f;
    params.keep_window_size = keep_window_size;

    // Dropout参数（禁用）
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    
    // 因果掩码参数
    params.is_causal = is_causal;
    
    // 添加这些重要的参数设置
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
    
    // 确保LSE指针设置正确
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (is_causal) {
        if (head_dim == 32)        { launch_attention_fwd_<scalar_t, 32, true>(params, stream); }
        else if (head_dim == 64)   { launch_attention_fwd_<scalar_t, 64, true>(params, stream); }
        else if (head_dim == 128)  { launch_attention_fwd_<scalar_t, 128, true>(params, stream); }
        else { TORCH_CHECK(false, "Unsupported head_dim for causal attention: ", head_dim); }
    } else {
        if (head_dim == 32)        { launch_attention_fwd_<scalar_t, 32, false>(params, stream); }
        else if (head_dim == 64)   { launch_attention_fwd_<scalar_t, 64, false>(params, stream); }
        else if (head_dim == 128)  { launch_attention_fwd_<scalar_t, 128, false>(params, stream); }
        else { TORCH_CHECK(false, "Unsupported head_dim for non-causal attention: ", head_dim); }
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    return {output, softmax_lse};
}

} // namespace FLASH_NAMESPACE

// CUDA入口点函数，从C++ API文件调用。它保持在全局命名空间中。
std::vector<torch::Tensor> apply_dynamic_mask_attention_cuda(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& zero_hold_states,
    float scale,
    int keep_window_size,
    bool is_causal,
    bool return_softmax
) {

    // 验证输入
    TORCH_CHECK(query_states.dim() == 4, "query_states must be a 4D tensor");
    TORCH_CHECK(key_states.dim() == 4, "key_states must be a 4D tensor");
    TORCH_CHECK(value_states.dim() == 4, "value_states must be a 4D tensor");
    TORCH_CHECK(zero_hold_states.dim() == 4, "zero_hold_states must be a 4D tensor");

    const int batch_size = query_states.size(0);
    const int seq_len_q = query_states.size(1);
    const int num_heads = query_states.size(2);
    const int head_dim = query_states.size(3);
    const int seq_len_k = key_states.size(1);
    const int num_kv_heads = key_states.size(2);

    TORCH_CHECK(key_states.size(0) == batch_size, "Q/K batch mismatch");
    TORCH_CHECK(value_states.size(1) == seq_len_k, "K/V seq mismatch");
    TORCH_CHECK(key_states.size(3) == head_dim, "Q/K head_dim mismatch");
    TORCH_CHECK(value_states.size(0) == batch_size, "Q/V batch mismatch");
    TORCH_CHECK(value_states.size(2) == num_kv_heads, "K/V kv_heads mismatch");
    TORCH_CHECK(value_states.size(3) == head_dim, "Q/V head_dim mismatch");

    TORCH_CHECK(query_states.scalar_type() == at::kHalf || query_states.scalar_type() == at::kBFloat16,
                "Only half/bfloat16 supported");
    TORCH_CHECK(key_states.scalar_type() == query_states.scalar_type(), "All inputs must have same dtype");
    TORCH_CHECK(value_states.scalar_type() == query_states.scalar_type(), "All inputs must have same dtype");

    TORCH_CHECK(head_dim == 32 || head_dim == 64 || head_dim == 128, "head_dim must be 32, 64, or 128");

    TORCH_CHECK(query_states.is_contiguous(), "query_states must be contiguous");
    TORCH_CHECK(key_states.is_contiguous(), "key_states must be contiguous");
    TORCH_CHECK(value_states.is_contiguous(), "value_states must be contiguous");
    TORCH_CHECK(zero_hold_states.is_contiguous(), "zero_hold_states must be contiguous");

    auto active_indices_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(query_states.device());
    auto active_indices = torch::zeros({batch_size, num_kv_heads, seq_len_q, keep_window_size},
                                       active_indices_options);
    auto output_options = torch::TensorOptions()
                              .dtype(query_states.dtype())
                              .device(query_states.device());
    auto output = torch::zeros({batch_size, seq_len_q, num_heads, head_dim}, output_options);

    auto softmax_lse_options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(query_states.device());
    auto softmax_lse = torch::zeros({batch_size, num_heads, seq_len_q}, softmax_lse_options);

    c10::cuda::CUDAGuard device_guard(query_states.device());

    std::vector<torch::Tensor> result_tensors;
    if (query_states.scalar_type() == at::kHalf) {
        result_tensors = FLASH_NAMESPACE::dynamic_mask_attention_dispatch<cutlass::half_t>(
            query_states, key_states, value_states, zero_hold_states, active_indices,
            output, softmax_lse, scale, keep_window_size, is_causal, return_softmax
        );
    } else if (query_states.scalar_type() == at::kBFloat16) {
        result_tensors = FLASH_NAMESPACE::dynamic_mask_attention_dispatch<cutlass::bfloat16_t>(
            query_states, key_states, value_states, zero_hold_states, active_indices,
            output, softmax_lse, scale, keep_window_size, is_causal, return_softmax
        );
    } else {
        TORCH_CHECK(false, "apply_attention only supports half and bfloat16");
    }

    if (return_softmax) {
        return {result_tensors[0], result_tensors[1]};
    } else {
        return {result_tensors[0]};
    }
}