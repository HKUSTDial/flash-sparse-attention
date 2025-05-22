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

// 首先定义Flash_fwd_params结构体，如果没有在其他头文件中定义的话
// 这里假设该结构体是在某个头文件中定义的，如果没有则需要在此添加定义
// struct Flash_fwd_params { ... };

template <typename scalar_t, int HeadDim, bool IsCausal>
__global__ void run_mha_fwd_kernel(Flash_fwd_params params) {
    // 修改块大小选择逻辑，确保与HeadDim兼容
    constexpr int kBlockM = 8;
    constexpr int kBlockN = HeadDim <= 32 ? 32 : 
                           (HeadDim == 64 ? 64 : 
                           (HeadDim == 128 ? 64 : 32));
    constexpr int kNWarps = 4;
    
    using Kernel_traits = Flash_fwd_kernel_traits<HeadDim, kBlockM, kBlockN, kNWarps, false, false, scalar_t>;
    
    constexpr bool kIsEvenMN = true; 
    constexpr bool kIsEvenK = true;  
    constexpr bool kReturnSoftmax = false; 
    
    compute_attn<Kernel_traits, IsCausal, kIsEvenMN, kIsEvenK, kReturnSoftmax>(params);
}

// 修改host-side启动函数
template <typename scalar_t, int HeadDim, bool IsCausal>
void launch_run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kBlockM = 8;
    constexpr int kBlockN = HeadDim <= 32 ? 32 : 
                           (HeadDim == 64 ? 64 : 
                           (HeadDim == 128 ? 64 : 32));
    constexpr int kNWarps = 4;

    using Kernel_traits = Flash_fwd_kernel_traits<HeadDim, kBlockM, kBlockN, kNWarps, false, false, scalar_t>;

    // 计算共享内存大小并添加额外的安全裕度
    size_t smem_size = Kernel_traits::kSmemSize;
    // 16字节对齐以确保正确访问
    smem_size = (smem_size + 15) & ~15;
    
    printf("需要的共享内存大小: %zu\n", smem_size);
    static_assert(Kernel_traits::kSmemSize <= 102 * 1024, "Shared memory usage exceeds hardware limit");

    // 查询设备支持的最大动态共享内存
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_dynamic_smem = prop.sharedMemPerBlockOptin;

    // 总是设置最大可用共享内存来避免边界问题
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            run_mha_fwd_kernel<scalar_t, HeadDim, IsCausal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    // 检查是否超出设备最大支持
    TORCH_CHECK(smem_size <= max_dynamic_smem,
        "需要的共享内存(", smem_size, ") 超过了设备最大支持(", max_dynamic_smem, ")");

    dim3 grid_dim(
        cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM),
        params.b,
        params.h
    );
    dim3 block_dim(Kernel_traits::kNThreads);

    run_mha_fwd_kernel<scalar_t, HeadDim, IsCausal><<<grid_dim, block_dim, smem_size, stream>>>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(err));
    }
}

// 动态掩码注意力调度函数
template <typename scalar_t>
std::vector<torch::Tensor> dynamic_mask_attention_dispatch(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& zero_hold_states,
    torch::Tensor& output,
    torch::Tensor& softmax_lse,
    float scale,
    int keep_window_size,
    bool is_causal,
    bool return_softmax
) {
    // 从输入张量获取head_dim - 修复第一个错误
    const int head_dim = query_states.size(3);
    
    // 设置内核参数和启动配置
    // 确保Flash_fwd_params在正确的命名空间中
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));
    
    // 设置参数
    params.q_ptr = query_states.data_ptr();
    params.k_ptr = key_states.data_ptr();
    params.v_ptr = value_states.data_ptr();
    params.o_ptr = output.data_ptr();
    params.zero_hold_ptr = zero_hold_states.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    
    // 设置维度参数
    params.b = query_states.size(0);
    params.h = query_states.size(1);
    params.h_k = key_states.size(1);
    params.seqlen_q = query_states.size(2);
    params.seqlen_k = key_states.size(2);
    params.d = head_dim;
    params.total_q = params.seqlen_q * params.b;
    
    // 设置其他参数
    params.scale_softmax = scale;
    params.scale_softmax_log2 = log2f(scale);
    params.keep_window_size = keep_window_size;
    params.h_h_k_ratio = params.h / params.h_k;
    
    // 设置stride
    params.q_batch_stride = query_states.stride(0);
    params.k_batch_stride = key_states.stride(0);
    params.v_batch_stride = value_states.stride(0);
    params.o_batch_stride = output.stride(0);
    params.zero_hold_batch_stride = zero_hold_states.stride(0);
    
    params.q_row_stride = query_states.stride(2);
    params.k_row_stride = key_states.stride(2);
    params.v_row_stride = value_states.stride(2);
    params.o_row_stride = output.stride(2);
    params.zero_hold_row_stride = zero_hold_states.stride(2);
    
    params.q_head_stride = query_states.stride(1);
    params.k_head_stride = key_states.stride(1);
    params.v_head_stride = value_states.stride(1);
    params.o_head_stride = output.stride(1);
    params.zero_hold_head_stride = zero_hold_states.stride(1);
    
    // 设置标志
    params.is_causal = is_causal;
    params.is_bf16 = query_states.scalar_type() == torch::kBFloat16;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 根据head_dim和是否因果选择合适的内核
    if (is_causal) {
        if (head_dim <= 32)        { launch_run_mha_fwd_<scalar_t, 32, true>(params, stream); }
        else if (head_dim <= 64)   { launch_run_mha_fwd_<scalar_t, 64, true>(params, stream); }
        else if (head_dim <= 128)  { launch_run_mha_fwd_<scalar_t, 128, true>(params, stream); }
        else { TORCH_CHECK(false, "不支持>128的head_dim"); }
    } else {
        if (head_dim <= 32)        { launch_run_mha_fwd_<scalar_t, 32, false>(params, stream); }
        else if (head_dim <= 64)   { launch_run_mha_fwd_<scalar_t, 64, false>(params, stream); }
        else if (head_dim <= 128)  { launch_run_mha_fwd_<scalar_t, 128, false>(params, stream); }
        else { TORCH_CHECK(false, "不支持>128的head_dim"); }
    }
    
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
    
    // 验证张量连续性
    TORCH_CHECK(query_states.is_contiguous(), "query_states must be contiguous");
    TORCH_CHECK(key_states.is_contiguous(), "key_states must be contiguous");
    TORCH_CHECK(value_states.is_contiguous(), "value_states must be contiguous");
    TORCH_CHECK(zero_hold_states.is_contiguous(), "zero_hold_states must be contiguous");
    
    const int batch_size = query_states.size(0);
    const int num_heads = query_states.size(1);
    const int query_len = query_states.size(2);
    const int head_dim = query_states.size(3);

    TORCH_CHECK(key_states.size(0) == batch_size, "Batch size mismatch between Q and K");
    const int num_kv_heads = key_states.size(1);
    const int key_len = key_states.size(2); // key_len from key_states
    
    // 验证维度参数
    TORCH_CHECK(value_states.size(2) == key_len, "Key length mismatch between K and V");
    TORCH_CHECK(key_states.size(3) == head_dim, "Head dimension mismatch between Q and K");
    TORCH_CHECK(value_states.size(0) == batch_size, "Batch size mismatch between Q and V");
    TORCH_CHECK(value_states.size(1) == num_kv_heads, "Num KV heads mismatch between K and V");
    TORCH_CHECK(value_states.size(3) == head_dim, "Head dimension mismatch between Q and V");

    TORCH_CHECK(zero_hold_states.size(0) == batch_size, "Batch size mismatch for zero_hold_states");
    TORCH_CHECK(zero_hold_states.size(1) == num_kv_heads, "Num KV heads mismatch for zero_hold_states");
    TORCH_CHECK(zero_hold_states.size(2) == query_len, "Query length mismatch for zero_hold_states");
    TORCH_CHECK(zero_hold_states.size(3) == key_len, "Key length mismatch for zero_hold_states");
    
    // 验证数据类型
    TORCH_CHECK(query_states.scalar_type() == at::kHalf || query_states.scalar_type() == at::kBFloat16, 
                "Dynamic Mask Attention只支持half (float16)和bfloat16数据类型");
    
    // 确保keep_window_size合法
    TORCH_CHECK(keep_window_size > 0 && keep_window_size <= key_len, 
               "keep_window_size必须是正数且不大于key_len");
    
    // 注意：使用与输入一致的维度顺序 [batch, num_heads, seq_len, head_dim]
    auto output_options = torch::TensorOptions()
                              .dtype(query_states.dtype())
                              .device(query_states.device());
    auto output = torch::empty({batch_size, num_heads, query_len, head_dim}, output_options);
    
    auto softmax_lse_options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(query_states.device());
    auto softmax_lse = torch::empty({batch_size, num_heads, query_len}, softmax_lse_options);
    
    c10::cuda::CUDAGuard device_guard(query_states.device());
    
    // 获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    std::vector<torch::Tensor> result_tensors;
    
    // 只分发半精度类型，不使用AT_DISPATCH_FLOATING_TYPES_AND2
    if (query_states.scalar_type() == at::kHalf) {
        result_tensors = FLASH_NAMESPACE::dynamic_mask_attention_dispatch<cutlass::half_t>(
            query_states, 
            key_states, 
            value_states,
            zero_hold_states,
            output, 
            softmax_lse, 
            scale, 
            keep_window_size,
            is_causal,
            return_softmax
        );
    } else if (query_states.scalar_type() == at::kBFloat16) {
        result_tensors = FLASH_NAMESPACE::dynamic_mask_attention_dispatch<cutlass::bfloat16_t>(
            query_states, 
            key_states, 
            value_states,
            zero_hold_states,
            output, 
            softmax_lse, 
            scale, 
            keep_window_size,
            is_causal,
            return_softmax
        );
    } else {
        TORCH_CHECK(false, "apply_dynamic_mask_attention仅支持half和bfloat16数据类型");
    }
    
    if (return_softmax) {
        return {result_tensors[0], result_tensors[1]};
    } else {
        return {result_tensors[0]};
    }
}