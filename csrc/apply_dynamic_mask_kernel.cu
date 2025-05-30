#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "src/namespace_config.h"
#include "src/mask.h"
#include "src/utils.h"
#include "src/hardware_info.h"
#include "src/static_switch.h"


using namespace FLASH_NAMESPACE;
using namespace cute;

// 重新设计的动态掩码CUDA内核，使用DynamicMask结构体
template <typename scalar_t, bool is_causal, int kBlockM, int kBlockN>
__global__ void apply_dynamic_mask_kernel(
    scalar_t* output_ptr,
    const scalar_t* zero_hold_states_ptr,
    const int batch_size,
    const int num_kv_heads,
    const int query_len,
    const int key_len,
    const int keep_window_size
) {
    // 使用mask.h中的DynamicMask结构体
    DynamicMask<is_causal> dynamic_mask(key_len, query_len, keep_window_size);
    
    // 动态分配共享内存
    extern __shared__ char smem[];
    scalar_t* smem_zero_hold = reinterpret_cast<scalar_t*>(smem);
    bool* smem_active_indices = reinterpret_cast<bool*>(smem_zero_hold + kBlockM * kBlockN);
    
    // 计算当前线程块处理的批次和头部索引
    const int batch_head_idx = blockIdx.y * gridDim.z + blockIdx.z;
    const int b_idx = batch_head_idx / num_kv_heads;
    const int kv_idx = batch_head_idx % num_kv_heads;
    
    if (b_idx >= batch_size) return;
    
    // 计算当前线程块处理的行和列索引
    const int row_idx_offset = blockIdx.x * kBlockM;
    const int col_idx_offset = 0;  // 处理整行
    
    // 计算全局内存偏移
    const int batch_head_offset = (b_idx * num_kv_heads + kv_idx) * query_len * key_len;
    
    // 创建共享内存张量 - 使用3D布局以匹配DynamicMask的期望
    // 布局: (MMA=4, MMA_M, MMA_N) 
    constexpr int MMA = 4;
    constexpr int MMA_M = kBlockM / (2 * 8);  // 2个外部行，每个8行
    constexpr int MMA_N = kBlockN / (2 * 1);  // 2列
    
    auto smem_zero_hold_tensor = make_tensor(
        make_smem_ptr(smem_zero_hold),
        make_shape(Int<MMA>{}, Int<MMA_M>{}, Int<MMA_N>{}),
        make_stride(Int<MMA_M * MMA_N>{}, Int<MMA_N>{}, Int<1>{})
    );
    
    auto smem_active_indices_tensor = make_tensor(
        make_smem_ptr(smem_active_indices),
        make_shape(Int<MMA>{}, Int<MMA_M>{}, Int<MMA_N>{}),
        make_stride(Int<MMA_M * MMA_N>{}, Int<MMA_N>{}, Int<1>{})
    );
    
    // 协作加载数据到共享内存
    const int tid = threadIdx.x;
    const int elements_per_thread = (kBlockM * kBlockN + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        int elem_idx = tid * elements_per_thread + i;
        if (elem_idx < kBlockM * kBlockN) {
            int local_row = elem_idx / kBlockN;
            int local_col = elem_idx % kBlockN;
            int global_row = row_idx_offset + local_row;
            int global_col = col_idx_offset + local_col;
            
            if (global_row < query_len && global_col < key_len) {
                smem_zero_hold[elem_idx] = zero_hold_states_ptr[
                    batch_head_offset + global_row * key_len + global_col
                ];
            } else {
                smem_zero_hold[elem_idx] = scalar_t(0.0f);
            }
            smem_active_indices[elem_idx] = true;
        }
    }
    __syncthreads();
    
    // 使用DynamicMask处理
    dynamic_mask.get_active_zerohold(
        smem_zero_hold_tensor,
        smem_active_indices_tensor,
        col_idx_offset,
        row_idx_offset,
        1  // warp_row_stride
    );
    
    // 将结果写回全局内存
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        int elem_idx = tid * elements_per_thread + i;
        if (elem_idx < kBlockM * kBlockN) {
            int local_row = elem_idx / kBlockN;
            int local_col = elem_idx % kBlockN;
            int global_row = row_idx_offset + local_row;
            int global_col = col_idx_offset + local_col;
            
            if (global_row < query_len && global_col < key_len) {
                output_ptr[batch_head_offset + global_row * key_len + global_col] = 
                    smem_zero_hold[elem_idx];
            }
        }
    }
}

template <typename scalar_t, bool is_causal>
void apply_dynamic_mask_cuda_impl(
    torch::Tensor& output,
    const torch::Tensor& zero_hold_states,
    const int keep_window_size
) {
    // 获取维度
    const int batch_size = zero_hold_states.size(0);
    const int num_kv_heads = zero_hold_states.size(1);
    const int query_len = zero_hold_states.size(2);
    const int key_len = zero_hold_states.size(3);
    
    // 使用较小的块尺寸以适应共享内存
    constexpr int kBlockM = 16;  // 处理16行
    constexpr int kBlockN = 128; // 直接使用 128，不用 min 函数
    
    // 计算共享内存大小
    const int smem_size = kBlockM * kBlockN * sizeof(scalar_t) + 
                         kBlockM * kBlockN * sizeof(bool);
    
    // 检查共享内存大小是否超过限制
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, zero_hold_states.device().index());
    TORCH_CHECK(smem_size <= props.sharedMemPerBlock, 
                "共享内存需求(", smem_size, "字节)超过设备限制(", 
                props.sharedMemPerBlock, "字节)");
    
    // 配置线程块和网格
    constexpr int threads_per_block = 256;
    dim3 block(threads_per_block);
    
    // 计算需要的块数
    const int grid_m = (query_len + kBlockM - 1) / kBlockM;
    const int batch_head_count = batch_size * num_kv_heads;
    
    // 使用y和z维度来处理批次和头部
    dim3 grid(
        grid_m,
        min(batch_head_count, 65535),
        (batch_head_count + 65534) / 65535
    );
    
    // 启动CUDA内核
    apply_dynamic_mask_kernel<scalar_t, is_causal, kBlockM, kBlockN>
        <<<grid, block, smem_size>>>(
        output.data_ptr<scalar_t>(),
        zero_hold_states.data_ptr<scalar_t>(),
        batch_size,
        num_kv_heads,
        query_len,
        key_len,
        keep_window_size
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

// 主接口函数
torch::Tensor apply_dynamic_mask_cuda(
    const torch::Tensor& zero_hold_states,
    const int keep_window_size,
    const bool is_causal
) {
    
    // 验证输入
    TORCH_CHECK(zero_hold_states.dim() == 4, "zero_hold_states必须是4D张量 [batch_size, num_kv_heads, query_len, key_len]");
    
    // 所有张量必须是CUDA张量
    TORCH_CHECK(zero_hold_states.is_cuda(), "zero_hold_states必须是CUDA张量");
    
    // 获取维度
    const int batch_size = zero_hold_states.size(0);
    const int num_kv_heads = zero_hold_states.size(1);
    const int query_len = zero_hold_states.size(2);
    const int key_len = zero_hold_states.size(3);
    
    // 创建输出张量并复制输入（因为需要原地修改）
    auto output = zero_hold_states.clone();
    
    // 设置当前设备
    c10::cuda::CUDAGuard device_guard(zero_hold_states.device());
    
    // 根据数据类型和因果掩码标志分发实现
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(zero_hold_states.scalar_type(), "apply_dynamic_mask", ([&] {
        if (is_causal) {
            apply_dynamic_mask_cuda_impl<scalar_t, true>(
                output, zero_hold_states, keep_window_size);
        } else {
            apply_dynamic_mask_cuda_impl<scalar_t, false>(
                output, zero_hold_states, keep_window_size);
        }
    }));
    
    return output;
}