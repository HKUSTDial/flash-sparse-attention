import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from flash_dma_cpp import apply_dynamic_mask

def calculate_zero_hold_states(value_states, dt_proj, A, causal_mask=None):
    """
    计算zero_hold_states
    
    参数:
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        causal_mask: [batch_size, 1, query_len, key_len]
    
    返回:
        zero_hold_states: [batch_size, num_kv_heads, query_len, key_len]
    """
    batch_size, num_kv_heads, key_len, head_dim = value_states.shape
    query_len = causal_mask.shape[2] if causal_mask is not None else key_len
    
    # 1. 转置和重塑value_states并与dt_proj.T矩阵相乘
    dt_result = torch.matmul(value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), dt_proj.T)
    
    # 2. 应用softplus激活函数和系数A
    dt_states = torch.exp(F.softplus(dt_result) * A)
    zero_hold_states = dt_states.transpose(-1, -2).unsqueeze(-2).expand(-1, -1, query_len, -1)
        
    return zero_hold_states

def dynamic_mask_python(
    value_states: torch.Tensor,          # [batch_size, num_kv_heads, key_len, head_dim]
    dt_proj: torch.Tensor,               # [num_kv_heads, num_kv_heads * head_dim]
    A: torch.Tensor,                     # [num_kv_heads]
    causal_mask: torch.Tensor = None,    # [batch_size, 1, query_len, key_len] 0 is keep, -inf is mask
    keep_window_size=2048,
    is_causal=True
):
    batch_size, num_kv_heads, key_len, head_dim = value_states.shape
    query_len = causal_mask.shape[2] if causal_mask is not None else key_len
    device = value_states.device
    dtype = value_states.dtype

    # 结果张量
    dynamic_mask_result = torch.zeros(batch_size, num_kv_heads, query_len, key_len, device=device, dtype=dtype)

    #  1. 计算zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)

    for b_idx in range(batch_size):
        for kv_idx in range(num_kv_heads):
            for q_idx in range(query_len):
                # 创建副本进行原地操作（模拟CUDA的原地修改）
                zero_hold_state = zero_hold_states[b_idx, kv_idx, q_idx, :].clone()
                
                #  应用因果掩码（原地修改）
                if is_causal:
                    # 修正：确保与CUDA实现一致的因果掩码逻辑
                    zero_hold_state[q_idx+1:] = 0.0  # 直接使用切片，更高效
                
                # 2. 应用top-k选择（原地修改）
                actual_keep_window_size = min(keep_window_size, key_len)
                if key_len > actual_keep_window_size:
                    # 获取top-k值和索引
                    topk_values, topk_indices = torch.topk(zero_hold_state, actual_keep_window_size, dim=-1)
                    # 原地修改：先清零，然后scatter回去
                    zero_hold_state.fill_(0)
                    zero_hold_state.scatter_(-1, topk_indices, topk_values)
                
                # 存储结果
                dynamic_mask_result[b_idx, kv_idx, q_idx, :] = zero_hold_state
    
    return dynamic_mask_result

def dynamic_mask_cuda(
    value_states: torch.Tensor,          # [batch_size, num_kv_heads, key_len, head_dim]
    dt_proj: torch.Tensor,               # [num_kv_heads, num_kv_heads * head_dim]
    A: torch.Tensor,                     # [num_kv_heads]
    causal_mask: torch.Tensor = None,    # [batch_size, 1, query_len, key_len] 0 is keep, -inf is mask
    keep_window_size=2048,
    is_causal=True
):
    # 预先在python中计算zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)
    
    # 调用CUDA实现（现在使用原地操作）
    return apply_dynamic_mask(
        zero_hold_states=zero_hold_states.contiguous(),
        causal_mask=causal_mask.contiguous() if causal_mask is not None else None,
        keep_window_size=keep_window_size,
        is_causal=is_causal
    )

def analyze_differences(original_result, cuda_result, value_states, dt_proj, A, batch_size, key_len, num_kv_heads, head_dim):
    """分析两种实现之间的差异"""
    # 计算整体差异
    diff = torch.abs(original_result - cuda_result)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"结果分析:")
    print(f"  最大差异: {max_diff:.8f}")
    print(f"  平均差异: {mean_diff:.8f}")
    
    # 检查是否基本相等（考虑浮点误差）
    is_equal = torch.allclose(original_result, cuda_result, rtol=1e-5, atol=1e-5)
    print(f"  两种实现的结果是否相等: {is_equal}")
    
    # 检查非零元素位置是否匹配（这对于稀疏掩码非常重要）
    original_nonzero = (original_result != 0)
    cuda_nonzero = (cuda_result != 0)
    
    # 使用逻辑运算符
    intersection = torch.sum((original_nonzero & cuda_nonzero).float()).item()
    union = torch.sum((original_nonzero | cuda_nonzero).float()).item()
    nonzero_match = intersection / max(1.0, union)
    
    print(f"  非零元素位置匹配率: {nonzero_match:.4f}")
    
    # 检查topk选择是否一致
    if original_result.shape[-1] > 1:
        _, original_topk_indices = torch.sort(original_result, dim=-1, descending=True)
        _, cuda_topk_indices = torch.sort(cuda_result, dim=-1, descending=True)
        
        k = min(128, original_result.shape[-1])
        original_topk = original_topk_indices[..., :k]
        cuda_topk = cuda_topk_indices[..., :k]
        
        topk_match_count = 0
        total_elements = original_topk.numel()
        
        for b in range(original_topk.shape[0]):
            for h in range(original_topk.shape[1]):
                for q in range(original_topk.shape[2]):
                    orig_indices = set(original_topk[b, h, q].tolist())
                    cuda_indices = set(cuda_topk[b, h, q].tolist())
                    topk_match_count += len(orig_indices.intersection(cuda_indices))
        
        topk_match_rate = topk_match_count / total_elements
        print(f"  Top-{k}元素匹配率: {topk_match_rate:.4f}")
    
    # 如果不相等，显示一些不一致的示例
    if not is_equal and max_diff > 1e-4:
        max_diff_idx = torch.argmax(diff.flatten())
        max_diff_idx = np.unravel_index(max_diff_idx.item(), diff.shape)
        
        print("\n差异最大的位置详情:")
        print(f"  位置坐标: {max_diff_idx}")
        print(f"  原始实现值: {original_result[max_diff_idx]:.6f}")
        print(f"  CUDA实现值: {cuda_result[max_diff_idx]:.6f}")
        print(f"  绝对差异: {diff[max_diff_idx]:.6f}")
        
        if max_diff > 0.1:
            print("\n注: 小的数值差异可能由浮点计算顺序不同导致，大的差异可能表示实现逻辑有问题。")

def test_cuda_equivalence():
    """测试Python原型和真实CUDA实现的等价性"""
    print("=" * 50)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 测试不同参数配置以覆盖更多场景
    test_configs = [
        # (batch_size, num_kv_heads, key_len, query_len, head_dim, keep_window_size)
        (1, 2, 8, 4, 4, 2),         # 小规模测试
        (1, 2, 200, 100, 8, 50),      # 中等规模，key_len > keep_window_size
        (1, 2, 2048, 2048, 128, 2048),   # 大规模测试
        (1, 1, 40, 40, 16, 100),      # key_len < keep_window_size 的情况
    ]
    
    for i, config in enumerate(test_configs):
        batch_size, num_kv_heads, key_len, query_len, head_dim, keep_window_size = config
        print(f"\n测试配置 #{i+1}:")
        print(f"  batch_size={batch_size}, num_kv_heads={num_kv_heads}, key_len={key_len}")
        print(f"  query_len={query_len}, head_dim={head_dim}, keep_window_size={keep_window_size}")
        
        # 创建测试数据
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
        dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device)
        A = torch.rand(num_kv_heads, device=device)
        
        # 创建带缓存位置的自定义因果掩码
        cache_position = torch.arange(0, query_len + 0, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
                    (query_len, key_len), fill_value=min_type, device=device, dtype=value_states.dtype
                )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        # 计算原始Python实现的结果
        print("计算原始Python实现结果...")
        start_time = time.time()
        original_result = dynamic_mask_python(value_states, dt_proj, A, causal_mask, keep_window_size)
        python_time = time.time() - start_time
        print(f"  Python耗时: {python_time:.6f}秒")
        
        # 计算真实CUDA实现的结果
        start_time = time.time()
        cuda_result = dynamic_mask_cuda(
            value_states,
            dt_proj,
            A,
            causal_mask=causal_mask,
            keep_window_size=keep_window_size,
            is_causal=True,
        )
        cuda_time = time.time() - start_time
        print(f"  CUDA耗时: {cuda_time:.6f}秒")
        print(f"  加速比: {python_time/cuda_time:.2f}x")
        
        # 计算和分析差异
        analyze_differences(original_result, cuda_result, value_states, dt_proj, A, batch_size, key_len, num_kv_heads, head_dim)
        
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

def test_performance():
    """测试CUDA实现的性能"""
    print("=" * 50)
    
    # 设置较大规模的测试
    configs = [
        # (batch_size, num_kv_heads, key_len, query_len, head_dim, keep_window_size)
        (1, 1, 8192, 1, 64, 2048),    # 典型的transformer配置
        (1, 2, 16384, 1, 128, 2048), # 大规模配置
        (1, 4, 32768, 1, 128, 2048), # 超大规模配置（如果内存允许）
    ]
    
    print("\n性能测试结果:")
    print("配置                                     Python耗时   CUDA耗时   加速比")
    print("-" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for config in configs:
        batch_size, num_kv_heads, key_len, query_len, head_dim, keep_window_size = config
        
        try:
            # 创建测试数据
            value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
            dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device)
            A = torch.rand(num_kv_heads, device=device)
            
            # 创建带缓存位置的自定义因果掩码
            cache_position = torch.arange(0, query_len + 0, device=device)
            min_type = torch.finfo(value_states.dtype).min
            causal_mask = torch.full(
                        (query_len, key_len), fill_value=min_type, device=device, dtype=value_states.dtype
                    )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            
            # 预热
            _ = dynamic_mask_python(value_states, dt_proj, A, causal_mask, keep_window_size)
            _ = dynamic_mask_cuda(
                value_states,
                dt_proj,
                A,
                causal_mask=causal_mask,
                keep_window_size=keep_window_size,
                is_causal=True,
            )
            
            # 测量Python时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = dynamic_mask_python(value_states, dt_proj, A, causal_mask, keep_window_size)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            python_time = time.time() - start_time
            
            # 测量CUDA时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = dynamic_mask_cuda(
                value_states,
                dt_proj,
                A,
                causal_mask=causal_mask,
                keep_window_size=keep_window_size,
                is_causal=True,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            cuda_time = time.time() - start_time
            
            # 计算加速比
            speedup = python_time / cuda_time
            
            # 打印结果
            config_str = f"b={batch_size}, h={num_kv_heads}, seq={key_len}, d={head_dim}, k={keep_window_size}"
            print(f"{config_str:<40} {python_time:9.6f}s {cuda_time:9.6f}s {speedup:8.2f}x")
            
        except RuntimeError as e:
            config_str = f"b={batch_size}, h={num_kv_heads}, seq={key_len}, d={head_dim}, k={keep_window_size}"
            print(f"{config_str:<40} 内存不足，跳过")
    
    print("-" * 70)

if __name__ == "__main__":
    """
    测试动态掩码 Python 原型与实际 CUDA 实现的等价性
    
    这个脚本验证两种实现的数值一致性:
    1. 原始 Python 实现 (torch.topk 方式)
    2. 实际 CUDA 实现 (flash_dma 扩展)
    
    脚本会输出详细的测试结果，包括:
    - 数值差异统计
    - 非零元素匹配率
    - Top-K 元素选择一致性
    - 性能比较
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试动态掩码 Python/CUDA 实现等价性')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--perf', action='store_true', help='运行性能测试')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 打印测试环境信息
    print(f"PyTorch 版本: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"随机种子: {args.seed}")
    
    # 运行等价性测试
    test_cuda_equivalence()

    # 如果需要，运行性能测试
    if args.perf:
        test_performance()




