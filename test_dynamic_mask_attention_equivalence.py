import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from flash_dma_cpp import apply_dynamic_mask_attention

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


def non_zero_mask_indices(
    dynamic_mask: torch.Tensor # [key_len]
):
    key_len = len(dynamic_mask)
    indices = []
    for i in range(key_len):
        if dynamic_mask[i] != 0:
            indices.append(i)
    return torch.tensor(indices)


def dynamic_mask_attention_python(
    query_states: torch.Tensor,          # [batch_size, num_heads, query_len, head_dim]
    key_states: torch.Tensor,            # [batch_size, num_kv_heads, key_len, head_dim]
    value_states: torch.Tensor,          # [batch_size, num_kv_heads, key_len, head_dim]
    dt_proj: torch.Tensor,               # [num_kv_heads, num_kv_heads * head_dim]
    A: torch.Tensor,                     # [num_kv_heads]
    scaling: float = None,
    causal_mask: torch.Tensor = None,    # [batch_size, 1, query_len, key_len] 0 is keep, -inf is mask
    keep_window_size=2048,
    is_causal=True
):
    batch_size, num_heads, query_len, head_dim = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape
    device = query_states.device
    dtype = query_states.dtype

    num_queries_per_kv = num_heads // num_kv_heads

    # 初始化结果张量
    attn_weights = torch.zeros((batch_size, num_heads, query_len, key_len), device=device, dtype=dtype)
    attn_outputs = torch.zeros((batch_size, num_heads, query_len, head_dim), device=device, dtype=dtype)
    dynamic_masks = torch.zeros(batch_size, num_kv_heads, query_len, key_len, device=device, dtype=dtype)

    #  1. 计算zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)

    for b_idx in range(batch_size):
        for kv_idx in range(num_kv_heads):
            for q_idx in range(query_len):
                #  应用因果掩码
                if is_causal and causal_mask is not None:
                    zero_hold_state = zero_hold_states[b_idx, kv_idx, q_idx, :].masked_fill(causal_mask[b_idx, 0, q_idx, :] != 0, 0)
                else:
                    zero_hold_state = zero_hold_states[b_idx, kv_idx, q_idx, :]
                
                # 2. 应用top-k选择
                actual_keep_window_size = min(keep_window_size, key_len)
                if key_len > actual_keep_window_size:
                    topk_values, topk_indices = torch.topk(zero_hold_state, actual_keep_window_size, dim=-1)
                    dynamic_mask = torch.zeros_like(zero_hold_state)
                    dynamic_mask.scatter_(-1, topk_indices, topk_values)
                else:
                    dynamic_mask = zero_hold_state
                dynamic_masks[b_idx, kv_idx, q_idx, :] = dynamic_mask

                non_mask_indices = non_zero_mask_indices(dynamic_mask)
                if len(non_mask_indices) == 0:
                    continue

                k_vecs = key_states[b_idx, kv_idx, non_mask_indices, :] # [keep_window_size, head_dim]
                v_vecs = value_states[b_idx, kv_idx, non_mask_indices, :] # [keep_window_size, head_dim]

                for q_group_idx in range(num_queries_per_kv):
                    h_idx = kv_idx * num_queries_per_kv + q_group_idx
                    q_vec = query_states[b_idx, h_idx, q_idx, :] # [head_dim]

                    attn_weight = torch.sum(q_vec.unsqueeze(0) * k_vecs, dim=-1)

                    if scaling is not None:
                        attn_weight = attn_weight * scaling

                    attn_weight = attn_weight + dynamic_mask[non_mask_indices]
                    attn_weight = F.softmax(attn_weight, dim=-1)
                    attn_weights[b_idx, h_idx, q_idx, non_mask_indices] = attn_weight
                    attn_output = torch.sum(attn_weight.unsqueeze(1) * v_vecs, dim=0)
                    attn_outputs[b_idx, h_idx, q_idx, :] = attn_output

    attn_outputs = attn_outputs.transpose(1, 2).contiguous()

    return dynamic_masks, attn_weights, attn_outputs

def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,          # [batch_size, num_heads, query_len, head_dim]
    key_states: torch.Tensor,            # [batch_size, num_kv_heads, key_len, head_dim]
    value_states: torch.Tensor,          # [batch_size, num_kv_heads, key_len, head_dim]
    dt_proj: torch.Tensor,               # [num_kv_heads, num_kv_heads * head_dim]
    A: torch.Tensor,                     # [num_kv_heads]
    scaling: float = None,
    causal_mask: torch.Tensor = None,    # [batch_size, 1, query_len, key_len] 0 is keep, -inf is mask
    keep_window_size=2048,
    is_causal=True,
    return_softmax=False
):
    """CUDA实现的动态掩码注意力"""
    batch_size, num_heads, query_len, head_dim = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape
    device = query_states.device
    orig_dtype = query_states.dtype  # 记录原始数据类型
    
    # 计算zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)
    
    # 确保数据类型和内存布局正确
    query_states = query_states.contiguous().to(torch.bfloat16)
    key_states = key_states.contiguous().to(torch.bfloat16)
    value_states = value_states.contiguous().to(torch.bfloat16)
    zero_hold_states = zero_hold_states.contiguous().to(torch.bfloat16)
    
    # 设置默认缩放因子
    if scaling is None:
        scaling = 1.0 / (head_dim ** 0.5)
    
    # 调用CUDA实现
    try:
        result = apply_dynamic_mask_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            zero_hold_states=zero_hold_states,
            scale=scaling,
            keep_window_size=min(keep_window_size, key_len),  # 确保窗口大小不超过序列长度
            is_causal=is_causal,
            return_softmax=return_softmax
        )
        
        # 将结果转换回原始数据类型
        output = result[0].to(orig_dtype)
        output = output.transpose(1, 2).contiguous()
        
        # 必须返回三个值以匹配Python实现
        return None, None, output
    except Exception as e:
        print(f"CUDA错误详情: {e}")
        # 打印更详细的调试信息
        print(f"query_states: shape={query_states.shape}, dtype={query_states.dtype}, contiguous={query_states.is_contiguous()}")
        print(f"key_states: shape={key_states.shape}, dtype={key_states.dtype}, contiguous={key_states.is_contiguous()}")
        print(f"value_states: shape={value_states.shape}, dtype={value_states.dtype}, contiguous={value_states.is_contiguous()}")
        print(f"zero_hold_states: shape={zero_hold_states.shape}, dtype={zero_hold_states.dtype}, contiguous={zero_hold_states.is_contiguous()}")
        raise

def analyze_differences(original_result, cuda_result, query_states, key_states, value_states):
    """分析两种实现之间的差异"""
    # 确保两个张量有相同的数据类型
    cuda_result = cuda_result.to(original_result.dtype)
    print(f"原始结果: {original_result.shape}, {original_result.dtype}")
    print(f"CUDA结果: {cuda_result.shape}, {cuda_result.dtype}")

    # 计算整体差异
    diff = torch.abs(original_result - cuda_result)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"结果分析:")
    print(f"  最大绝对差异: {max_diff:.8f}")
    print(f"  平均绝对差异: {mean_diff:.8f}")
    
    # 检查是否基本相等（考虑浮点误差）
    is_close = torch.allclose(original_result, cuda_result, rtol=1e-3, atol=1e-3)
    print(f"  两种实现的结果是否相等 (rtol=1e-3, atol=1e-3): {'是' if is_close else '否'}")
    
    # 如果误差较大，进一步分析
    if not is_close and max_diff > 1e-3:
        # 显示最大差异的位置
        max_diff_indices = torch.where(diff == torch.max(diff))
        if len(max_diff_indices[0]) > 0:
            b_idx, q_idx, h_idx, d_idx = [idx[0].item() for idx in max_diff_indices]
            
            print(f"\n  最大差异位置: batch={b_idx}, query={q_idx}, head={h_idx}, dim={d_idx}")
            print(f"    Python值: {original_result[b_idx, q_idx, h_idx, d_idx].item():.6f}")
            print(f"    CUDA值:   {cuda_result[b_idx, q_idx, h_idx, d_idx].item():.6f}")
            print(f"    差异:     {diff[b_idx, q_idx, h_idx, d_idx].item():.6f}")
            
            # 检查特定head在该批次和查询位置的整体差异
            head_diff = diff[b_idx, q_idx, h_idx, :].mean().item()
            print(f"  该head在该位置的平均差异: {head_diff:.8f}")
    
    return is_close, max_diff, mean_diff

def test_cuda_equivalence():
    """测试Python原型和CUDA实现的等价性"""
    print("\n" + "=" * 70)
    print("测试Python原型和CUDA实现的等价性")
    print("=" * 70)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 测试不同参数配置
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 1, 1, 32, 32, 32, True),     # 小规模测试，因果掩码
        (1, 8, 4, 32, 32, 64, False),    # 小规模测试，非因果掩码
        (2, 12, 6, 64, 128, 64, True),   # 中等规模，GQA模式
        (2, 8, 8, 64, 64, 128, True),    # 中等规模，MHA模式
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal = config
        
        print(f"\n测试配置 {i+1}/{len(test_configs)}:")
        print(f"  batch_size={batch_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  query_len={query_len}, key_len={key_len}, head_dim={head_dim}")
        print(f"  is_causal={is_causal}")
        
        # 创建随机输入数据
        query_states = torch.randn(batch_size, num_heads, query_len, head_dim, device=device)
        key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
        value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
        
        # 创建投影矩阵和缩放系数
        dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device)
        A = torch.rand(num_kv_heads, device=device) * 0.5 + 0.5  # 生成0.5到1.0之间的随机系数
    
        # 创建带缓存位置的自定义因果掩码
        cache_position = torch.arange(0, query_len + 0, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
                    (query_len, key_len), fill_value=min_type, device=device, dtype=value_states.dtype
                )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # 设置缩放因子和保留窗口大小
        scaling = 1.0 / (head_dim ** 0.5)
        keep_window_size = min(64, key_len)  # 使用较小的窗口大小用于测试
        
        # 运行Python实现
        start_time = time.time()
        _, _, py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        py_time = time.time() - start_time
        
        # 运行CUDA实现
        start_time = time.time()
        _, _, cuda_output = dynamic_mask_attention_cuda(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        cuda_time = time.time() - start_time
        
        # 分析差异
        is_close, max_diff, mean_diff = analyze_differences(
            py_output, cuda_output, query_states, key_states, value_states
        )
        
        # 报告性能差异
        speedup = py_time / cuda_time if cuda_time > 0 else float('inf')
        print(f"\n  性能对比:")
        print(f"    Python实现: {py_time*1000:.2f} ms")
        print(f"    CUDA实现:   {cuda_time*1000:.2f} ms")
        print(f"    加速比:     {speedup:.2f}x")
        
        # 更新测试结果
        test_result = "通过" if is_close else "失败"
        all_passed = all_passed and is_close
        print(f"\n  测试结果: {test_result}")
        
        # 如果测试失败，可以提前退出
        if not is_close and max_diff > 1e-2:
            print("  差异过大，停止后续测试。")
            break
    
    print("\n" + "=" * 70)
    print(f"等价性测试总结: {'全部通过' if all_passed else '有测试失败'}")
    print("=" * 70)
    
    return all_passed

def test_performance():
    """测试CUDA实现的性能"""
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)
    
    # 设置较大规模的测试
    configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim)
        (1, 16, 16, 256, 256, 64),      # 典型的中等序列长度，MHA
        (2, 16, 8, 512, 512, 64),       # 较长序列，GQA
        (1, 32, 8, 1024, 1024, 128),    # 长序列，大幅GQA
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n性能测试结果:")
    print(f"{'配置':<45}{'Python (ms)':<15}{'CUDA (ms)':<15}{'加速比':<10}")
    print("-" * 85)
    
    for config in configs:
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim = config
        
        config_str = f"b={batch_size}, h={num_heads}, kv={num_kv_heads}, q={query_len}, k={key_len}, d={head_dim}"
        
        # 创建随机输入数据
        query_states = torch.randn(batch_size, num_heads, query_len, head_dim, device=device)
        key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
        value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device)
        
        # 创建投影矩阵和缩放系数
        dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device)
        A = torch.rand(num_kv_heads, device=device) * 0.5 + 0.5
        
        # 创建带缓存位置的自定义因果掩码
        cache_position = torch.arange(0, query_len + 0, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
                    (query_len, key_len), fill_value=min_type, device=device, dtype=value_states.dtype
                )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # 设置缩放因子和保留窗口大小
        scaling = 1.0 / (head_dim ** 0.5)
        keep_window_size = min(key_len // 4, 256)  # 使用较大的窗口大小
        
        # 预热CUDA
        for _ in range(3):
            _ = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
        torch.cuda.synchronize()
        
        # 测量Python实现的时间（仅对小规模配置）
        if query_len * key_len <= 256 * 256:
            start_time = time.time()
            _ = dynamic_mask_attention_python(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
            torch.cuda.synchronize()
            py_time = (time.time() - start_time) * 1000  # ms
        else:
            py_time = float('inf')  # 对于大规模测试，Python实现太慢
        
        # 测量CUDA实现的时间
        start_time = time.time()
        for _ in range(5):  # 运行多次以获得更准确的时间
            _ = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) * 1000 / 5  # ms (平均每次运行时间)
        
        # 计算加速比
        speedup = py_time / cuda_time if cuda_time > 0 and py_time != float('inf') else "N/A"
        if isinstance(speedup, float):
            speedup = f"{speedup:.1f}x"
        
        # 格式化输出
        py_time_str = f"{py_time:.2f}" if py_time != float('inf') else "超时"
        print(f"{config_str:<45}{py_time_str:<15}{cuda_time:.2f:<15}{speedup:<10}")
    
    print("-" * 85)

if __name__ == "__main__":
    """
    测试动态掩码注意力Python原型与实际CUDA实现的等价性
    
    这个脚本验证两种实现的数值一致性:
    1. 原始Python实现
    2. 实际CUDA实现 (flash_dma 扩展)
    
    测试包括:
    - 不同批次大小、头数、序列长度和维度的多种配置
    - 因果和非因果掩码选项
    - 结果的数值等价性分析
    - 性能比较
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试动态掩码注意力Python/CUDA实现等价性')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--perf', action='store_true', help='运行性能测试')
    parser.add_argument('--equiv', action='store_true', help='运行等价性测试')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 打印测试环境信息
    print(f"PyTorch 版本: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 默认运行两种测试，或根据命令行参数选择
    run_equiv = args.equiv or not (args.equiv or args.perf)
    run_perf = args.perf or not (args.equiv or args.perf)
    
    if run_equiv:
        test_cuda_equivalence()
    
    if run_perf:
        test_performance()