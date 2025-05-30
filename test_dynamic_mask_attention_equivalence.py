import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from flash_dma_cpp import apply_dynamic_mask_attention
# from flash_dma_cpp import apply_attention

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
    batch_size, _, key_len, _ = value_states.shape
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

def non_neginf_mask_indices(
    dynamic_mask: torch.Tensor # [key_len]
):
    key_len = len(dynamic_mask)
    indices = []
    for i in range(key_len):
        if dynamic_mask[i] != torch.finfo(dynamic_mask.dtype).min:
            indices.append(i)
    return torch.tensor(indices)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def dynamic_mask_attention_python(
#     query_states: torch.Tensor,
#     key_states: torch.Tensor,
#     value_states: torch.Tensor,
#     dt_proj: torch.Tensor,
#     A: torch.Tensor,
#     scaling: float = None,
#     causal_mask: torch.Tensor = None,
#     keep_window_size=2048,
#     is_causal=True,
#     debug_print=True  # 添加调试打印开关
# ):  
#     batch_size, num_heads, query_len, head_dim = query_states.shape
#     _, num_kv_heads, key_len, _ = key_states.shape
#     device = query_states.device
#     dtype = query_states.dtype

#     num_queries_per_kv = num_heads // num_kv_heads

#     if debug_print:
#         print(f"\n=== Python实现调试信息 ===")
#         print(f"输入形状: Q={query_states.shape}, K={key_states.shape}, V={value_states.shape}")
#         print(f"scaling={scaling}, keep_window_size={keep_window_size}")
#         print(f"Q范围: [{query_states.min():.6f}, {query_states.max():.6f}]")
#         print(f"K范围: [{key_states.min():.6f}, {key_states.max():.6f}]")
#         print(f"V范围: [{value_states.min():.6f}, {value_states.max():.6f}]")

#     # 1. 初始化attn_weights 和 attn_outputs
#     attn_weights = torch.zeros((batch_size, num_heads, query_len, key_len), device=device, dtype=dtype)
#     attn_outputs = torch.zeros((batch_size, num_heads, query_len, head_dim), device=device, dtype=dtype)

#     # 2. 转置和重塑value_states 并与 dt_proj.T 矩阵相乘
#     value_reshaped = value_states.transpose(-2, -3).reshape(batch_size, key_len, -1)
#     dt_result = torch.matmul(value_reshaped, dt_proj.T)

#     # 3. 应用softplus激活函数和系数A
#     softplus_result = F.softplus(dt_result)
#     scaled_result = softplus_result * A
#     exp_result = torch.exp(scaled_result)
#     zero_hold_states = exp_result.transpose(-1, -2).unsqueeze(-2).expand(-1, -1, query_len, -1)

#     # 只处理第一个batch和第一个query进行详细调试
#     debug_batch = 0
#     debug_kv = 0  
#     debug_query = 0

#     for b_idx in range(batch_size):
#         for kv_idx in range(num_kv_heads):
#             for q_idx in range(query_len):
#                 if debug_print and b_idx == debug_batch and kv_idx == debug_kv and q_idx == debug_query:
#                     print(f"\n--- Step 1: 处理第一个block ---")
#                     print(f"Query_states[{b_idx},{kv_idx},{q_idx},:8]: {query_states[b_idx, kv_idx, q_idx, :8]}")
#                     print(f"Key_states[{b_idx},{kv_idx},{q_idx},:8]: {key_states[b_idx, kv_idx, q_idx, :8]}")
#                     print(f"Value_states[{b_idx},{kv_idx},{q_idx},:8]: {value_states[b_idx, kv_idx, q_idx, :8]}")
#                     print(f"zero_hold_states[{b_idx},{kv_idx},{q_idx},:8]: {zero_hold_states[b_idx, kv_idx, q_idx, :8]}")
#                 # 3. 应用因果掩码
#                 if is_causal and causal_mask is not None:
#                     zero_hold_state = zero_hold_states[b_idx, kv_idx, q_idx, :].masked_fill(causal_mask[b_idx, 0, q_idx, :] != 0, torch.finfo(value_states.dtype).min)
#                 else:
#                     zero_hold_state = zero_hold_states[b_idx, kv_idx, q_idx, :]
      
#                 # 4. 应用top-k选择
#                 keep_window_size = min(keep_window_size, key_len)
#                 if key_len > keep_window_size:
#                     topk_values, topk_indices = torch.topk(zero_hold_state, keep_window_size, dim=-1)
#                     dynamic_mask = zero_hold_state.clone()
#                     dynamic_mask.scatter_(-1, topk_indices, topk_values)
#                 else:
#                     dynamic_mask = zero_hold_state

#                 # 5. 稀疏注意力权重计算
#                 non_mask_indices = non_neginf_mask_indices(dynamic_mask)
#                 if len(non_mask_indices) == 0:
#                     continue
                    
#                 if debug_print and b_idx == debug_batch and kv_idx == debug_kv and q_idx == debug_query:
#                     print(f"\n--- Step 2: dynamic mask处理后 [{b_idx},{kv_idx},{q_idx}] ---")
#                     print(f"non_mask_indices: {non_mask_indices[:8]}")
#                     print(f"zero_hold_state[前8个]: {dynamic_mask[:8]}")

#                 k_vecs = key_states[b_idx, kv_idx, non_mask_indices, :] # [keep_window_size, head_dim]
#                 v_vecs = value_states[b_idx, kv_idx, non_mask_indices, :] # [keep_window_size, head_dim]

#                 for q_group_idx in range(num_queries_per_kv):
#                     h_idx = kv_idx * num_queries_per_kv + q_group_idx
#                     q_vec = query_states[b_idx, h_idx, q_idx, :] # [head_dim]

#                     # QK点积
#                     attn_weight = torch.sum(q_vec.unsqueeze(0) * k_vecs, dim=-1)
                    
#                     if debug_print and b_idx == debug_batch and kv_idx == debug_kv and q_idx == debug_query and q_group_idx == 0:
#                         print(f"\n--- Step 3: 注意力计算 head[{h_idx}] ---")
#                         print(f"QK点积attn_weight: {attn_weight[:8]}")

#                     # 应用scaling和dynamic_mask
#                     attn_weight = attn_weight * scaling + dynamic_mask[non_mask_indices]
                    
#                     if debug_print and b_idx == debug_batch and kv_idx == debug_kv and q_idx == debug_query and q_group_idx == 0:
#                         print(f"\n--- Step 4: 应用scaling和mask后 ---")
#                         print(f"scaling mask后attn_weight: {attn_weight[:8]}")
    
                    
#                     # Softmax
#                     attn_weight = F.softmax(attn_weight, dim=-1)
#                     attn_weights[b_idx, h_idx, q_idx, non_mask_indices] = attn_weight
                    
#                     if debug_print and b_idx == debug_batch and kv_idx == debug_kv and q_idx == debug_query and q_group_idx == 0:
#                         print(f"\n--- Step 5: softmax后 ---")
#                         print(f"softmax后attn_weight: {attn_weight[:8]}")

#                     # 计算输出
#                     attn_output = torch.sum(attn_weight.unsqueeze(1) * v_vecs, dim=0)
#                     attn_outputs[b_idx, h_idx, q_idx, :] = attn_output

#     attn_outputs = attn_outputs.transpose(1, 2).contiguous()
    
#     if debug_print:
#         print(f"\n--- 最终输出 ---")
#         print(f"attn_outputs shape: {attn_outputs.shape}")
#         print(f"attn_outputs范围: [{attn_outputs.min():.6f}, {attn_outputs.max():.6f}]")
#         print(f"第一个输出[0,0,0,:8]: {attn_outputs[0,0,0,:8]}")

#     return attn_outputs

def dynamic_mask_attention_python(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float = None,
    causal_mask: torch.Tensor = None,
    keep_window_size=2048,
    is_causal=True,
):  
    batch_size, num_heads, query_len, head_dim = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape
    device = query_states.device
    dtype = query_states.dtype

    num_queries_per_kv = num_heads // num_kv_heads


    # 2. 转置和重塑value_states 并与 dt_proj.T 矩阵相乘
    value_reshaped = value_states.transpose(-2, -3).reshape(batch_size, key_len, -1)
    dt_result = torch.matmul(value_reshaped, dt_proj.T)

    # 3. 应用softplus激活函数和系数A
    softplus_result = F.softplus(dt_result)
    scaled_result = softplus_result * A
    exp_result = torch.exp(scaled_result)
    zero_hold_states = exp_result.transpose(-1, -2).unsqueeze(-2).expand(-1, -1, query_len, -1)

    if is_causal and causal_mask is not None:
        zero_hold_states = zero_hold_states.masked_fill(causal_mask != 0, torch.finfo(value_states.dtype).min)
    else:
        zero_hold_states = zero_hold_states

    # 4. 应用top-k选择
    keep_window_size = min(keep_window_size, key_len)
    if key_len > keep_window_size:
        topk_values, topk_indices = torch.topk(zero_hold_states, keep_window_size, dim=-1)
        dynamic_mask = zero_hold_states.clone()
        dynamic_mask.scatter_(-1, topk_indices, topk_values)
    else:
        dynamic_mask = zero_hold_states
    
    # 5. 稀疏注意力权重计算
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    dynamic_mask = repeat_kv(dynamic_mask, num_queries_per_kv)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))  # [batch_size, num_heads, query_len, key_len]
    attn_weights = attn_weights * scaling + dynamic_mask  # 应用scaling和dynamic_mask
    attn_weights = F.softmax(attn_weights, dim=-1)  # Softmax归一化
    attn_outputs = torch.matmul(attn_weights, value_states)  # [batch_size, num_heads, query_len, head_dim]
    attn_outputs = attn_outputs.transpose(1, 2).contiguous()  # 转置为 [batch_size, query_len, num_heads, head_dim]
    # print(f"\n--- 最终输出 ---")
    # print(f"attn_outputs shape: {attn_outputs.shape}")
    # print(f"attn_outputs范围: [{attn_outputs.min():.6f}, {attn_outputs.max():.6f}]")
    # print(f"第一个输出[0,0,0,:8]: {attn_outputs[0,0,0,:8]}")
    return attn_outputs

def flash_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float = None,
    causal_mask: torch.Tensor = None,
    is_causal=True,
):
    """CUDA实现的Flash Attention"""
    _, _, query_len, _ = query_states.shape
    _, _, key_len, _ = key_states.shape
    if query_len > 32768 or key_len > 32768:
        return "OOM"

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    attn_outputs = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        # is_causal=is_causal,
        scale=scaling,
        enable_gqa=True
    )
    attn_outputs = attn_outputs.transpose(1, 2).contiguous()  # 转置为 [batch_size, query_len, num_heads, head_dim]
    return attn_outputs

def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float = None,
    causal_mask: torch.Tensor = None,
    keep_window_size=2048,
    is_causal=True,
    return_softmax=False
):
    """CUDA实现的动态掩码注意力"""
    
    # 计算zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask).contiguous()
    print(zero_hold_states)

    # 确保数据类型和内存布局正确
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()
    zero_hold_states = zero_hold_states.contiguous()

    result = apply_dynamic_mask_attention(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        zero_hold_states=zero_hold_states,
        scale=scaling,
        keep_window_size=keep_window_size,  # 确保窗口大小不超过序列长度
        is_causal=is_causal,
        return_softmax=return_softmax
    )
    
    # 将结果转换回原始数据类型
    attn_outputs = result[0]
    # print(f"\n--- 最终输出 ---")
    # print(f"attn_outputs shape: {attn_outputs.shape}")
    # print(f"attn_outputs范围: [{attn_outputs.min():.6f}, {attn_outputs.max():.6f}]")
    # print(f"第一个输出[0,0,0,:8]: {attn_outputs[0,0,0,:8]}")
    return attn_outputs
  

def analyze_differences(original_result, cuda_result):
    """分析两种实现之间的差异"""
    # 确保两个张量有相同的数据类型
    cuda_result = cuda_result.to(original_result.dtype)
    print(f"原始结果: {original_result.shape}, {original_result.dtype}")
    print(f"CUDA结果: {cuda_result.shape}, {cuda_result.dtype}")

    # 添加详细的调试信息
    print(f"\n调试信息:")
    print(f"  原始结果范围: [{torch.min(original_result):.6f}, {torch.max(original_result):.6f}]")
    print(f"  CUDA结果范围: [{torch.min(cuda_result):.6f}, {torch.max(cuda_result):.6f}]")
    
    # 检查是否有NaN或Inf值
    original_has_nan = torch.isnan(original_result).any()
    cuda_has_nan = torch.isnan(cuda_result).any()
    original_has_inf = torch.isinf(original_result).any()
    cuda_has_inf = torch.isinf(cuda_result).any()
    
    print(f"  原始结果包含NaN: {original_has_nan}, Inf: {original_has_inf}")
    print(f"  CUDA结果包含NaN: {cuda_has_nan}, Inf: {cuda_has_inf}")

    # 计算整体差异
    diff = torch.abs(original_result - cuda_result)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    # 找到最大差异的位置
    max_diff_idx = torch.argmax(diff.flatten())
    max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
    orig_val = original_result[max_diff_pos].item()
    cuda_val = cuda_result[max_diff_pos].item()
    
    print(f"\n结果分析:")
    print(f"  最大绝对差异: {max_diff:.8f}")
    print(f"  平均绝对差异: {mean_diff:.8f}")
    print(f"  最大差异位置: {max_diff_pos}")
    print(f"  该位置原始值: {orig_val:.8f}")
    print(f"  该位置CUDA值: {cuda_val:.8f}")
    
    # 计算相对差异
    relative_diff = diff / (torch.abs(original_result) + 1e-8)
    max_rel_diff = torch.max(relative_diff).item()
    mean_rel_diff = torch.mean(relative_diff).item()
    print(f"  最大相对差异: {max_rel_diff:.8f}")
    print(f"  平均相对差异: {mean_rel_diff:.8f}")
    
    # 根据数据类型调整容差
    if original_result.dtype == torch.bfloat16:
        # bfloat16 的有效精度约为 3-4 位小数，所以放宽容差
        rtol, atol = 1e-2, 1e-2
        tolerance_note = "bfloat16 容差"
    elif original_result.dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
        tolerance_note = "float16 容差"
    else:
        rtol, atol = 1e-3, 1e-3
        tolerance_note = "float32 容差"
    
    # 检查是否基本相等
    is_close = torch.allclose(original_result, cuda_result, rtol=rtol, atol=atol)
    print(f"  两种实现的结果是否相等 ({tolerance_note}: rtol={rtol}, atol={atol}): {'是' if is_close else '否'}")
    
    # 统计在容差范围内的元素比例
    close_mask = torch.abs(original_result - cuda_result) <= (atol + rtol * torch.abs(cuda_result))
    close_ratio = torch.sum(close_mask).float() / close_mask.numel()
    print(f"  在容差范围内的元素比例: {close_ratio:.4f} ({torch.sum(close_mask)}/{close_mask.numel()})")
    
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
        # (1, 1, 1, 8, 8, 32, True),     # 小规模测试，因果掩码
        (1, 1, 1, 64, 64, 32, True),    # 小规模测试，非因果掩码
        # (1, 2, 1, 8, 8, 32, True),     # 小规模测试，GQA模式
        # (1, 1, 1, 128, 128, 32, True),  # 中等规模测试，因果掩码
        # (1, 1, 1, 128, 128, 32, False), # 中等规模测试，非因果掩码
        # (1, 2, 1, 128, 128, 32, True),   # 中等规模测试，GQA模式
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
        query_states = torch.randn(batch_size, num_heads, query_len, head_dim, device=device, dtype=torch.bfloat16)
        key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device, dtype=torch.bfloat16)
        value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device, dtype=torch.bfloat16)
        dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
        A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
        
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
        scaling = head_dim ** -0.5
        keep_window_size = min(32, key_len)  # 使用较小的窗口大小用于测试

        # 运行Python实现
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        py_output_copy = py_output.clone()
        py_time = time.time() - start_time
        # exit(0)
        # 运行CUDA实现
        start_time = time.time()
        cuda_output = dynamic_mask_attention_cuda(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        cuda_time = time.time() - start_time

        # 分析差异
        is_close, max_diff, mean_diff = analyze_differences(
            py_output_copy, cuda_output
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
    """测试CUDA实现的性能，每种配置运行3次取平均值"""
    print("\n" + "=" * 70)
    print("性能测试 (每种配置运行3次取平均值)")
    print("=" * 70)
    
    # 设置较大规模的测试
    configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim)
        (1, 2, 1, 256, 256, 32),
        (1, 2, 1, 512, 512, 32),
        (1, 2, 1, 1024, 1024, 32),
        (1, 2, 1, 2048, 2048, 32),
        (1, 2, 1, 4096, 4096, 32),
        (1, 2, 1, 8192, 8192, 32),
        (1, 2, 1, 16384, 16384, 32),
        (1, 2, 1, 32768, 32768, 32),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_runs = 3  # 运行3次取平均值
    
    print("\n性能测试结果:")
    print(f"{'配置':<45}{'Flash Attention (ms)':<25}{'Dynamic Mask Attention (ms)':<32}{'加速比':<20}")
    print("-" * 120)
    
    for config in configs:
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim = config
        
        config_str = f"b={batch_size}, h={num_heads}, kv={num_kv_heads}, q={query_len}, k={key_len}, d={head_dim}"
        
        # 创建随机输入数据
        query_states = torch.randn(batch_size, num_heads, query_len, head_dim, device=device, dtype=torch.bfloat16)
        key_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device, dtype=torch.bfloat16)
        value_states = torch.randn(batch_size, num_kv_heads, key_len, head_dim, device=device, dtype=torch.bfloat16)
        dt_proj = torch.randn(num_kv_heads, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
        A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
        
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
        scaling = head_dim ** 0.5
        keep_window_size = 2048
        
        # 测量Python实现的时间（进行多次测试）
        py_times = []
        for _ in range(num_runs):
            # 预热运行
            result = flash_attention_cuda(
                query_states, key_states, value_states,
                scaling, causal_mask,
                True
            )
            if result == "OOM":
                py_times = ["OOM"] * num_runs
                break
            torch.cuda.synchronize()

            
            # 计时运行
            start_time = time.time()
            result = flash_attention_cuda(
                query_states, key_states, value_states,
                scaling, causal_mask,
                True
            )
            torch.cuda.synchronize()
            py_times.append((time.time() - start_time) * 1000)  # ms
        
        # 计算Python实现的平均时间
        if py_times and py_times[0] == "OOM":
            py_time = "OOM"
        elif all(t != float('inf') for t in py_times):
            py_time = sum(py_times) / len(py_times)
        else:
            py_time = float('inf')
        
        # 测量CUDA实现的时间（进行多次测试）
        cuda_times = []
        for _ in range(num_runs):
            # 预热运行
            _ = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
            torch.cuda.synchronize()
            
            # 计时运行
            start_time = time.time()
            _ = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
            torch.cuda.synchronize()
            cuda_times.append((time.time() - start_time) * 1000)  # ms
        
        # 计算CUDA实现的平均时间
        cuda_time = sum(cuda_times) / len(cuda_times)
        
        # 计算加速比
        if py_time == "OOM":
            speedup = "N/A (OOM)"
        elif cuda_time > 0 and py_time != float('inf'):
            speedup = f"{py_time / cuda_time:.1f}x"
        else:
            speedup = "N/A"
        
        # 格式化输出
        if py_time == "OOM":
            py_time_str = "OOM"
        elif py_time != float('inf'):
            py_time_str = f"{py_time:.2f}"
        else:
            py_time_str = "超时"
            
        cuda_time_str = f"{cuda_time:.2f}"
        print(f"{config_str:<55}{py_time_str:<25}{cuda_time_str:<25}{speedup:<10}")
    
    print("-" * 120)

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
    
    # if run_perf:
    #     test_performance()