import torch
import torch.nn.functional as F
import numpy as np


def prepare_dynamic_mask(
    hidden_states: torch.Tensor,
    zoh_states: torch.Tensor,
    keep_window_size: int = 2048,
    attention_mask: torch.Tensor | None = None,
):
    """
    Calculate dynamic attention mask to mask tokens for sparse attention.

    Combine `zoh_states` with `attention_mask` to generate the final `attn_mask`.

    Args:
        hidden_states: Input hidden states to determine dtype minimum value
        zoh_states: zoh_states of shape (batch_size, num_kv_heads, key_sequence_length)
        keep_window_size: Window size of tokens not dynamically masked
        attention_mask: Optional attention mask of shape (batch_size, 1, query_len, key_len)
    
    Returns:
        tuple: (attn_bias, attn_mask)
    """
    min_dtype = torch.finfo(hidden_states.dtype).min
    dtype = hidden_states.dtype
    attn_bias = zoh_states[:, :, None, :].expand(
        -1, -1, hidden_states.shape[2], -1
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask, 
                torch.tensor(0.0, device=attention_mask.device, dtype=dtype), 
                min_dtype
            )
        attn_bias = attn_bias.masked_fill(
            attention_mask[:, :, :, : attn_bias.shape[-1]] != 0, min_dtype
        )
    
    if attn_bias.shape[-1] > keep_window_size:
        topk_indices = torch.topk(
            attn_bias, keep_window_size, dim=-1, largest=True, sorted=False
        ).indices
        attn_mask = torch.zeros_like(attn_bias, dtype=dtype, device=attn_bias.device)
        attn_mask = attn_mask.scatter(-1, topk_indices, 1.0)
        attn_bias = attn_bias.masked_fill(attn_mask == 0.0, min_dtype)
    else:
        attn_mask = torch.ones_like(attn_bias, dtype=dtype, device=attn_bias.device)
    return attn_bias, attn_mask


def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
):  
    batch_size, num_heads, query_len, head_dim = query_states.shape
    key_len = key_states.shape[2]

    # Initialize attention weights and attention outputs
    attn_weights = torch.zeros((batch_size, num_heads, query_len, key_len), device=query_states.device, dtype=query_states.dtype)
    attn_outputs = torch.zeros((batch_size, num_heads, query_len, head_dim), device=query_states.device, dtype=query_states.dtype)

    dt_states = torch.matmul(value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), dt_proj.T)
    dt_states = torch.exp(A * F.softplus(dt_states)).transpose(-1, -2)
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states, dt_states, keep_window_size=keep_window_size, attention_mask=causal_mask
    )  # [batch_size, num_kv_heads, query_len, key_len]

    for b_idx in range(batch_size):
        for h_idx in range(num_heads):
            for q_idx in range(query_len):

                # Sparse attention weight calculation
                non_mask_indices = []
                for k in range(key_len):
                    if attn_mask[b_idx, h_idx, q_idx, k] != torch.finfo(value_states.dtype).min:
                        non_mask_indices.append(k)
                if len(non_mask_indices) == 0:
                    continue

                q_vec = query_states[b_idx, h_idx, q_idx, :]                # [head_dim]
                k_vecs = key_states[b_idx, h_idx, non_mask_indices, :]      # [keep_window_size, head_dim]
                v_vecs = value_states[b_idx, h_idx, non_mask_indices, :]    # [keep_window_size, head_dim]

                # QK dot product
                attn_weight = torch.sum(q_vec.unsqueeze(0) * k_vecs, dim=-1)

                # Apply scaling and dynamic_mask
                attn_weight = attn_weight * scaling + attn_bias[b_idx, h_idx, q_idx, non_mask_indices]

                # Softmax
                attn_weight = F.softmax(attn_weight, dim=-1)
                
                # Use non-inplace operation instead
                attn_weights = attn_weights.clone()
                attn_weights[b_idx, h_idx, q_idx, non_mask_indices] = attn_weight

                attn_output = torch.sum(attn_weight.unsqueeze(1) * v_vecs, dim=0)
                attn_outputs = attn_outputs.clone()
                attn_outputs[b_idx, h_idx, q_idx, :] = attn_output

    attn_outputs = attn_outputs.transpose(1, 2).contiguous()
    
    return attn_outputs


def dynamic_mask_attention_python(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
):  
    batch_size, num_heads, query_len, head_dim = query_states.shape
    key_len = key_states.shape[2]

    dt_states = torch.matmul(value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), dt_proj.T)
    dt_states = torch.exp(A * F.softplus(dt_states)).transpose(-1, -2)
    attn_bias, _ = prepare_dynamic_mask(
        query_states, dt_states, keep_window_size=keep_window_size, attention_mask=causal_mask
    )  # [batch_size, num_kv_heads, query_len, key_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))  # [batch_size, num_heads, query_len, key_len]
    attn_weights = attn_weights * scaling + attn_bias  # Apply scaling and dynamic_mask

    attn_weights = F.softmax(attn_weights, dim=-1)  # Softmax normalization
    attn_outputs = torch.matmul(attn_weights, value_states)  # [batch_size, num_heads, query_len, head_dim]
    attn_outputs = attn_outputs.transpose(1, 2).contiguous()
    return attn_outputs


def test_equivalence():
    """Test equivalence of outputs and gradients between CUDA and Python implementations"""
    print("🔬" + "=" * 76 + "🔬")
    print("🧠 Testing Equivalence of Dynamic Mask Attention Functions (Gradients) 🧠")
    print("🔬" + "=" * 76 + "🔬")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    batch_size, num_heads, query_len, key_len, head_dim = 1, 1, 64, 64, 64
    keep_window_size = 32  # Set a smaller keep_window_size for testing
    
    # Create tensors with requires_grad=True for gradient testing
    query_states = torch.randn(batch_size, num_heads, query_len, head_dim, requires_grad=True)
    key_states = torch.randn(batch_size, num_heads, key_len, head_dim, requires_grad=True)
    value_states = torch.randn(batch_size, num_heads, key_len, head_dim, requires_grad=True)
    dt_proj = torch.randn(num_heads, num_heads * head_dim, requires_grad=True)
    A = torch.randn(num_heads, requires_grad=True)

    # Create causal mask
    cache_position = torch.arange(key_len - query_len, query_len + key_len - query_len)
    min_type = torch.finfo(value_states.dtype).min
    causal_mask = torch.full(
                (query_len, key_len), fill_value=min_type, dtype=value_states.dtype
            )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(key_len) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    scaling = head_dim ** -0.5
    
    # Test CUDA function
    print("\n⚡ Testing CUDA implementation...")
    query_cuda = query_states.clone().detach().requires_grad_(True)
    key_cuda = key_states.clone().detach().requires_grad_(True)
    value_cuda = value_states.clone().detach().requires_grad_(True)
    dt_proj_cuda = dt_proj.clone().detach().requires_grad_(True)
    A_cuda = A.clone().detach().requires_grad_(True)
    
    cuda_outputs = dynamic_mask_attention_cuda(
        query_cuda,
        key_cuda,
        value_cuda,
        dt_proj_cuda,
        A_cuda,
        scaling=scaling,
        causal_mask=causal_mask,
        keep_window_size=keep_window_size,
    )
    
    # Test Python function
    print("🐍 Testing Python implementation...")
    query_python = query_states.clone().detach().requires_grad_(True)
    key_python = key_states.clone().detach().requires_grad_(True)
    value_python = value_states.clone().detach().requires_grad_(True)
    dt_proj_python = dt_proj.clone().detach().requires_grad_(True)
    A_python = A.clone().detach().requires_grad_(True)
    
    python_outputs = dynamic_mask_attention_python(
        query_python,
        key_python,
        value_python,
        dt_proj_python,
        A_python,
        scaling=scaling,
        causal_mask=causal_mask,
        keep_window_size=keep_window_size,
    )
    
    # Compare outputs
    print("\n📊 Comparing outputs...")
    output_diff = torch.abs(cuda_outputs - python_outputs)
    max_output_diff = torch.max(output_diff)
    mean_output_diff = torch.mean(output_diff)
    
    print(f"   📌 Max output difference: {max_output_diff:.2e}")
    print(f"   📍 Mean output difference: {mean_output_diff:.2e}")
    print(f"   📋 Output shapes - CUDA: {cuda_outputs.shape}, Python: {python_outputs.shape}")
    
    outputs_equal = torch.allclose(cuda_outputs, python_outputs, atol=1e-5, rtol=1e-4)
    output_icon = "✅" if outputs_equal else "❌"
    print(f"   {output_icon} Outputs are equal (atol=1e-5, rtol=1e-4): {outputs_equal}")
    
    # Test gradients
    print("\n🧮 Testing gradients...")
    
    # Create a simple loss function (sum of outputs)
    cuda_loss = cuda_outputs.sum()
    python_loss = python_outputs.sum()
    
    # Compute gradients
    cuda_loss.backward()
    python_loss.backward()
    
    # Compare gradients for each parameter
    grad_comparisons = [
        ("query_states", query_cuda.grad, query_python.grad),
        ("key_states", key_cuda.grad, key_python.grad),
        ("value_states", value_cuda.grad, value_python.grad),
        ("dt_proj", dt_proj_cuda.grad, dt_proj_python.grad),
        ("A", A_cuda.grad, A_python.grad),
    ]
    
    all_grads_equal = True
    for param_name, cuda_grad, python_grad in grad_comparisons:
        if cuda_grad is None or python_grad is None:
            print(f"   ⚠️ {param_name}: One or both gradients are None")
            if cuda_grad != python_grad:
                all_grads_equal = False
            continue
            
        grad_diff = torch.abs(cuda_grad - python_grad)
        max_grad_diff = torch.max(grad_diff)
        mean_grad_diff = torch.mean(grad_diff)
        
        grads_equal = torch.allclose(cuda_grad, python_grad, atol=1e-5, rtol=1e-4)
        all_grads_equal = all_grads_equal and grads_equal
        grad_icon = "✅" if grads_equal else "❌"
        
        print(f"   📊 {param_name}:")
        print(f"     📌 Max gradient difference: {max_grad_diff:.2e}")
        print(f"     📍 Mean gradient difference: {mean_grad_diff:.2e}")
        print(f"     {grad_icon} Gradients equal (atol=1e-5, rtol=1e-4): {grads_equal}")
    
    # Summary
    print("\n" + "🏁" + "=" * 76 + "🏁")
    print("📋 SUMMARY 📋")
    print("🏁" + "=" * 76 + "🏁")
    output_summary_icon = "✅" if outputs_equal else "❌"
    grad_summary_icon = "✅" if all_grads_equal else "❌"
    print(f"{output_summary_icon} Outputs equivalent: {outputs_equal}")
    print(f"{grad_summary_icon} Gradients equivalent: {all_grads_equal}")
    
    if outputs_equal and all_grads_equal:
        print("🎉 Both functions are mathematically equivalent!")
    else:
        print("😞 Functions are NOT equivalent. Check implementation differences.")
        
    return outputs_equal, all_grads_equal


if __name__ == "__main__":
    test_equivalence()