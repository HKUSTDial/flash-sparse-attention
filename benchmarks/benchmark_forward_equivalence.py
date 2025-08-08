#!/usr/bin/env python3
"""
Forward Equivalence Benchmark for Dynamic Mask Attention

This script validates the numerical consistency between Python prototype 
and CUDA implementation of dynamic mask attention for forward pass only.

Tests include:
- Multiple configurations of batch size, head count, sequence length, and dimensions
- Causal and non-causal mask options  
- Numerical equivalence analysis
- Group Query Attention (GQA) mode testing
"""

import torch
import torch.nn.functional as F
import argparse
import time
import gc
import sys

# Import the compiled CUDA extension
try:
    from flash_dmattn.flash_dmattn_interface import flash_dmattn_func
    print("✅ Successfully imported flash_dmattn interface")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn interface: {e}")
    print("Please make sure the package is properly installed with: pip install .")
    # Don't exit here, just warn
    flash_dmattn_func = None

# Import the Triton implementation
try:
    from flash_dmattn.flash_dmattn_triton import triton_dmattn_func
    print("✅ Successfully imported flash_dmattn_triton")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn_triton: {e}")
    print("Please make sure the Triton implementation is available.")
    # Don't exit here, just warn
    triton_dmattn_func = None

# Import the Flex Attention implementation
try:
    from flash_dmattn.flash_dmattn_flex import flex_dmattn_func
    print("✅ Successfully imported flash_dmattn_flex")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn_flex: {e}")
    print("Please make sure the Flex Attention implementation is available.")
    # Don't exit here, just warn
    flex_dmattn_func = None


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


def calculate_zoh_states(value_states, dt_proj, A):
    """
    Calculate zoh states for dynamic mask attention.
    
    Args:
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        causal_mask: Optional causal mask
    
    Returns:
        zoh_states: [batch_size, num_kv_heads, key_len]
    """
    batch_size, _, key_len, _ = value_states.shape
    
    # Transpose and reshape value_states, then matrix multiply with dt_proj.T
    dt_result = torch.matmul(
        value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), 
        dt_proj.T
    )
    
    # Apply softplus activation and coefficient A
    dt_states = torch.exp(F.softplus(dt_result) * A)
    zoh_states = dt_states.transpose(-1, -2)  # [batch_size, num_kv_heads, key_len]

    return zoh_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    Transform from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def dynamic_mask_attention_python(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
):
    """
    Python reference implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape

    num_queries_per_kv = num_heads // num_kv_heads

    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask function to process dynamic mask
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        causal_mask if is_causal else None
    )
    
    # Sparse attention weight calculation
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
    attn_weights = attn_weights * scaling + attn_bias           # Apply scaling and zoh
    attn_weights = F.softmax(attn_weights, dim=-1)              # Softmax normalization
    attn_outputs = torch.matmul(attn_weights, value_states)
    attn_outputs = attn_outputs.transpose(1, 2).contiguous()    # Transpose to [batch, query_len, num_heads, head_dim]

    return attn_outputs


def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
    return_softmax=False
):
    """
    CUDA implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
        return_softmax: Whether to return softmax weights
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flash_dmattn_func is None:
        raise RuntimeError("flash_dmattn_func not available")

    # Calculate zoh_states
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask to get the processed attention mask  
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        causal_mask if is_causal else None
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    # Ensure correct data types and memory layout for CUDA function
    # CUDA function expects: q, k, v in [batch, seqlen, num_heads, head_dim] format
    query_states = query_states.transpose(1, 2)     # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2)         # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2)     # [batch, key_len, num_kv_heads, head_dim]

    # Call the new flash_dmattn_func interface
    attn_outputs = flash_dmattn_func(
        query_states,               # [batch, query_len, num_heads, head_dim]
        key_states,                 # [batch, key_len, num_kv_heads, head_dim]
        value_states,               # [batch, key_len, num_kv_heads, head_dim]
        attn_mask=attn_mask,        # [batch, num_kv_heads, query_len, key_len]
        attn_bias=attn_bias,        # [batch, num_kv_heads, query_len, key_len]
        is_causal=is_causal,
        scale=scaling,
        softcap=0.0,
        deterministic=True,
        return_attn_probs=return_softmax
    )
    
    return attn_outputs  # [batch, query_len, num_heads, head_dim]


def dynamic_mask_attention_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
):
    """
    Triton implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if triton_dmattn_func is None:
        raise RuntimeError("Triton implementation not available")
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    # Calculate zoh_states
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask to get the processed attention mask  
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        causal_mask if is_causal else None
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    # Repeat KV for multi-head attention (GQA support)
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
    
    # Triton function expects: q, k, v in [batch, seqlen, num_heads, head_dim] format
    query_states = query_states.transpose(1, 2).contiguous()    # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()    # [batch, key_len, num_heads, head_dim]
    attn_mask = attn_mask.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]
    attn_bias = attn_bias.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]
    
    # Call the Triton implementation
    attn_outputs = triton_dmattn_func(
        query_states,               # q: [batch, seqlen_q, num_heads, head_dim]
        key_states,                 # k: [batch, seqlen_k, num_heads, head_dim]
        value_states,               # v: [batch, seqlen_k, num_heads, head_dim]
        attn_mask=attn_mask,        # mask: [batch, num_heads, seqlen_q, seqlen_k]
        attn_bias=attn_bias,        # bias: [batch, num_heads, seqlen_q, seqlen_k]
        is_causal=is_causal,        # causal masking
        scale=scaling               # scaling factor
    )
    
    return attn_outputs  # [batch, query_len, num_heads, head_dim]


def dynamic_mask_attention_flex(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
):
    """
    Flex Attention implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flex_dmattn_func is None:
        raise RuntimeError("Flex Attention implementation not available")
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    # Calculate zoh_states
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask to get the processed attention mask  
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        causal_mask if is_causal else None
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    # Repeat KV for multi-head attention (GQA support)
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
    
    # Flex attention expects: q, k, v in [batch, num_heads, seqlen, head_dim] format
    # But attention_mask and attention_bias in [batch, num_heads, query_len, key_len] format
    
    # Call the Flex Attention implementation
    attn_outputs = flex_dmattn_func(
        query_states.transpose(1, 2),               # q: [batch, query_len, num_heads, head_dim]
        key_states.transpose(1, 2),                 # k: [batch, key_len, num_heads, head_dim]
        value_states.transpose(1, 2),               # v: [batch, key_len, num_heads, head_dim]
        attn_mask=attn_mask,                        # attn_mask: [batch, num_heads, query_len, key_len]
        attn_bias=attn_bias,                        # attn_bias: [batch, num_heads, query_len, key_len]
        is_causal=is_causal,                        # is_causal: whether to apply causal masking
        scale=scaling                               # scaling factor
    )
    
    return attn_outputs  # [batch, query_len, num_heads, head_dim]


def analyze_differences(original_result, cuda_result, accuracy_threshold=0.95):
    """
    Analyze differences between two implementations.
    
    Args:
        original_result: Python implementation result
        cuda_result: CUDA implementation result
        accuracy_threshold: Minimum ratio of elements within tolerance to pass (default: 0.95)
    
    Returns:
        tuple: (is_close, max_diff, mean_diff)
    """
    # Ensure both tensors have same data type
    cuda_result = cuda_result.to(original_result.dtype)
    print(f"📋 Original result: {original_result.shape}, {original_result.dtype}")
    print(f"⚡ CUDA result: {cuda_result.shape}, {cuda_result.dtype}")

    # Add detailed debugging information
    print(f"\n🔍 Debugging info:")
    print(f"  📈 Original result range: [{torch.min(original_result):.6f}, {torch.max(original_result):.6f}]")
    print(f"  ⚡ CUDA result range: [{torch.min(cuda_result):.6f}, {torch.max(cuda_result):.6f}]")
    
    # Check for NaN or Inf values
    original_has_nan = torch.isnan(original_result).any()
    cuda_has_nan = torch.isnan(cuda_result).any()
    original_has_inf = torch.isinf(original_result).any()
    cuda_has_inf = torch.isinf(cuda_result).any()
    
    nan_icon = "⚠️" if original_has_nan or cuda_has_nan else "✅"
    inf_icon = "⚠️" if original_has_inf or cuda_has_inf else "✅"
    print(f"  {nan_icon} Original result contains NaN: {original_has_nan}, Inf: {original_has_inf}")
    print(f"  {nan_icon} CUDA result contains NaN: {cuda_has_nan}, Inf: {cuda_has_inf}")

    # Calculate overall differences
    diff = torch.abs(original_result - cuda_result)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    # Find position of maximum difference
    max_diff_idx = torch.argmax(diff.flatten())
    max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
    orig_val = original_result[max_diff_pos].item()
    cuda_val = cuda_result[max_diff_pos].item()
    
    print(f"\n📊 Result analysis:")
    print(f"  📌 Maximum absolute difference: {max_diff:.8f}")
    print(f"  📍 Mean absolute difference: {mean_diff:.8f}")
    print(f"  📍 Position of maximum difference: {max_diff_pos}")
    print(f"  📋 Original value at position: {orig_val:.8f}")
    print(f"  ⚡ CUDA value at position: {cuda_val:.8f}")
    
    # Calculate relative differences
    relative_diff = diff / (torch.abs(original_result) + 1e-8)
    max_rel_diff = torch.max(relative_diff).item()
    mean_rel_diff = torch.mean(relative_diff).item()
    print(f"  📏 Maximum relative difference: {max_rel_diff:.8f}")
    print(f"  📏 Mean relative difference: {mean_rel_diff:.8f}")
    
    # Adjust tolerance based on data type
    if original_result.dtype == torch.bfloat16:
        # bfloat16 effective precision is about 3-4 decimal places
        rtol, atol = 1e-2, 1e-2
        tolerance_note = "bfloat16 tolerance"
    elif original_result.dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
        tolerance_note = "float16 tolerance"
    else:
        rtol, atol = 1e-3, 1e-3
        tolerance_note = "float32 tolerance"
    
    # Statistics of elements within tolerance
    close_mask = torch.abs(original_result - cuda_result) <= (atol + rtol * torch.abs(cuda_result))
    close_ratio = torch.sum(close_mask).float() / close_mask.numel()
    ratio_icon = "🎯" if close_ratio >= 0.99 else "📊" if close_ratio >= 0.95 else "⚠️"
    print(f"  {ratio_icon} Elements within tolerance ratio: {close_ratio:.4f} ({torch.sum(close_mask)}/{close_mask.numel()})")
    
    # Check if accuracy meets threshold (95% default)
    accuracy_pass = close_ratio >= accuracy_threshold
    accuracy_icon = "✅" if accuracy_pass else "❌"
    print(f"  {accuracy_icon} Accuracy threshold ({accuracy_threshold*100:.1f}%): {'Pass' if accuracy_pass else 'Fail'}")
    
    # Also check strict allclose for reference
    strict_close = torch.allclose(original_result, cuda_result, rtol=rtol, atol=atol)
    strict_icon = "✅" if strict_close else "❌"
    print(f"  {strict_icon} Strict allclose ({tolerance_note}: rtol={rtol}, atol={atol}): {'Yes' if strict_close else 'No'}")
    
    # Use accuracy threshold as the primary criteria
    is_close = accuracy_pass
    
    return is_close, max_diff, mean_diff


def test_cuda_forward_equivalence(accuracy_threshold=0.95):
    """Test forward pass equivalence between Python prototype and CUDA implementation."""
    print("\n" + "🚀" + "=" * 76 + "🚀")
    print("🔬 Testing Forward Pass Equivalence: Python Prototype vs CUDA Implementation 🔬")
    print("🚀" + "=" * 76 + "🚀")
    
    # Check if CUDA implementation is available
    if flash_dmattn_func is None:
        print("❌ CUDA implementation not available, skipping test.")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test different parameter configurations
    # If you encounter NAN issues when running multiple configurations, try running a single configuration
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 1, 1, 64, 64, 32, True),
        (1, 1, 1, 64, 64, 32, False),
        (1, 1, 1, 128, 128, 32, True),
        (1, 1, 1, 128, 128, 32, False),
        (1, 1, 1, 256, 256, 32, True),
        (1, 1, 1, 256, 256, 32, False),
        (1, 1, 1, 512, 512, 32, True),
        (1, 1, 1, 512, 512, 32, False),
        (1, 1, 1, 1024, 1024, 32, True),
        (1, 1, 1, 1024, 1024, 32, False),
        (1, 1, 1, 2048, 2048, 32, True),
        (1, 1, 1, 2048, 2048, 32, False),
        (1, 1, 1, 4096, 4096, 32, True),
        (1, 1, 1, 4096, 4096, 32, False),
        (1, 2, 1, 64, 64, 32, True),
        (2, 1, 1, 128, 128, 32, True),
        (2, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 64, 64, 128, True),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 512, 512, 128, True),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Using device: {device}")
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal = config
        
        # Progress indicator
        progress_filled = "█" * (i + 1)
        progress_empty = "░" * (len(test_configs) - i - 1)
        progress_bar = f"[{progress_filled}{progress_empty}]"
        
        print(f"\n🧪 Test configuration {i+1}/{len(test_configs)} {progress_bar}")
        print(f"  📊 batch_size={batch_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  📏 query_len={query_len}, key_len={key_len}, head_dim={head_dim}")
        print(f"  🔒 is_causal={is_causal}")
        print(f"  🎯 Accuracy threshold: {accuracy_threshold*100:.1f}%")
        
        # Create random input data
        query_states = torch.randn(
            batch_size, num_heads, query_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        key_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        value_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        dt_proj = torch.randn(
            num_kv_heads, num_kv_heads * head_dim, 
            device=device, dtype=torch.bfloat16
        )
        A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
        
        # Create custom causal mask with cache position
        cache_position = torch.arange(key_len - query_len, key_len, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
            (query_len, key_len), fill_value=min_type, 
            device=device, dtype=value_states.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        keep_window_size = 64

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time
        
        # Run CUDA implementation
        start_time = time.time()
        cuda_output = dynamic_mask_attention_cuda(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        
        
        # Analyze differences
        py_output_copy = py_output.clone()
        cuda_output_copy = cuda_output.clone()
        is_close, max_diff, mean_diff = analyze_differences(py_output_copy, cuda_output_copy, accuracy_threshold)
        
        # Report performance difference
        speedup = py_time / cuda_time if cuda_time > 0 else float('inf')
        print(f"\n⚡ Performance comparison:")
        print(f"    🐍 Python implementation: {py_time*1000:.2f} ms")
        print(f"    🚀 CUDA implementation:   {cuda_time*1000:.2f} ms")
        print(f"    📈 Speedup:               {speedup:.2f}x")
        
        # Update test results
        test_result = "Passed" if is_close else "Failed"
        result_icon = "✅" if is_close else "❌"
        all_passed = all_passed and is_close
        print(f"\n{result_icon} Test result: {test_result}")
        
        # If test fails with large difference, can exit early
        if not is_close and max_diff > 1e-2:
            print("  ⚠️ Difference too large, stopping subsequent tests.")
            break
        del query_states, key_states, value_states, dt_proj, A, causal_mask, py_output, cuda_output, py_output_copy, cuda_output_copy
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    print("\n" + "🏁" + "=" * 76 + "🏁")
    summary_icon = "🎉" if all_passed else "😞"
    print(f"{summary_icon} Forward Equivalence Test Summary: {'All Passed' if all_passed else 'Some Tests Failed'}")
    print("🏁" + "=" * 76 + "🏁")
    
    return all_passed


def test_triton_forward_equivalence(accuracy_threshold=0.95):
    """Test forward pass equivalence between Python and Triton implementations."""
    print("\n" + "🔥" + "=" * 76 + "🔥")
    print("🔬 Testing Forward Pass Equivalence: Python vs Triton 🔬")
    print("🔥" + "=" * 76 + "🔥")
    
    if triton_dmattn_func is None:
        print("❌ Triton implementation not available, skipping Triton tests")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # If you encounter NAN issues when running multiple configurations, try running a single configuration
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 1, 1, 64, 64, 32, True),
        (1, 1, 1, 64, 64, 32, False),
        (1, 1, 1, 128, 128, 32, True),
        (1, 1, 1, 128, 128, 32, False),
        (1, 1, 1, 256, 256, 32, True),
        (1, 1, 1, 256, 256, 32, False),
        (1, 1, 1, 512, 512, 32, True),
        (1, 1, 1, 512, 512, 32, False),
        (1, 1, 1, 1024, 1024, 32, True),
        (1, 1, 1, 1024, 1024, 32, False),
        (1, 1, 1, 2048, 2048, 32, True),
        (1, 1, 1, 2048, 2048, 32, False),
        (1, 1, 1, 4096, 4096, 32, True),
        (1, 1, 1, 4096, 4096, 32, False),
        (1, 2, 1, 64, 64, 32, True),
        (2, 1, 1, 128, 128, 32, True),
        (2, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 64, 64, 128, True),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 512, 512, 128, True),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Using device: {device}")
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal = config
        
        # Progress indicator
        progress_filled = "█" * (i + 1)
        progress_empty = "░" * (len(test_configs) - i - 1)
        progress_bar = f"[{progress_filled}{progress_empty}]"
        
        print(f"\n🧪 Test configuration {i+1}/{len(test_configs)} {progress_bar}")
        print(f"  📊 batch_size={batch_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  📏 query_len={query_len}, key_len={key_len}, head_dim={head_dim}")
        print(f"  🔒 is_causal={is_causal}")
        print(f"  🎯 Accuracy threshold: {accuracy_threshold*100:.1f}%")
        
        # Create random input data
        query_states = torch.randn(
            batch_size, num_heads, query_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        key_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        value_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        dt_proj = torch.randn(
            num_kv_heads, num_kv_heads * head_dim, 
            device=device, dtype=torch.bfloat16
        )
        A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
        
        # Create custom causal mask with cache position
        cache_position = torch.arange(0, query_len + 0, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
            (query_len, key_len), fill_value=min_type, 
            device=device, dtype=value_states.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        keep_window_size = 64

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time
        
        # Run Triton implementation
        start_time = time.time()
        try:
            triton_output = dynamic_mask_attention_triton(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, is_causal
            )
            torch.cuda.synchronize()
            triton_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Triton implementation failed: {e}")
            triton_output = None
            triton_time = float('inf')
        
        # Analyze differences
        py_output_copy = py_output.clone()
        
        if triton_output is not None:
            triton_output_copy = triton_output.clone()
            
            print("\n📊 Python vs Triton comparison:")
            triton_vs_py_close, triton_max_diff, triton_mean_diff = analyze_differences(py_output_copy, triton_output_copy, accuracy_threshold)
        else:
            triton_vs_py_close = False
        
        # Report performance differences
        print(f"\n⚡ Performance comparison:")
        print(f"    🐍 Python implementation: {py_time*1000:.2f} ms")
        if triton_output is not None:
            print(f"    🔥 Triton implementation: {triton_time*1000:.2f} ms")
            
            triton_speedup = py_time / triton_time if triton_time > 0 else float('inf')
            print(f"    📈 Triton speedup vs Python: {triton_speedup:.2f}x")
        
        # Update test results
        test_passed = triton_vs_py_close if triton_output is not None else False
        test_result = "Passed" if test_passed else "Failed"
        result_icon = "✅" if test_passed else "❌"
        all_passed = all_passed and test_passed
        print(f"\n{result_icon} Overall test result: {test_result}")
        
        # If test fails with large difference, can exit early
        if not test_passed:
            if triton_output is not None:
                if triton_max_diff > 1e-2:
                    print("  ⚠️ Difference too large, stopping subsequent tests.")
                    break
        
        del query_states, key_states, value_states, dt_proj, A, causal_mask, py_output, py_output_copy
        if triton_output is not None:
            del triton_output, triton_output_copy
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    print("\n" + "🏁" + "=" * 76 + "🏁")
    summary_icon = "🎉" if all_passed else "😞"
    print(f"{summary_icon} Python vs Triton Test Summary: {'All Passed' if all_passed else 'Some Tests Failed'}")
    print("🏁" + "=" * 76 + "🏁")
    
    return all_passed


def test_flex_forward_equivalence(accuracy_threshold=0.95):
    """Test forward pass equivalence between Python and Flex Attention implementations."""
    print("\n" + "🌟" + "=" * 76 + "🌟")
    print("🔬 Testing Forward Pass Equivalence: Python vs Flex Attention 🔬")
    print("🌟" + "=" * 76 + "🌟")
    
    if flex_dmattn_func is None:
        print("❌ Flex Attention implementation not available, skipping Flex Attention tests")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test configurations for Flex Attention
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 1, 1, 64, 64, 32, True),
        (1, 1, 1, 64, 64, 32, False),
        (1, 1, 1, 128, 128, 32, True),
        (1, 1, 1, 128, 128, 32, False),
        (1, 1, 1, 256, 256, 32, True),
        (1, 1, 1, 256, 256, 32, False),
        (1, 1, 1, 512, 512, 32, True),
        (1, 1, 1, 512, 512, 32, False),
        (1, 1, 1, 1024, 1024, 32, True),
        (1, 1, 1, 1024, 1024, 32, False),
        (1, 1, 1, 2048, 2048, 32, True),
        (1, 1, 1, 2048, 2048, 32, False),
        (1, 1, 1, 4096, 4096, 32, True),
        (1, 1, 1, 4096, 4096, 32, False),
        (1, 2, 1, 64, 64, 32, True),
        (2, 1, 1, 128, 128, 32, True),
        (2, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 64, 64, 128, True),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 512, 512, 128, True),
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Using device: {device}")
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal = config
        
        # Progress indicator
        progress_filled = "█" * (i + 1)
        progress_empty = "░" * (len(test_configs) - i - 1)
        progress_bar = f"[{progress_filled}{progress_empty}]"
        
        print(f"\n🧪 Test configuration {i+1}/{len(test_configs)} {progress_bar}")
        print(f"  📊 batch_size={batch_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  📏 query_len={query_len}, key_len={key_len}, head_dim={head_dim}")
        print(f"  🔒 is_causal={is_causal}")
        print(f"  🎯 Accuracy threshold: {accuracy_threshold*100:.1f}%")
        
        # Create random input data
        query_states = torch.randn(
            batch_size, num_heads, query_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        key_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        value_states = torch.randn(
            batch_size, num_kv_heads, key_len, head_dim, 
            device=device, dtype=torch.bfloat16
        )
        dt_proj = torch.randn(
            num_kv_heads, num_kv_heads * head_dim, 
            device=device, dtype=torch.bfloat16
        )
        A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
        
        # Create custom causal mask with cache position
        cache_position = torch.arange(0, query_len + 0, device=device)
        min_type = torch.finfo(value_states.dtype).min
        causal_mask = torch.full(
            (query_len, key_len), fill_value=min_type, 
            device=device, dtype=value_states.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        keep_window_size = 64

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time
        
        # Run Flex Attention implementation
        start_time = time.time()
        try:
            flex_output = dynamic_mask_attention_flex(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, is_causal
            )
            torch.cuda.synchronize()
            flex_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Flex Attention implementation failed: {e}")
            flex_output = None
            flex_time = float('inf')
        
        # Analyze differences
        py_output_copy = py_output.clone()
        
        if flex_output is not None:
            flex_output_copy = flex_output.clone()
            
            print("\n📊 Python vs Flex Attention comparison:")
            flex_vs_py_close, flex_max_diff, flex_mean_diff = analyze_differences(py_output_copy, flex_output_copy, accuracy_threshold)
        else:
            flex_vs_py_close = False
        
        # Report performance differences
        print(f"\n⚡ Performance comparison:")
        print(f"    🐍 Python implementation: {py_time*1000:.2f} ms")
        if flex_output is not None:
            print(f"    🌟 Flex Attention implementation: {flex_time*1000:.2f} ms")
            
            flex_speedup = py_time / flex_time if flex_time > 0 else float('inf')
            print(f"    📈 Flex Attention speedup vs Python: {flex_speedup:.2f}x")
        
        # Update test results
        test_passed = flex_vs_py_close if flex_output is not None else False
        test_result = "Passed" if test_passed else "Failed"
        result_icon = "✅" if test_passed else "❌"
        all_passed = all_passed and test_passed
        print(f"\n{result_icon} Overall test result: {test_result}")
        
        # If test fails with large difference, can exit early
        if not test_passed:
            if flex_output is not None:
                if flex_max_diff > 1e-2:
                    print("  ⚠️ Difference too large, stopping subsequent tests.")
                    break
        
        del query_states, key_states, value_states, dt_proj, A, causal_mask, py_output, py_output_copy
        if flex_output is not None:
            del flex_output, flex_output_copy
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    print("\n" + "🏁" + "=" * 76 + "🏁")
    summary_icon = "🎉" if all_passed else "😞"
    print(f"{summary_icon} Python vs Flex Attention Test Summary: {'All Passed' if all_passed else 'Some Tests Failed'}")
    print("🏁" + "=" * 76 + "🏁")
    
    return all_passed


def main():
    """
    Test forward pass equivalence between Python prototype and CUDA implementation
    of dynamic mask attention.
    
    This script validates numerical consistency including:
    - Standard forward pass (fwd)
    - Different batch sizes, head counts, sequence lengths and dimensions
    - Causal and non-causal mask options
    - Numerical equivalence analysis
    - Performance comparison
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test forward equivalence between Python/CUDA dynamic mask attention'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--accuracy-threshold', type=float, default=0.95, 
                        help='Minimum accuracy ratio to pass test (default: 0.95)')
    parser.add_argument('--test-type', type=str, default='all', 
                        choices=['all', 'cuda', 'triton', 'flex'],
                        help='Type of test to run (default: all)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Print test environment information
    print("🧬" + "=" * 78 + "🧬")
    print("🔬 Dynamic Mask Attention Forward Pass Equivalence Test Suite 🔬")
    print("🧬" + "=" * 78 + "🧬")
    print(f"🐍 PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA device: {torch.cuda.get_device_name()}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Test type: {args.test_type}")
    print(f"🎯 Accuracy threshold: {args.accuracy_threshold*100:.1f}%")
    
    # Track overall test results
    test_results = {}
    
    # Run tests based on user selection
    if args.test_type in ['all', 'cuda']:
        print("\n" + "📍" + " Starting Standard Forward Pass Tests " + "📍")
        test_results['cuda'] = test_cuda_forward_equivalence(args.accuracy_threshold)
    
    if args.test_type in ['all', 'triton']:
        print("\n" + "🔥" + " Starting Python vs Triton Tests " + "🔥")
        test_results['triton'] = test_triton_forward_equivalence(args.accuracy_threshold)

    if args.test_type in ['all', 'flex']:
        print("\n" + "🌟" + " Starting Python vs Flex Attention Tests " + "🌟")
        test_results['flex'] = test_flex_forward_equivalence(args.accuracy_threshold)


    # Print overall summary
    print("\n" + "🏆" + "=" * 78 + "🏆")
    print("🔬 FINAL TEST SUMMARY 🔬")
    print("🏆" + "=" * 78 + "🏆")
    
    all_passed = True
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        status_text = "PASSED" if result else "FAILED"
        print(f"  {status_icon} {test_name.upper():12} : {status_text}")
        all_passed = all_passed and result
    
    # Overall result
    overall_icon = "🎉" if all_passed else "😞"
    overall_text = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\n{overall_icon} OVERALL RESULT: {overall_text}")
    print("🏆" + "=" * 78 + "🏆")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main() 