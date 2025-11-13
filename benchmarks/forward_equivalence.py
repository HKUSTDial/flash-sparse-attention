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

from flash_sparse_attn.utils.mask import create_mask

# Import the compiled CUDA extension
try:
    from flash_sparse_attn.flash_sparse_attn_interface import flash_sparse_attn_func
    print("✅ Successfully imported flash_sparse_attn interface")
except ImportError as e:
    print(f"❌ Failed to import flash_sparse_attn interface: {e}")
    print("Please make sure the package is properly installed with: pip install .")
    # Don't exit here, just warn
    flash_sparse_attn_func = None

# Import the Triton implementation
try:
    from flash_sparse_attn.flash_sparse_attn_triton import triton_sparse_attn_func
    print("✅ Successfully imported flash_sparse_attn_triton")
except ImportError as e:
    print(f"❌ Failed to import flash_sparse_attn_triton: {e}")
    print("Please make sure the Triton implementation is available.")
    # Don't exit here, just warn
    triton_sparse_attn_func = None

# Import the Flex Attention implementation
try:
    from flash_sparse_attn.flash_sparse_attn_flex import flex_sparse_attn_func
    print("✅ Successfully imported flash_sparse_attn_flex")
except ImportError as e:
    print(f"❌ Failed to import flash_sparse_attn_flex: {e}")
    print("Please make sure the Flex Attention implementation is available.")
    # Don't exit here, just warn
    flex_sparse_attn_func = None


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
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size: int,
    is_causal: bool,
):
    """
    Python reference implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_len, key_len] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    batch_size, num_heads, query_len, _ = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape

    num_queries_per_kv = num_heads // num_kv_heads

    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=causal_mask if is_causal else None,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        window_size=window_size,
        min_dtype=torch.finfo(query_states.dtype).min,
        type="topk"
    )

    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv) if attn_mask is not None else None

    # Sparse attention weight calculation
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))     # Dot product weights
    attn_weights = attn_weights * scaling + attn_bias                           # Apply scaling and bias
    if attn_mask is not None:
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))      # Apply mask
    attn_weights = F.softmax(attn_weights, dim=-1)                              # Softmax normalization
    attn_outputs = torch.matmul(attn_weights, value_states)                     # Weighted sum of values
    attn_outputs = attn_outputs.transpose(1, 2).contiguous()                    # Transpose to [batch, query_len, num_heads, head_dim]

    return attn_outputs


def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size: int,
    is_causal: bool,
):
    """
    CUDA implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_len, key_len] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flash_sparse_attn_func is None:
        raise RuntimeError("flash_sparse_attn_func not available")

    batch_size, num_heads, query_len, _ = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape

    num_queries_per_kv = num_heads // num_kv_heads

    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=causal_mask if is_causal else None,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        window_size=window_size,
        min_dtype=torch.finfo(query_states.dtype).min,
        type="topk"
    )
    
    # Ensure correct data types and memory layout for CUDA function
    query_states = query_states.transpose(1, 2)     # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2)         # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2)     # [batch, key_len, num_kv_heads, head_dim]

    # Call the flash_sparse_attn_func interface
    attn_outputs = flash_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        is_causal=is_causal,
        softmax_scale=scaling,
        softcap=0.0,
        deterministic=True,
        return_attn_probs=False,
    )
    
    return attn_outputs  # [batch, query_len, num_heads, head_dim]


def dynamic_mask_attention_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size: int,
    is_causal: bool,
):
    """
    Triton implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_len, key_len] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if triton_sparse_attn_func is None:
        raise RuntimeError("Triton implementation not available")
    
    batch_size, num_heads, query_len, _ = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape

    num_queries_per_kv = num_heads // num_kv_heads

    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=causal_mask if is_causal else None,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        window_size=window_size,
        min_dtype=torch.finfo(query_states.dtype).min,
        type="topk"
    )
    
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
    attn_outputs = triton_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        is_causal=is_causal,
        softmax_scale=scaling,
    )
    
    return attn_outputs  # [batch, query_len, num_heads, head_dim]


def dynamic_mask_attention_flex(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size: int,
    is_causal: bool,
):
    """
    Flex Attention implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_len, key_len] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flex_sparse_attn_func is None:
        raise RuntimeError("Flex Attention implementation not available")
    
    batch_size, num_heads, query_len, _ = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape

    num_queries_per_kv = num_heads // num_kv_heads

    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=causal_mask if is_causal else None,
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        window_size=window_size,
        min_dtype=torch.finfo(query_states.dtype).min,
        type="topk"
    )
    
    # Repeat KV for multi-head attention (GQA support)
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
    
    # Ensure correct data types and memory layout for Flex function
    query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]
    attn_mask = attn_mask.contiguous()                              # [batch, num_heads, seqlen_q, seqlen_k]
    attn_bias = attn_bias.contiguous()                              # [batch, num_heads, seqlen_q, seqlen_k]

    # Call the Flex Attention implementation
    attn_outputs = flex_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        is_causal=is_causal,
        softmax_scale=scaling,
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
    if flash_sparse_attn_func is None:
        print("❌ CUDA implementation not available, skipping test.")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test different parameter configurations
    # If you encounter NAN issues when running multiple configurations, try running a single configuration
    # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
    test_configs = [
        # Head dim 32
        (1, 2, 1, 128, 128, 32, False),
        (1, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 256, 256, 32, False),
        (1, 2, 1, 256, 256, 32, True),
        (1, 2, 1, 512, 512, 32, False),
        (1, 2, 1, 512, 512, 32, True),
        (1, 2, 1, 1024, 1024, 32, False),
        (1, 2, 1, 1024, 1024, 32, True),
        (1, 2, 1, 2048, 2048, 32, False),
        (1, 2, 1, 2048, 2048, 32, True),
        (1, 2, 1, 4096, 4096, 32, False),
        (1, 2, 1, 4096, 4096, 32, True),

        # Head dim 64
        (1, 2, 1, 128, 128, 64, False),
        (1, 2, 1, 128, 128, 64, True),
        (1, 2, 1, 256, 256, 64, False),
        (1, 2, 1, 256, 256, 64, True),
        (1, 2, 1, 512, 512, 64, False),
        (1, 2, 1, 512, 512, 64, True),
        (1, 2, 1, 1024, 1024, 64, False),
        (1, 2, 1, 1024, 1024, 64, True),
        (1, 2, 1, 2048, 2048, 64, False),
        (1, 2, 1, 2048, 2048, 64, True),
        (1, 2, 1, 4096, 4096, 64, False),
        (1, 2, 1, 4096, 4096, 64, True),

        # Head dim 96
        (1, 2, 1, 128, 128, 96, False),
        (1, 2, 1, 128, 128, 96, True),
        (1, 2, 1, 256, 256, 96, False),
        (1, 2, 1, 256, 256, 96, True),
        (1, 2, 1, 512, 512, 96, False),
        (1, 2, 1, 512, 512, 96, True),
        (1, 2, 1, 1024, 1024, 96, False),
        (1, 2, 1, 1024, 1024, 96, True),
        (1, 2, 1, 2048, 2048, 96, False),
        (1, 2, 1, 2048, 2048, 96, True),
        (1, 2, 1, 4096, 4096, 96, False),
        (1, 2, 1, 4096, 4096, 96, True),

        # Head dim 128
        (1, 2, 1, 128, 128, 128, False),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, False),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 512, 512, 128, False),
        (1, 2, 1, 512, 512, 128, True),
        (1, 2, 1, 1024, 1024, 128, False),
        (1, 2, 1, 1024, 1024, 128, True),
        (1, 2, 1, 2048, 2048, 128, False),
        (1, 2, 1, 2048, 2048, 128, True),
        (1, 2, 1, 4096, 4096, 128, False),
        (1, 2, 1, 4096, 4096, 128, True),

        # Head dim 192
        (1, 2, 1, 128, 128, 192, False),
        (1, 2, 1, 128, 128, 192, True),
        (1, 2, 1, 256, 256, 192, False),
        (1, 2, 1, 256, 256, 192, True),
        (1, 2, 1, 512, 512, 192, False),
        (1, 2, 1, 512, 512, 192, True),
        (1, 2, 1, 1024, 1024, 192, False),
        (1, 2, 1, 1024, 1024, 192, True),
        (1, 2, 1, 2048, 2048, 192, False),
        (1, 2, 1, 2048, 2048, 192, True),
        (1, 2, 1, 4096, 4096, 192, False),
        (1, 2, 1, 4096, 4096, 192, True),

        # Head dim 256
        (1, 2, 1, 128, 128, 256, False),
        (1, 2, 1, 128, 128, 256, True),
        (1, 2, 1, 256, 256, 256, False),
        (1, 2, 1, 256, 256, 256, True),
        (1, 2, 1, 512, 512, 256, False),
        (1, 2, 1, 512, 512, 256, True),
        (1, 2, 1, 1024, 1024, 256, False),
        (1, 2, 1, 1024, 1024, 256, True),
        (1, 2, 1, 2048, 2048, 256, False),
        (1, 2, 1, 2048, 2048, 256, True),
        (1, 2, 1, 4096, 4096, 256, False),
        (1, 2, 1, 4096, 4096, 256, True),
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
        attn_bias = torch.randn(
            batch_size, num_kv_heads, query_len, key_len,
            device=device, dtype=torch.bfloat16
        )
        cache_position = torch.arange(key_len - query_len, key_len, device=device)
        causal_mask = torch.arange(key_len, device=device) <= cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        window_size = 1024

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            attn_bias, causal_mask, scaling,
            window_size, is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time

        # Run CUDA implementation
        start_time = time.time()
        cuda_output = dynamic_mask_attention_cuda(
            query_states, key_states, value_states,
            attn_bias, causal_mask, scaling,
            window_size, is_causal
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
        del query_states, key_states, value_states, attn_bias, causal_mask, py_output, cuda_output, py_output_copy, cuda_output_copy
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
    
    if triton_sparse_attn_func is None:
        print("❌ Triton implementation not available, skipping Triton tests")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # If you encounter NAN issues when running multiple configurations, try running a single configuration
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 128, 128, 32, False),
        (1, 2, 1, 256, 256, 32, True),
        (1, 2, 1, 256, 256, 32, False),
        (1, 2, 1, 512, 512, 32, True),
        (1, 2, 1, 512, 512, 32, False),
        (1, 2, 1, 1024, 1024, 32, True),
        (1, 2, 1, 1024, 1024, 32, False),
        (1, 2, 1, 2048, 2048, 32, True),
        (1, 2, 1, 2048, 2048, 32, False),
        (1, 2, 1, 4096, 4096, 32, True),
        (1, 2, 1, 4096, 4096, 32, False),

        (1, 2, 1, 128, 128, 64, True),
        (1, 2, 1, 128, 128, 64, False),
        (1, 2, 1, 256, 256, 64, True),
        (1, 2, 1, 256, 256, 64, False),
        (1, 2, 1, 512, 512, 64, True),
        (1, 2, 1, 512, 512, 64, False),
        (1, 2, 1, 1024, 1024, 64, True),
        (1, 2, 1, 1024, 1024, 64, False),
        (1, 2, 1, 2048, 2048, 64, True),
        (1, 2, 1, 2048, 2048, 64, False),
        (1, 2, 1, 4096, 4096, 64, True),
        (1, 2, 1, 4096, 4096, 64, False),

        (1, 2, 1, 128, 128, 96, True),
        (1, 2, 1, 128, 128, 96, False),
        (1, 2, 1, 256, 256, 96, True),
        (1, 2, 1, 256, 256, 96, False),
        (1, 2, 1, 512, 512, 96, True),
        (1, 2, 1, 512, 512, 96, False),
        (1, 2, 1, 1024, 1024, 96, True),
        (1, 2, 1, 1024, 1024, 96, False),
        (1, 2, 1, 2048, 2048, 96, True),
        (1, 2, 1, 2048, 2048, 96, False),
        (1, 2, 1, 4096, 4096, 96, True),
        (1, 2, 1, 4096, 4096, 96, False),

        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 256, 256, 128, False),
        (1, 2, 1, 512, 512, 128, True),
        (1, 2, 1, 512, 512, 128, False),
        (1, 2, 1, 1024, 1024, 128, True),
        (1, 2, 1, 1024, 1024, 128, False),
        (1, 2, 1, 2048, 2048, 128, True),
        (1, 2, 1, 2048, 2048, 128, False),
        (1, 2, 1, 4096, 4096, 128, True),
        (1, 2, 1, 4096, 4096, 128, False),

        # Not support head_dim > 128 in triton yet
        # (1, 2, 1, 128, 128, 256, True),
        # (1, 2, 1, 128, 128, 256, False),
        # (1, 2, 1, 256, 256, 256, True),
        # (1, 2, 1, 256, 256, 256, False),
        # (1, 2, 1, 512, 512, 256, True),
        # (1, 2, 1, 512, 512, 256, False),
        # (1, 2, 1, 1024, 1024, 256, True),
        # (1, 2, 1, 1024, 1024, 256, False),
        # (1, 2, 1, 2048, 2048, 256, True),
        # (1, 2, 1, 2048, 2048, 256, False),
        # (1, 2, 1, 4096, 4096, 256, True),
        # (1, 2, 1, 4096, 4096, 256, False),
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
        attn_bias = torch.randn(
            batch_size, num_kv_heads, query_len, key_len,
            device=device, dtype=torch.bfloat16
        )
        cache_position = torch.arange(key_len - query_len, key_len, device=device)
        causal_mask = torch.arange(key_len, device=device) <= cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        window_size = 1024

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            attn_bias, causal_mask, scaling,
            window_size, is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time
        
        # Run Triton implementation
        start_time = time.time()
        try:
            triton_output = dynamic_mask_attention_triton(
                query_states, key_states, value_states,
                attn_bias, causal_mask, scaling,
                window_size, is_causal
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

        del query_states, key_states, value_states, attn_bias, causal_mask, py_output, py_output_copy
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
    
    if flex_sparse_attn_func is None:
        print("❌ Flex Attention implementation not available, skipping Flex Attention tests")
        return False
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test configurations for Flex Attention
    test_configs = [
        # (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, is_causal)
        (1, 2, 1, 128, 128, 32, True),
        (1, 2, 1, 128, 128, 32, False),
        (1, 2, 1, 256, 256, 32, True),
        (1, 2, 1, 256, 256, 32, False),
        (1, 2, 1, 512, 512, 32, True),
        (1, 2, 1, 512, 512, 32, False),
        (1, 2, 1, 1024, 1024, 32, True),
        (1, 2, 1, 1024, 1024, 32, False),
        (1, 2, 1, 2048, 2048, 32, True),
        (1, 2, 1, 2048, 2048, 32, False),
        (1, 2, 1, 4096, 4096, 32, True),
        (1, 2, 1, 4096, 4096, 32, False),

        (1, 2, 1, 128, 128, 64, True),
        (1, 2, 1, 128, 128, 64, False),
        (1, 2, 1, 256, 256, 64, True),
        (1, 2, 1, 256, 256, 64, False),
        (1, 2, 1, 512, 512, 64, True),
        (1, 2, 1, 512, 512, 64, False),
        (1, 2, 1, 1024, 1024, 64, True),
        (1, 2, 1, 1024, 1024, 64, False),
        (1, 2, 1, 2048, 2048, 64, True),
        (1, 2, 1, 2048, 2048, 64, False),
        (1, 2, 1, 4096, 4096, 64, True),
        (1, 2, 1, 4096, 4096, 64, False),

        (1, 2, 1, 128, 128, 96, True),
        (1, 2, 1, 128, 128, 96, False),
        (1, 2, 1, 256, 256, 96, True),
        (1, 2, 1, 256, 256, 96, False),
        (1, 2, 1, 512, 512, 96, True),
        (1, 2, 1, 512, 512, 96, False),
        (1, 2, 1, 1024, 1024, 96, True),
        (1, 2, 1, 1024, 1024, 96, False),
        (1, 2, 1, 2048, 2048, 96, True),
        (1, 2, 1, 2048, 2048, 96, False),
        (1, 2, 1, 4096, 4096, 96, True),
        (1, 2, 1, 4096, 4096, 96, False),

        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 256, 256, 128, True),
        (1, 2, 1, 256, 256, 128, False),
        (1, 2, 1, 512, 512, 128, True),
        (1, 2, 1, 512, 512, 128, False),
        (1, 2, 1, 1024, 1024, 128, True),
        (1, 2, 1, 1024, 1024, 128, False),
        (1, 2, 1, 2048, 2048, 128, True),
        (1, 2, 1, 2048, 2048, 128, False),
        (1, 2, 1, 4096, 4096, 128, True),
        (1, 2, 1, 4096, 4096, 128, False),

        (1, 2, 1, 128, 128, 128, True),
        (1, 2, 1, 128, 128, 128, False),
        (1, 2, 1, 256, 256, 256, True),
        (1, 2, 1, 256, 256, 256, False),
        (1, 2, 1, 512, 512, 256, True),
        (1, 2, 1, 512, 512, 256, False),
        (1, 2, 1, 1024, 1024, 256, True),
        (1, 2, 1, 1024, 1024, 256, False),
        (1, 2, 1, 2048, 2048, 256, True),
        (1, 2, 1, 2048, 2048, 256, False),
        (1, 2, 1, 4096, 4096, 256, True),
        (1, 2, 1, 4096, 4096, 256, False),
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
        attn_bias = torch.randn(
            batch_size, num_kv_heads, query_len, key_len,
            device=device, dtype=torch.bfloat16
        )
        cache_position = torch.arange(key_len - query_len, key_len, device=device)
        causal_mask = torch.arange(key_len, device=device) <= cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        
        # Set scaling factor and keep window size
        scaling = head_dim ** -0.5
        window_size = 1024

        # Run Python implementation
        start_time = time.time()
        py_output = dynamic_mask_attention_python(
            query_states, key_states, value_states,
            window_size, attn_bias, causal_mask, scaling,
            is_causal
        )
        torch.cuda.synchronize()
        py_time = time.time() - start_time
        
        # Run Flex Attention implementation
        start_time = time.time()
        try:
            flex_output = dynamic_mask_attention_flex(
                query_states, key_states, value_states,
                attn_bias, causal_mask, scaling,
                window_size, is_causal
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

        del query_states, key_states, value_states, attn_bias, causal_mask, py_output, py_output_copy
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
