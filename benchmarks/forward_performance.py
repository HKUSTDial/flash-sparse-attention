#!/usr/bin/env python3
"""
Performance Benchmark for Dynamic Mask Attention

This script measures and compares the performance of multiple Dynamic Mask Attention 
implementations against SDPA baseline across various configurations.

Implementations tested:
- PyTorch SDPA - Baseline
- Dynamic Mask Attention CUDA - Custom CUDA kernel implementation  
- Dynamic Mask Attention Triton - Triton kernel implementation
- Dynamic Mask Attention Flex - Flex Attention implementation

Benchmark includes:
- Multiple sequence lengths and batch sizes
- Head count and dimension variations
- Throughput and latency measurements
- Memory usage analysis
- Speedup comparisons across all implementations
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import gc

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


def prepare_mask(
    hidden_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor = None,
    window_size: int = None,
):
    """
    Args:
        hidden_states: Input hidden states to determine dtype minimum value
        attn_bias: Attention bias of shape (batch_size, num_heads, query_length, key_length)
        causal_mask: Optional causal mask to apply
        window_size: Window size of tokens not masked
    
    Returns:
        tuple: (attn_bias, attn_mask)
    """
    dtype = hidden_states.dtype
    min_dtype = torch.finfo(dtype).min

    if attn_bias.shape[-1] > window_size:
        if causal_mask is not None:
            topk_values, topk_indices = torch.topk(
                attn_bias.masked_fill(~causal_mask, min_dtype).detach(),
                window_size, dim=-1, largest=True, sorted=False
            )
        else:
            topk_values, topk_indices = torch.topk(
                attn_bias,
                window_size, dim=-1, largest=True, sorted=False
            )
        attn_mask = torch.zeros_like(attn_bias, dtype=torch.bool, device=attn_bias.device).scatter_(-1, topk_indices, topk_values != min_dtype)
    else:
        attn_mask = causal_mask.expand_as(attn_bias) if causal_mask is not None else torch.ones_like(attn_bias, dtype=torch.bool, device=attn_bias.device)
    return attn_bias, attn_mask


def scaled_dot_product_attention_cuda(
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
    CUDA implementation of SDPA baseline.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_heads, query_length, key_length]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        window_size: Number of tokens to keep in attention window
        scaling: Attention scaling factor
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads
 
    attn_bias, attn_mask = prepare_mask(
        query_states,
        attn_bias,
        causal_mask if is_causal else None,
        window_size,
    )

    # Repeat KV for multi-head attention (GQA support)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    attn_bias = attn_bias.masked_fill(~attn_mask, torch.finfo(query_states.dtype).min).contiguous()

    try:
        torch.cuda.synchronize()
        start_time = time.time()

        attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_bias,
            scale=scaling,
            # is_causal=is_causal,
            enable_gqa=True,
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        return attn_outputs, (end_time - start_time) * 1000
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size=2048,
    is_causal=True,
):
    """
    CUDA implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_heads, query_length, key_length]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        window_size: Number of tokens to keep in attention window
        scaling: Attention scaling factor
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if flash_sparse_attn_func is None:
        return "Not Available", 0

    attn_bias, attn_mask = prepare_mask(
        query_states,
        attn_bias,
        causal_mask if is_causal else None,
        window_size,
    )
    
    # Ensure correct data types and memory layout for CUDA function
    query_states = query_states.transpose(1, 2)     # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2)         # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2)     # [batch, key_len, num_kv_heads, head_dim]

    try:
        torch.cuda.synchronize()
        start_time = time.time()

        attn_outputs = flash_sparse_attn_func(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=is_causal,
            softmax_scale=scaling,
            softcap=0.0,
            deterministic=False,
            return_attn_probs=False
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size=2048,
    is_causal=True,
):
    """
    Triton implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_heads, query_length, key_length]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        window_size: Number of tokens to keep in attention window
        scaling: Attention scaling factor
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if triton_sparse_attn_func is None:
        return "Not Available", 0
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    attn_bias, attn_mask = prepare_mask(
        query_states,
        attn_bias,
        causal_mask if is_causal else None,
        window_size,
    )

    # Repeat KV for multi-head attention (GQA support)
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)

    # Ensure correct data types and memory layout for Triton function
    query_states = query_states.transpose(1, 2).contiguous()    # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()    # [batch, key_len, num_heads, head_dim]
    attn_mask = attn_mask.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]
    attn_bias = attn_bias.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]

    try:
        torch.cuda.synchronize()
        start_time = time.time()
        
        attn_outputs = triton_sparse_attn_func(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=is_causal,
            softmax_scale=scaling,
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_flex(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_bias: torch.Tensor,
    causal_mask: torch.Tensor,
    scaling: float,
    window_size=2048,
    is_causal=True,
):
    """
    Flex Attention implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [batch_size, num_heads, query_length, key_length]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        window_size: Number of tokens to keep in attention window
        scaling: Attention scaling factor
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if flex_sparse_attn_func is None:
        return "Not Available", 0
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    attn_bias, attn_mask = prepare_mask(
        query_states,
        attn_bias,
        causal_mask if is_causal else None,
        window_size,
    )

    # Repeat KV for multi-head attention (GQA support)
    key_states = repeat_kv(key_states, num_queries_per_kv)
    value_states = repeat_kv(value_states, num_queries_per_kv)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)

    # Ensure correct data types and memory layout for Flex function
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()
    attn_mask = attn_mask.contiguous()
    attn_bias = attn_bias.contiguous()

    try:
        torch.cuda.synchronize()
        start_time = time.time()
        
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
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms  
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def measure_memory_usage():
    """
    Measure current GPU memory usage.
    
    Returns:
        tuple: (allocated_mb, reserved_mb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        return allocated, reserved
    return 0, 0


def benchmark_attention_performance(config, test_type='all', num_runs=5, warmup_runs=2):
    """
    Benchmark attention performance for a given configuration.
    
    Args:
        config: Tuple of (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Performance metrics
    """
    batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    # Set scaling factor from config
    scaling = head_dim ** -0.5
    
    results = {
        'config': config,
        'sdpa_forward_times': [],
        'fdma_cuda_forward_times': [],
        'fdma_triton_forward_times': [],
        'fdma_flex_forward_times': [],
        'sdpa_forward_memory': 0,
        'fdma_cuda_forward_memory': 0,
        'fdma_triton_forward_memory': 0,
        'fdma_flex_forward_memory': 0,
        'sdpa_forward_status': 'success',
        'fdma_cuda_forward_status': 'success',
        'fdma_triton_forward_status': 'success',
        'fdma_flex_forward_status': 'success'
    }
    
    # Determine which implementations to run
    run_flash = test_type in ['all', 'sdpa', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex']
    run_cuda = test_type in ['all', 'cuda', 'sdpa-vs-cuda']
    run_triton = test_type in ['all', 'triton', 'sdpa-vs-triton']
    run_flex = test_type in ['all', 'flex', 'sdpa-vs-flex']
    
    # Benchmark SDPA
    if run_flash:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = scaled_dot_product_attention_cuda(
                query_states, key_states, value_states,
                attn_bias, causal_mask,
                scaling, window_size, is_causal
            )
            if result[0] == "OOM":
                results['sdpa_forward_status'] = 'OOM'
                break
            torch.cuda.synchronize()

        if results['sdpa_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = scaled_dot_product_attention_cuda(
                    query_states, key_states, value_states,
                    attn_bias, causal_mask,
                    scaling, window_size, is_causal
                )
                
                if result[0] == "OOM":
                    results['sdpa_forward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                results['sdpa_forward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['sdpa_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['sdpa_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention
    if run_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                attn_bias, causal_mask,
                scaling, window_size, is_causal
            )
            if result[0] == "OOM":
                results['fdma_cuda_forward_status'] = 'OOM'
                break
            torch.cuda.synchronize()
        
        if results['fdma_cuda_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_cuda(
                    query_states, key_states, value_states,
                    attn_bias, causal_mask,
                    scaling, window_size, is_causal
                )
                
                if result[0] == "OOM":
                    results['fdma_cuda_forward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_cuda_forward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_cuda_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_cuda_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention (Triton)
    if run_triton:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_triton(
                query_states, key_states, value_states,
                attn_bias, causal_mask,
                scaling, window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_triton_forward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_triton_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_triton(
                    query_states, key_states, value_states,
                    attn_bias, causal_mask,
                    scaling, window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_triton_forward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_triton_forward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_triton_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_triton_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention (Flex)
    if run_flex:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_flex(
                query_states, key_states, value_states,
                attn_bias, causal_mask,
                scaling, window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_flex_forward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_flex_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_flex(
                    query_states, key_states, value_states,
                    attn_bias, causal_mask, scaling, window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_flex_forward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_flex_forward_times'].append(result[1])  # ms

            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_flex_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_flex_forward_status'] = 'N/A'

    return results


def run_performance_benchmark(test_type='all', num_runs=3, warmup_runs=2):
    """Run comprehensive performance benchmark across different configurations."""
    print("\n" + "🏆" + "=" * 76 + "🏆")
    
    # Update title based on test type
    if test_type == 'all':
        title = "⚡ Performance Benchmark: SDPA vs CUDA vs Triton vs Flex ⚡"
    elif test_type == 'sdpa-vs-cuda':
        title = "⚡ Performance Benchmark: SDPA Attention vs CUDA ⚡"
    elif test_type == 'sdpa-vs-triton':
        title = "⚡ Performance Benchmark: SDPA Attention vs Triton ⚡"
    elif test_type == 'sdpa-vs-flex':
        title = "⚡ Performance Benchmark: SDPA Attention vs Flex ⚡"
    elif test_type == 'sdpa':
        title = "⚡ Performance Benchmark: SDPA Attention Only ⚡"
    elif test_type == 'cuda':
        title = "⚡ Performance Benchmark: CUDA Implementations ⚡"
    elif test_type == 'triton':
        title = "⚡ Performance Benchmark: Triton Implementation ⚡"
    elif test_type == 'flex':
        title = "⚡ Performance Benchmark: Flex Implementation ⚡"
    else:
        title = "⚡ Performance Benchmark ⚡"
    
    print(title)
    print("🏆" + "=" * 76 + "🏆")
    
    # Test configurations: (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal)
    configs = [
        # Vary sequence length
        (1, 2, 1, 256, 256, 64, 1024, True),
        (1, 2, 1, 512, 512, 64, 1024, True),
        (1, 2, 1, 1024, 1024, 64, 1024, True),
        (1, 2, 1, 2048, 2048, 64, 1024, True),
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (1, 2, 1, 8192, 8192, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 32768, 32768, 64, 1024, True),

        # Inference
        (1, 2, 1, 1, 256, 64, 1024, True),
        (1, 2, 1, 1, 512, 64, 1024, True),
        (1, 2, 1, 1, 1024, 64, 1024, True),
        (1, 2, 1, 1, 2048, 64, 1024, True),
        (1, 2, 1, 1, 4096, 64, 1024, True),
        (1, 2, 1, 1, 8192, 64, 1024, True),
        (1, 2, 1, 1, 16384, 64, 1024, True),
        (1, 2, 1, 1, 32768, 64, 1024, True),
        (1, 2, 1, 1, 65536, 64, 1024, True),
        (1, 2, 1, 1, 131072, 64, 1024, True),
        (1, 2, 1, 1, 262144, 64, 1024, True),
        (1, 2, 1, 1, 524288, 64, 1024, True),
        
        # Vary batch size
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (2, 2, 1, 4096, 4096, 64, 1024, True),
        (4, 2, 1, 4096, 4096, 64, 1024, True),
        (8, 2, 1, 4096, 4096, 64, 1024, True),
        
        # Vary head count
        (1, 1, 1, 4096, 4096, 64, 1024, True),
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (1, 4, 1, 4096, 4096, 64, 1024, True),
        (1, 8, 2, 4096, 4096, 64, 1024, True),
        
        # Vary head dimension
        (1, 2, 1, 4096, 4096, 32, 1024, True),
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (1, 2, 1, 4096, 4096, 96, 1024, True),
        (1, 2, 1, 4096, 4096, 128, 1024, True),
        (1, 2, 1, 4096, 4096, 192, 1024, True),
        (1, 2, 1, 4096, 4096, 256, 1024, True),
        
        # Vary window_size
        (1, 2, 1, 32768, 32768, 64, 32, True),
        (1, 2, 1, 32768, 32768, 64, 64, True),
        (1, 2, 1, 32768, 32768, 64, 128, True),
        (1, 2, 1, 32768, 32768, 64, 256, True),
        (1, 2, 1, 32768, 32768, 64, 512, True),
        (1, 2, 1, 32768, 32768, 64, 1024, True),
        (1, 2, 1, 32768, 32768, 64, 2048, True),
        (1, 2, 1, 32768, 32768, 64, 4096, True),
        (1, 2, 1, 32768, 32768, 64, 8192, True),
        (1, 2, 1, 32768, 32768, 64, 16384, True),
        (1, 2, 1, 32768, 32768, 64, 32768, True),
    ]
    
    print(f"\n📊 Benchmark Results (averaged over {num_runs} runs):")
    print(f"🔧 {'Configuration':<60} ⚡ {'SDPA':<10} 🚀 {'CUDA':<10} 🌟 {'Triton':<10} 🌟 {'Flex':<15} 📈 {'Speedup':<15}")
    print("🔄" + "-" * 150 + "🔄")
    
    all_results = []
    
    for config in configs:
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal = config
        
        results = benchmark_attention_performance(config, test_type, num_runs, warmup_runs)
        all_results.append(results)
        
        # Calculate averages for all implementations
        implementations = {
            'sdpa': ('sdpa_forward', results['sdpa_forward_status'], results['sdpa_forward_times']),
            'cuda': ('fdma_cuda_forward', results['fdma_cuda_forward_status'], results['fdma_cuda_forward_times']),
            'triton': ('fdma_triton_forward', results['fdma_triton_forward_status'], results['fdma_triton_forward_times']),
            'flex': ('fdma_flex_forward', results['fdma_flex_forward_status'], results['fdma_flex_forward_times'])
        }
        
        # Calculate time strings and averages
        time_strs = {}
        time_avgs = {}
        
        for impl_key, (_, status, times) in implementations.items():
            if status == 'success' and times:
                avg_time = sum(times) / len(times)
                time_strs[impl_key] = f"{avg_time:.2f}"
                time_avgs[impl_key] = avg_time
            else:
                time_strs[impl_key] = status[:8]  # Truncate status for display
                time_avgs[impl_key] = float('inf')
        
        # Calculate speedups
        speedup_strs = {}
        flash_avg = time_avgs.get('sdpa', float('inf'))
        
        for impl_key in ['cuda', 'triton', 'flex']:
            impl_avg = time_avgs.get(impl_key, float('inf'))
            if flash_avg != float('inf') and impl_avg != float('inf') and impl_avg > 0:
                speedup = flash_avg / impl_avg
                speedup_strs[impl_key] = f"{speedup:.2f}x"
            else:
                speedup_strs[impl_key] = "N/A"
        
        # Format output with shorter config string
        config_short = f" B{batch_size} Hq{num_heads} Hkv{num_kv_heads} Q{query_len} K{key_len} D{head_dim} W{window_size} "
        if not is_causal:
            config_short += "N"
        else:
            config_short += "C"
        
        # Add status icons
        icons = ""
        for impl_key, (_, status, _) in implementations.items():
            if status == 'success':
                icons += " ✅ "
            elif status in ['OOM', 'Not Available']:
                icons += " ❌ "
            else:
                icons += " ⚠️ "
        
        # Create speedup summary (best performing implementation)
        best_speedup = "N/A"
        best_impl = "N/A"
        for impl_key, speedup_str in speedup_strs.items():
            if speedup_str != "N/A":
                try:
                    speedup_val = float(speedup_str.replace('x', ''))
                    if best_speedup == "N/A" or speedup_val > float(best_speedup.replace('x', '')):
                        best_speedup = speedup_str
                        best_impl = impl_key.upper()
                except:
                    continue
        
        speedup_summary = f"{best_impl}:{best_speedup}" if best_speedup != "N/A" else "N/A"
        
        print(f"{icons} {config_short:<48} {time_strs['sdpa']:<12} {time_strs['cuda']:<12} {time_strs['triton']:<12} {time_strs['flex']:<18} {speedup_summary:<15}")
    
    print("🔄" + "-" * 150 + "🔄")
    
    # Summary statistics
    implementation_speedups = {
        'cuda': [],
        'triton': [],
        'flex': []
    }
    
    for results in all_results:
        if results['sdpa_forward_status'] == 'success' and results['sdpa_forward_times']:
            flash_avg = sum(results['sdpa_forward_times']) / len(results['sdpa_forward_times'])

            # Calculate speedups for each implementation
            for impl_key in implementation_speedups.keys():
                # Map implementation keys to actual result keys
                if impl_key == 'cuda':
                    status_key = 'fdma_cuda_forward_status'
                    times_key = 'fdma_cuda_forward_times'
                else:
                    status_key = f'fdma_{impl_key}_forward_status'
                    times_key = f'fdma_{impl_key}_forward_times'

                if (status_key in results and results[status_key] == 'success' and 
                    times_key in results and results[times_key]):
                    
                    impl_avg = sum(results[times_key]) / len(results[times_key])
                    if impl_avg > 0:
                        implementation_speedups[impl_key].append(flash_avg / impl_avg)
    
    print(f"\n🏆 Summary:")
    
    # Display statistics for each implementation
    for impl_key, speedups in implementation_speedups.items():
        if speedups:
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            min_speedup = np.min(speedups)
            
            # Choose appropriate icon based on performance
            if avg_speedup > 2.0:
                icon = "🔥"
            elif avg_speedup > 1.5:
                icon = "🚀"
            elif avg_speedup > 1.0:
                icon = "📈"
            else:
                icon = "😐"
            
            impl_name = impl_key.replace('_', '-').upper()
            print(f"  {icon} {impl_name:10} vs SDPA - Avg: {avg_speedup:.2f}x (Best: {max_speedup:.2f}x, Worst: {min_speedup:.2f}x)")
        else:
            print(f"  ❌ {impl_key.replace('_', '-').upper():10} vs SDPA - No successful runs")


def main():
    """
    Run performance benchmarks for Dynamic Mask Attention.
    
    This script measures and compares performance including:
    - Latency measurements across different configurations
    - Memory usage analysis
    - Throughput comparisons
    - Scalability analysis
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Performance benchmark for Dynamic Mask Attention'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
    parser.add_argument('--test-type', type=str, default='all', 
                        choices=['all', 'sdpa', 'cuda', 'triton', 'flex', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex'],
                        help='Type of benchmark to run (default: all)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Print test environment information
    print(f"🐍 PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA device: {torch.cuda.get_device_name()}")
        print(f"💾 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Test type: {args.test_type}")
    print(f"🔄 Runs: {args.runs}, Warmup: {args.warmup}")
    
    # Run performance benchmark
    run_performance_benchmark(args.test_type, args.runs, args.warmup)


if __name__ == "__main__":
    main()