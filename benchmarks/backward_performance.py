#!/usr/bin/env python3
"""
Backward Performance Benchmark for Dynamic Mask Attention

This script measures and compares the backward pass performance of multiple Dynamic Mask Attention 
implementations against SDPA baseline across various configurations.

Implementations tested:
- PyTorch SDPA - Baseline (backward pass)
- Dynamic Mask Attention CUDA - Custom CUDA kernel implementation (backward pass)
- Dynamic Mask Attention Triton - Triton kernel implementation (backward pass)
- Dynamic Mask Attention Flex - Flex Attention implementation (backward pass)

Benchmark includes:
- Multiple sequence lengths and batch sizes
- Head count and dimension variations
- Backward pass throughput and latency measurements
- Memory usage analysis during backward pass
- Speedup comparisons across all implementations for gradient computation
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


def scaled_dot_product_attention_backward_cuda(
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
        scaling: Attention scaling factor
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
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

    # Repeat KV for multi-head attention (GQA support)
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    attn_bias = attn_bias.masked_fill(~attn_mask, torch.finfo(query_states.dtype).min).contiguous()

    try:
        attn_outputs = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_bias,
            scale=scaling,
            # is_causal=is_causal,
            enable_gqa=True
        )
  
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()  # [batch, query_len, num_heads, head_dim]

        torch.cuda.synchronize()
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()

        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_cuda(
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
    CUDA implementation of dynamic mask attention backward pass.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if flash_sparse_attn_func is None:
        return "Not Available", 0

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
    query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_kv_heads, head_dim]

    try:
        attn_outputs = flash_sparse_attn_func(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=is_causal,
            softmax_scale=scaling,
            softcap=0.0,
            deterministic=False,
            return_attn_probs=False
        )

        torch.cuda.synchronize()
        start_time = time.time()
        
        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_triton(
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
    Triton implementation of dynamic mask attention backward pass.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if triton_sparse_attn_func is None:
        return "Not Available", 0
    
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
        block_size=64,
        type="topk"
    )

    # Ensure correct data types and memory layout for Triton function
    query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]  
    key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_heads, head_dim]  
    value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]  

    try:
        attn_outputs = triton_sparse_attn_func(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            is_causal=is_causal,
            softmax_scale=scaling,
        )

        torch.cuda.synchronize()
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_flex(
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
    Flex Attention implementation of dynamic mask attention backward pass.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        attn_bias: [num_kv_heads, query_len, key_len]
        causal_mask: [batch_size, 1, query_length, key_length] or None
        scaling: Attention scaling factor
        window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
    """
    if flex_sparse_attn_func is None:
        return "Not Available", 0
    
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
    attn_mask = repeat_kv(attn_mask, num_queries_per_kv) if attn_mask is not None else None
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv)

    # Ensure correct data types and memory layout for Flex function
    query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]

    try:
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
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def measure_memory_usage():
    """
    Measure current GPU memory usage.
    
    Returns:
        tuple: (allocated_mb, reserved_mb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        return allocated, reserved
    return 0, 0


def benchmark_backward_attention_performance(config, test_type='all', num_runs=5, warmup_runs=2):
    """
    Benchmark backward attention performance for a given configuration.
    
    Args:
        config: Tuple of (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal)
        test_type: Type of test to run ('all', 'sdpa', 'cuda', 'triton', 'flex', etc.)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Performance metrics
    """
    batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random input data (requires_grad=True for backward pass)
    query_states = torch.randn(
        batch_size, num_heads, query_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    key_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    value_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    attn_bias = torch.randn(
        batch_size, num_kv_heads, 1, key_len,
        device=device, dtype=torch.bfloat16
    )
    cache_position = torch.arange(key_len - query_len, key_len, device=device)
    causal_mask = torch.arange(key_len, device=device) <= cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    
    # Set scaling factor from config
    scaling = head_dim ** -0.5
    
    results = {
        'config': config,
        'sdpa_backward_times': [],
        'fdma_cuda_backward_times': [],
        'fdma_triton_backward_times': [],
        'fdma_flex_backward_times': [],
        'sdpa_backward_memory': 0,
        'fdma_cuda_backward_memory': 0,
        'fdma_triton_backward_memory': 0,
        'fdma_flex_backward_memory': 0,
        'sdpa_backward_status': 'success',
        'fdma_cuda_backward_status': 'success',
        'fdma_triton_backward_status': 'success',
        'fdma_flex_backward_status': 'success'
    }
    
    # Determine which implementations to run
    run_sdpa = test_type in ['all', 'sdpa', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex']
    run_cuda = test_type in ['all', 'cuda', 'sdpa-vs-cuda']
    run_triton = test_type in ['all', 'triton', 'sdpa-vs-triton']
    run_flex = test_type in ['all', 'flex', 'sdpa-vs-flex']
    
    # Benchmark SDPA Backward
    if run_sdpa:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            query_sdpa = query_states.clone().detach().requires_grad_(True)
            key_sdpa = key_states.clone().detach().requires_grad_(True)
            value_sdpa = value_states.clone().detach().requires_grad_(True)
            attn_bias_sdpa = attn_bias.clone().detach().requires_grad_(True)
            causal_mask_sdpa = causal_mask.clone().detach()

            result = scaled_dot_product_attention_backward_cuda(
                query_sdpa, key_sdpa, value_sdpa,
                attn_bias_sdpa, causal_mask_sdpa,
                scaling, window_size, is_causal
            )
            if result[0] == "OOM":
                results['sdpa_backward_status'] = 'OOM'
                break
            torch.cuda.synchronize()
        
        if results['sdpa_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                query_sdpa = query_states.clone().detach().requires_grad_(True)
                key_sdpa = key_states.clone().detach().requires_grad_(True)
                value_sdpa = value_states.clone().detach().requires_grad_(True)
                attn_bias_sdpa = attn_bias.clone().detach().requires_grad_(True)
                causal_mask_sdpa = causal_mask.clone().detach()

                result = scaled_dot_product_attention_backward_cuda(
                    query_sdpa, key_sdpa, value_sdpa,
                    attn_bias_sdpa, causal_mask_sdpa,
                    scaling, window_size, is_causal
                )
                
                if result[0] == "OOM":
                    results['sdpa_backward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                results['sdpa_backward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['sdpa_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['sdpa_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention CUDA Backward
    if run_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            query_cuda = query_states.clone().detach().requires_grad_(True)
            key_cuda = key_states.clone().detach().requires_grad_(True)
            value_cuda = value_states.clone().detach().requires_grad_(True)
            attn_bias_cuda = attn_bias.clone().detach().requires_grad_(True)
            causal_mask_cuda = causal_mask.clone().detach()

            result = dynamic_mask_attention_backward_cuda(
                query_cuda, key_cuda, value_cuda, attn_bias_cuda, causal_mask_cuda,
                scaling, window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_cuda_backward_status'] = result[0]
                break
            torch.cuda.synchronize()
        
        if results['fdma_cuda_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                query_cuda = query_states.clone().detach().requires_grad_(True)
                key_cuda = key_states.clone().detach().requires_grad_(True)
                value_cuda = value_states.clone().detach().requires_grad_(True)
                attn_bias_cuda = attn_bias.clone().detach().requires_grad_(True)
                causal_mask_cuda = causal_mask.clone().detach()

                result = dynamic_mask_attention_backward_cuda(
                    query_cuda, key_cuda, value_cuda, attn_bias_cuda, causal_mask_cuda,
                    scaling, window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_cuda_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_cuda_backward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_cuda_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_cuda_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention Triton Backward
    if run_triton:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            query_triton = query_states.clone().detach().requires_grad_(True)
            key_triton = key_states.clone().detach().requires_grad_(True)
            value_triton = value_states.clone().detach().requires_grad_(True)
            attn_bias_triton = attn_bias.clone().detach().requires_grad_(True)
            causal_mask_triton = causal_mask.clone().detach()

            result = dynamic_mask_attention_backward_triton(
                query_triton, key_triton, value_triton, attn_bias_triton, causal_mask_triton,
                scaling, window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_triton_backward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_triton_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                query_triton = query_states.clone().detach().requires_grad_(True)
                key_triton = key_states.clone().detach().requires_grad_(True)
                value_triton = value_states.clone().detach().requires_grad_(True)
                attn_bias_triton = attn_bias.clone().detach().requires_grad_(True)
                causal_mask_triton = causal_mask.clone().detach()

                result = dynamic_mask_attention_backward_triton(
                    query_triton, key_triton, value_triton, attn_bias_triton, causal_mask_triton,
                    scaling, window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_triton_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_triton_backward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_triton_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_triton_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention Flex Backward
    if run_flex:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            query_flex = query_states.clone().detach().requires_grad_(True)
            key_flex = key_states.clone().detach().requires_grad_(True)
            value_flex = value_states.clone().detach().requires_grad_(True)
            attn_bias_flex = attn_bias.clone().detach().requires_grad_(True)
            causal_mask_flex = causal_mask.clone().detach()

            result = dynamic_mask_attention_backward_flex(
                query_flex, key_flex, value_flex, attn_bias_flex, causal_mask_flex,
                scaling, window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_flex_backward_status'] = result[0]
                break
            torch.cuda.synchronize()
        
        if results['fdma_flex_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                # Clone inputs for each run
                query_flex = query_states.clone().detach().requires_grad_(True)
                key_flex = key_states.clone().detach().requires_grad_(True)
                value_flex = value_states.clone().detach().requires_grad_(True)
                attn_bias_flex = attn_bias.clone().detach().requires_grad_(True)
                causal_mask_flex = causal_mask.clone().detach()

                result = dynamic_mask_attention_backward_flex(
                    query_flex, key_flex, value_flex, attn_bias_flex, causal_mask_flex,
                    scaling, window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_flex_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                results['fdma_flex_backward_times'].append(result[1])  # ms
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_flex_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_flex_backward_status'] = 'N/A'
    
    return results


def run_backward_performance_benchmark(test_type='all', num_runs=3, warmup_runs=2):
    """Run comprehensive backward pass performance benchmark across different configurations."""
    print("\n" + "🏆" + "=" * 76 + "🏆")
    
    # Update title based on test type
    if test_type == 'all':
        title = "🔥 Backward Pass Performance Benchmark: All Implementations 🔥"
    elif test_type == 'sdpa-vs-cuda':
        title = "🚀 Backward Pass Performance: SDPA vs CUDA 🚀"
    elif test_type == 'sdpa-vs-triton':
        title = "🌟 Backward Pass Performance: SDPA vs Triton 🌟"
    elif test_type == 'sdpa-vs-flex':
        title = "✨ Backward Pass Performance: SDPA vs Flex ✨"
    elif test_type == 'sdpa':
        title = "📊 Backward Pass Performance: SDPA Only 📊"
    elif test_type == 'cuda':
        title = "🚀 Backward Pass Performance: CUDA Only 🚀"
    elif test_type == 'triton':
        title = "🌟 Backward Pass Performance: Triton Only 🌟"
    elif test_type == 'flex':
        title = "✨ Backward Pass Performance: Flex Only ✨"
    else:
        title = "🔥 Backward Pass Performance Benchmark 🔥"
    
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
        (1, 2, 1, 16384, 16384, 32, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 96, 1024, True),
        (1, 2, 1, 16384, 16384, 128, 1024, True),

        # Vary window_size
        (1, 2, 1, 16384, 16384, 64, 32, True),
        (1, 2, 1, 16384, 16384, 64, 64, True),
        (1, 2, 1, 16384, 16384, 64, 128, True),
        (1, 2, 1, 16384, 16384, 64, 256, True),
        (1, 2, 1, 16384, 16384, 64, 512, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 2048, True),
        (1, 2, 1, 16384, 16384, 64, 4096, True),
        (1, 2, 1, 16384, 16384, 64, 8192, True),
        (1, 2, 1, 16384, 16384, 64, 16384, True),
    ]
    
    print(f"\n📊 Backward Pass Benchmark Results (averaged over {num_runs} runs):")
    print(f"🔧 {'Configuration':<60} ⚡ {'SDPA-BWD':<12} 🚀 {'CUDA-BWD':<12} 🌟 {'Triton-BWD':<12} ✨ {'Flex-BWD':<15} 📈 {'Speedup':<15}")
    print("🔄" + "-" * 160 + "🔄")
    
    all_results = []
    
    for config in configs:
        try:
            results = benchmark_backward_attention_performance(config, test_type, num_runs, warmup_runs)
            all_results.append(results)
            
            # Format configuration string
            batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, window_size, is_causal = config
            config_str = f"B{batch_size} Hq{num_heads} Hkv{num_kv_heads} Q{query_len} K{key_len} D{head_dim} W{window_size} {'C' if is_causal else 'N'}"
            
            # Calculate averages and format results
            sdpa_avg = f"{sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times']):.2f}ms" if results['sdpa_backward_times'] else results['sdpa_backward_status']
            cuda_avg = f"{sum(results['fdma_cuda_backward_times'])/len(results['fdma_cuda_backward_times']):.2f}ms" if results['fdma_cuda_backward_times'] else results['fdma_cuda_backward_status']
            triton_avg = f"{sum(results['fdma_triton_backward_times'])/len(results['fdma_triton_backward_times']):.2f}ms" if results['fdma_triton_backward_times'] else results['fdma_triton_backward_status']
            flex_avg = f"{sum(results['fdma_flex_backward_times'])/len(results['fdma_flex_backward_times']):.2f}ms" if results['fdma_flex_backward_times'] else results['fdma_flex_backward_status']

            # Calculate speedup (SDPA vs others)
            speedup_info = []
            if results['sdpa_backward_times'] and results['fdma_cuda_backward_times']:
                sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
                cuda_time = sum(results['fdma_cuda_backward_times'])/len(results['fdma_cuda_backward_times'])
                speedup_info.append(f"CUDA: {sdpa_time/cuda_time:.1f}x")
            
            if results['sdpa_backward_times'] and results['fdma_triton_backward_times']:
                sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
                triton_time = sum(results['fdma_triton_backward_times'])/len(results['fdma_triton_backward_times'])
                speedup_info.append(f"Tri: {sdpa_time/triton_time:.1f}x")

            if results['sdpa_backward_times'] and results['fdma_flex_backward_times']:
                sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
                flex_time = sum(results['fdma_flex_backward_times'])/len(results['fdma_flex_backward_times'])
                speedup_info.append(f"Flex: {sdpa_time/flex_time:.1f}x")
            
            speedup_str = ", ".join(speedup_info) if speedup_info else "N/A"
            
            print(f"📊 {config_str:<60} ⚡ {sdpa_avg:<12} 🚀 {cuda_avg:<12} 🌟 {triton_avg:<12} ✨ {flex_avg:<15} 📈 {speedup_str:<15}")
            
        except Exception as e:
            print(f"❌ Error in config {config}: {e}")
            continue
    
    print("🔄" + "-" * 160 + "🔄")
    
    # Summary statistics
    implementation_speedups = {
        'cuda': [],
        'triton': [],
        'flex': []
    }
    
    for results in all_results:
        if results['sdpa_backward_times'] and results['fdma_cuda_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            cuda_time = sum(results['fdma_cuda_backward_times'])/len(results['fdma_cuda_backward_times'])
            implementation_speedups['cuda'].append(sdpa_time/cuda_time)

        if results['sdpa_backward_times'] and results['fdma_triton_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            triton_time = sum(results['fdma_triton_backward_times'])/len(results['fdma_triton_backward_times'])
            implementation_speedups['triton'].append(sdpa_time/triton_time)

        if results['sdpa_backward_times'] and results['fdma_flex_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            flex_time = sum(results['fdma_flex_backward_times'])/len(results['fdma_flex_backward_times'])
            implementation_speedups['flex'].append(sdpa_time/flex_time)
    
    print(f"\n🏆 Backward Pass Summary:")
    
    # Display statistics for each implementation
    for impl_key, speedups in implementation_speedups.items():
        if speedups:
            impl_name = impl_key.upper()
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            print(f"  🚀 {impl_name:7} speedup: avg={avg_speedup:.2f}x, max={max_speedup:.2f}x, min={min_speedup:.2f}x ({len(speedups)} configs)")
        else:
            print(f"  ❌ {impl_key.upper():7} : No valid measurements")


def main():
    """
    Run backward pass performance benchmarks for Dynamic Mask Attention.
    
    This script measures and compares backward pass performance including:
    - Gradient computation latency measurements across different configurations
    - Memory usage analysis during backward pass
    - Throughput comparisons for gradient computation
    - Scalability analysis for large sequences
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Backward pass performance benchmark for Dynamic Mask Attention'
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
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Test type: {args.test_type}")
    print(f"🔄 Runs: {args.runs}, Warmup: {args.warmup}")
    
    # Run backward pass performance benchmark
    run_backward_performance_benchmark(args.test_type, args.runs, args.warmup)


if __name__ == "__main__":
    main()
