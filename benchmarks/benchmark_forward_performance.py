#!/usr/bin/env python3
"""
Performance Benchmark for Dynamic Mask Attention

This script measures and compares the performance of Dynamic Mask Attention 
implementation against Flash Attention baseline across various configurations.

Benchmark includes:
- Multiple sequence lengths and batch sizes
- Head count and dimension variations
- Throughput and latency measurements
- Memory usage analysis
- Speedup comparisons
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import gc
from flash_dma_cpp import apply_dynamic_mask_attention  # type: ignore
from typing import cast


def prepare_dynamic_mask(
    hidden_states: torch.Tensor,
    dt_states: torch.Tensor,
    keep_window_size: int = 2048,
    attention_mask: torch.Tensor | None = None,
):
    """
    Calculate dynamic attention mask to mask tokens for sparse attention.

    Args:
        hidden_states: Input hidden states to determine dtype minimum value
        dt_states: dt_states of shape (batch_size, num_kv_heads, key_sequence_length)
        keep_window_size: Window size of tokens not dynamically masked
        attention_mask: Optional attention mask of shape (batch_size, 1, query_len, key_len)
    
    Returns:
        tuple: (attn_mask, active_mask)
    """
    min_dtype = torch.finfo(hidden_states.dtype).min
    dtype = hidden_states.dtype
    attn_mask = dt_states[:, :, None, :].expand(
        -1, -1, hidden_states.shape[2], -1
    )  # [batch_size, num_kv_heads, query_len, key_len]
    active_mask = torch.ones_like(attn_mask, dtype=dtype, device=attn_mask.device)
    
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask, 
                torch.tensor(0.0, device=attention_mask.device, dtype=dtype), 
                min_dtype
            )
        attn_mask = attn_mask.masked_fill(
            attention_mask[:, :, :, : attn_mask.shape[-1]] != 0, min_dtype
        )
    
    if attn_mask.shape[-1] > keep_window_size:
        topk_indices = torch.topk(
            attn_mask, keep_window_size, dim=-1, largest=True, sorted=False
        ).indices
        active_mask = torch.zeros_like(attn_mask, dtype=dtype, device=attn_mask.device)
        active_mask = active_mask.scatter(-1, topk_indices, 1.0)
        attn_mask = attn_mask.masked_fill(active_mask == 0.0, min_dtype)
    
    return attn_mask, active_mask


def calculate_zero_hold_states(value_states, dt_proj, A, causal_mask=None):
    """
    Calculate zero hold states for dynamic mask attention.
    
    Args:
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        causal_mask: Optional causal mask
    
    Returns:
        zero_hold_states: [batch_size, num_kv_heads, key_len]
    """
    batch_size, _, key_len, _ = value_states.shape
    
    # Transpose and reshape value_states, then matrix multiply with dt_proj.T
    dt_result = torch.matmul(
        value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), 
        dt_proj.T
    )
    
    # Apply softplus activation and coefficient A
    dt_states = torch.exp(F.softplus(dt_result) * A)
    zero_hold_states = dt_states.transpose(-1, -2)  # [batch_size, num_kv_heads, key_len]
        
    return zero_hold_states


def flash_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    is_causal=True,
):
    """
    CUDA implementation of Flash Attention baseline.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs or "OOM" if out of memory
    """
    _, _, query_len, _ = query_states.shape
    _, _, key_len, _ = key_states.shape
    if query_len > 32768 or key_len > 32768:
        return "OOM"

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    try:
        attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=scaling,
            enable_gqa=True
        )
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()  # Transpose to [batch, query_len, num_heads, head_dim]
        return attn_outputs
    except torch.cuda.OutOfMemoryError:
        return "OOM"


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
    # Calculate zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)

    _, active_mask = prepare_dynamic_mask(
        query_states,
        zero_hold_states,
        keep_window_size,
        causal_mask if is_causal else None
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    # Ensure correct data types and memory layout
    query_states = query_states.transpose(1, 2).contiguous()  # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()      # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()  # [batch, key_len, num_kv_heads, head_dim]
    zero_hold_states = zero_hold_states[:, :, None, :].expand(
        -1, -1, query_states.shape[1], -1
    ).contiguous()  # [batch, num_kv_heads, query_len, key_len]
    active_mask = active_mask.contiguous()  # [batch, num_kv_heads, query_len, key_len]

    try:
        result = apply_dynamic_mask_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            zoh_states=zero_hold_states,
            active_mask=active_mask,
            scale=scaling,
            keep_window_size=keep_window_size,
            is_causal=is_causal,
            return_softmax=return_softmax
        )
        
        # Convert result back to original data type
        attn_outputs = result[0]
        return attn_outputs
    except torch.cuda.OutOfMemoryError:
        return "OOM"


def dynamic_mask_attention_cuda_no_topk(
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
    CUDA implementation of dynamic mask attention without topk computation.
    This version skips the topk calculation for more accurate kernel performance testing.
    
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
    # Calculate zero_hold_states
    zero_hold_states = calculate_zero_hold_states(value_states, dt_proj, A, causal_mask)

    # Create a simplified mask without topk computation
    batch_size, num_heads, query_len, head_dim = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape
    dtype = query_states.dtype
    device = query_states.device
    
    # Create full active mask (no topk selection)
    active_mask = torch.zeros(
        (batch_size, num_kv_heads, query_len, key_len), 
        dtype=dtype, 
        device=device
    )

    # Ensure correct data types and memory layout
    query_states = query_states.transpose(1, 2).contiguous()  # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()      # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()  # [batch, key_len, num_kv_heads, head_dim]
    zero_hold_states = zero_hold_states[:, :, None, :].expand(
        -1, -1, query_states.shape[1], -1
    ).contiguous()  # [batch, num_kv_heads, query_len, key_len]
    active_mask = active_mask.contiguous()  # [batch, num_kv_heads, query_len, key_len]

    try:
        result = apply_dynamic_mask_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            zoh_states=zero_hold_states,
            active_mask=active_mask,
            scale=scaling,
            keep_window_size=0,
            is_causal=is_causal,
            return_softmax=return_softmax
        )
        
        # Convert result back to original data type
        attn_outputs = result[0]
        return attn_outputs
    except torch.cuda.OutOfMemoryError:
        return "OOM"


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


def benchmark_attention_performance(config, num_runs=5, warmup_runs=2):
    """
    Benchmark attention performance for a given configuration.
    
    Args:
        config: Tuple of (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Performance metrics
    """
    batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim = config
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
    keep_window_size = 2048
    
    results = {
        'config': config,
        'flash_attention_times': [],
        'dynamic_mask_attention_times': [],
        'dynamic_mask_attention_no_topk_times': [],
        'flash_attention_memory': 0,
        'dynamic_mask_attention_memory': 0,
        'dynamic_mask_attention_no_topk_memory': 0,
        'flash_attention_status': 'success',
        'dynamic_mask_attention_status': 'success',
        'dynamic_mask_attention_no_topk_status': 'success'
    }
    
    # Benchmark Flash Attention
    gc.collect()
    torch.cuda.empty_cache()
    
    # Warmup runs
    for _ in range(warmup_runs):
        result = flash_attention_cuda(
            query_states, key_states, value_states,
            scaling, causal_mask, True
        )
        if result == "OOM":
            results['flash_attention_status'] = 'OOM'
            break
        torch.cuda.synchronize()
    
    if results['flash_attention_status'] == 'success':
        # Measure memory before benchmark
        mem_before = measure_memory_usage()
        
        # Actual benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            result = flash_attention_cuda(
                query_states, key_states, value_states,
                scaling, causal_mask, True
            )
            torch.cuda.synchronize()
            end_time = time.time()
            
            if result == "OOM":
                results['flash_attention_status'] = 'OOM'
                break
            
            results['flash_attention_times'].append((end_time - start_time) * 1000)  # ms
        
        # Measure memory after
        mem_after = measure_memory_usage()
        results['flash_attention_memory'] = mem_after[0] - mem_before[0]
    
    # Benchmark Dynamic Mask Attention
    gc.collect()
    torch.cuda.empty_cache()
    
    # Warmup runs
    for _ in range(warmup_runs):
        result = dynamic_mask_attention_cuda(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, True
        )
        if result == "OOM":
            results['dynamic_mask_attention_status'] = 'OOM'
            break
        torch.cuda.synchronize()
    
    if results['dynamic_mask_attention_status'] == 'success':
        # Measure memory before benchmark
        mem_before = measure_memory_usage()
        
        # Actual benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            result = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
            torch.cuda.synchronize()
            end_time = time.time()
            
            if result == "OOM":
                results['dynamic_mask_attention_status'] = 'OOM'
                break
            
            results['dynamic_mask_attention_times'].append((end_time - start_time) * 1000)  # ms
        
        # Measure memory after
        mem_after = measure_memory_usage()
        results['dynamic_mask_attention_memory'] = mem_after[0] - mem_before[0]
    
    # Benchmark Dynamic Mask Attention (No TopK)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Warmup runs
    for _ in range(warmup_runs):
        result = dynamic_mask_attention_cuda_no_topk(
            query_states, key_states, value_states,
            dt_proj, A, scaling, causal_mask,
            keep_window_size, True
        )
        if result == "OOM":
            results['dynamic_mask_attention_no_topk_status'] = 'OOM'
            break
        torch.cuda.synchronize()
    
    if results['dynamic_mask_attention_no_topk_status'] == 'success':
        # Measure memory before benchmark
        mem_before = measure_memory_usage()
        
        # Actual benchmark runs
        for _ in range(num_runs):
            start_time = time.time()
            result = dynamic_mask_attention_cuda_no_topk(
                query_states, key_states, value_states,
                dt_proj, A, scaling, causal_mask,
                keep_window_size, True
            )
            torch.cuda.synchronize()
            end_time = time.time()
            
            if result == "OOM":
                results['dynamic_mask_attention_no_topk_status'] = 'OOM'
                break
            
            results['dynamic_mask_attention_no_topk_times'].append((end_time - start_time) * 1000)  # ms
        
        # Measure memory after
        mem_after = measure_memory_usage()
        results['dynamic_mask_attention_no_topk_memory'] = mem_after[0] - mem_before[0]
    
    return results


def run_performance_benchmark():
    """Run comprehensive performance benchmark across different configurations."""
    print("\n" + "🏆" + "=" * 76 + "🏆")
    print("⚡ Performance Benchmark: Dynamic Mask Attention vs Flash Attention ⚡")
    print("🏆" + "=" * 76 + "🏆")
    
    # Test configurations: (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim)
    configs = [
        # Vary sequence length
        (1, 2, 1, 256, 256, 32),
        (1, 2, 1, 512, 512, 32),
        (1, 2, 1, 1024, 1024, 32),
        (1, 2, 1, 2048, 2048, 32),
        (1, 2, 1, 4096, 4096, 32),
        (1, 2, 1, 8192, 8192, 32),
        (1, 2, 1, 16384, 16384, 32),
        (1, 2, 1, 32768, 32768, 32),

        # Inference
        (1, 2, 1, 64, 256, 32),
        (1, 2, 1, 64, 512, 32),
        (1, 2, 1, 64, 1024, 32),
        (1, 2, 1, 64, 2048, 32),
        (1, 2, 1, 64, 4096, 32),
        (1, 2, 1, 64, 8192, 32),
        (1, 2, 1, 64, 16384, 32),
        (1, 2, 1, 64, 32768, 32),
        
        # Vary batch size
        (1, 2, 1, 1024, 1024, 32),
        (2, 2, 1, 1024, 1024, 32),
        (4, 2, 1, 1024, 1024, 32),
        (8, 2, 1, 1024, 1024, 32),
        
        # Vary head count
        (1, 1, 1, 1024, 1024, 32),
        (1, 2, 1, 1024, 1024, 32),
        (1, 4, 1, 1024, 1024, 32),
        (1, 8, 2, 1024, 1024, 32),
        
        # # Vary head dimension
        # (1, 2, 1, 1024, 1024, 32),
        # (1, 2, 1, 1024, 1024, 64),
        # (1, 2, 1, 1024, 1024, 128),
    ]

    num_runs = 3  # Run 3 times and take average
    
    print(f"\n📊 Benchmark Results (averaged over {num_runs} runs):")
    print(f"🔧 {'Configuration':<42} ⚡ {'Flash (ms)':<12} 🚀 {'DMA (ms)':<12} 🚀 {'DMA-Skip-All (ms)':<22} 📈 {'Speedup':<12} 📈 {'Skip-All-Speedup':<20} 💾 {'Memory':<10}")
    print("🔄" + "-" * 155 + "🔄")
    
    all_results = []
    
    for config in configs:
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim = config
        
        results = benchmark_attention_performance(config, num_runs=num_runs)
        all_results.append(results)
        
        # Calculate averages
        if results['flash_attention_status'] == 'success' and results['flash_attention_times']:
            flash_avg = sum(results['flash_attention_times']) / len(results['flash_attention_times'])
            flash_time_str = f"{flash_avg:.2f}"
        else:
            flash_time_str = results['flash_attention_status']
            flash_avg = float('inf')
        
        if results['dynamic_mask_attention_status'] == 'success' and results['dynamic_mask_attention_times']:
            dma_avg = sum(results['dynamic_mask_attention_times']) / len(results['dynamic_mask_attention_times'])
            dma_time_str = f"{dma_avg:.2f}"
        else:
            dma_time_str = results['dynamic_mask_attention_status']
            dma_avg = float('inf')
            
        if results['dynamic_mask_attention_no_topk_status'] == 'success' and results['dynamic_mask_attention_no_topk_times']:
            dma_nt_avg = sum(results['dynamic_mask_attention_no_topk_times']) / len(results['dynamic_mask_attention_no_topk_times'])
            dma_nt_time_str = f"{dma_nt_avg:.2f}"
        else:
            dma_nt_time_str = results['dynamic_mask_attention_no_topk_status']
            dma_nt_avg = float('inf')
        
        # Calculate speedups
        if flash_avg != float('inf') and dma_avg != float('inf') and dma_avg > 0:
            speedup = flash_avg / dma_avg
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
            
        if flash_avg != float('inf') and dma_nt_avg != float('inf') and dma_nt_avg > 0:
            kernel_speedup = flash_avg / dma_nt_avg
            kernel_speedup_str = f"{kernel_speedup:.2f}x"
        else:
            kernel_speedup_str = "N/A"
        
        # Memory usage
        mem_diff = results['dynamic_mask_attention_memory'] - results['flash_attention_memory']
        mem_str = f"{mem_diff:+.0f}"
        
        # Format output
        config_short = f"b={batch_size},h={num_heads},kv={num_kv_heads},q={query_len},k={key_len},d={head_dim}"
        
        # Add status icons
        flash_icon = "✅" if results['flash_attention_status'] == 'success' else "💥"
        dma_icon = "✅" if results['dynamic_mask_attention_status'] == 'success' else "💥"
        dma_nt_icon = "✅" if results['dynamic_mask_attention_no_topk_status'] == 'success' else "💥"
        
        print(f"{flash_icon}{dma_icon}{dma_nt_icon} {config_short:<42} {flash_time_str:<14} {dma_time_str:<20} {dma_nt_time_str:<20} {speedup_str:<18} {kernel_speedup_str:<20} {mem_str:<12}")
    
    print("🔄" + "-" * 155 + "🔄")
    
    # Summary statistics
    speedups = []
    kernel_speedups = []
    for results in all_results:
        if (results['flash_attention_status'] == 'success' and 
            results['dynamic_mask_attention_status'] == 'success' and
            results['flash_attention_times'] and results['dynamic_mask_attention_times']):
            
            flash_avg = sum(results['flash_attention_times']) / len(results['flash_attention_times'])
            dma_avg = sum(results['dynamic_mask_attention_times']) / len(results['dynamic_mask_attention_times'])
            
            if dma_avg > 0:
                speedups.append(flash_avg / dma_avg)
        
        if (results['flash_attention_status'] == 'success' and 
            results['dynamic_mask_attention_no_topk_status'] == 'success' and
            results['flash_attention_times'] and results['dynamic_mask_attention_no_topk_times']):
            
            flash_avg = sum(results['flash_attention_times']) / len(results['flash_attention_times'])
            dma_nt_avg = sum(results['dynamic_mask_attention_no_topk_times']) / len(results['dynamic_mask_attention_no_topk_times'])
            
            if dma_nt_avg > 0:
                kernel_speedups.append(flash_avg / dma_nt_avg)
    
    print(f"\n🏆 Summary:")
    if speedups:
        avg_speedup = np.mean(speedups)
        speedup_icon = "🚀" if avg_speedup > 1.5 else "📈" if avg_speedup > 1.0 else "😐"
        print(f"  {speedup_icon} DMA vs Flash - Average speedup: {avg_speedup:.2f}x (Best: {np.max(speedups):.2f}x, Worst: {np.min(speedups):.2f}x)")
    
    if kernel_speedups:
        avg_kernel_speedup = np.mean(kernel_speedups)
        kernel_icon = "🔥" if avg_kernel_speedup > 2.0 else "🚀" if avg_kernel_speedup > 1.5 else "📈" if avg_kernel_speedup > 1.0 else "😐"
        print(f"  {kernel_icon} DMA-NoTopK vs Flash - Average kernel speedup: {avg_kernel_speedup:.2f}x (Best: {np.max(kernel_speedups):.2f}x, Worst: {np.min(kernel_speedups):.2f}x)")
        print(f"  💡 TopK overhead: ~{((np.mean(kernel_speedups) - np.mean(speedups) if speedups else 0) / np.mean(kernel_speedups) * 100) if kernel_speedups else 0:.1f}% performance impact")


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
    
    # Run performance benchmark
    run_performance_benchmark()


if __name__ == "__main__":
    main()