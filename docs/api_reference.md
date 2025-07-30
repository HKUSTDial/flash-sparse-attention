# Flash Dynamic Mask Attention API Reference

## Overview

Flash Dynamic Mask Attention is a high-performance implementation that combines Flash Attention's memory efficiency with Dynamic Mask Attention's sparse computation capabilities. This API provides CUDA-accelerated attention computation with dynamic masking for handling extremely long sequences efficiently.

The library provides multiple interfaces:
- **High-level Functions**: Easy-to-use functions with automatic backend selection
- **Specific Implementations**: Direct access to CUDA, Triton, and Flex Attention backends
- **Packed Variants**: Optimized functions for QKV-packed and KV-packed tensors
- **Variable Length**: Support for variable sequence lengths within batches

## Table of Contents

1. [Installation](#installation)
2. [High-Level Interface](#high-level-interface)
3. [Core Functions](#core-functions)
4. [Packed Variants](#packed-variants)
5. [Variable Length Functions](#variable-length-functions)
6. [Backend Selection](#backend-selection)

## Installation

### Prerequisites

- **Python**: 3.8 or later
- **PyTorch**: 2.0.0 or later with CUDA support
- **CUDA**: 11.8 or later
- **NVIDIA GPU**: Compute Capability 8.0 or higher
- **Dependencies**: `packaging`, `torch`

### Install from Source

```bash
git clone https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn
git submodule update --init --recursive
pip install -e .
```

## High-Level Interface

### Automatic Backend Selection

```python
from flash_dmattn import flash_dmattn_func_auto, get_available_backends

# Check available backends
backends = get_available_backends()
print(f"Available backends: {backends}")

# Use with automatic backend selection
output = flash_dmattn_func_auto(
    q=query, k=key, v=value,
    attn_mask=attention_mask,
    attn_bias=attention_bias
)

# Force specific backend
output = flash_dmattn_func_auto(
    backend="cuda",  # or "triton", "flex"
    q=query, k=key, v=value,
    attn_mask=attention_mask,
    attn_bias=attention_bias
)
```

## Core Functions

### flash_dmattn_func

The main attention function supporting multi-head and grouped-query attention.

```python
def flash_dmattn_func(
    q: torch.Tensor,                               # Query tensor
    k: torch.Tensor,                               # Key tensor  
    v: torch.Tensor,                               # Value tensor
    attn_mask: Optional[torch.Tensor] = None,      # Attention mask
    attn_bias: Optional[torch.Tensor] = None,      # Attention bias
    dropout_p: Optional[float] = None,             # Dropout probability
    softmax_scale: Optional[float] = None,         # Scaling factor
    is_causal: Optional[bool] = None,              # Causal masking
    softcap: Optional[float] = None,               # Soft capping
    deterministic: Optional[bool] = None,          # Deterministic mode
    return_attn_probs: Optional[bool] = None,      # Return attention weights
) -> torch.Tensor
```

#### Parameters

- **q** (`torch.Tensor`): Query tensor of shape `(batch_size, seqlen_q, num_heads, head_dim)`
  - Must be contiguous and on CUDA device
  - Supported dtypes: `torch.float16`, `torch.bfloat16`

- **k** (`torch.Tensor`): Key tensor of shape `(batch_size, seqlen_k, num_heads_k, head_dim)`
  - Same dtype and device as `q`
  - Supports grouped-query attention when `num_heads_k < num_heads`

- **v** (`torch.Tensor`): Value tensor of shape `(batch_size, seqlen_k, num_heads_k, head_dim)`
  - Same dtype and device as `q`
  - Supports grouped-query attention when `num_heads_k < num_heads`

- **attn_mask** (`Optional[torch.Tensor]`): Attention mask of shape `(batch_size, num_heads, seqlen_q, seqlen_k)`
  - Binary mask: 1.0 for positions to attend, 0.0 for masked positions
  - If `None`, no masking is applied

- **attn_bias** (`Optional[torch.Tensor]`): Attention bias of shape `(batch_size, num_heads, seqlen_q, seqlen_k)`
  - Added to attention scores before softmax
  - If `None`, no bias is applied

- **dropout_p** (`Optional[float]`): Dropout probability (default: 0.0)
  - Range: [0.0, 1.0]
  - Applied to attention weights

- **softmax_scale** (`Optional[float]`): Scaling factor for attention scores
  - If `None`, defaults to `1.0 / sqrt(head_dim)`

- **is_causal** (`Optional[bool]`): Whether to apply causal masking (default: False)
  - When True, applies lower triangular mask

- **softcap** (`Optional[float]`): Soft capping value (default: 0.0)
  - If > 0, applies `softcap * tanh(score / softcap)`

- **deterministic** (`Optional[bool]`): Use deterministic backward pass (default: True)
  - Slightly slower but more memory efficient

- **return_attn_probs** (`Optional[bool]`): Return attention probabilities (default: False)
  - For debugging only

#### Returns

- **output** (`torch.Tensor`): Attention output of shape `(batch_size, seqlen_q, num_heads, head_dim)`
- **softmax_lse** (optional): Log-sum-exp of attention weights
- **attn_probs** (optional): Attention probabilities (if `return_attn_probs=True`)

## Packed Variants

### flash_dmattn_qkvpacked_func

Optimized function for QKV-packed tensors.

```python
def flash_dmattn_qkvpacked_func(
    qkv: torch.Tensor,                             # Packed QKV tensor
    attn_mask: Optional[torch.Tensor] = None,      # Attention mask
    attn_bias: Optional[torch.Tensor] = None,      # Attention bias
    dropout_p: Optional[float] = None,             # Dropout probability
    softmax_scale: Optional[float] = None,         # Scaling factor
    is_causal: Optional[bool] = None,              # Causal masking
    softcap: Optional[float] = None,               # Soft capping
    deterministic: Optional[bool] = None,          # Deterministic mode
    return_attn_probs: Optional[bool] = None,      # Return attention weights
) -> torch.Tensor
```

**Parameters:**
- **qkv** (`torch.Tensor`): Packed tensor of shape `(batch_size, seqlen, 3, num_heads, head_dim)`
  - Contains query, key, and value tensors stacked along dimension 2

### flash_dmattn_kvpacked_func

Optimized function for KV-packed tensors.

```python
def flash_dmattn_kvpacked_func(
    q: torch.Tensor,                               # Query tensor
    kv: torch.Tensor,                              # Packed KV tensor
    attn_mask: Optional[torch.Tensor] = None,      # Attention mask
    attn_bias: Optional[torch.Tensor] = None,      # Attention bias
    dropout_p: Optional[float] = None,             # Dropout probability
    softmax_scale: Optional[float] = None,         # Scaling factor
    is_causal: Optional[bool] = None,              # Causal masking
    softcap: Optional[float] = None,               # Soft capping
    deterministic: Optional[bool] = None,          # Deterministic mode
    return_attn_probs: Optional[bool] = None,      # Return attention weights
) -> torch.Tensor
```

**Parameters:**
- **q** (`torch.Tensor`): Query tensor of shape `(batch_size, seqlen_q, num_heads, head_dim)`
- **kv** (`torch.Tensor`): Packed tensor of shape `(batch_size, seqlen_k, 2, num_heads_k, head_dim)`

## Variable Length Functions

### flash_dmattn_varlen_func

Attention function supporting variable sequence lengths within a batch.

```python
def flash_dmattn_varlen_func(
    q: torch.Tensor,                               # Query tensor
    k: torch.Tensor,                               # Key tensor
    v: torch.Tensor,                               # Value tensor
    attn_mask: Optional[torch.Tensor] = None,      # Attention mask
    attn_bias: Optional[torch.Tensor] = None,      # Attention bias
    cu_seqlens_q: torch.Tensor = None,             # Cumulative sequence lengths (query)
    cu_seqlens_k: torch.Tensor = None,             # Cumulative sequence lengths (key)
    max_seqlen_q: int = None,                      # Maximum query sequence length
    max_seqlen_k: int = None,                      # Maximum key sequence length
    dropout_p: Optional[float] = None,             # Dropout probability
    softmax_scale: Optional[float] = None,         # Scaling factor
    is_causal: Optional[bool] = None,              # Causal masking
    softcap: Optional[float] = None,               # Soft capping
    deterministic: Optional[bool] = None,          # Deterministic mode
    return_attn_probs: Optional[bool] = None,      # Return attention weights
    block_table: Optional[torch.Tensor] = None,    # Block table for paged attention
) -> torch.Tensor
```

**Additional Parameters:**
- **cu_seqlens_q** (`torch.Tensor`): Cumulative sequence lengths for queries, shape `(batch_size + 1,)`
- **cu_seqlens_k** (`torch.Tensor`): Cumulative sequence lengths for keys, shape `(batch_size + 1,)`
- **max_seqlen_q** (`int`): Maximum sequence length in the batch for queries
- **max_seqlen_k** (`int`): Maximum sequence length in the batch for keys
- **block_table** (`Optional[torch.Tensor]`): Block table for paged attention (experimental)

## Backend Selection

### Available Backends

```python
from flash_dmattn import get_available_backends, CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE

# Check which backends are available
backends = get_available_backends()
print(f"Available: {backends}")

# Check individual backend availability
print(f"CUDA: {CUDA_AVAILABLE}")
print(f"Triton: {TRITON_AVAILABLE}")  
print(f"Flex: {FLEX_AVAILABLE}")
```

### Backend-Specific Functions

```python
# Direct access to specific implementations
from flash_dmattn import flash_dmattn_func        # CUDA backend
from flash_dmattn import triton_dmattn_func       # Triton backend  
from flash_dmattn import flex_dmattn_func         # Flex Attention backend
```

### Data Types and Memory Layout

- **Supported dtypes**: `torch.float16`, `torch.bfloat16`
- **Recommended**: `torch.bfloat16` for better numerical stability
- **Device**: CUDA tensors only
- **Memory**: All tensors must be contiguous in the last dimension

### Basic Usage Examples

#### Standard Attention

```python
import torch
from flash_dmattn import flash_dmattn_func

# Setup
batch_size, seq_len, num_heads, head_dim = 2, 4096, 12, 128
device = torch.device('cuda')
dtype = torch.bfloat16

# Create input tensors
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)  
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

# Basic attention
output = flash_dmattn_func(q=q, k=k, v=v, is_causal=True)
print(f"Output shape: {output.shape}")  # [2, 4096, 12, 128]
```

#### Dynamic Mask Attention

```python
import torch
from flash_dmattn import flash_dmattn_func
import math

# Create attention mask and bias for dynamic masking
batch_size, num_heads, seq_len = 2, 12, 4096
keep_window_size = 1024
device = torch.device('cuda')
dtype = torch.bfloat16

# Create sparse attention mask (attend to top-k positions)
attention_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype)
attention_mask = torch.zeros_like(attention_bias)

# Keep top-k positions per query
if seq_len > keep_window_size:
    topk_indices = torch.topk(attention_bias, keep_window_size, dim=-1, largest=True).indices
    attention_mask.scatter(-1, topk_indices, 1.0)
else:
    attention_mask.fill_(1.0)

# Run attention with dynamic masking
output = flash_dmattn_func(
    q=q, k=k, v=v,
    attn_mask=attention_mask,
    attn_bias=attention_bias,
    is_causal=True,
    softmax_scale=1.0/math.sqrt(head_dim)
)
```

#### Grouped-Query Attention (GQA)

```python
import torch
from flash_dmattn import flash_dmattn_func

# GQA setup: fewer key/value heads than query heads
batch_size, seq_len, num_heads, num_kv_heads, head_dim = 2, 2048, 32, 8, 128
device = torch.device('cuda')
dtype = torch.bfloat16

q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

# Attention mask for GQA
attn_mask = torch.ones(batch_size, num_kv_heads, seq_len, seq_len, device=device, dtype=dtype)

output = flash_dmattn_func(q=q, k=k, v=v, attn_mask=attn_mask, is_causal=True)
```

#### Variable Length Sequences

```python
import torch
from flash_dmattn import flash_dmattn_varlen_func

# Variable length setup
batch_size = 3
seq_lens = [512, 1024, 768]  # Different lengths per batch
total_tokens = sum(seq_lens)
num_heads, head_dim = 16, 64
device = torch.device('cuda')
dtype = torch.bfloat16

# Concatenated tensors
q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

# Cumulative sequence lengths
cu_seqlens = torch.tensor([0] + seq_lens, device=device, dtype=torch.int32).cumsum(0)

# Variable length attention
output = flash_dmattn_varlen_func(
    q=q, k=k, v=v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max(seq_lens),
    max_seqlen_k=max(seq_lens),
    is_causal=True
)
```

### Performance Optimization

#### Memory Efficiency

```python
# Use gradient checkpointing for long sequences
import torch.utils.checkpoint as checkpoint

def attention_checkpoint(q, k, v, *args, **kwargs):
    return checkpoint.checkpoint(flash_dmattn_func, q, k, v, *args, **kwargs)

# Process very long sequences in chunks
def chunked_attention(q, k, v, chunk_size=8192):
    seq_len = q.shape[1]
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        q_chunk = q[:, i:i+chunk_size]
        output_chunk = flash_dmattn_func(q=q_chunk, k=k, v=v, is_causal=True)
        outputs.append(output_chunk)
    
    return torch.cat(outputs, dim=1)
```

#### Backend Selection for Performance

```python
from flash_dmattn import flash_dmattn_func_auto

# Automatic selection (CUDA > Triton > Flex)
output = flash_dmattn_func_auto(q=q, k=k, v=v)

# Force specific backend for performance testing
backends = ["cuda", "triton", "flex"]
for backend in backends:
    try:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = flash_dmattn_func_auto(backend=backend, q=q, k=k, v=v)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time)
        print(f"{backend}: {elapsed:.2f} ms")
    except RuntimeError as e:
        print(f"{backend}: not available - {e}")
```

### Common Issues and Solutions

#### Import Errors

```python
# Test basic import
try:
    from flash_dmattn import flash_dmattn_func, get_available_backends
    print("✅ Flash Dynamic Mask Attention imported successfully")
    print(f"Available backends: {get_available_backends()}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure the package is properly installed with: pip install -e .")
```

#### Performance Issues

1. **Slow Execution**
   - Ensure tensors are contiguous and on the same GPU
   - Use optimal head dimensions (multiples of 8)
   - Check that CUDA backend is being used
   
2. **High Memory Usage**
   - Use gradient checkpointing for training
   - Process sequences in chunks for very long sequences
   - Consider using variable length functions for batches with mixed lengths

3. **Numerical Instability**
   - Use `torch.bfloat16` instead of `torch.float16`
   - Check attention mask and bias values for NaN/Inf
   - Monitor gradient norms during training

#### Debugging

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check intermediate values  
output = flash_dmattn_func(
    q=q, k=k, v=v,
    attn_mask=attn_mask,
    attn_bias=attn_bias,
    return_attn_probs=True  # Get attention weights for debugging
)

if isinstance(output, tuple):
    attn_output, softmax_lse, attn_weights = output
    print(f"Attention weights range: [{attn_weights.min():.6f}, {attn_weights.max():.6f}]")
    print(f"LSE stats: mean={softmax_lse.mean():.6f}, std={softmax_lse.std():.6f}")
else:
    attn_output = output

# Check for NaN values
if torch.isnan(attn_output).any():
    print("⚠️ NaN detected in attention output")
```

#### Memory Monitoring

```python
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        print(f"GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB max allocated")

# Monitor memory usage
print_memory_stats()
output = flash_dmattn_func(q=q, k=k, v=v)
print_memory_stats()

# Clear cache if needed
torch.cuda.empty_cache()
```

<!-- ### Backward -->
<!-- TODO -->

---

For more information, see the [integration documentation](integration.md) and [benchmarking results](../benchmarks/).
