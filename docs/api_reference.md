# Flash Dynamic Mask Attention API Reference

## Overview

Flash Dynamic Mask Attention is a high-performance attention implementation that combines the memory efficiency of Flash Attention with the sparse compute benefits of Dynamic Mask Attention. It supports CUDA, Triton, and Flex Attention backends and dynamic masking for very long sequences.

Interfaces provided:
- High-level: simple entry point with automatic backend selection
- Backend-specific: direct access to CUDA, Triton, and Flex implementations
- Packed variants: optimized paths for QKV-packed and KV-packed inputs
- Variable length: support for batches with different sequence lengths

## Table of Contents

1. [Installation](#installation)
2. [High-Level Interface](#high-level-interface)
3. [Core Functions](#core-functions)
4. [Packed Variants](#packed-variants)
5. [Variable Length Functions](#variable-length-functions)
6. [Backend Selection](#backend-selection)

## Installation

### Prerequisites

- Python: 3.8+
- PyTorch: 2.0.0+ with CUDA
- CUDA: 11.8+
- NVIDIA GPU: Compute Capability 8.0+
- Dependencies: `packaging`, `torch`

### Install from Source

```bash
git clone https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn
git submodule update --init --recursive
pip install -e .
```

## High-Level Interface

### Automatic Backend Selection

Note: `flash_dmattn_func_auto` returns a callable attention function, not the attention output.

```python
from flash_dmattn import flash_dmattn_func_auto, get_available_backends

# Check available backends
backends = get_available_backends()
print(f"Available backends: {backends}")

# Auto-select (priority: cuda > triton > flex)
attn = flash_dmattn_func_auto()
output = attn(q, k, v, attn_mask=attention_mask, attn_bias=attention_bias, is_causal=True, scale=None)

# Force a specific backend
attn = flash_dmattn_func_auto(backend="cuda")  # or "triton", "flex"
output = attn(q, k, v, attn_mask=attention_mask, attn_bias=attention_bias, is_causal=True, scale=None)
```

## Core Functions

### flash_dmattn_func (CUDA backend)

Main attention function. Supports multi-head and grouped-query attention (when the number of KV heads is smaller than the number of Q heads). Requires the CUDA extension to be built and available.

```python
def flash_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_kv_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_kv_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    scale: Optional[float] = None,                  # score scaling, defaults to 1/sqrt(head_dim)
    is_causal: Optional[bool] = None,               # causal mask
    softcap: Optional[float] = None,                # CUDA-only
    deterministic: Optional[bool] = None,           # CUDA-only
) -> torch.Tensor
```

#### Parameters

- query: (B, Q, H, D). CUDA tensor, fp16/bf16, last dim contiguous
- key: (B, K, H_kv, D). Same dtype/device as query; GQA when H_kv <= H
- value: (B, K, H_kv, D). Same dtype/device as query; GQA when H_kv <= H
- attn_mask: (B, H, Q, K). 1.0 = visible, 0.0 = masked. None to disable
- attn_bias: (B, H, Q, K). Added to scores before softmax. None to disable
- scale: score scaling; default 1/sqrt(D)
- is_causal: apply lower-triangular mask
- softcap, deterministic: only effective on the CUDA backend; ignored on others

#### Returns

- output: (B, Q, H, D)

## Packed Variants (CUDA backend)

### flash_dmattn_qkvpacked_func

Optimized function for QKV-packed input.

```python
def flash_dmattn_qkvpacked_func(
    qkv: torch.Tensor,                             # (batch, seqlen, 3, num_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,               # CUDA-only
    deterministic: Optional[bool] = None,          # CUDA-only
) -> torch.Tensor
```

### flash_dmattn_kvpacked_func

Optimized function for KV-packed input.

```python
def flash_dmattn_kvpacked_func(
    q: torch.Tensor,                               # (batch, seqlen_q, num_heads, head_dim)
    kv: torch.Tensor,                              # (batch, seqlen_k, 2, num_kv_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,               # CUDA-only
    deterministic: Optional[bool] = None,          # CUDA-only
) -> torch.Tensor
```

## Variable Length Functions (CUDA backend)

### flash_dmattn_varlen_func

Variable length attention for batches with mixed sequence lengths.

```python
def flash_dmattn_varlen_func(
    query: torch.Tensor,                            # (total_q, H, D) or (B, Q, H, D)
    key: torch.Tensor,                              # same layout as query
    value: torch.Tensor,                            # same layout as query
    attn_mask: Optional[torch.Tensor] = None,       # (B, H, Q, K)
    attn_bias: Optional[torch.Tensor] = None,       # (B, H, Q, K)
    cu_seqlens_q: torch.Tensor = None,              # (B+1,)
    cu_seqlens_k: torch.Tensor = None,              # (B+1,)
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,                # CUDA-only
    deterministic: Optional[bool] = None,           # CUDA-only
    block_table: Optional[torch.Tensor] = None,     # experimental: paged attention
) -> torch.Tensor
```

- cu_seqlens_q/k: cumulative sequence lengths for query/key
- max_seqlen_q/k: max sequence lengths per batch
- block_table: experimental support for paged attention

## Backend Selection

### Available Backends

```python
from flash_dmattn import get_available_backends, CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE

print(get_available_backends())   # e.g., ["cuda", "triton", "flex"]
print(CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE)
```

### Backend-Specific Functions

```python
# Direct access to specific backends
from flash_dmattn import flash_dmattn_func        # CUDA backend (requires compiled extension)
from flash_dmattn import triton_dmattn_func       # Triton backend
from flash_dmattn import flex_dmattn_func         # Flex Attention backend

# Unified call signature (public layer)
# query/key/value: (B, L{q/k}, H, D)
# attn_mask/attn_bias: (B, H, Lq, Lk)
# is_causal: bool, scale: Optional[float]
output = triton_dmattn_func(q, k, v, attn_mask=mask, attn_bias=bias, is_causal=True, scale=None)
output = flex_dmattn_func(q, k, v, attn_mask=mask, attn_bias=bias, is_causal=True, scale=None)
```

Notes:
- Triton returns only the attention output tensor.
- Flex currently uses causal masking and score_mod with bias; provided attn_mask is not applied in the kernel at the moment (subject to change in future versions).

### Data Types and Memory Layout

- dtypes: `torch.float16`, `torch.bfloat16` (bf16 recommended for stability)
- device: CUDA tensors only
- memory: last dimension must be contiguous (`stride(-1) == 1`); call `.contiguous()` if needed

## Basic Usage Examples

Prefer the high-level automatic interface for cross-backend portability.

### Standard Attention

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

B, L, H, D = 2, 4096, 12, 128
device = torch.device('cuda')
dtype = torch.bfloat16

q = torch.randn(B, L, H, D, device=device, dtype=dtype)
k = torch.randn(B, L, H, D, device=device, dtype=dtype)
v = torch.randn(B, L, H, D, device=device, dtype=dtype)

attn = flash_dmattn_func_auto()
output = attn(q, k, v, is_causal=True)
print(output.shape)  # [2, 4096, 12, 128]
```

### Dynamic Mask Attention

```python
import torch, math
from flash_dmattn import flash_dmattn_func_auto

B, H, L = 2, 12, 4096
keep_window_size = 1024
device = torch.device('cuda')
dtype = torch.bfloat16

q = torch.randn(B, L, H, 128, device=device, dtype=dtype)
k = torch.randn(B, L, H, 128, device=device, dtype=dtype)
v = torch.randn(B, L, H, 128, device=device, dtype=dtype)

attention_bias = torch.randn(B, H, L, L, device=device, dtype=dtype)
attention_mask = torch.zeros_like(attention_bias)

if L > keep_window_size:
    topk_indices = torch.topk(attention_bias, keep_window_size, dim=-1, largest=True).indices
    attention_mask.scatter_(-1, topk_indices, 1.0)
else:
    attention_mask.fill_(1.0)

attn = flash_dmattn_func_auto()
output = attn(q, k, v, attn_mask=attention_mask, attn_bias=attention_bias, is_causal=True, scale=1.0/math.sqrt(128))
```

### Grouped-Query Attention (GQA)

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

B, L, H, H_kv, D = 2, 2048, 32, 8, 128
device = torch.device('cuda')
dtype = torch.bfloat16

q = torch.randn(B, L, H, D, device=device, dtype=dtype)
k = torch.randn(B, L, H_kv, D, device=device, dtype=dtype)
v = torch.randn(B, L, H_kv, D, device=device, dtype=dtype)

attn_mask = torch.ones(B, H, L, L, device=device, dtype=dtype)

attn = flash_dmattn_func_auto()
output = attn(q, k, v, attn_mask=attn_mask, is_causal=True)
```

### Variable Length Sequences (CUDA backend)

```python
import torch
from flash_dmattn import flash_dmattn_varlen_func

B = 3
seq_lens = [512, 1024, 768]
T = sum(seq_lens)
H, D = 16, 64
device = torch.device('cuda')
dtype = torch.bfloat16

q = torch.randn(T, H, D, device=device, dtype=dtype)
k = torch.randn(T, H, D, device=device, dtype=dtype)
v = torch.randn(T, H, D, device=device, dtype=dtype)

cu = torch.tensor([0] + seq_lens, device=device, dtype=torch.int32).cumsum(0)

output = flash_dmattn_varlen_func(
    q=q, k=k, v=v,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=max(seq_lens), max_seqlen_k=max(seq_lens),
    is_causal=True
)
```

## Performance Optimization

### Memory Efficiency

```python
# Gradient checkpointing for long sequences
import torch.utils.checkpoint as checkpoint
from flash_dmattn import flash_dmattn_func_auto

attn = flash_dmattn_func_auto()

def attention_checkpoint(q, k, v, *args, **kwargs):
    return checkpoint.checkpoint(lambda *a, **kw: attn(*a, **kw), q, k, v, *args, **kwargs)

# Process very long sequences in chunks
def chunked_attention(q, k, v, chunk_size=8192, **kwargs):
    L = q.shape[1]
    outs = []
    for i in range(0, L, chunk_size):
        outs.append(attn(q[:, i:i+chunk_size], k, v, **kwargs))
    return torch.cat(outs, dim=1)
```

### Backend Selection for Performance

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

backends = ["cuda", "triton", "flex"]
for backend in backends:
    try:
        attn = flash_dmattn_func_auto(backend=backend)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = attn(q, k, v, is_causal=True)
        end.record()
        torch.cuda.synchronize()
        print(f"{backend}: {start.elapsed_time(end):.2f} ms")
    except RuntimeError as e:
        print(f"{backend}: not available - {e}")
```

## Common Issues and Solutions

### Import Errors

```python
try:
    from flash_dmattn import flash_dmattn_func_auto, get_available_backends
    print("✅ Imported successfully", get_available_backends())
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please install with: pip install -e .")
```

### Performance Issues

1. Slow execution: ensure all tensors are on the same GPU and last dim is contiguous; use head dims multiple of 8; prefer CUDA backend when available
2. High memory: use gradient checkpointing; chunk long sequences; use varlen for mixed-length batches
3. Numerical stability: prefer bfloat16; check mask/bias for NaN/Inf; monitor gradient norms

### Debugging

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

torch.autograd.set_detect_anomaly(True)
attn = flash_dmattn_func_auto()
output = attn(q, k, v, attn_mask=attn_mask, attn_bias=attn_bias, is_causal=True)
if torch.isnan(output).any():
    print("⚠️ NaN detected in attention output")
```

### Memory Monitoring

```python
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"max alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print_memory_stats()
attn = flash_dmattn_func_auto()
output = attn(q, k, v)
print_memory_stats()

torch.cuda.empty_cache()
```

