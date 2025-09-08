# Flash Dynamic Mask Attention API Reference


## Overview

Flash Dynamic Mask Attention is a high-performance attention implementation that combines the memory efficiency of Flash Attention with the sparse compute benefits of Dynamic Mask Attention. It supports CUDA, Triton, and Flex Attention backends and dynamic masking for very long sequences.

Interfaces provided:
- High-level: simple entry point with automatic backend selection
- Backend-specific: direct access to CUDA, Triton, and Flex implementations
- Transformers Integration: seamless integration with HuggingFace Transformers models


## Table of Contents

1. [Installation](#installation)
2. [High-Level Interface](#high-level-interface)
3. [Core Functions](#core-functions)
4. [Transformers Integration](#transformers-integration)
5. [Backend Selection](#backend-selection)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Summary](#summary)


## Installation

### Prerequisites

- Python: 3.8+
- PyTorch: 2.0.0+ with CUDA
- CUDA: 11.8+ for CUDA backend
- NVIDIA GPU: Compute Capability 8.0+ for CUDA backend
- Optional: `triton` for Triton backend, `transformers` for Flex backend and integrations

### Install from Source

```bash
git clone https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn
MAX_JOBS=4 pip install . --no-build-isolation
```


## High-Level Interface

### Automatic Backend Selection

Note: `flash_dmattn_func_auto` returns a callable attention function, not the attention output.

```python
from flash_dmattn import get_available_backends, flash_dmattn_func_auto

# Check available backends
backends = get_available_backends()
print(f"Available backends: {backends}")

# Auto-select (priority: cuda > triton > flex)
dmattn_func = flash_dmattn_func_auto()
output = dmattn_func(q, k, v, attn_mask=attention_mask, attn_bias=attention_bias, is_causal=True, scale=None)

# Force a specific backend
dmattn_func = flash_dmattn_func_auto(backend="cuda")  # or "triton", "flex"
output = dmattn_func(q, k, v, attn_mask=attention_mask, attn_bias=attention_bias, is_causal=True, scale=None)
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
    return_attn_probs: Optional[bool] = None,       # CUDA-only, for testing
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
- softcap, deterministic, return_attn_probs: only effective on the CUDA backend; ignored on others

#### Returns

- output: (B, Q, H, D)

### triton_dmattn_func (Triton backend)

Triton-based implementation that provides good performance without requiring custom CUDA kernels.

```python
def triton_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    is_causal: bool = False,                        # causal mask
    scale: Optional[float] = None,                  # score scaling, defaults to 1/sqrt(head_dim)
) -> torch.Tensor
```

### flex_dmattn_func (Flex Attention backend)

Flex Attention-based implementation using PyTorch's native flex attention with dynamic masking support.

```python
def flex_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    is_causal: Optional[bool] = None,               # causal mask
    scale: Optional[float] = None,                  # score scaling, defaults to 1/sqrt(head_dim)
) -> torch.Tensor
```


## Transformers Integration

Integration function for HuggingFace Transformers models that provides seamless flash dynamic mask attention support.

### flash_dynamic_mask_attention_forward


```python
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

def flash_dynamic_mask_attention_forward(
    module: torch.nn.Module,                        # The attention module
    query: torch.Tensor,                            # (batch_size, num_heads, query_len, head_dim)
    key: torch.Tensor,                              # (batch_size, num_kv_heads, key_len, head_dim)
    value: torch.Tensor,                            # (batch_size, num_kv_heads, key_len, head_dim)
    attention_mask: Optional[torch.Tensor],         # (batch_size, num_kv_heads, query_len, key_len)
    attention_bias: Optional[torch.Tensor],         # (batch_size, num_kv_heads, query_len, key_len)
    scaling: Optional[float] = None,                # score scaling
    softcap: Optional[float] = None,                # softcap value
    **kwargs,
) -> tuple[torch.Tensor, None]
```

#### Parameters

- module: The attention module instance
- query: Query tensor with head-first layout (B, H, Q, D)
- key: Key tensor with head-first layout (B, H_kv, K, D)
- value: Value tensor with head-first layout (B, H_kv, K, D)
- attention_mask: Boolean attention mask
- attention_bias: Attention bias to add to scores
- scaling: Score scaling factor
- softcap: Softcap value for attention scores
- **kwargs: Additional arguments including:
  - is_causal: Whether to apply causal mask
  - keep_window_size: Size of window to keep
  - layer_idx: Layer index for logging
  - implementation: Implementation to use ("flash_dmattn" or None)

#### Returns

- tuple[torch.Tensor, None]: Output tensor (B, Q, H, D) and None for compatibility

### Usage with Transformers

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, tuple
from transformers.cache_utils import Cache
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

class DynamicMaskAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.keep_window_size = config.keep_window_size
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        # Dynamic mask for the QK^T attention weights matrix
        self.A = nn.Parameter(torch.zeros(config.num_key_value_heads))
        self.dt_proj = nn.Linear(
            config.num_key_value_heads * self.head_dim, config.num_key_value_heads, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = DogeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DogeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Sampling dt_states from value_states to generate attention bias
        dt_states = self.dt_proj(
            value_states.transpose(1, 2).reshape(value_states.shape[0], value_states.shape[-2], -1)
        )
        dt_states = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
        attn_bias = dt_states[:, :, None, :].expand(
            -1, -1, hidden_states.shape[1], -1
        ).to(hidden_states.dtype)  # [batch_size, num_heads, query_len, key_len]

        # Choose attention implementation: fallback to eager if flash_dmattn is not available
        attention_interface: Callable = eager_attention_forward
        if flash_dynamic_mask_attention_forward is not None:
            attention_interface = flash_dynamic_mask_attention_forward

        # Expand attention mask to match the expected shape
        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1, attn_bias.shape[1], -1, -1)
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attn_bias,
            scale=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

This example shows:
- **Dynamic attention bias generation**: Using learnable parameters to create attention bias
- **Flexible backend selection**: Graceful fallback to standard attention when flash_dmattn is unavailable
- **Proper tensor reshaping**: Converting between different tensor layouts as needed
- **Integration with caching**: Support for key-value caching in generation scenarios


## Backend Selection

### Available Backends

```python
from flash_dmattn import get_available_backends, CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE

print(get_available_backends())   # e.g., ["cuda", "triton", "flex"]
print(CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE)
```

### Available Functions

The library exports the following functions:

```python
from flash_dmattn import (
    # High-level interface
    get_available_backends,     # Get list of available backends
    flash_dmattn_func_auto,     # Automatic backend selection
    
    
    # Backend-specific functions
    flash_dmattn_func,          # CUDA backend (if available)
    triton_dmattn_func,         # Triton backend (if available)
    flex_dmattn_func,           # Flex Attention backend (if available)
    
    # Backend availability flags
    CUDA_AVAILABLE,
    TRITON_AVAILABLE,
    FLEX_AVAILABLE,
)

# Transformers integration
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward
```

### Backend-Specific Functions

```python
# Direct access to specific backends
from flash_dmattn import flash_dmattn_func        # CUDA backend
from flash_dmattn import triton_dmattn_func       # Triton backend
from flash_dmattn import flex_dmattn_func         # Flex Attention backend

# Unified call signature (public layer)
# query/key/value: (B, L{q/k}, H, D)
# attn_mask/attn_bias: (B, H, Lq, Lk)
# is_causal: bool, scale: Optional[float]
output = flash_dmattn_func(q, k, v, attn_mask=mask, attn_bias=bias, is_causal=True, scale=None)
output = triton_dmattn_func(q, k, v, attn_mask=mask, attn_bias=bias, is_causal=True, scale=None)
output = flex_dmattn_func(q, k, v, attn_mask=mask, attn_bias=bias, is_causal=True, scale=None)
```

Notes:
- All backends support the same unified interface for seamless switching
- Flex backend currently uses causal masking and score_mod with bias; provided attn_mask is not applied in the kernel at the moment, subject to change in future versions
- CUDA backend supports additional parameters like softcap, deterministic, and return_attn_probs

### When to Use Each Backend

**CUDA Backend:**
- ✅ Training workloads requiring full gradient support
- ✅ Production inference requiring maximum performance
- ✅ Applications needing deterministic behavior
- ❌ Avoid if you cannot build custom CUDA extensions

**Triton Backend:**
- ✅ Training workloads when CUDA extension is not available
- ✅ Development and prototyping
- ✅ Cross-platform compatibility needs
- ✅ Good balance of performance and ease of installation

**Flex Backend:**
- ✅ Inference-only applications
- ✅ Research with latest PyTorch features
- ✅ Quick experimentation without custom builds
- ❌ Avoid for training due to limited backward support
- ❌ Avoid when strict attention mask compliance is required


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
2. High memory: use gradient checkpointing; chunk long sequences; consider Triton or Flex backends for very long sequences
3. Numerical stability: prefer bfloat16; check mask/bias for NaN/Inf; monitor gradient norms

### Transformers Integration Issues

1. Model compatibility: ensure your model supports custom attention implementations
2. Shape mismatches: check that tensor layouts match expected formats
3. Gradient flow: verify that gradients flow properly through the custom attention function

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
```

## Summary

Flash Dynamic Mask Attention provides a unified interface for high-performance attention computation with the following key features:

- **Multiple Backends**: CUDA for best performance, Triton for good compatibility, and Flex Attention for native PyTorch support
- **Automatic Backend Selection**: Seamless fallback between available backends
- **Dynamic Masking**: Efficient sparse attention with arbitrary attention masks
- **GQA Support**: Grouped-query attention for efficient inference
- **Transformers Integration**: Direct integration with HuggingFace models
- **Memory Efficiency**: Optimized memory usage for very long sequences

Choose the backend that best fits your needs:
- **CUDA**: For maximum performance and full feature support, especially for training
- **Triton**: For good performance without custom CUDA compilation, supports both training and inference
- **Flex**: For inference scenarios and compatibility with latest PyTorch features, but limited backward support for training yet

### Backend Comparison

| Feature | CUDA | Triton | Flex |
|---------|------|--------|------|
| Performance | Highest | Good | Good |
| Memory Efficiency | Best | Good | Good |
| Build Requirements | Custom CUDA extension | triton package | transformers package |
| GQA Support | ✅ | ✅ | ✅ |
| Attention Mask | ✅ | ✅ | ⚠️ |
| Attention Bias | ✅ | ✅ | ✅ |
| Causal Mask | ✅ | ✅ | ✅ |
| Softcap | ✅ | ❌ | ❌ |
| Deterministic | ✅ | ❌ | ❌ |
| Return Attention Probs | ✅ | ❌ | ❌ |
| Backward Support | ✅ | ✅ | ⚠️ |

Notes:
- ✅ = Fully supported
- ⚠️ = Limited support or workarounds needed  
- ❌ = Not supported

