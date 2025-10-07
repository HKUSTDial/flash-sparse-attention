# Flash Dynamic Mask Attention API Reference


## Overview

Flash Dynamic Mask Attention is a high-performance attention implementation that combines the memory efficiency of Flash Attention with the sparse compute benefits of Dynamic Mask Attention. It supports CUDA, Triton, and Flex Attention backends and dynamic masking for very long sequences.


## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Backend Selection and Comparison](#backend-selection-and-comparison)
4. [API Reference](#api-reference)
   - [CUDA Backend: flash_dmattn_func](#flash_dmattn_func-cuda-backend)
   - [Triton Backend: triton_dmattn_func](#triton_dmattn_func-triton-backend)
   - [Flex Backend: flex_dmattn_func](#flex_dmattn_func-flex-backend)
5. [Integrations](#integrations)
   - [Transformers Integration](#transformers-integration)
6. [Common Issues and Solutions](#common-issues-and-solutions)


## Installation

Please refer to the [README](https://github.com/SmallDoges/flash-dmattn/blob/main/README.md#install) for detailed installation instructions.

```bash
# With CUDA backend
pip install flash-dmattn

# Or install from source
pip install -e .

# Triton/Flex only
FLASH_DMATTN_SKIP_CUDA_BUILD=1 pip install -e .
```


## Quick Start

Use `flash_dmattn_func_auto` to automatically select the best available backend without manual checking.

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

# Prepare input tensors
batch, seqlen, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

# Get attention function (auto-select backend, priority: cuda > triton > flex)
attn_func = flash_dmattn_func_auto()

# Compute attention
output = attn_func(q, k, v, is_causal=True)
print(f"Output shape: {output.shape}")  # (2, 1024, 8, 64)

# Or force a specific backend
attn_func = flash_dmattn_func_auto(backend="cuda")  # or "triton", "flex"
output = attn_func(q, k, v, is_causal=True)
```

> [!NOTE]
> `flash_dmattn_func_auto` returns a callable attention function, not the attention output.


## Backend Selection and Comparison

### Check Available Backends

```python
from flash_dmattn import get_available_backends, CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE

# List all available backends
print(get_available_backends())  # e.g., ["cuda", "triton", "flex"]

# Check specific backend availability
print(f"CUDA: {CUDA_AVAILABLE}, Triton: {TRITON_AVAILABLE}, Flex: {FLEX_AVAILABLE}")
```

### Backend Feature Comparison

| Feature | CUDA | Triton | Flex |
|---------|------|--------|------|
| **Performance** | Highest | Good | Good |
| **Memory Efficiency** | Best | Good | Good |
| **Build Requirements** | Custom CUDA extension | triton package | transformers package |
| **GQA Support** | ✅ | ✅ | ✅ |
| **Attention Mask** | ✅ | ✅ | ⚠️ |
| **Attention Bias** | ✅ | ✅ | ✅ |
| **Causal Mask** | ✅ | ✅ | ✅ |
| **Softcap** | ✅ | ❌ | ❌ |
| **Deterministic** | ✅ | ❌ | ❌ |
| **Return Attention Probs** | ✅ | ❌ | ❌ |
| **Backward Support** | ✅ | ✅ | ⚠️ |

> [!NOTE]
> ✅ Fully supported | ⚠️ Limited support | ❌ Not supported

### When to Use Each Backend

**CUDA Backend** ([details](#flash_dmattn_func-cuda-backend))
- ✅ Training workloads requiring full gradient support
- ✅ Production inference requiring maximum performance
- ✅ Applications needing deterministic behavior
- ❌ Avoid: when custom CUDA extensions cannot be built

**Triton Backend** ([details](#triton_dmattn_func-triton-backend))
- ✅ Training when CUDA extension unavailable
- ✅ Development and prototyping
- ✅ Cross-platform compatibility needs
- ✅ Good balance of performance and ease of installation

**Flex Backend** ([details](#flex_dmattn_func-flex-backend))
- ✅ Inference-only applications
- ✅ Research with latest PyTorch features
- ✅ Quick experimentation without custom builds
- ❌ Avoid: training (limited backward support)
- ❌ Avoid: when strict attention mask compliance required

### Import Available Functions

```python
from flash_dmattn import (
    # Automatic backend selection
    get_available_backends,
    flash_dmattn_func_auto,
    
    # Backend-specific functions
    flash_dmattn_func,      # CUDA backend
    triton_dmattn_func,     # Triton backend
    flex_dmattn_func,       # Flex backend
    
    # Backend availability flags
    CUDA_AVAILABLE,
    TRITON_AVAILABLE,
    FLEX_AVAILABLE,
)

# Transformers integration
from flash_dmattn.integrations.flash_dynamic_mask_attention import (
    flash_dynamic_mask_attention_forward
)
```


## API Reference

### flash_dmattn_func (CUDA backend)

Main attention function. Supports multi-head and grouped-query attention (when the number of KV heads is smaller than the number of Q heads). Requires the CUDA extension to be built and available.

```python
def flash_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_kv_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_kv_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, {num_heads, num_kv_heads, 1}, {seqlen_q, 0}, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, {num_heads, num_kv_heads, 1}, {seqlen_q, 0}, seqlen_k)
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
- attn_mask: (B, {H, H_kv, 1}, {Q, 0}, K). 1.0 = visible, 0.0 = masked. None to disable
- attn_bias: (B, {H, H_kv, 1}, {Q, 0}, K). Added to scores before softmax. None to disable
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


## Integrations

### Transformers Integration

Integration function for HuggingFace Transformers models that provides seamless flash dynamic mask attention support.

#### flash_dynamic_mask_attention_forward


```python
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

def flash_dynamic_mask_attention_forward(
    module: torch.nn.Module,                        # The attention module
    query: torch.Tensor,                            # (batch_size, num_heads, query_len, head_dim)
    key: torch.Tensor,                              # (batch_size, num_kv_heads, key_len, head_dim)
    value: torch.Tensor,                            # (batch_size, num_kv_heads, key_len, head_dim)
    attention_mask: Optional[torch.Tensor],         # (batch_size, {num_heads, num_kv_heads, 1}, {query_len, 0}, key_len)
    attention_bias: Optional[torch.Tensor],         # (batch_size, {num_heads, num_kv_heads, 1}, {query_len, 0}, key_len)
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
- attention_mask: Boolean attention mask (B, {H, H_kv, 1}, {Q, 0}, K)
- attention_bias: Attention bias to add to scores (B, {H, H_kv, 1}, {Q, 0}, K)
- scaling: Score scaling factor
- softcap: Softcap value for attention scores
- **kwargs: Additional arguments including:
  - is_causal: Whether to apply causal mask
  - keep_window_size: Size of window to keep
  - layer_idx: Layer index for logging
  - implementation: Implementation to use ("flash_dmattn" or None)

#### Returns

- tuple[torch.Tensor, None]: Output tensor (B, Q, H, D) and None for compatibility

#### Usage Example

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
            # sin and cos are specific to RoPE models; static cache needs cache_position
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Sample dt_states from value_states to generate attention_bias
        dt_states = self.dt_proj(
            value_states.transpose(1, 2).reshape(value_states.shape[0], value_states.shape[-2], -1)
        )
        attn_bias = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2).to(hidden_states.dtype)

        # Choose attention implementation
        attention_interface: Callable = flash_dynamic_mask_attention_forward
        
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
- **Flexible backend selection**: Easily switch attention implementations via `attention_interface`
- **Proper tensor reshaping**: Converting between different tensor layouts as needed
- **Integration with caching**: Support for key-value caching in generation scenarios


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

