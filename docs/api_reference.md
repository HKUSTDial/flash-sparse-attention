# Flash Dynamic Mask Attention API Reference

## Overview

Flash Dynamic Mask Attention is a high-performance implementation that combines Flash Attention's memory efficiency with Dynamic Mask Attention's sparse computation capabilities. This API provides CUDA-accelerated attention computation with dynamic masking for handling extremely long sequences efficiently.

## Table of Contents

1. [Installation](#installation)
2. [Forward](#forward)

## Installation

### Prerequisites

- CUDA >= 11.8
- PyTorch >= 2.0
- CUTLASS library
- GPU with compute capability >= 8.0 (Ampere architecture or newer)

### Build from Source

```bash
cd flash_dma
pip install -e .
```

## Forward

```python
def fwd(
    q: torch.Tensor,                      # Query tensor
    k: torch.Tensor,                      # Key tensor  
    v: torch.Tensor,                      # Value tensor
    zoh: torch.Tensor,                    # ZOH states tensor
    active_mask: torch.Tensor,            # Active mask tensor
    out: Optional[torch.Tensor] = None,   # Output tensor (optional)
    p_dropout: float = 0.0,               # Dropout probability
    softmax_scale: float = None,          # Scaling factor for attention
    is_causal: bool = False,              # Whether to apply causal mask
    keep_window_size: int = 2048,         # Window size for dynamic masking
    softcap: float = 0.0,                 # Soft capping for attention scores
    return_softmax: bool = False,         # Whether to return softmax weights
    gen: Optional[torch.Generator] = None # Random generator for dropout
) -> List[torch.Tensor]
```

#### Parameters

- **q** (`torch.Tensor`): Query tensor of shape `(batch_size, seqlen_q, num_heads, head_dim)`
  - Must be contiguous in the last dimension
  - Supported dtypes: `torch.float16`, `torch.bfloat16`
  - Must be on CUDA device

- **k** (`torch.Tensor`): Key tensor of shape `(batch_size, seqlen_k, num_heads_k, head_dim)`
  - Must be contiguous in the last dimension
  - Same dtype and device as `q`
  - `num_heads_k` can be different from `num_heads` for grouped-query attention

- **v** (`torch.Tensor`): Value tensor of shape `(batch_size, seqlen_k, num_heads_k, head_dim)`
  - Must be contiguous in the last dimension
  - Same dtype and device as `q`

- **zoh** (`torch.Tensor`): Zero-Order Hold states tensor of shape `(batch_size, num_heads_k, seqlen_q, seqlen_k)`
  - Contains the dynamic attention bias values
  - Same dtype and device as `q`
  - Used for dynamic masking computation

- **active_mask** (`torch.Tensor`): Active mask tensor of shape `(batch_size, num_heads_k, seqlen_q, seqlen_k)`
  - Binary mask indicating which positions should be computed
  - Same dtype and device as `q`
  - 1.0 for active positions, 0.0 for masked positions

- **out** (`Optional[torch.Tensor]`): Pre-allocated output tensor
  - If provided, must have shape `(batch_size, seqlen_q, num_heads, head_dim)`
  - If `None`, will be allocated automatically

- **p_dropout** (`float`): Dropout probability (default: 0.0)
  - Range: [0.0, 1.0]
  - Applied to attention weights

- **softmax_scale** (`float`): Scaling factor for attention scores
  - If `None`, defaults to `1.0 / sqrt(head_dim)`
  - Applied before softmax

- **is_causal** (`bool`): Whether to apply causal (lower triangular) mask (default: False)
  - Combined with dynamic masking

- **keep_window_size** (`int`): Maximum number of tokens to keep per query (default: 2048)
  - Controls sparsity level of attention
  - Dynamic masking only applied when `seqlen_k > keep_window_size`

- **softcap** (`float`): Soft capping value for attention scores (default: 0.0)
  - If > 0, applies `softcap * tanh(score / softcap)`

- **return_softmax** (`bool`): Whether to return attention weights (default: False)
  - Only for debugging purposes

- **gen** (`Optional[torch.Generator]`): Random number generator for dropout
  - Used for reproducible dropout

#### Returns

Returns a list of tensors:
- `output`: Attention output of shape `(batch_size, seqlen_q, num_heads, head_dim)`
- `softmax_lse`: Log-sum-exp of attention weights for numerical stability
- `p`: Attention weights (if `return_softmax=True`)
- `rng_state`: Random number generator state

### Data Types

- **Supported**: `torch.float16`, `torch.bfloat16`
- **Recommended**: `torch.bfloat16` for better numerical stability
- All input tensors must have the same dtype

### Memory Layout

- All tensors must be contiguous in the last dimension
- CUDA tensors only
- Optimal performance with tensors already on the same GPU

### Basic Usage

```python
import torch
import flash_dma
import math

# Setup
batch_size, seq_len, num_heads, head_dim = 2, 4096, 12, 128
device = torch.device('cuda')
dtype = torch.bfloat16

# Create input tensors
query = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                   device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_heads, head_dim,
                 device=device, dtype=dtype)  
value = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device=device, dtype=dtype)

# Prepare ZOH states and active mask
zoh_states = torch.randn(batch_size, num_heads, seq_len, seq_len,
                        device=device, dtype=dtype)
active_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len,
                        device=device, dtype=dtype)

# Apply sparsity (keep top-k per row)
keep_window_size = 1024
if seq_len > keep_window_size:
    # Select top-k most important keys for each query
    topk_indices = torch.topk(zoh_states, keep_window_size, dim=-1, 
                             largest=True, sorted=False).indices
    active_mask.scatter(-1, topk_indices, 1.0)

# Run attention
output = flash_dma.fwd(
    q=query,
    k=key, 
    v=value,
    zoh=zoh_states,
    active_mask=active_mask,
    softmax_scale=1.0/math.sqrt(head_dim),
    keep_window_size=keep_window_size
)[0]

print(f"Output shape: {output.shape}")
```

### Performance Issues

1. **Slow Execution**
   - Ensure tensors are contiguous
   - Use optimal head dimensions (multiples of 8)
   - Check GPU utilization with `nvidia-smi`

2. **High Memory Usage**
   - Reduce `keep_window_size`
   - Use gradient checkpointing
   - Process sequences in chunks

3. **Numerical Instability**
   - Use `torch.bfloat16` instead of `torch.float16`
   - Check attention mask values
   - Monitor gradient norms

### Debug Mode

```python
# Enable debug output
torch.autograd.set_detect_anomaly(True)

# Check intermediate values
output, softmax_lse, attn_weights, _ = flash_dma.fwd(
    query, key, value, zoh_states, active_mask,
    return_softmax=True
)

print(f"Attention weights range: [{attn_weights.min()}, {attn_weights.max()}]")
print(f"LSE stats: mean={softmax_lse.mean()}, std={softmax_lse.std()}")
```

<!-- ### Backward -->
<!-- TODO -->

---

For more information, see the [integration documentation](integration.md) and [benchmarking results](../benchmarks/).
