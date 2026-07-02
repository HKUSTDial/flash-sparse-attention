<div align="center">
  <img src="https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/logo.png" alt="flash-algo" width="100%">
</div>

<div align="center">


**English** | [简体中文](./README_zh.md)

</div>


Flash-Sparse-Attention is a high-performance trainable sparse attention implementation that combines Flash Attention's memory efficiency with sparse computation for handling extremely long sequences in Transformer models.


# Key Features

> [!NOTE]
> Support for arbitrary mask and bias shapes is available in [this branch](https://github.com/HKUSTDial/flash-sparse-attention/tree/final_mask_version). The current main branch no longer maintains that feature set.

## Supported Features

- Forward and backward passes for dense attention, sparse attention, and gated attention
- Regular batched inputs and varlen inputs
- Causal attention and local window attention
- Arbitrary combinations of Q and KV sequence lengths, with head dimensions up to 256
- Grouped Query Attention and Multi Query Attention
- Sparse softmax threshold control
- Gated attention with gate inputs and configurable gating sparsity
- Flex Local Window Attention with per-head arbitrary window sizes and local ranges
- Split-KV for workload balancing in forward and decode workloads
- Split-QO for workload balancing in backward workloads
- Fused Quant for low-precision computation on hardware without native FP8 support
- Top-k gather KV-cache decode
- Paged Attention

**For complete API documentation, please refer to [here](https://hkustdial.github.io/flash-sparse-attention/)**

## Features We Aim to Support

- KV-Cache Manager
- [TLE](https://github.com/flagos-ai/FlagTree/wiki/TLE) backend support
- [Gluon](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon) backend support


# Installation

## Requirements

- **Linux**: Ubuntu 22.04 or later
- **Device**: GPU, XPU, NPU, or PPU
- **Python**: 3.9 or later
- **PyTorch**: 2.5.1 or later
- **Triton**: 3.6.0 or later
- **Triton Kernels**: 3.6.0 or later

## Install

Install from PyPI:

```bash
pip install flash-sparse-attn
```

Additionally, install `triton_kernels`:

```bash
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@v3.6.0#subdirectory=python/triton_kernels"
```

To install from source (includes all dependencies automatically):

```bash
git clone https://github.com/flash-algo/flash-sparse-attn.git
cd flash-sparse-attn
pip install .
```


## Install via HuggingFace Kernel

You can also load the kernels directly from [HuggingFace Kernel](https://github.com/huggingface/kernels) without installing the package:

```python
from kernels import get_kernel

fsa = get_kernel("JingzeShi/flash-sparse-attn", version=1, trust_remote_code=True)

# Forward
out = fsa.flash_sparse_attn_func(q, k, v, is_causal=True)
# Backward
out.sum().backward()
# Decode
out = fsa.flash_sparse_attn_with_kvcache_func(q, k_cache, v_cache)
```


# Quick Start

## Basic Usage

Below are examples for forward, backward, and decode.

```python
import torch
from flash_sparse_attn.ops.triton.interface import (
    flash_sparse_attn_func,
    flash_sparse_attn_with_kvcache_func,
)

dtype = torch.bfloat16
device = torch.device("cuda")
batch_size, seqlen, num_heads, num_kv_heads, head_dim = 2, 4096, 32, 8, 128
```

### Forward

Combine flex window, split-KV, fused quant, and sparse softmax for maximum performance.

```python
query = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
key = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
value = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)

output = flash_sparse_attn_func(
    query, key, value,
    is_causal=True,
    softmax_threshold=128.0 / seqlen,
    is_local=True,
    is_quant=True,
    is_split_kv=True,
)
```

### Backward

Combine flex window, split-QO, fused quant, and low-contribution skipping for maximum backward performance.

```python
query = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
key = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
value = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)

output = flash_sparse_attn_func(
    query, key, value,
    is_causal=True,
    softmax_threshold=1.0 / seqlen,
    is_local=True,
    is_quant=True,
    is_split_kv=True,
    is_split_qo=True,
)

output.sum().backward()
```

### Decode

Combine flex window, split-KV, fused quant, sparse softmax, packed GQA, and Graph for maximum decode performance.

```python
query = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
key = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
value = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)

def fsa_decode_fn():
    return flash_sparse_attn_with_kvcache_func(
        query, key, value,
        softmax_threshold=128.0 / seqlen,
        is_local=True,
        is_quant=True,
    )

# Warmup
for _ in range(3):
    fsa_decode_fn()
torch.cuda.synchronize()

# Capture Graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = fsa_decode_fn()

# Replay
graph.replay()
```


# Performance

The following benchmarks cover forward, backward, and decode workloads, using FlashAttention as the baseline.

## NVIDIA GPU


### A100

**Forward Performance**

![Attention forward speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_a100sxm4.png)

**Backward Performance**

![Attention backward speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_a100sxm4.png)

**Decode Performance**

![Attention decode speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_a100sxm4.png)


### H20

**Forward Performance**

![Attention forward speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_h203e.png)

**Backward Performance**

![Attention backward speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_h203e.png)

**Decode Performance**

![Attention decode speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_h203e.png)


### RTX PRO 6000

**Forward Performance**

![Attention forward speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_rtxpro6000.png)

**Backward Performance**

![Attention backward speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_rtxpro6000.png)

**Decode Performance**

![Attention decode speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_rtxpro6000.png)


# Benchmarking

Benchmark scripts are located under [tests](tests/), covering forward, backward, and decoding performance.

## Forward Performance

```bash
python tests/benchmark_forward.py
```

## Backward Performance

```bash
python tests/benchmark_backward.py
```

## Decode Performance

```bash
python tests/benchmark_decode.py
```


# Citation

If you use FSA in your research, please cite:

```bibtex
@misc{shi2025trainabledynamicmasksparse,
      title={Trainable Dynamic Mask Sparse Attention},
      author={Jingze Shi and Yifan Wu and Bingheng Wu and Yiran Peng and Liangdong Wang and Guang Liu and Yuyu Luo},
      year={2025},
      eprint={2508.02124},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.02124},
}
```


# Acknowledgments

This project builds upon and integrates several excellent works:

- **[OpenSeek](https://github.com/FlagAI-Open/OpenSeek)** - Kernel development support
- **[Flash-Attention](https://github.com/Dao-AILab/flash-attention)** - Memory-efficient attention computation
- **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** - High-performance matrix operations library

We thank the open-source community for its contributions to efficient Transformer implementations. 🤗
