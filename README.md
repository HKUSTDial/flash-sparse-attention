<!-- <div align="center">
  <img src="./assets/logo.png" alt="SmallDoges" width="100%">
</div> -->

<div align="center">


**English** | [简体中文](./README_zh.md)

</div>


![Flash-Sparse-Attention Banner](assets/flash_sparse_attention_banner.png)

Flash-Sparse-Attention is a high-performance trainable sparse attention implementation that integrates Flash Attention's memory efficiency with Dynamic Mask Attention's sparse computation capabilities for processing extremely long sequences in transformer models.


## Why Flash-Sparse-Attention

In large-scale Transformer training and inference, the dominant bottlenecks diverge:

- **Training-side compute bottleneck**: The computational complexity of full attention grows quadratically with sequence length, and backpropagation requires repeating computations of the same order, leading to massive compute consumption on key-value pairs that contribute very little.
- **Inference-side memory bottleneck**: Full attention requires repeated reading and writing of Q, K, V, and intermediate variables, making memory access to the KV-cache the dominant factor in the computation flow, hindering full utilization of compute resources.

Thus, a more effective approach is sparse attention: interacting each query with only the $w$ most relevant keys, reducing computation and memory access from $O(N^2)$ to $O(N \cdot w)$ where $w \ll N$. If the sparse pattern can adapt to the task, it has the potential to be both fast and accurate, addressing bottlenecks in both training and inference. For more details, please refer to the paper [Trainable Dynamic Mask Sparse Attention](https://arxiv.org/abs/2508.02124).


## Key Features

### Supported Features

- Forward and backward passes with causal mask
- Arbitrary Q and KV sequence lengths
- Arbitrary number of heads and head dimensions up to 256
- Grouped Query Attention and Multi Query Attention
- Flexible Mask and Bias
- Skipping memory access and computation for masked regions
- Gradient computation for bias

### Features We Aim to Support

- Paged Attention
- TMA, WGMMA, and FP8 low-precision
- Sequence Parallelism
- Further performance improvements for skipping memory access and computation


## Performance

We present the expected speedup of FSA over standard PyTorch SDPA under mask and bias conditions.

![FSA Performance Overview](assets/performance_overview.png)

---

### Forward Pass Performance

The following table shows the forward pass performance comparison between FSA and standard PyTorch SDPA on an NVIDIA A100-SXM4-80GB. Results are averaged over 3 runs after 2 warmup runs.

| Mode   | Q len | K len  | Window W | SDPA (ms) | FSA (ms) | Speedup |
|--------|-------|--------|----------|-----------|-----------|---------|
| Train  | 256   | 256    | 1024     | 0.29      | 0.19      | 1.58x   |
| Train  | 512   | 512    | 1024     | 0.35      | 0.19      | 1.86x   |
| Train  | 1024  | 1024   | 1024     | 0.51      | 0.18      | 2.81x   |
| Train  | 2048  | 2048   | 1024     | 1.04      | 0.18      | 5.68x   |
| Train  | 4096  | 4096   | 1024     | 2.53      | 0.24      | 10.41x  |
| Train  | 8192  | 8192   | 1024     | 9.38      | 0.36      | 25.93x  |
| Train  | 16384 | 16384  | 1024     | 28.39     | 0.81      | 35.25x  |
| Train  | 32768 | 32768  | 1024     | 111.87    | 2.25      | 49.78x  |
| Train  | 32768 | 32768  | 32       | 113.19    | 2.10      | 53.97x  |
| Train  | 32768 | 32768  | 64       | 113.17    | 2.12      | 53.32x  |
| Train  | 32768 | 32768  | 128      | 113.14    | 2.10      | 53.78x  |
| Train  | 32768 | 32768  | 256      | 113.18    | 2.13      | 53.18x  |
| Train  | 32768 | 32768  | 512      | 113.19    | 2.17      | 52.17x  |
| Train  | 32768 | 32768  | 1024     | 113.19    | 2.24      | 50.45x  |
| Train  | 32768 | 32768  | 2048     | 113.15    | 2.39      | 47.35x  |
| Train  | 32768 | 32768  | 4096     | 113.16    | 2.67      | 42.39x  |
| Train  | 32768 | 32768  | 8192     | 113.11    | 3.20      | 35.29x  |
| Train  | 32768 | 32768  | 16384    | 113.15    | 3.97      | 28.51x  |
| Train  | 32768 | 32768  | 32768    | 113.11    | 4.90      | 23.10x  |
| Infer  | 1     | 256    | 1024     | 0.25      | 0.19      | 1.28x   |
| Infer  | 1     | 512    | 1024     | 0.25      | 0.19      | 1.27x   |
| Infer  | 1     | 1024   | 1024     | 0.25      | 0.20      | 1.28x   |
| Infer  | 1     | 2048   | 1024     | 0.25      | 0.20      | 1.24x   |
| Infer  | 1     | 4096   | 1024     | 0.25      | 0.19      | 1.29x   |
| Infer  | 1     | 8192   | 1024     | 0.25      | 0.20      | 1.25x   |
| Infer  | 1     | 16384  | 1024     | 0.25      | 0.19      | 1.29x   |
| Infer  | 1     | 32768  | 1024     | 0.27      | 0.20      | 1.33x   |
| Infer  | 1     | 65536  | 1024     | 0.42      | 0.20      | 2.10x   |
| Infer  | 1     | 131072 | 1024     | 0.72      | 0.20      | 3.65x   |
| Infer  | 1     | 262144 | 1024     | 1.31      | 0.22      | 6.06x   |
| Infer  | 1     | 524288 | 1024     | 2.49      | 0.24      | 10.45x  |
| Infer  | 1     | 524288 | 32       | 2.48      | 0.21      | 11.60x  |
| Infer  | 1     | 524288 | 64       | 2.44      | 0.21      | 11.66x  |
| Infer  | 1     | 524288 | 128      | 2.45      | 0.21      | 11.47x  |
| Infer  | 1     | 524288 | 256      | 2.43      | 0.21      | 11.47x  |
| Infer  | 1     | 524288 | 512      | 2.44      | 0.22      | 10.89x  |
| Infer  | 1     | 524288 | 1024     | 2.44      | 0.24      | 10.31x  |
| Infer  | 1     | 524288 | 2048     | 2.44      | 0.27      | 9.07x   |
| Infer  | 1     | 524288 | 4096     | 2.45      | 0.33      | 7.41x   |
| Infer  | 1     | 524288 | 8192     | 2.44      | 0.35      | 6.93x   |
| Infer  | 1     | 524288 | 16384    | 2.44      | 0.35      | 6.93x   |
| Infer  | 1     | 524288 | 32768    | 2.45      | 0.35      | 6.96x   |
| Infer  | 1     | 524288 | 65536    | 2.44      | 0.35      | 6.88x   |

---

### Backward Pass Performance

The following table shows the backward pass performance comparison between FSA and standard PyTorch SDPA on an NVIDIA A100-SXM4-80GB. Results are averaged over 3 runs after 2 warmup runs.

| Mode  | Q len | K len  | Window W | SDPA-BWD (ms) | FSA-BWD (ms) | Speedup |
|-------|-------|--------|----------|---------------|---------------|---------|
| Train | 256   | 256    | 1024     | 0.42          | 0.62          | 0.7x    |
| Train | 512   | 512    | 1024     | 0.56          | 0.60          | 0.9x    |
| Train | 1024  | 1024   | 1024     | 0.94          | 0.61          | 1.5x    |
| Train | 2048  | 2048   | 1024     | 1.79          | 0.69          | 2.6x    |
| Train | 4096  | 4096   | 1024     | 3.76          | 1.08          | 3.5x    |
| Train | 8192  | 8192   | 1024     | 14.39         | 2.06          | 7.0x    |
| Train | 16384 | 16384  | 1024     | 39.56         | 4.97          | 8.0x    |
| Train | 32768 | 32768  | 1024     | 142.07        | 25.63         | 5.5x    |
| Train | 32768 | 32768  | 32       | 142.70        | 21.91         | 6.5x    |
| Train | 32768 | 32768  | 64       | 142.65        | 22.29         | 6.4x    |
| Train | 32768 | 32768  | 128      | 142.69        | 23.04         | 6.2x    |
| Train | 32768 | 32768  | 256      | 142.69        | 24.27         | 5.9x    |
| Train | 32768 | 32768  | 512      | 142.67        | 25.12         | 5.7x    |
| Train | 32768 | 32768  | 1024     | 142.55        | 25.58         | 5.6x    |
| Train | 32768 | 32768  | 2048     | 142.75        | 25.64         | 5.6x    |
| Train | 32768 | 32768  | 4096     | 142.61        | 24.84         | 5.7x    |
| Train | 32768 | 32768  | 8192     | 142.33        | 25.63         | 5.6x    |
| Train | 32768 | 32768  | 16384    | 142.40        | 25.62         | 5.6x    |
| Train | 32768 | 32768  | 32768    | 142.43        | 25.63         | 5.6x    |

---


## Installation

### Requirements

- **Linux**: Ubuntu 22.04 or later
- **NVIDIA GPU**: Compute Capability 8.0 or higher
- **C++ Compiler**: GCC 7+
- **CUDA**: 11.8 or later
- **Python**: 3.9 or later
- **PyTorch**: 2.5.1 or later  

### Install

You can install FSA via pre-compiled wheels:

```bash
pip install flash_sparse_attn --no-build-isolation
```

Alternatively, you can compile and install from source:

```bash
git clone https://github.com/SmallDoges/flash_sparse_attn.git
cd flash_sparse_attn
pip install . --no-build-isolation
```


## Quick Start

### Basic Usage

```python
import torch
from flash_sparse_attn import flash_sparse_attn_func_auto
from flash_sparse_attn.utils.mask import create_mask
import math

# Setup
batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 256, 2, 1, 64
window_size = 128
device = torch.device('cuda')
dtype = torch.bfloat16
min_dtype = torch.finfo(dtype).min  # dtype minimum value

# Input tensors
query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

# Create bias for sparse attention
attn_bias = torch.randn(batch_size, num_kv_heads, 1, seq_len, device=device, dtype=dtype)

# Generate dynamic mask based on bias
if seq_len > window_size:
    attn_mask = create_mask(
        attention_bias=attn_bias,
        attention_mask=None,
        batch_size=batch_size,
        query_len=seq_len,
        key_len=seq_len,
        window_size=window_size,
        min_dtype=min_dtype,
    )

# Select FSA kernel
flash_sparse_attn_func = flash_sparse_attn_func_auto(backend="cuda")

# Run Flash-Sparse-Attention
output = flash_sparse_attn_func(
    query=query,
    key=key,
    value=value,
    attn_mask=attn_mask,
    attn_bias=attn_bias,
    is_causal=True,
    softmax_scale=1.0/math.sqrt(head_dim),
)

print(f"Output shape: {output.shape}")  # [1, 256, 2, 64]
```

### Gradient Computation Example

```python
# Enable gradient computation
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
attn_bias.requires_grad_(True)

# Forward pass
output = flash_sparse_attn_func(
    query=query, key=key, value=value,
    attn_mask=attn_mask,
    attn_bias=attn_bias,
    is_causal=True,
    softmax_scale=1.0/math.sqrt(head_dim)
)

# Backward pass
loss = output.sum()
loss.backward()

print(f"Query gradient shape: {query.grad.shape}")
print(f"Key gradient shape: {key.grad.shape}")
print(f"Value gradient shape: {value.grad.shape}")
print(f"Bias gradient shape: {attn_bias.grad.shape}")
```


## Benchmarking

FSA provides comprehensive benchmarking tools to evaluate performance across different configurations:

### Forward Pass Equivalence
```bash
python benchmarks/forward_equivalence.py
```
Validates numerical consistency between Python reference and CUDA implementation.

### Forward Pass Performance Benchmarking
```bash
python benchmarks/forward_performance.py
```
Compares FSA against standard SDPA across various sequence lengths and batch sizes.

### Backward Pass Equivalence
```bash
python benchmarks/backward_equivalence.py
```
Validates numerical consistency between Python reference and CUDA implementation.

### Backward Pass Performance Benchmarking
```bash
python benchmarks/backward_performance.py
```
Compares FSA against standard SDPA across various sequence lengths and batch sizes.

### Gradient Computation
```bash
python benchmarks/grad_equivalence.py
```
Tests backward pass implementation and gradient equivalence.


## Documentation

📚 **Complete documentation is available in the [docs](docs/) directory:**

- **[API Reference](docs/api_reference.md)** - Complete function documentation and usage examples


## Contributing

We welcome contributions from the community! FSA is an open-source project and we value all types of contributions.

### How to Contribute

- **Report bugs**: Found a bug? Please [open an issue](https://github.com/SmallDoges/flash_sparse_attn/issues/new/choose)
- **Request features**: Have an idea for improvement? [Let us know](https://github.com/SmallDoges/flash_sparse_attn/issues/new/choose)
- **Submit code**: Ready to contribute code? Check our [Contributing Guide](CONTRIBUTING.md)
- **Improve docs**: Help us make the documentation better

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Submit a pull request

For detailed instructions, see our [Contributing Guide](CONTRIBUTING.md).

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.


## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.


## Citation

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


## Acknowledgments

This project builds upon and integrates several excellent works:

- **[OpenSeek](https://github.com/FlagAI-Open/OpenSeek)** - Kernel development support
- **[Flash-Attention](https://github.com/Dao-AILab/flash-attention)** - Memory-efficient attention computation
- **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** - High-performance matrix operations library

We thank the open-source community for their contributions to efficient transformer implementations. 🤗
