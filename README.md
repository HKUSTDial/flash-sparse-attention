# Flash Dynamic Mask Attention

![Flash-DMA Banner](assets/flash_dmattn_banner.jpg)

Flash-DMA is a high-performance attention implementation that integrates Flash Attention's memory efficiency with Dynamic Mask Attention's computational efficiency for processing extremely long sequences in transformer models.

## Key Features

- **Sparse Attention Computation**: Dynamically selects the most important keys for each query, reducing computation from $O(N^2)$ to $O(N \cdot k)$ where $k \ll N$.
- **Memory Efficiency**: Maintains Flash Attention's $O(N)$ memory complexity without materializing the full attention matrix.
- **CUDA-Accelerated**: Deep integration at the CUDA kernel level for maximum performance.
- **Long Sequence Support**: Efficiently handles sequences of 128K+ tokens that would be impractical with standard attention.
- **Backward Compatible**: API compatible with existing Flash Attention implementations.

## Installation

### Prerequisites

- **Python**: 3.7 or later
- **PyTorch**: 1.10.0 or later
- **CUDA**: 11.0 or later (for GPU acceleration)
- **NVIDIA GPU**: Compute Capability 6.0 or higher
- **C++ Compiler**: GCC 7+ or compatible

### Install from Source

```bash
git clone https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn
git submodule update --init --recursive
pip install .
```

### CUDA Environment Setup

Ensure your CUDA environment is properly configured:

```bash
# Check CUDA installation
nvcc --version

# Set CUDA_HOME if needed
export CUDA_HOME=/usr/local/cuda
```

## Quick Start

### Basic Usage

```python
import torch
from flash_dma_cpp import apply_dynamic_mask_attention

# Input tensors
batch_size, num_heads, seq_len, head_dim = 1, 8, 4096, 64
query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# Dynamic mask parameters
dt_proj = torch.randn(num_heads, num_heads * head_dim, device='cuda', dtype=torch.float16)
A = torch.randn(num_heads, device='cuda', dtype=torch.float16)

# Apply Flash-DMA attention
output = apply_dynamic_mask_attention(
    query, key, value,
    dt_proj=dt_proj,
    A=A,
    keep_window_size=2048,
    is_causal=True
)
```

### Performance Comparison

Flash-DMA achieves significant speedups for long sequences:

| Sequence Length | Selection Ratio | Theoretical Speedup | Practical Speedup |
|-----------------|----------------|---------------------|-------------------|
| 4,096          | 0.25           | 4.0×                | 2.5-3.0×          |
| 16,384         | 0.125          | 8.0×                | 4.0-5.0×          |
| 65,536         | 0.0625         | 16.0×               | 6.0-8.0×          |

## How It Works

Flash-DMA combines two complementary techniques:

- **Dynamic Mask Attention**: Computes relevance scores for keys and selects only the most important ones for attention computation
- **Flash Attention**: Processes attention in blocks to reduce memory usage and HBM access


### The Integration Approach

The integration happens at the CUDA kernel level with several key components:

- **ZOH States**: Pre-computed importance scores for key selection
- **Active Masks**: Binary masks indicating which keys should be considered for each query
- **Sparse Matrix Multiplication**: Custom CUDA kernels for efficient sparse attention computation
- **Block-Based Processing**: Maintains Flash Attention's block-based approach for memory efficiency

This creates a hybrid attention mechanism that achieves both memory and computational efficiency.

## Documentation

For detailed technical documentation, see:
- [Integration Guide](docs/integration.md) - Comprehensive technical details
- [API Reference](#api-reference) - Function signatures and parameters

### API Reference

#### `apply_dynamic_mask_attention`

```python
apply_dynamic_mask_attention(
    query: torch.Tensor,           # [batch, heads, seq_len, head_dim]
    key: torch.Tensor,             # [batch, heads, seq_len, head_dim] 
    value: torch.Tensor,           # [batch, heads, seq_len, head_dim]
    dt_proj: torch.Tensor,         # [heads, heads * head_dim]
    A: torch.Tensor,               # [heads]
    causal_mask: torch.Tensor = None,  # [batch, 1, seq_len, seq_len]
    keep_window_size: int = 2048,
    is_causal: bool = True
) -> torch.Tensor
```

**Parameters:**
- `query, key, value`: Input attention tensors in BHSD format
- `dt_proj`: Projection matrix for dynamic mask computation
- `A`: Scaling coefficients for each attention head
- `causal_mask`: Optional causal mask (0 = keep, -inf = mask)
- `keep_window_size`: Maximum number of keys to keep per query
- `is_causal`: Enable causal attention masking

## Building from Source

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn

# Build in development mode
pip install -e .
```

### Build Requirements

- CUDA Toolkit 11.0+
- CUTLASS library (included as submodule)
- CUB library (included as submodule)

### Supported Architectures

- SM 6.0+ (Pascal, Volta, Turing, Ampere, Ada Lovelace)
- Optimized for SM 8.0+ (Ampere and newer)

## Testing

### Run Tests

```bash
# Basic functionality test
python test_dynamic_mask_equivalence.py

# Comprehensive attention test
python test_dynamic_mask_attention_equivalence.py

# Performance benchmarks
python benchmarks/benchmark_grad.py
```

### Test Options

```bash
# Test with different parameters
python test_dynamic_mask_equivalence.py --seed 42 --perf

# Custom test configuration
python test_dynamic_mask_attention_equivalence.py --batch-size 2 --seq-len 8192
```

## Performance

### Memory Efficiency

- **Standard Attention**: O(N²) memory for N sequence length
- **Flash-DMA**: O(N) memory complexity
- **Reduced Peak Memory**: Up to 8× reduction for very long sequences

### Computational Efficiency  

- **Dynamic Selection**: Only computes attention for k most relevant keys
- **Sparse Operations**: Custom CUDA kernels for sparse matrix operations
- **Block Processing**: Maintains cache-friendly memory access patterns

### Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| PyTorch | 1.10.0+ |
| CUDA | 11.0+ |
| Python | 3.7+ |
| GPU Arch | SM 6.0+ |

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or sequence length
# Use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Compilation Errors**
```bash
# Ensure CUDA_HOME is set
export CUDA_HOME=/usr/local/cuda
# Update NVCC if needed
which nvcc
```

**Performance Issues**
- Ensure GPU has sufficient compute capability (6.0+)
- Use appropriate data types (float16 recommended)
- Verify CUDA kernels are being used (not CPU fallback)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use Flash-DMA in your research, please cite:

```bibtex
@software{flash_dma_2025,
  title={Flash Dynamic Mask Attention: Efficient Long Sequence Processing},
  author={SmallDoges Contributors},
  year={2025},
  url={https://github.com/SmallDoges/flash-dmattn}
}
```

## Acknowledgments

This project builds upon the excellent work of:
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention) by Tri Dao et al.
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) library for efficient matrix operations