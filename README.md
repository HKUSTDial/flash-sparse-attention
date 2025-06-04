# Flash Dynamic Mask Attention

![Flash-DMA Banner](assets/flash_dmattn_banner.jpg)

Flash-DMA is a high-performance attention implementation that integrates Flash Attention's memory efficiency with Dynamic Mask Attention's computational efficiency for processing extremely long sequences in transformer models.

## Key Features

- **Sparse Attention Computation**: Dynamically selects the most important keys for each query, reducing computation from $O(N^2)$ to $O(N \cdot k)$ where $k \ll N$.
- **Memory Efficiency**: Maintains Flash Attention's $O(N)$ memory complexity without materializing the full attention matrix.
- **CUDA-Accelerated**: Deep integration at the CUDA kernel level for maximum performance.
- **Long Sequence Support**: Efficiently handles sequences of 128K+ tokens that would be impractical with standard attention.
- **Backward Compatible**: API compatible with existing Flash Attention implementations.

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


## Acknowledgments

This project builds upon the excellent work of:
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention) by Tri Dao et al.
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) library for efficient matrix operations