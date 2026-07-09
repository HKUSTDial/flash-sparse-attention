<div align="center">
  <img src="https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/logo.png" alt="flash-algo" width="100%">
</div>

<div align="center">


[English](./README.md) | **简体中文**

</div>


Flash-Sparse-Attention 是一个高性能的可训练稀疏注意力实现, 将 Flash Attention 的内存效率与稀疏计算能力相结合, 用于在 Transformer 模型中处理超长序列. 


# 主要特性

> [!NOTE]
> 支持任意形状 mask 和 bias 的版本为[这个分支](https://github.com/HKUSTDial/flash-sparse-attention/tree/final_mask_version), 当前主分支不再覆盖这部分功能.

## 支持的功能

- Dense Attention, Sparse attention 和 Gated attention 的前向与反向
- 常规批次输入与 varlen 输入
- 因果注意力与局部窗口注意力
- 任意 Q / KV 序列长度组合, 以及小于等于 256 的头维度
- 分组查询注意力和多查询注意力
- 稀疏 softmax 阈值控制
- gated attention 支持门控输入, 以及控制门控稀疏程度
- Flex Local Window Attention 支持逐head的任意窗口大小和局部范围
- Split-KV 适用于前向和解码的工作负载均衡
- Split-QO 适用于反向的工作负载均衡
- Fused Quant 支持非FP8原生支持的硬件使用低精度计算
- Top-k gather KV-cache 解码
- 分页注意力

**完整API文档请参考 [这里](https://hkustdial.github.io/flash-sparse-attention/)**

## 我们想要支持的功能

- KV-Cache 管理器
- [TLE](https://github.com/flagos-ai/FlagTree/wiki/TLE) 后端支持
- [Gluon](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon) 后端支持


# 安装

## 依赖

- **Linux**: Ubuntu 22.04 或更高版本
- **Device**: GPU, XPU, NPU, 或 PPU
- **Python**: 3.9 或更高版本
- **PyTorch**: 2.5.1 或更高版本
- **Triton**: 3.6.0 或更高版本
- **Triton Kernels**: 3.6.0 或更高版本

## 安装

直接安装：

```bash
pip install flash-sparse-attn
```

此外，需要安装 `triton_kernels`：

```bash
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@v3.6.0#subdirectory=python/triton_kernels"
```

如果您希望从源码安装（自动包含所有依赖）：

```bash
git clone https://github.com/flash-algo/flash-sparse-attn.git
cd flash-sparse-attn
pip install .
```


## 通过 HuggingFace Kernel 使用

也可以直接从 [HuggingFace Kernel](https://github.com/huggingface/kernels) 加载 kernel，无需安装本包：

```python
from kernels import get_kernel

fsa = get_kernel("JingzeShi/flash-sparse-attn", version=1, trust_remote_code=True)

# 前向
out = fsa.flash_sparse_attn_func(q, k, v, is_causal=True)
# 反向
out.sum().backward()
# 解码
out = fsa.flash_sparse_attn_with_kvcache_func(q, k_cache, v_cache)
```

需要先安装 `pip install kernels`。


# 快速开始

## 基本用法

以下是前向、反向和解码的示例。

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

### 前向

组合 flex window, split-KV, fused quant, 和 sparse softmax 以获得最大性能。

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

### 反向

组合 flex window, split-QO, split-KV, fused quant, 和 low-contribution skipping 以获得最大反向性能。

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

### 解码

组合 flex window, split-KV, fused quant, sparse softmax, packed GQA 和 Graph 以获得最大解码性能。

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

# 预热
for _ in range(3):
    fsa_decode_fn()
torch.cuda.synchronize()

# 捕获 Graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = fsa_decode_fn()

# 重放
graph.replay()
```


# 性能

以下基准测试涵盖前向、后向和解码工作负载, 以FlashAttention作为基线。

## NVIDIA GPU


### A100

**前向传播性能**

![Attention forward speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_a100sxm4.png)

**反向传播性能**

![Attention backward speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_a100sxm4.png)

**解码性能**

![Attention decode speed, head dim 128, a100](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_a100sxm4.png)


### H20

**前向传播性能**

![Attention forward speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_h203e.png)

**反向传播性能**

![Attention backward speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_h203e.png)

**解码性能**

![Attention decode speed, head dim 128, h20-3e](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_h203e.png)


### RTX PRO 6000

**前向传播性能**

![Attention forward speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_forward_rtxpro6000.png)

**反向传播性能**

![Attention backward speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_backward_rtxpro6000.png)

**解码性能**

![Attention decode speed, head dim 128, rtx pro 6000](https://github.com/HKUSTDial/flash-sparse-attention/releases/download/v2.0.5/latency_decode_rtxpro6000.png)


# 基准测试

基准测试脚本位于 [tests](tests/) 下, 用于评估前向、反向和解码三类场景下的性能。

## 前向传播性能

```bash
python tests/benchmark_forward.py
```

## 反向传播性能

```bash
python tests/benchmark_backward.py
```

## 解码性能

```bash
python tests/benchmark_decode.py
```


# 引用

如果您在研究中使用 FSA, 请引用：

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

# 致谢

本项目基于并集成了几个优秀的工作：

- **[OpenSeek](https://github.com/FlagAI-Open/OpenSeek)** - 内核开发支持
- **[Flash-Attention](https://github.com/Dao-AILab/flash-attention)** - 内存高效的注意力计算
- **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** - 高性能矩阵运算库

我们感谢开源社区对高效 Transformer 实现的贡献. 🤗
