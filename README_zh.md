<div align="center">
  <img src="./assets/logo.png" alt="SmallDoges" width="100%">
</div>

<div align="center">


[English](./README.md) | **简体中文**

</div>

**可训练的动态掩码稀疏注意力**

> Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo

> 论文: https://huggingface.co/papers/2508.02124

![Flash-DMA Banner](assets/flash_dmattn_banner.png)

Flash-DMA 是一个高性能的注意力实现，将 Flash Attention 的内存效率与动态掩码注意力的稀疏计算能力相结合，用于在 Transformer 模型中处理超长序列。


## 主要特性

- **稀疏注意力计算**: 为每个查询动态选择最重要的键，将计算复杂度从 $O(N^2)$ 降低到 $O(N \cdot w)$，其中 $w \ll N$。
- **内存效率**: 保持 Flash Attention 的 $O(N)$ 内存复杂度，无需实例化完整的注意力矩阵。
- **CUDA 加速**: 在 CUDA 内核层面深度集成，采用自定义稀疏 GEMM 运算以获得最佳性能。
- **长序列支持**: 当序列长度超过 `keep_window_size` 时，通过动态掩码高效处理 128K+ 标记的序列。
- **高级集成**: 从 Python 前端到 CUDA 后端的完整集成，具有优化的内存布局和稀疏计算策略。


## 性能

我们展示了 Flash-DMA 相对于标准 PyTorch SDPA 的预期加速效果。

![Speedup](assets/speedup.png)


## 安装

### 先决条件

- **Python**: 3.8 或更高版本
- **PyTorch**: 2.0.0 或更高版本  
- **CUDA**: 11.8 或更高版本
- **NVIDIA GPU**: 计算能力 8.0 或更高
- **C++ 编译器**: GCC 7+

### CUDA 环境设置

确保您的 CUDA 环境已正确配置：

```bash
# 检查 CUDA 安装
nvcc --version

# 如需要，设置 CUDA_HOME
export CUDA_HOME=/usr/local/cuda
```

### 从源码安装

```bash
git clone https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn
git submodule update --init --recursive
pip install .
```


## 快速开始

```python
import torch
from flash_dmattn import flash_dmattn_func_auto
import math

# 设置
batch_size, seq_len, num_heads, head_dim = 2, 4096, 16, 128
device = torch.device('cuda')
dtype = torch.bfloat16

# 输入张量
query = torch.randn(batch_size, seq_len, num_heads, head_dim, 
                   device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_heads, head_dim,
                 device=device, dtype=dtype)
value = torch.randn(batch_size, seq_len, num_heads, head_dim,
                   device=device, dtype=dtype)

# 为稀疏注意力创建掩码和偏置
attention_bias = torch.randn(batch_size, num_heads, seq_len, seq_len,
                           device=device, dtype=dtype)
attention_mask = torch.ones(batch_size, num_heads, seq_len, seq_len,
                          device=device, dtype=dtype)

# 应用动态掩码（为长序列保留 top-k）
keep_window_size = 2048
if seq_len > keep_window_size:
    # 为每个查询选择 top-k 最重要的键
    topk_indices = torch.topk(attention_bias, keep_window_size, dim=-1, 
                             largest=True, sorted=False).indices
    attention_mask.zero_()
    attention_mask.scatter(-1, topk_indices, 1.0)

# 选择后端
flash_dmattn_func = flash_dmattn_func_auto(backend="cuda")

# 运行 Flash 动态掩码注意力
output = flash_dmattn_func(
    query=query,
    key=key, 
    value=value,
    attn_mask=attention_mask,
    attn_bias=attention_bias,
    is_causal=True,
    scale=1.0/math.sqrt(head_dim),
)

print(f"输出形状: {output.shape}")  # [2, 4096, 16, 128]
```


## 工作原理

Flash-DMA 结合了两种互补的技术：

- **动态掩码注意力**: 计算键的相关性分数，并仅选择最重要的键进行注意力计算
- **Flash Attention**: 分块处理注意力以减少内存使用和 HBM 访问

### 集成方法

集成发生在 CUDA 内核层面，具有几个关键组件：

- **ZOH 状态**: 预计算的键选择重要性分数
- **活跃掩码**: 指示每个查询应考虑哪些键的二进制掩码
- **稀疏矩阵乘法**: 高效稀疏注意力计算的自定义 CUDA 内核
- **分块处理**: 保持 Flash Attention 的分块方法以提高内存效率

这创建了一种混合注意力机制，为长序列实现了内存和计算效率。


## 文档

📚 **完整文档可在 [docs](docs/) 目录中找到：**

- **[API 参考](docs/api_reference.md)** - 完整的函数文档和使用示例
- **[集成指南](docs/integration.md)** - Flash Attention 集成的详细技术文档


## 从源码构建

### 开发环境设置

```bash
# 克隆包含子模块
git clone --recursive https://github.com/SmallDoges/flash-dmattn.git
cd flash-dmattn

# 在开发模式下构建
pip install -e .

# 运行测试以验证安装
python -c "import flash_dma_cuda; print('✅ Flash DMA CUDA 扩展导入成功')"
```

### 构建要求

- CUDA Toolkit 11.8+
- CUTLASS 库
- 支持 CUDA 的 PyTorch

### 支持的架构

- **SM 8.0** 
- **SM 9.0**
- **SM 10.0**
- **SM 12.0**

**注意**: Flash 动态掩码注意力需要 CUDA 计算能力 8.0+ 才能获得最佳性能。不支持更早的架构。

## 基准测试

Flash-DMA 提供全面的基准测试工具，用于评估不同配置下的性能：

### 前向传播等效性
```bash
python benchmarks/benchmark_forward_equivalence.py
```
验证 Python 参考实现与 CUDA 实现之间的数值一致性。

### 性能基准测试  
```bash
python benchmarks/benchmark_forward_performance.py
```
在各种序列长度和批大小下比较 Flash-DMA 与标准 Flash Attention。

### 梯度计算
```bash
python benchmarks/benchmark_grad.py
```
测试反向传播实现和梯度等效性。

### 多查询联想回忆
```bash
python benchmarks/benchmark_mqar.py
```
评估长程推理任务的性能。


## 故障排除

### 常见问题

**编译错误**
```bash
# 确保 CUDA_HOME 设置正确
echo $CUDA_HOME  # Linux/Mac
echo $env:CUDA_HOME  # Windows PowerShell

# 检查 CUDA 工具包版本
nvcc --version

# 验证 PyTorch CUDA 支持
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

**导入错误**
```python
# 测试基本导入
try:
    from flash_dmattn import flash_dmattn_func, get_available_backends
    print("✅ Flash 动态掩码注意力导入成功")
    print(f"可用后端: {get_available_backends()}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保包已正确安装，使用: pip install -e .")
```

**性能问题**
```python
# 监控 GPU 内存使用
from flash_dmattn import flash_dmattn_func

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"GPU 内存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print_memory_stats()
output = flash_dmattn_func(q=query, k=key, v=value, is_causal=True)
print_memory_stats()

# 如需要，清除缓存
torch.cuda.empty_cache()
```

## 许可证

本项目采用 BSD 3-Clause 许可证。详情请参见 [LICENSE](LICENSE)。

## 引用

如果您在研究中使用 Flash-DMA，请引用：

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

## 致谢

本项目基于并集成了几个优秀的工作：

- **[OpenSeek](https://github.com/FlagAI-Open/OpenSeek)** - 内核开发支持
- **[Flash-Attention](https://github.com/Dao-AILab/flash-attention)** - 内存高效的注意力计算
- **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)** - 高性能矩阵运算库

我们感谢开源社区对高效 Transformer 实现的贡献。🤗
