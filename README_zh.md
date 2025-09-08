<div align="center">
  <img src="./assets/logo.png" alt="SmallDoges" width="100%">
</div>

<div align="center">


[English](./README.md) | **简体中文**

</div>


![Flash-DMA Banner](assets/flash_dmattn_banner.png)

Flash-DMA 是一个高性能的注意力实现，将 Flash Attention 的内存效率与动态掩码注意力的稀疏计算能力相结合，用于在 Transformer 模型中处理超长序列。


## 主要特性

### 🎯 核心内核优势
- **4D Mask & Bias 支持**: 原生支持 `(batch_size, num_kv_heads, query_len, key_len)` 形状的 attention_mask 和 attention_bias 张量
- **智能计算跳过**: 基于 attention_mask 的 block-level 自动跳过机制，完全跳过全零 mask 区块的计算和内存访问
- **完整梯度支持**: 内置 attention_bias 的完整梯度计算路径，支持端到端训练

### 🚀 性能与效率
- **动态稀疏注意力**: 为每个查询动态选择最重要的键，将计算复杂度从 $O(N^2)$ 降低到 $O(N \cdot w)$，其中 $w \ll N$， 支持可训练的稀疏结构
- **内存效率**: 保持 Flash Attention 的 $O(N)$ 内存复杂度，无需实例化完整的注意力矩阵
- **CUDA 深度优化**: 自定义 CUDA 内核，含共享内存别名、流水线预取、按块跳过，实现高吞吐与低访存开销
- **超长上下文支持**: 通过动态掩码窗口裁剪，在保持精度的前提下支撑 128K+ 令牌级别的上下文处理


## 性能

我们展示了 Flash-DMA 相对于标准 PyTorch SDPA 的预期加速效果。

---

### 前向传播性能

以下表格是我们在NVIDIA A100-SXM4-80GB上对Flash-DMA与标准PyTorch SDPA在不同配置下的前向性能对比测试结果。结果为预热两次, 运行三次的平均值。

| Mode   | Q len | K len  | Window W | SDPA (ms) | FDMA (ms) | Speedup |
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

### 反向传播性能

以下表格是我们在NVIDIA A100-SXM4-80GB上对Flash-DMA与标准PyTorch SDPA在不同配置下的反向性能对比测试结果。结果为预热两次, 运行三次的平均值。

| Mode  | Q len | K len  | Window W | SDPA-BWD (ms) | FDMA-BWD (ms) | Speedup |
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
MAX_JOBS=4 pip install . --no-build-isolation
```


## 快速开始

### 基本用法

```python
import torch
from flash_dmattn import flash_dmattn_func_auto
import math

# 设置
batch_size, seq_len, num_heads, num_kv_heads, head_dim = 1, 256, 2, 1, 64
keep_window_size = 128
device = torch.device('cuda')
dtype = torch.bfloat16
min_dtype = torch.finfo(dtype).min  # dtype 的最小值

# 输入张量
query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

# 为稀疏注意力创建 mask 和 bias
attention_mask = torch.ones(batch_size, num_kv_heads, seq_len, seq_len, device=device, dtype=dtype)
attention_bias = torch.randn(batch_size, num_kv_heads, seq_len, seq_len, device=device, dtype=dtype)

# 基于 bias 生成稀疏 mask
if seq_len > keep_window_size:
    # 为每个查询选择 top-k 最重要的键
    topk_values, topk_indices = torch.topk(
        attention_bias, keep_window_size, dim=-1, 
        largest=True, sorted=False
    )
    # 生成有效的 top-k mask
    valid_topk = (topk_values != min_dtype).to(dtype)
    attention_mask = torch.zeros_like(attention_bias, dtype=dtype, device=attention_bias.device)
    attention_mask = attention_mask.scatter(-1, topk_indices, valid_topk)
    attention_bias = attention_bias.masked_fill(attention_mask == 0.0, min_dtype)

# 选择 FDMA 内核
flash_dmattn_func = flash_dmattn_func_auto(backend="cuda")

# 运行 FDMA
output = flash_dmattn_func(
    query=query,
    key=key, 
    value=value,
    attn_mask=attention_mask,
    attn_bias=attention_bias,
    is_causal=True,
    scale=1.0/math.sqrt(head_dim),
)

print(f"输出形状: {output.shape}")  # [1, 256, 2, 64]
```

### 梯度计算示例

```python
# 开启梯度计算
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
attention_bias.requires_grad_(True)

# 前向传播
output = flash_dmattn_func(
    query=query, key=key, value=value,
    attn_mask=attention_mask,
    attn_bias=attention_bias,
    is_causal=True,
    scale=1.0/math.sqrt(head_dim)
)

# 反向传播
loss = output.sum()
loss.backward()

print(f"Query 梯度形状: {query.grad.shape}")
print(f"Key 梯度形状: {key.grad.shape}")
print(f"Value 梯度形状: {value.grad.shape}")
print(f"Bias 梯度形状: {attention_bias.grad.shape}")
```


## 工作原理

Flash-DMA 通过将 Flash Attention 的高效内存访问模式与动态掩码注意力的稀疏计算能力相结合，实现了高效的注意力机制。

### 核心技术融合

- **🎯 4D Mask & Bias 原生支持**: 内核直接处理 `(batch_size, num_kv_heads, query_len, key_len)` 形状的张量
- **⚡ Block-level 智能跳过**: 基于 mask 的统一 OR-reduction 跳过逻辑，完全避免全零区块的计算和内存访问
- **🔄 完整梯度链路**: 内置 attention bias 梯度计算，支持端到端可微分训练

### 关键优化策略

1. **统一跳过逻辑**: 前向和反向过程使用相同的 block-level 跳过决策
2. **内存访问优化**: 只有当 `OR(mask_block) == true` 时才加载 K/V 数据
3. **梯度路径完整性**: dbias 梯度计算完全融合在反向内核中
4. **共享内存复用**: sMask ↔ sP, sBias ↔ sdS 智能别名化


## 文档

📚 **完整文档可在 [docs](docs/) 目录中找到：**

- **[API 参考](docs/api_reference.md)** - 完整的函数文档和使用示例
- **[集成指南](docs/integration.md)** - Flash Attention 集成的详细技术文档


## 从源码构建

### 开发环境设置

```bash
# 克隆包含子模块
git clone https://github.com/SmallDoges/flash-dmattn.git
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
python benchmarks/forward_equivalence.py
```
验证 Python 参考实现与 CUDA 实现之间的数值一致性。

### 前向传播性能基准测试  
```bash
python benchmarks/forward_performance.py
```
在各种序列长度和批大小下比较 Flash-DMA 与标准 SDPA。

### 反向传播等效性
```bash
python benchmarks/backward_equivalence.py
```
验证 Python 参考实现与 CUDA 实现之间的数值一致性。

### 反向传播性能基准测试
```bash
python benchmarks/backward_performance.py
```
比较 Flash-DMA 与标准 SDPA 在各种序列长度和批大小下的性能。

### 梯度计算
```bash
python benchmarks/grad_equivalence.py
```
测试反向传播实现和梯度等效性。


## 故障排除

### 常见问题

**编译错误**
```bash
# 确保 CUDA_HOME 设置正确
echo $CUDA_HOME         # Linux/Mac
echo $env:CUDA_HOME     # Windows PowerShell

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


## 贡献

我们欢迎社区的贡献！Flash-DMA 是一个开源项目，我们重视所有类型的贡献。

### 如何贡献

- **报告错误**: 发现了错误？请[提交 issue](https://github.com/SmallDoges/flash-dmattn/issues/new/choose)
- **功能请求**: 有改进想法？[告诉我们](https://github.com/SmallDoges/flash-dmattn/issues/new/choose)
- **提交代码**: 准备贡献代码？查看我们的[贡献指南](CONTRIBUTING.md)
- **改进文档**: 帮助我们完善文档

### 贡献者快速入门

1. Fork 仓库
2. 创建功能分支: `git checkout -b feature-name`
3. 进行修改并测试
4. 提交 Pull Request

详细说明请参见我们的[贡献指南](CONTRIBUTING.md)。

### 行为准则

本项目遵循[贡献者公约行为准则](CODE_OF_CONDUCT.md)。参与时，您需要遵守此准则。

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
