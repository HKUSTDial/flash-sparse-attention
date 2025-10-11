# Flash Dynamic Mask Attention API 参考文档


## 概述

Flash Dynamic Mask Attention 是一个高性能注意力实现，结合了 Flash Attention 的内存效率和 Dynamic Mask Attention 的稀疏计算优势。它支持 CUDA、Triton 和 Flex Attention 后端，并支持超长序列的动态掩码。


## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [后端选择与比较](#后端选择与比较)
4. [接口函数详解](#接口函数详解)
   - [CUDA 后端：flash_dmattn_func](#flash_dmattn_func-cuda-后端)
   - [Triton 后端：triton_dmattn_func](#triton_dmattn_func-triton-后端)
   - [Flex 后端：flex_dmattn_func](#flex_dmattn_func-flex-后端)
5. [集成](#集成)
   - [Transformers 集成](#transformers-集成)
6. [常见问题与解决方案](#常见问题与解决方案)


## 安装

请参考 [README](https://github.com/SmallDoges/flash-dmattn/blob/main/README_zh.md#%E5%AE%89%E8%A3%85-1) 以获取详细的安装说明和依赖项。

```bash
# 使用 CUDA 后端
pip install flash-dmattn

# 或从源码安装
pip install -e .

# 仅使用 Triton/Flex 后端
FLASH_DMATTN_SKIP_CUDA_BUILD=1 pip install -e .
```


## 快速开始

使用 `flash_dmattn_func_auto` 可以自动选择最佳可用后端，无需手动判断。

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

# 准备输入张量
batch, seqlen, num_heads, head_dim = 2, 1024, 8, 64
q = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
k = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
v = torch.randn(batch, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

# 获取注意力函数（自动选择后端，优先级: cuda > triton > flex）
attn_func = flash_dmattn_func_auto()

# 调用注意力计算
output = attn_func(q, k, v, is_causal=True)
print(f"输出形状: {output.shape}")  # (2, 1024, 8, 64)

# 也可以强制使用特定后端
attn_func = flash_dmattn_func_auto(backend="cuda")  # 或 "triton", "flex"
output = attn_func(q, k, v, is_causal=True)
```

> [!NOTE]
> `flash_dmattn_func_auto` 返回一个可调用的注意力函数，而不是注意力输出。


## 后端选择与比较

### 可用后端检查

```python
from flash_dmattn import get_available_backends, CUDA_AVAILABLE, TRITON_AVAILABLE, FLEX_AVAILABLE

# 查看所有可用后端
print(get_available_backends())  # 例如：["cuda", "triton", "flex"]

# 检查特定后端是否可用
print(f"CUDA: {CUDA_AVAILABLE}, Triton: {TRITON_AVAILABLE}, Flex: {FLEX_AVAILABLE}")
```

### 后端特性对比

| 特性 | CUDA | Triton | Flex |
|------|------|--------|------|
| **性能** | 最高 | 良好 | 良好 |
| **内存效率** | 最佳 | 良好 | 良好 |
| **构建要求** | 自定义 CUDA 扩展 | triton 包 | transformers 包 |
| **GQA 支持** | ✅ | ✅ | ✅ |
| **注意力掩码** | ✅ | ✅ | ⚠️ |
| **注意力偏置** | ✅ | ✅ | ✅ |
| **因果掩码** | ✅ | ✅ | ✅ |
| **Softcap** | ✅ | ❌ | ❌ |
| **确定性** | ✅ | ❌ | ❌ |
| **返回注意力概率** | ✅ | ❌ | ❌ |
| **反向传播支持** | ✅ | ✅ | ⚠️ |

> [!NOTE]
> ✅ 完全支持 | ⚠️ 有限支持 | ❌ 不支持

### 何时使用各个后端

**CUDA 后端** ([详细说明](#flash_dmattn_func-cuda-后端))
- ✅ 完整梯度支持的训练工作负载
- ✅ 最大性能生产推理
- ✅ 需要确定性行为的应用
- ❌ 避免：无法构建自定义 CUDA 扩展时

**Triton 后端** ([详细说明](#triton_dmattn_func-triton-后端))
- ✅ CUDA 扩展不可用时的训练工作负载
- ✅ 开发和原型设计
- ✅ 跨平台兼容性需求
- ✅ 性能和易安装性的良好平衡

**Flex 后端** ([详细说明](#flex_dmattn_func-flex-后端))
- ✅ 仅推理应用
- ✅ 使用最新 PyTorch 特性的研究
- ✅ 无需自定义构建的快速实验
- ❌ 避免：训练
- ❌ 避免：需要严格的注意力掩码遵从时

### 导入可用函数

```python
from flash_dmattn import (
    # 自动后端选择
    get_available_backends,
    flash_dmattn_func_auto,
    
    # 后端特定函数
    flash_dmattn_func,      # CUDA 后端
    triton_dmattn_func,     # Triton 后端
    flex_dmattn_func,       # Flex 后端
    
    # 后端可用性标志
    CUDA_AVAILABLE,
    TRITON_AVAILABLE,
    FLEX_AVAILABLE,
)

# Transformers 集成
from flash_dmattn.integrations.flash_dynamic_mask_attention import (
    flash_dynamic_mask_attention_forward
)
```


## 接口函数详解

### flash_dmattn_func (CUDA 后端)

主要的注意力函数。支持多头注意力和分组查询注意力（当 KV 头数少于 Q 头数时）。需要 CUDA 扩展已构建并可用。

```python
def flash_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_kv_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_kv_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, {num_heads, num_kv_heads, 1}, {seqlen_q, 0}, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, {num_heads, num_kv_heads, 1}, {seqlen_q, 0}, seqlen_k)
    softmax_scale: Optional[float] = None,                  # 分数缩放，默认为 1/sqrt(head_dim)
    is_causal: Optional[bool] = None,               # 因果掩码
    softcap: Optional[float] = None,                # 仅 CUDA 支持
    deterministic: Optional[bool] = None,           # 仅 CUDA 支持
    return_attn_probs: Optional[bool] = None,       # 仅 CUDA 支持，用于测试
) -> torch.Tensor
```

#### 参数

- query: (B, Q, H, D). CUDA 张量，fp16/bf16，最后一维连续
- key: (B, K, H_kv, D). 与 query 相同的数据类型/设备；当 H_kv <= H 时为 GQA
- value: (B, K, H_kv, D). 与 query 相同的数据类型/设备；当 H_kv <= H 时为 GQA
- attn_mask: (B, {H, H_kv, 1}, {Q, 0}, K). 1.0 = 可见，0.0 = 被掩码。None 表示禁用
- attn_bias: (B, {H, H_kv, 1}, {Q, 0}, K). 在 softmax 前加到分数上。None 表示禁用
- softmax_scale: 分数缩放；默认为 1/sqrt(D)
- is_causal: 应用因果掩码
- softcap, deterministic, return_attn_probs: 仅在 CUDA 后端有效；在其他后端被忽略

#### 返回值

- output: (B, Q, H, D)

### triton_dmattn_func (Triton 后端)

基于 Triton 的实现，无需自定义 CUDA 内核即可提供良好性能。

```python
def triton_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    is_causal: bool = False,                        # 因果掩码
    softmax_scale: Optional[float] = None,                  # 分数缩放，默认为 1/sqrt(head_dim)
) -> torch.Tensor
```

### flex_dmattn_func (Flex Attention 后端)

基于 Flex Attention 的实现，使用 PyTorch 原生 flex attention 并支持动态掩码。

```python
def flex_dmattn_func(
    query: torch.Tensor,                            # (batch, seqlen_q, num_heads, head_dim)
    key: torch.Tensor,                              # (batch, seqlen_k, num_heads, head_dim)
    value: torch.Tensor,                            # (batch, seqlen_k, num_heads, head_dim)
    attn_mask: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    attn_bias: Optional[torch.Tensor] = None,       # (batch, num_heads, seqlen_q, seqlen_k)
    is_causal: Optional[bool] = None,               # 因果掩码
    softmax_scale: Optional[float] = None,                  # 分数缩放，默认为 1/sqrt(head_dim)
) -> torch.Tensor
```


## 集成

### Transformers 集成

为 HuggingFace Transformers 模型提供的集成函数，提供无缝的 flash dynamic mask attention 支持。

#### flash_dynamic_mask_attention_forward

```python
from flash_dmattn.integrations.flash_dynamic_mask_attention import flash_dynamic_mask_attention_forward

def flash_dynamic_mask_attention_forward(
    module: torch.nn.Module,                        # 注意力模块
    query: torch.Tensor,                            # (batch_size, num_heads, query_len, head_dim)
    key: torch.Tensor,                              # (batch_size, num_kv_heads, key_len, head_dim)
    value: torch.Tensor,                            # (batch_size, num_kv_heads, key_len, head_dim)
    attention_mask: Optional[torch.Tensor],         # (batch_size, {num_heads, num_kv_heads, 1}, {query_len, 0}, key_len)
    attention_bias: Optional[torch.Tensor],         # (batch_size, {num_heads, num_kv_heads, 1}, {query_len, 0}, key_len)
    scaling: Optional[float] = None,                # 分数缩放
    softcap: Optional[float] = None,                # softcap 值
    **kwargs,
) -> tuple[torch.Tensor, None]
```

#### 参数

- module: 注意力模块实例
- query: 查询张量 (B, H, Q, D)
- key: 键张量 (B, H_kv, K, D)
- value: 值张量 (B, H_kv, K, D)
- attention_mask: 布尔注意力掩码 (B, {H, H_kv, 1}, {Q, 0}, K)
- attention_bias: 加到分数上的注意力偏置 (B, {H, H_kv, 1}, {Q, 0}, K)
- scaling: 分数缩放因子
- softcap: 注意力分数的 softcap 值
- **kwargs: 额外参数，包括：
  - is_causal: 是否应用因果掩码
  - window_size: 保持的窗口大小
  - layer_idx: 用于日志的层索引
  - implementation: 使用的实现（"flash_dmattn" 或 None）

#### 返回值

- tuple[torch.Tensor, None]: 输出张量 (B, Q, H, D) 和 None（用于兼容性）

#### 使用示例

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
        self.window_size = config.window_size
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
            # sin 和 cos 是 RoPE 模型特有的；static cache 需要 cache_position
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 从 value_states 采样 dt_states 以生成 attention_bias
        dt_states = self.dt_proj(
            value_states.transpose(1, 2).reshape(value_states.shape[0], value_states.shape[-2], -1)
        )
        attn_bias = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2).to(hidden_states.dtype)

        # 选择注意力实现
        attention_interface: Callable = flash_dynamic_mask_attention_forward
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            attention_bias=attn_bias,
            softmax_scale=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

这个示例展示了：
- **动态注意力偏置生成**: 使用可学习参数创建注意力偏置
- **灵活的后端选择**: 通过 `attention_interface` 轻松切换注意力实现
- **正确的张量重塑**: 根据需要在不同的张量布局之间转换
- **与缓存的集成**: 在生成场景中支持键值缓存


## 常见问题与解决方案

### 导入错误

```python
try:
    from flash_dmattn import flash_dmattn_func_auto, get_available_backends
    print("✅ 导入成功", get_available_backends())
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请使用以下命令安装: pip install -e .")
```

### 性能问题

1. 执行缓慢：确保所有张量在同一个 GPU 上且最后一维是连续的；使用 8 的倍数的头维度；尽可能使用 CUDA 后端
2. 高内存：使用梯度检查点；分块长序列；考虑对超长序列使用 Triton 或 Flex 后端
3. 数值稳定性：优先使用 bfloat16；检查掩码/偏置是否有 NaN/Inf；监控梯度范数

### Transformers 集成问题

1. 模型兼容性：确保您的模型支持自定义注意力实现
2. 形状不匹配：检查张量布局是否匹配预期格式
3. 梯度流：验证梯度是否正确地通过自定义注意力函数流动

### 调试

```python
import torch
from flash_dmattn import flash_dmattn_func_auto

torch.autograd.set_detect_anomaly(True)
attn = flash_dmattn_func_auto()
output = attn(q, k, v, attn_mask=attn_mask, attn_bias=attn_bias, is_causal=True)
if torch.isnan(output).any():
    print("⚠️ 注意力输出中检测到 NaN")
```

### 内存监控

```python
def print_memory_stats():
    if torch.cuda.is_available():
        print(f"已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"已预留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"最大分配: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print_memory_stats()
attn = flash_dmattn_func_auto()
output = attn(q, k, v)
print_memory_stats()
```
