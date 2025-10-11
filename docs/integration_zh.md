# Flash 动态掩码注意力集成指南

## 概述

本文档阐述了如何在 Flash Attention 框架中集成 Dynamic Mask Attention（动态掩码注意力）。通过将 Flash Attention 的高效显存利用方式与动态稀疏掩码结合，这一集成能够在极长序列场景下实现稀疏注意力的高效计算。

该集成方案采用统一的稀疏计算路径：Python 端负责预计算注意力掩码与偏置张量，CUDA 后端在前向与反向两个阶段执行基于块的跳过逻辑与稀疏算子调度。

## 目录

1. [集成架构](#集成架构)
2. [核心改动](#核心改动)
3. [实现细节](#实现细节)
4. [稀疏计算策略](#稀疏计算策略)
5. [内存布局](#内存布局)
6. [性能考量](#性能考量)
7. [API 变化](#api-变化)

## 集成架构

### 高层设计

动态掩码注意力的集成在前向与反向过程中统一采用块级稀疏执行路径：

1. **动态掩码计算**：Python 端预先生成注意力掩码（mask）与注意力偏置（bias）张量。
2. **统一稀疏执行**：CUDA 后端在块粒度上决定是否跳过计算，并执行稀疏化的注意力与梯度算子。
3. **内存优化**：通过共享内存别名与显式同步实现更高的共享内存复用率。

### 关键组件

- **注意力掩码**：形状为 `(batch, num_kv_heads, query_len, key_len)` 的二值张量（1.0 表示保留，0.0 表示跳过）。
- **注意力偏置**：与掩码形状一致的张量，在 Softmax 前加性注入。
- **块级跳过逻辑**：对 `(BlockM × BlockN)` tile 做 OR 归约判断是否执行计算。
- **LSE 缓存**：前向阶段缓存 log-sum-exp 结果，反向阶段复用以保持数值稳定。
- **共享内存别名**：动态复用共享内存缓冲区，配合 `__syncthreads()` 控制生命周期。
- **完备梯度链路**：在保留稀疏跳过能力的同时，确保梯度流动正确。

## 核心改动

### 1. 参数结构扩展（`flash.h`）

**目的**：扩展参数结构体以支持动态掩码与偏置信息，同时保留对 QKV 的统一访问接口。

```cpp
struct QKV_params {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    index_t q_batch_stride, k_batch_stride, v_batch_stride;
    index_t q_row_stride, k_row_stride, v_row_stride;
    index_t q_head_stride, k_head_stride, v_head_stride;
    int h, h_k;
    int h_h_k_ratio;
};

struct Mask_params {
    void *__restrict__ mask_ptr;
    index_t mask_batch_stride;
    index_t mask_head_stride;
    index_t mask_row_stride;
};

struct Bias_params {
    void *__restrict__ bias_ptr;
    index_t bias_batch_stride;
    index_t bias_head_stride;
    index_t bias_row_stride;
};

struct Flash_fwd_params : public QKV_params, public Mask_params, public Bias_params {
    // ...existing code...
    bool seqlenq_ngroups_swapped;
};
```

**设计要点**：
- 多重继承将 QKV、掩码、偏置的参数维度拆分，保持接口清晰。
- 为掩码与偏置提供完整的 stride 信息，以便在 CUDA 中高效寻址。
- 与原有 Flash Attention 的内存布局保持兼容，避免性能回退。

### 2. 内核特性与内存布局（`kernel_traits.h`）

**目的**：根据架构（SM75 / SM80+）选择合适的 MMA 原子与内存拷贝路径，为动态掩码操作提供最佳性能。

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kNWarps = kNWarps_;
    // ...existing code...
    using SmemCopyAtomMask = SmemCopyAtom;
    using SmemCopyAtomBias = SmemCopyAtom;
};
```

**设计要点**：
- 根据编译目标自动选择 `cp.async` 与 LDSM 指令路径。
- 统一掩码与偏置的共享内存加载策略，避免额外的 bank conflict。
- 模板化的类型安全保证不同精度（FP16/BF16）路径一致。

### 3. 块级信息扩展（`block_info.h`）

**目的**：在可变长度场景下计算掩码与偏置的块级偏移量，保证全局内存访问有序。

```cpp
template<bool Varlen = true>
struct BlockInfo {
    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb) {
        // ...existing code...
    }

    template <typename index_t>
    __forceinline__ __device__ index_t mask_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        index_t offset = sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
        sum_s_k == -1 ? offset += leftpad_k : offset += uint32_t(sum_s_k + leftpad_k);
        return offset;
    }

    // ...existing code...
};
```

**设计要点**：
- 提供统一的偏移量计算方法，简化内核中的地址计算。
- 同时支持固定长度与可变长度两种输入形式。
- 将左侧填充（left pad）纳入偏移量，保证稀疏掩码与 KV 缓存对齐。

### 4. 内存拷贝与算子工具（`utils.h`）

**目的**：提供布局转换、类型转换、warp 归约与通用 GEMM 包装，适配 Flash Attention 的内存层次结构。

```cpp
namespace FLASH_NAMESPACE {

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    // ...existing code...
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

// ...existing code...

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(/* ... */) {
    // ...existing code...
}

} // namespace FLASH_NAMESPACE
```

**设计要点**：
- 通过布局转换统一 MMA 累加器的访问方式，方便掩码逻辑在寄存器中操作。
- 提供针对 BF16 的专用类型转换，避免额外的精度损耗。
- Warp 归约与 GEMM 包装均支持将数据留在寄存器中，降低共享内存压力。

### 5. 动态掩码核心逻辑（`mask.h`）

**目的**：在寄存器层面将掩码与偏置应用到注意力得分上，同时处理因果掩码与边界情况。

```cpp
template <bool Causal_mask = false, typename TensorType, typename MaskType, typename BiasType>
__forceinline__ __device__ void apply_mask(
    TensorType &tensor,
    MaskType &mask,
    BiasType &bias,
    const float scale_softmax,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset,
    const int max_seqlen_q,
    const int warp_row_stride) {
    // ...existing code...
}
```

**设计要点**：
- 在 `tensor` 保持 MMA 布局的情况下，逐元素应用掩码、偏置与缩放因子。
- 因果掩码通过列索引上限裁剪实现，与动态掩码兼容。
- 被掩盖的位置直接写入 `-INFINITY`，防止 Softmax 后出现数值污染。

### 6. 反向链路扩展（`flash_bwd_kernel.h`）

**目的**：在反向传播中复用动态掩码逻辑，确保梯度仅在活跃 tile 上计算。

```cpp
struct Flash_bwd_params : public Flash_fwd_params {
    // ...existing code...
};

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel = false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb,
                                                  const int bidh, const int n_block) {
    // ...existing code...
}
```

**设计要点**：
- 反向路径沿用前向阶段的 tile 活跃性判断，跳过完全被掩码的块。
- 结合 LSE 缓存，重算前向 Softmax 时保持数值稳定。
- 保证五个梯度 GEMM 在活跃 tile 上依旧串联执行，避免梯度缺失。

### 7. 前向内核改造（`flash_fwd_kernel.h`）

**目的**：在主注意力内核中插入动态掩码流程，同时保持 Flash Attention 的高并发与共享内存利用率。

```cpp
template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb,
                                              const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    // ...existing code...
}
```

**设计要点**：
- 按 tile 裁剪逻辑提前判断是否加载 K/V，降低无效内存访问。
- 仅在提供掩码/偏置时启用相应的分支，保持向后兼容。
- 通过模板参数在编译期裁剪分支，减少运行期开销。

### 8. 启动模板更新（`flash_fwd_launch_template.h`）

**目的**：在 kernel launch 阶段配置共享内存需求、模板实例化与错误处理，适配动态掩码的新资源需求。

```cpp
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel,
                            bool Is_causal, bool Is_even_MN, bool Is_even_K,
                            bool Is_softcap, bool Return_softmax) {
    // ...existing code...
}

// ...existing code...

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // ...existing code...
}
```

**设计要点**：
- 统一宏定义减少重复代码，便于扩展到新的 kernel 变体。
- 针对不支持的架构给出明确的构建期/运行期错误提示。
- 在 launch 前计算共享内存需求，必要时启用 `cudaFuncSetAttribute` 进行配置。

### 9. Python 接口扩展（`flash_api.cpp`）

**目的**：扩展 C++/PyBind11 接口以接受掩码与偏置张量，并提供全面的数据校验。

```cpp
void set_params_fprop(
    Flash_fwd_params &params,
    // ...existing code...
) {
    // ...existing code...
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,
    // ...existing code...
    const bool return_softmax) {
    // ...existing code...
    return {out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashDynamicMaskAttention";
    // ...existing code...
}
```

**设计要点**：
- 对输入张量的形状、dtype、device 进行全面校验。
- 保持原有参数顺序，新增参数保持向后兼容的默认行为。
- 当掩码或偏置未提供时，自动填充零值张量以保证接口易用性。

## 实现细节

### C++ API 接口

C++ 端对外暴露如下核心函数，用于前向、可变长度前向与反向计算：

```cpp
namespace FLASH_NAMESPACE {

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    // ...existing code...
    const bool return_softmax);

std::vector<at::Tensor> mha_varlen_fwd(/* ... */);

std::vector<at::Tensor> mha_bwd(/* ... */);

} // namespace FLASH_NAMESPACE
```

- `mha_fwd`：标准批量前向，支持稀疏掩码与偏置。
- `mha_varlen_fwd`：支持变长序列并使用累计长度数组。
- `mha_bwd`：完成梯度计算，返回 dQ / dK / dV / dBias / dMask 等张量。

### 参数设置与校验

`set_params_fprop` 会在调用前:

- 重置 `Flash_fwd_params` 并写入基本维度信息。
- 将掩码与偏置的设备指针、stride、批次数等全部注册。
- 基于输入 `dtype` 设置缩放因子与 `softcap`，同时准备缓存指针。

### Python 绑定与接口

PyBind11 模块对外暴露 `mha_fwd`、`mha_bwd`、`varlen_fwd` 等接口，文档字符串说明了参数要求与返回值。用户可通过 Python 直接调用 C++/CUDA 实现。

### Python 前端集成示例

```python
import torch
import torch.nn as nn
import flash_dmattn_cuda as flash_dmattn

class DynamicMaskAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ...existing code...

    def forward(self, query_states, key_states, value_states, attn_mask, attn_bias):
        out, softmax_lse = flash_dmattn.fwd(
            query_states, key_states, value_states,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            return_softmax=True,
        )
        return out, softmax_lse
```

- 前端模块负责生成 `attn_mask`（布尔）与 `attn_bias`（与 Q/K/V dtype 相同）。
- 内部 `_flash_dynamic_mask_attention_forward` 会根据需要补零偏置并调用后端。
- 输入张量默认为 `(batch, seq_len, num_heads, head_dim)` 排列，内部会自动转置到后端期望格式。

## 稀疏计算策略

### 块级跳过逻辑

- 在加载 Q tile 后，先将掩码 tile 拷贝到共享内存并执行 OR 归约。
- 若整块被掩盖，则跳过 K/V 加载与后续计算，只推进指针。
- 对活跃块执行常规注意力流程，并复用共享内存保存 Softmax 结果。

### 前向算法

```pseudo
for m_block in M_tiles:
    load Q_tile
    load mask_tile -> shared
    any_active = or_reduce(mask_tile)
    if not any_active:
        continue
    load K_tile, V_tile
    compute scaled dot product
    apply mask & bias in registers
    softmax -> write O_tile
```

- 掩码裁剪保证 Tile 内所有无效位置直接输出 `-INF`。
- Softmax 前的缩放与偏置添加与密集版本保持一致。
- 通过共享内存别名（sMask ↔ sP）减少显存占用。

### 反向算法

```pseudo
for m_block in reversed(M_tiles):
    load Q_tile, dO_tile
    load mask_tile -> shared
    if tile inactive:
        continue
    recompute scores with cached LSE
    propagate gradients for dS, dV, dK, dQ
```

- 仅对活跃块执行五个 GEMM 组合，减少稀疏场景下的冗余计算。
- 使用前向缓存的 LSE 确保 Softmax 反向的数值稳定性。
- 对被跳过的块梯度自然为零，避免写入污染。

### 跳过逻辑正确性

- 若 tile 全部被掩码，输出必为零，跳过计算不会影响结果。
- 反向阶段活跃性与前向保持一致，保证梯度对应关系不被破坏。
- 由于被掩盖位置在 Softmax 前已写入 `-INF`，LSE 亦不受影响。

## 内存布局

### 全局内存组织

```
Q:      [batch, seqlen_q, num_heads, head_dim]
K:      [batch, seqlen_k, num_kv_heads, head_dim]
V:      [batch, seqlen_k, num_kv_heads, head_dim]
Mask:   [batch, num_kv_heads, seqlen_q, seqlen_k]
Bias:   [batch, num_kv_heads, seqlen_q, seqlen_k]
Output: [batch, seqlen_q, num_heads, head_dim]
```

### 共享内存布局（每个线程块）

```
Q Tile   : [kBlockM, head_dim]
K Tile   : [kBlockN, head_dim]
V Tile   : [kBlockN, head_dim]
S Tile   : [kBlockM, kBlockN]
Mask Tile: [kBlockM, kBlockN]
Bias Tile: [kBlockM, kBlockN]
```

### 寄存器布局（每个线程）

```
Q Frag   : [MMA_M, head_dim / N]
K Frag   : [MMA_N, head_dim / N]
V Frag   : [MMA_N, head_dim / N]
S Frag   : [MMA_M, MMA_N]
Mask Frag: [MMA_M, MMA_N]
Bias Frag: [MMA_M, MMA_N]
Acc Frag : [MMA_M, head_dim / N]
```

### 内存访问模式

- 掩码与偏置与 K/V 共享相同的 `Copy_Atom` 配置，确保 128-bit 对齐、最大化带宽。
- 共享内存拷贝后通过 `local_partition` 分配给线程，避免 bank conflict。
- `convert_layout_acc_rowcol` 将 MMA 布局转换为行/列布局，方便寄存器操作。

### 共享内存优化

- **别名复用**：`sMask` 在使用后可重用为 `sP`（Softmax 输出），`sBias` 可重用为 `sdS`。
- **同步屏障**：在重用前使用 `__syncthreads()` 确保所有线程完成对旧数据的使用。
- **块尺寸选择**：根据稀疏度与共享内存限制调整 tile 尺寸，提高 SM 占用率。

## 性能考量

- **共享内存复用**：别名策略可将共享内存占用削减约 30%。
- **块级跳过**：当稀疏度为 75% 时，可获得约 3× 的前向提速；稀疏度 90% 时可达到 ~6×。
- **带宽优化**：跳过无效 tile 可以线性降低全局内存带宽需求。
- **同步开销**：跳过路径的额外 OR 归约占总时间 <5%，可忽略不计。
- **硬件自适应**：针对 SM75/SM80+ 的不同指令集做了专门优化，确保跨架构稳定收益。

## API 变化

### 新增必要参数

- `attn_mask` (`torch.Tensor`): 形状 `(batch, num_kv_heads, seqlen_q, seqlen_k)` 的布尔张量，决定稀疏模式。
- `attn_bias` (`torch.Tensor`): 形状与掩码一致的加性偏置张量，dtype 与 Q/K/V 保持一致。

### 更新的函数签名

```python
def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    attn_bias: torch.Tensor,
    is_causal: bool = False,
    return_softmax: bool = False,
    **kwargs
) -> List[torch.Tensor]:
    ...
```

### 向后兼容说明

- 这是一个破坏性更新，旧的 Flash Attention 调用需显式提供掩码与偏置。
- 若业务场景不需要稀疏掩码，可传入全 1 掩码与全 0 偏置实现与旧版一致的行为。
- 缺省值在 Python 前端会自动补齐，降低迁移的代码改动。

### 完整用法示例

```python
import torch
from flash_dmattn.integrations.flash_dynamic_mask_attention import (
    flash_dynamic_mask_attention_forward,
)

batch, seq_q, seq_k, n_heads, head_dim = 2, 4096, 4096, 16, 128
q = torch.randn(batch, seq_q, n_heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
mask = torch.ones(batch, n_heads, seq_q, seq_k, device=q.device, dtype=torch.bool)
bias = torch.zeros(batch, n_heads, seq_q, seq_k, device=q.device, dtype=q.dtype)

out = flash_dynamic_mask_attention_forward(
    query_states=q,
    key_states=k,
    value_states=v,
    attention_mask=mask,
    attention_bias=bias,
    return_attn_probs=False,
)
```

- `flash_dynamic_mask_attention_forward` 会自动完成张量转置、补零偏置等准备工作。
- 若指定 `return_attn_probs=True`，将返回经过 Softmax 的注意力概率，用于调试或可视化。
- 稀疏模式的 mask 可通过 `flash_dmattn.utils.mask.MaskMod` 组合生成。

## 附加建议

- 修改 CUDA 核心代码后，至少运行 `benchmarks/forward_equivalence.py` 与 `benchmarks/grad_equivalence.py` 进行回归验证。
- 构建扩展时可使用 `pip install -e . --no-build-isolation`，必要时设置 `FLASH_DMATTN_CUDA_ARCHS` 指定目标架构。
- 若仅依赖 Triton/Flex 后端，可通过环境变量 `FLASH_DMATTN_SKIP_CUDA_BUILD=1` 跳过 CUDA 构建。
