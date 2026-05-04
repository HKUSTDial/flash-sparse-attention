import torch
import triton

from dataclasses import dataclass, fields
from flash_sparse_attn.ops.gluon import cache_utils


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_blackwell_ultra():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 3)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    TILE_M: int = 256
    TILE_N: int = 128
    GROUP_SIZE_N: int | None = None
    SPLIT_EXP_FACTOR: int | None = None
    NUM_WARPS: int = 4
    MAXNREG: int = 128
    OCCUPANCY: int = 1
    USE_TMEM_RED: bool = False
    NUM_KV_BUFFERS: int | None = None
    USE_EXP2_TURNSTILE: bool | None = None


def _default_split_exp_factor(head_dim: int) -> int:
    return max(1, 256 // head_dim)


def _default_num_kv_buffers(head_dim: int, dtype: torch.dtype) -> int:
    is_fp16 = dtype in [torch.float16, torch.bfloat16]
    if is_fp16:
        return 3 if head_dim == 128 else 6
    return 4 if head_dim == 128 else 8


def get_fwd_launch_config(
    head_dim: int,
    seqlen: int,
    dtype: torch.dtype,
    is_causal: bool,
    use_tmem_red: bool,
    override: KernelConfig | None = None,
) -> KernelConfig:
    is_fp8 = dtype == torch.float8_e5m2
    is_bf16 = dtype == torch.bfloat16
    is_bwu = is_blackwell_ultra()

    block_m = 256
    block_n = 128
    group_size_n = 1
    split_exp_factor = _default_split_exp_factor(head_dim)
    num_warps = 4
    maxnreg = 128
    occupancy = 1
    use_selected_tmem_red = (
        use_tmem_red or (is_bwu and not is_causal)
    ) and not is_causal
    num_kv_buffers = _default_num_kv_buffers(head_dim, dtype)
    use_exp2_turnstile = head_dim == 64

    if is_causal:
        group_size_n = 8 if head_dim == 64 or seqlen <= 2048 else 4

    if head_dim == 128:
        split_exp_factor = 4
        if not is_causal and is_bf16 and seqlen <= 2048:
            group_size_n = 4
    elif not is_causal and head_dim == 64 and use_selected_tmem_red:
        split_exp_factor = 1
        if seqlen <= 1024:
            num_kv_buffers = 2
        elif seqlen >= 8192:
            maxnreg = 112
    elif is_causal and head_dim == 64:
        num_kv_buffers = 2
        if seqlen <= 1024:
            split_exp_factor = 2
        else:
            use_exp2_turnstile = False

    if is_fp8:
        if is_causal and head_dim == 64:
            group_size_n = 8 if seqlen <= 2048 else 4
            split_exp_factor = 4 if seqlen <= 2048 else 2
            maxnreg = 112 if seqlen >= 4096 else 128
            use_selected_tmem_red = False
            num_kv_buffers = 2
            use_exp2_turnstile = seqlen <= 1024
        elif is_causal and head_dim == 128:
            group_size_n = 8 if seqlen <= 2048 else 4
            split_exp_factor = 2 if seqlen <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = False
            num_kv_buffers = 4
            use_exp2_turnstile = False
        elif not is_causal and head_dim == 64:
            group_size_n = 1
            split_exp_factor = 2
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 2 if seqlen <= 1024 else 8
            use_exp2_turnstile = True
        elif not is_causal and head_dim == 128:
            group_size_n = 1
            split_exp_factor = 4 if seqlen <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 4
            use_exp2_turnstile = False
        else:
            group_size_n = 4 if is_causal else 1
            split_exp_factor = _default_split_exp_factor(head_dim)
            use_selected_tmem_red = use_tmem_red and not is_causal

    config = KernelConfig(
        TILE_M=block_m,
        TILE_N=block_n,
        GROUP_SIZE_N=group_size_n,
        SPLIT_EXP_FACTOR=split_exp_factor,
        NUM_WARPS=num_warps,
        MAXNREG=maxnreg,
        OCCUPANCY=occupancy,
        USE_TMEM_RED=use_selected_tmem_red,
        NUM_KV_BUFFERS=num_kv_buffers,
        USE_EXP2_TURNSTILE=use_exp2_turnstile,
    )
    if override is None:
        return config

    values = {
        field.name: getattr(override, field.name) for field in fields(KernelConfig)
    }
    values = {
        name: getattr(config, name) if value is None else value
        for name, value in values.items()
    }
    return KernelConfig(**values)


get_fwd_launch_config = cache_utils.cache_launch_config(get_fwd_launch_config)
