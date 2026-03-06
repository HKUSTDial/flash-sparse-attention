import functools
from typing import Optional
import torch
import triton


def get_device():
    """
    Get the appropriate device for computation.

    :return device: torch.device object
    """
    # TODO: add NPU
    # Works for both NVIDIA and AMD
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel XPU if available
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_arch(device: torch.device):
    """
    Get the architecture string for the given device.

    :param device: torch.device object

    :return arch: Architecture string
    """
    if device == torch.device("cuda"):
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        return f"{sm}" if sm >= 80 else "N/A"
    elif device == torch.device("xpu"):
        return "N/A"
    elif device == torch.device("mps"):
        return "N/A"
    elif device == torch.device("cpu"):
        return "N/A"
    else:
        raise ValueError(f"Unsupported device: {device}")


def ensure_contiguous(fn):
    """
    Decorator to ensure that all tensor inputs to the decorated function are contiguous.

    :param fn: Function to be decorated

    :return wrapper: Wrapped function
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def num_splits_heuristic(
    total_mblocks: int,
    num_SMs: int,
    num_n_blocks: int,
    max_splits: int = 128,
) -> int:
    """
    Determine the number of KV splits for FlashDecoding.

    Splits only when there are enough KV blocks to benefit from parallelism,
    and targets full SM occupancy by over-subscribing the M-block count.

    :param total_mblocks: Total number of M-blocks across batch and heads.
    :param num_SMs: Number of streaming multiprocessors on the device.
    :param num_n_blocks: Number of N-blocks.
    :param max_splits: Hard upper bound on number of splits.

    :return: Number of splits.
    """
    if num_n_blocks <= 4:
        # 1 means no splitting
        return 1
    return min(num_SMs // max(total_mblocks, 1), max_splits, num_n_blocks)


FWD_BASE_AUTOTUNE_KEYS = ["seqlen_q", "seqlen_k", "IS_CAUSAL", "IS_LOCAL", "TILE_K"]


FWD_SM90_AUTOTUNE_KEYS = ["seqlen_q", "seqlen_k", "IS_CAUSAL", "IS_LOCAL", "TILE_K"]


FWD_COMBINE_AUTOTUNE_KEYS = ["seqlen_q", "head_dim", "MAX_SPLITS"]


def get_fwd_base_autotune_configs(autotune: bool):
    """
    Get autotuning configurations for the forward base kernel.

    :param autotune: Whether to perform autotuning

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    if not autotune:
        if arch == "cuda:sm80":
            return [
                triton.Config(
                    {"TILE_M": 128, "TILE_N": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm90":
            return [
                triton.Config(
                    {"TILE_M": 128, "TILE_N": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm100":
            return [
                triton.Config(
                    {"TILE_M": 128, "TILE_N": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm120":
            return [
                triton.Config(
                    {"TILE_M": 128, "TILE_N": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        else:
            raise ValueError(f"Unsupported architecture for default config: {arch}")

    configs = []
    BLOCK_M_OPTIONS = [256, 128, 64, 32]
    BLOCK_N_OPTIONS = [256, 128, 64, 32]
    NUM_WARPS_OPTIONS = [4, 8]
    NUM_STAGES_OPTIONS = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTIONS:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_fwd_sm90_autotune_configs(autotune: bool):
    """
    Get autotuning configurations for the forward SM90 kernel.

    :param autotune: Whether to perform autotuning

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")
    if arch != "cuda:sm90":
        raise ValueError(
            "Autotuning for the SM90-specific kernel is only supported on SM90 architecture."
        )

    if not autotune:
        return [
            triton.Config(
                {"TILE_M": 128, "TILE_N": 64},
                num_warps=4,
                num_stages=1,
            )
        ]

    configs = []
    BLOCK_M_OPTIONS = [256, 128, 64, 32]
    BLOCK_N_OPTIONS = [256, 128, 64, 32]
    NUM_WARPS_OPTIONS = [4, 8]
    NUM_STAGES_OPTIONS = [1, 2, 3]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTIONS:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_fwd_combine_autotune_configs(autotune: bool):
    """
    Get autotuning configurations for the forward combine kernel.

    :param autotune: Whether to perform autotuning

    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    if not autotune:
        if arch == "cuda:sm80":
            return [
                triton.Config(
                    {"TILE_M": 32, "TILE_K": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm90":
            return [
                triton.Config(
                    {"TILE_M": 32, "TILE_K": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm100":
            return [
                triton.Config(
                    {"TILE_M": 32, "TILE_K": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        elif arch == "cuda:sm120":
            return [
                triton.Config(
                    {"TILE_M": 32, "TILE_K": 128},
                    num_warps=4,
                    num_stages=1,
                )
            ]
        else:
            raise ValueError(f"Unsupported architecture for default config: {arch}")

    configs = []
    BLOCK_M_OPTIONS = [64, 32]
    BLOCK_K_OPTIONS = [256, 128, 64, 32]
    NUM_WARPS_OPTIONS = [4, 8]
    NUM_STAGES_OPTIONS = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bk in BLOCK_K_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTIONS:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_K": bk,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_fwd_base_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    pack_gqa: bool,
    num_splits: int,
):
    """
    Get the grid function for the forward base kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param pack_gqa: Whether GQA packing is used
    :param num_splits: Number of KV splits

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(
                seqlen_q * (num_heads_q // num_heads_kv) if pack_gqa else seqlen_q,
                META["TILE_M"],
            ),
            num_heads_kv if pack_gqa else num_heads_q,
            batch_size * num_splits,
        )

    return grid


def get_fwd_sm90_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    pack_gqa: bool,
    num_splits: int,
):
    """
    Get the grid function for the forward SM90 kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param pack_gqa: Whether GQA packing is used
    :param num_splits: Number of KV splits

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(
                seqlen_q * (num_heads_q // num_heads_kv) if pack_gqa else seqlen_q,
                META["TILE_M"],
            ),
            num_heads_kv if pack_gqa else num_heads_q,
            batch_size * num_splits,
        )

    return grid


def get_fwd_combine_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
    head_dim: int,
):
    """
    Get the grid function for the forward combine kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads
    :param head_dim: Head dimension

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(seqlen_q, META["TILE_M"]),
            triton.cdiv(head_dim, META["TILE_K"]),
            batch_size * num_heads_q,
        )

    return grid


def assert_fwd_base_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
):
    """
    Assert the validity of inputs for the forward base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension

    :raises AssertionError: If any of the assertions fail
    """
    assert query.is_cuda and key.is_cuda and value.is_cuda, (
        "All inputs must be on CUDA device"
    )
    assert query.dtype in [torch.float16, torch.bfloat16], (
        "Input dtype must be float16 or bfloat16"
    )
    assert query.dtype == key.dtype == value.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )


def assert_fwd_sm90_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
):
    """
    Assert the validity of inputs for the forward base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension

    :raises AssertionError: If any of the assertions fail
    """
    assert query.is_cuda and key.is_cuda and value.is_cuda, (
        "All inputs must be on CUDA device"
    )
    assert query.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2], (
        "Input dtype must be float16, bfloat16, or float8_e5m2"
    )
    assert query.dtype == key.dtype == value.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )


def assert_fwd_sparse_base_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    seqlen_k: int = None,
    head_dim: int = None,
    gate_scale: float = None,
):
    """
    Assert the validity of inputs for the forward sparse base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param alpha: Alpha tensor
    :param delta: Delta tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param seqlen_k: Sequence length for keys
    :param head_dim: Head dimension
    :param gate_scale: Gate scaling factor

    :raises AssertionError: If any of the assertions fail
    """
    assert (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and alpha.is_cuda
        and delta.is_cuda
    ), "All inputs must be on CUDA device"
    assert query.dtype in [torch.float16, torch.bfloat16], (
        "Input dtype must be float16 or bfloat16"
    )
    assert query.dtype == key.dtype == value.dtype == alpha.dtype == delta.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    assert 0.0 < gate_scale * seqlen_k < 1.0, (
        "gate_scale must be in the range (0.0, 1.0 / seqlen_k) for valid gating behavior"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )
