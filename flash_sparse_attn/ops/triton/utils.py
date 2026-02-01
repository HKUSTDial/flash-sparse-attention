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
        capability = torch.cuda.get_device_capability(device)
        sm = f"sm{capability[0]}{capability[1]}"
        return f"cuda:{sm}"
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


FWD_BASE_AUTOTUNE_KAYS = ["IS_CAUSAL", "IS_LOCAL", "TILE_K"]


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
    BLCOK_N_OPTIONS = [256, 128, 64, 32]
    NUM_WARPS_OPTIONS = [2, 4, 8]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLCOK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
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


def get_fwd_base_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    pack_gqa: bool,
):
    """
    Get the grid function for the forward base kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param pack_gqa: Whether GQA packing is used

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(
                seqlen_q * (num_heads_q // num_heads_kv) if pack_gqa else seqlen_q,
                META["TILE_M"],
            ),
            num_heads_kv if pack_gqa else num_heads_q,
            batch_size,
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
        "head_dim must be less than or equal to 256 for efficient memory access`"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )
