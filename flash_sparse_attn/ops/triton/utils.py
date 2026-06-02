import functools
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


@functools.lru_cache(maxsize=4096)
def num_splits_heuristic(
    seqlen_q: int,
    seqlen_k: int,
    num_SMs: int,
    TILE_M: int,
    TILE_N: int,
) -> int:
    """
    Determine the number of KV splits for FlashDecoding.

    Splits only when there are enough KV blocks to benefit from parallelism,
    and targets full SM occupancy by over-subscribing the M-block count.

    :param seqlen_q: Sequence length of queries.
    :param seqlen_k: Sequence length of keys.
    :param num_SMs: Number of streaming multiprocessors on the device.
    :param TILE_M: Tile size for M dimension.
    :param TILE_N: Tile size for N dimension.

    :return: Number of splits.
    """
    total_mblocks = triton.cdiv(seqlen_q, TILE_M)
    num_n_blocks = triton.cdiv(seqlen_k, TILE_N)
    max_splits = 1 << (max(num_SMs, 1).bit_length() - 1)
    if num_n_blocks <= 4:
        # 1 means no splitting
        return 1
    return max(1, min(num_SMs // max(total_mblocks, 1), max_splits, num_n_blocks))


@functools.lru_cache(maxsize=4096)
def window_sizes_heuristic(
    seqlen_k: int,
    num_heads_kv: int,
    device: torch.device,
    equal_bandwidth: bool = True,
    num_heads_kv_global: int = 0,
    tp_rank: int = 0,
) -> torch.Tensor:
    """
    Compute window sizes that partition the causal triangle into bands.

    :param seqlen_k: Sequence length of keys.
    :param num_heads_kv: Number of local KV heads on this rank.
    :param device: Target device.
    :param equal_bandwidth: If True, use equal-bandwidth partitioning for balanced decode load. If False, use equal-area partitioning for balanced forward and backward load.
    :param num_heads_kv_global: Total KV heads across all TP ranks. 0 means no TP (same as num_heads_kv).
    :param tp_rank: Tensor parallel rank of this process.

    :return: (num_heads_kv, 2) int32 tensor, columns are [left, right].
    """
    if num_heads_kv_global <= 0:
        num_heads_kv_global = num_heads_kv
    head_offset = tp_rank * num_heads_kv
    head_kv_idx = torch.arange(num_heads_kv + 1, dtype=torch.float32) + head_offset
    if equal_bandwidth:
        breakpoints = (seqlen_k * head_kv_idx / num_heads_kv_global).to(torch.int32)
    else:
        breakpoints = (
            seqlen_k * (1.0 - torch.sqrt(1.0 - head_kv_idx / num_heads_kv_global))
        ).to(torch.int32)
    window_size_left = torch.clamp(breakpoints[1:] - 1, min=0)
    window_size_right = breakpoints[:-1]
    return torch.stack([window_size_left, window_size_right], dim=1).to(device)
