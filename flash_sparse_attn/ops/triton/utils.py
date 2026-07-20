import functools
import torch
import triton
from typing import Optional


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
def window_sizes_heuristic(
    seqlen_k: int,
    num_heads_kv: int,
    device: torch.device,
    equal_bandwidth: bool = True,
    window_dist: int = 1024,
    window_sink: int = 64,
    num_heads_kv_global: Optional[int] = None,
    tp_rank: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute token count window sizes that partition non-diagonal causal distances into bands.

    :param seqlen_k: Sequence length of keys.
    :param num_heads_kv: Number of local KV heads on this rank.
    :param device: Target device.
    :param equal_bandwidth: If True, use equal-bandwidth partitioning for balanced decode load. If False, use equal-area partitioning for balanced forward and backward load.
    :param window_dist: Near-diagonal token count shared by all KV heads.
    :param window_sink: Sink token count shared by all KV heads.
    :param num_heads_kv_global: Total KV heads across all TP ranks.
    :param tp_rank: Tensor parallel rank of this process.

    :return: int32 tensor with shape [num_heads_kv, 4], columns are [window_sink, window_left, window_right, window_dist]. window_sink is the prefix sink token count, window_left is the distant local band token count, window_right is the token count gap after the near-diagonal window before the distant band, and window_dist is the near-diagonal token count.
    """
    if num_heads_kv_global is None:
        num_heads_kv_global = num_heads_kv
    head_offset = (tp_rank or 0) * num_heads_kv
    head_kv_idx = torch.arange(num_heads_kv + 1, dtype=torch.float32) + head_offset
    window_dist = min(max(window_dist, 0), seqlen_k)
    window_sink = min(max(window_sink, 0), max(seqlen_k - window_dist, 0))
    distance_span = max(seqlen_k - window_sink - window_dist, 0)
    if equal_bandwidth:
        breakpoints = distance_span * head_kv_idx / num_heads_kv_global
    else:
        breakpoints = distance_span * (
            1.0 - torch.sqrt(1.0 - head_kv_idx / num_heads_kv_global)
        )
    window_size_left = breakpoints[1:] - breakpoints[:-1]
    window_size_right = breakpoints[:-1]
    window_size_dist = torch.full_like(window_size_left, window_dist, dtype=torch.int32)
    window_size_sink = torch.full_like(window_size_left, window_sink, dtype=torch.int32)
    return torch.stack(
        [window_size_sink, window_size_left, window_size_right, window_size_dist], dim=1
    ).to(device)


@functools.lru_cache(maxsize=4096)
def num_splits_heuristic(
    seqlen_q: int,
    seqlen_k: int,
    num_SMs: int,
    TILE_M: int,
    TILE_N: int,
    is_local: bool = False,
    num_heads_kv: int = 1,
    window_dist: int = 1024,
    window_sink: int = 64,
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
    :param is_local: Whether local attention is used.
    :param num_heads_kv: Number of KV heads.
    :param window_dist: Near-diagonal token count shared by all KV heads.
    :param window_sink: Sink token count shared by all KV heads.

    :return: Number of splits.
    """
    total_mblocks = triton.cdiv(seqlen_q, TILE_M)
    num_n_blocks = triton.cdiv(seqlen_k, TILE_N)
    effective_n_blocks = num_n_blocks
    if is_local:
        window_dist = min(max(window_dist, 0), seqlen_k)
        window_sink = min(max(window_sink, 0), max(seqlen_k - window_dist, 0))
        distance_span = max(seqlen_k - window_sink - window_dist, 0)
        max_window_left = triton.cdiv(distance_span, max(num_heads_kv, 1))
        distant_blocks = triton.cdiv(max_window_left, TILE_N)
        near_blocks = 1 if window_dist > 0 else 0
        sink_blocks = triton.cdiv(window_sink, TILE_N)
        max_split_blocks = max(distant_blocks + near_blocks + sink_blocks, 1)
        effective_n_blocks = min(num_n_blocks, max_split_blocks)
    max_splits = 1 << (max(num_SMs, 1).bit_length() - 1)
    if effective_n_blocks <= 4:
        # 1 means no splitting
        return 1
    return max(1, min(num_SMs // max(total_mblocks, 1), max_splits, effective_n_blocks))


def alloc_fn(size: int, alignment: int, stream):
    """
    TMA descriptors require a global memory allocation

    :param size: Size of the allocation in bytes.
    :param alignment: Alignment requirement in bytes.
    :param stream: CUDA stream for the allocation.

    :return: A torch.Tensor representing the allocated memory.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)
