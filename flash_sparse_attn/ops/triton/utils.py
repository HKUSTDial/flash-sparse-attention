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
def window_sizes_heuristic(
    seqlen_k: int,
    num_heads_kv: int,
    device: torch.device,
    equal_bandwidth: bool = True,
    window_dist: int = 1024,
    window_sink: int = 64,
    num_heads_kv_global: int = 0,
    tp_rank: int = 0,
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
    if num_heads_kv_global <= 0:
        num_heads_kv_global = num_heads_kv
    window_dist = min(max(window_dist, 0), seqlen_k)
    window_sink = min(max(window_sink, 0), max(seqlen_k - window_dist, 0))
    head_offset = tp_rank * num_heads_kv
    head_kv_idx = torch.arange(num_heads_kv + 1, dtype=torch.float32) + head_offset
    distance_span = max(seqlen_k - window_sink - window_dist, 0)
    if equal_bandwidth:
        breakpoints = (distance_span * head_kv_idx / num_heads_kv_global).to(
            torch.int32
        )
    else:
        breakpoints = (
            distance_span
            * (1.0 - torch.sqrt(1.0 - head_kv_idx / num_heads_kv_global))
        ).to(torch.int32)
    window_size_left = breakpoints[1:] - breakpoints[:-1]
    window_size_right = breakpoints[:-1]
    window_size_dist = torch.full_like(window_size_left, window_dist)
    window_size_sink = torch.full_like(window_size_left, window_sink)
    return torch.stack(
        [window_size_sink, window_size_left, window_size_right, window_size_dist], dim=1
    ).to(device)


def max_split_blocks_from_window_sizes(
    window_sizes: torch.Tensor, TILE_N: int
) -> int | None:
    """
    Estimate useful split-axis blocks for local attention.

    The exact useful split count depends on the max values stored in window_sizes.

    :param window_sizes: Int32 tensor with shape [num_heads_kv, 4], columns are [sink, left, right, dist]
    :param TILE_N: Tile size for N dimension.

    :return: Useful split-axis blocks, or None when CUDA Graph capture needs a static launch-grid bound.
    """
    if window_sizes is None or window_sizes.numel() == 0:
        return 1
    if (
        window_sizes.is_cuda
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        return None
    window_sizes = torch.clamp(window_sizes, min=0)
    window_sink = torch.max(window_sizes[:, 0])
    window_left = torch.max(window_sizes[:, 1])
    window_dist = torch.max(window_sizes[:, 3])
    distant_blocks = torch.div(window_left + TILE_N - 1, TILE_N, rounding_mode="floor")
    near_blocks = (window_dist > 0).to(window_left.dtype)
    sink_blocks = torch.div(window_sink + TILE_N - 1, TILE_N, rounding_mode="floor")
    split_blocks = torch.maximum(
        distant_blocks + near_blocks + sink_blocks,
        torch.ones((), dtype=window_left.dtype, device=window_sizes.device),
    )
    return int(split_blocks)


@functools.lru_cache(maxsize=4096)
def num_splits_heuristic(
    seqlen_q: int,
    seqlen_k: int,
    num_SMs: int,
    TILE_M: int,
    TILE_N: int,
    max_split_blocks: int | None = None,
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
    :param max_split_blocks: Optional cap on useful split-axis blocks. Local two-band attention should pass this to avoid assigning splits to empty gap regions.

    :return: Number of splits.
    """
    total_mblocks = triton.cdiv(seqlen_q, TILE_M)
    num_n_blocks = triton.cdiv(seqlen_k, TILE_N)
    effective_n_blocks = num_n_blocks
    if max_split_blocks is not None:
        effective_n_blocks = min(num_n_blocks, max(max_split_blocks, 1))
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
