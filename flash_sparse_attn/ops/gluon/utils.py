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


@functools.lru_cache(maxsize=8)
def get_device_arch(device: torch.device) -> int:
    """
    Get the architecture for a given device.

    :param device: torch device
    :type device: torch.device

    :return arch: architecture model as a number.
    """
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        return sm if sm >= 80 else -1
    if device.type in {"xpu", "mps", "cpu"}:
        return -1
    raise ValueError(f"Unsupported device: {device}")


@functools.lru_cache(maxsize=8)
def get_device_num_sms(device: torch.device) -> int:
    """
    Get the SM count for a given device.

    :param device: torch device
    :type device: torch.device

    :return num_sms: number of streaming multiprocessors.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


def ensure_contiguous(fn):
    """
    Decorator to ensure that all tensor inputs to the decorated function are contiguous.

    :param fn: function to be decorated
    :type fn: Callable[..., Any]

    :return wrapper: wrapped function
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def cache_launch_grid(fn, maxsize: int = 512):
    """
    Bounded cache for Triton grid-factory closures.

    :param fn: grid-factory function to wrap
    :type fn: Callable[..., Any]
    :param maxsize: maximum number of cached entries.
    :type maxsize: int

    :return wrapper: wrapped function.
    """
    return functools.lru_cache(maxsize=maxsize)(fn)


@functools.lru_cache(maxsize=4096)
def window_sizes_heuristic(
    seqlen_k: int,
    num_heads_kv: int,
    device: torch.device,
    equal_bandwidth: bool = True,
    window_near: int = 1024,
    window_sink: int = 64,
    num_heads_kv_global: Optional[int] = None,
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute token count window sizes that partition non-diagonal causal distances into bands.

    :param seqlen_k: Sequence length of keys.
    :param num_heads_kv: Number of local KV heads on this rank.
    :param device: Target device.
    :param equal_bandwidth: If True, use equal-bandwidth partitioning for balanced decode load. If False, use equal-area partitioning for balanced forward and backward load.
    :param window_near: Near-diagonal token count shared by all KV heads.
    :param window_sink: Sink token count shared by all KV heads.
    :param num_heads_kv_global: Total KV heads across all TP ranks.
    :param tp_rank: Tensor parallel rank of this process.
    :param tp_size: Tensor parallel world size.

    :return: int32 tensor with shape [num_heads_kv, 4], columns are [window_sink, window_left, window_right, window_near]. window_sink is the prefix-sink token count, window_left is the distant band token count, window_right is the token count gap after the near-diagonal window before the distant band, and window_near is the near-diagonal token count.
    """
    if num_heads_kv_global is None:
        num_heads_kv_global = num_heads_kv

    if tp_size is not None and tp_size > num_heads_kv_global:
        # Calculate how many ranks share each KV head
        ranks_per_kv_group = tp_size // num_heads_kv_global
        # Map tp_rank to the logical KV head group index
        logical_kv_head_group = (tp_rank or 0) // ranks_per_kv_group
        head_offset = logical_kv_head_group * num_heads_kv
    else:
        head_offset = (tp_rank or 0) * num_heads_kv

    head_kv_idx = torch.arange(num_heads_kv + 1, dtype=torch.float32) + head_offset
    window_near = min(max(window_near, 0), seqlen_k)
    window_sink = min(max(window_sink, 0), max(seqlen_k - window_near, 0))
    distance_span = max(seqlen_k - window_sink - window_near, 0)
    if equal_bandwidth:
        breakpoints = (distance_span * head_kv_idx / num_heads_kv_global).to(
            torch.int32
        )
    else:
        breakpoints = (
            distance_span * (1.0 - torch.sqrt(1.0 - head_kv_idx / num_heads_kv_global))
        ).to(torch.int32)
    window_size_left = breakpoints[1:] - breakpoints[:-1]
    window_size_right = breakpoints[:-1]
    window_size_near = torch.full_like(window_size_left, window_near)
    window_size_sink = torch.full_like(window_size_left, window_sink)
    return torch.stack(
        [window_size_sink, window_size_left, window_size_right, window_size_near], dim=1
    ).to(device)


@functools.lru_cache(maxsize=4096)
def num_splits_heuristic(
    batch_size: int,
    seqlen_parallel: int,
    seqlen_loop: int,
    num_heads_parallel: int,
    num_heads_kv: int,
    num_SMs: int,
    TILE_PARALLEL: int,
    TILE_LOOP: int,
    is_local: bool = False,
    window_near: int = 1024,
    window_sink: int = 64,
) -> int:
    """
    Determine the number of splits.

    :param batch_size: Batch size.
    :param seqlen_parallel: Sequence length mapped across program instances.
    :param seqlen_loop: Sequence length traversed inside each program instance.
    :param num_heads_parallel: Number of heads mapped across program instances.
    :param num_heads_kv: Number of KV heads.
    :param num_SMs: Number of streaming multiprocessors on the device.
    :param TILE_PARALLEL: Tile size for the program-parallel sequence dimension.
    :param TILE_LOOP: Tile size for the loop sequence dimension.
    :param is_local: Whether local attention is used.
    :param window_near: Near-diagonal token count shared by all KV heads.
    :param window_sink: Sink token count shared by all KV heads.

    :return: Number of splits.
    """
    total_parallel_blocks = (
        batch_size * num_heads_parallel * triton.cdiv(seqlen_parallel, TILE_PARALLEL)
    )
    num_loop_blocks = triton.cdiv(seqlen_loop, TILE_LOOP)
    effective_loop_blocks = num_loop_blocks
    if is_local:
        window_near = min(max(window_near, 0), seqlen_loop)
        window_sink = min(max(window_sink, 0), max(seqlen_loop - window_near, 0))
        distance_span = max(seqlen_loop - window_sink - window_near, 0)
        max_window_left = triton.cdiv(distance_span, max(num_heads_kv, 1))
        distant_blocks = triton.cdiv(max_window_left, TILE_LOOP)
        near_blocks = 1 if window_near > 0 else 0
        sink_blocks = triton.cdiv(window_sink, TILE_LOOP)
        max_split_blocks = max(distant_blocks + near_blocks + sink_blocks, 1)
        effective_loop_blocks = min(num_loop_blocks, max_split_blocks)
    max_splits = 1 << (max(num_SMs, 1).bit_length() - 1)
    if effective_loop_blocks <= 4:
        # 1 means no splitting
        return 1
    return max(
        1,
        min(
            num_SMs // max(total_parallel_blocks, 1),
            max_splits,
            effective_loop_blocks,
        ),
    )
