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


def get_arch(device: torch.device):
    """
    Get the architecture string for the given device.

    :param device: torch.device object

    :return arch: Architecture model as a number
    """
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        return sm if sm >= 80 else -1
    elif device.type == "xpu":
        return -1
    elif device.type == "mps":
        return -1
    elif device.type == "cpu":
        return -1
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
