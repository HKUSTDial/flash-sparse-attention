import functools

import torch

from flash_sparse_attn.ops.triton import utils


_COMPILED_KERNEL_CACHE: dict = {}


@functools.lru_cache(maxsize=8)
def _arch_key(device_type: str, device_index: int) -> tuple[str, int]:
    """
    Get a stable (device_type, arch) key for a specific device.

    :param device_type: Device type string.
    :param device_index: Device index.

    :return key: Tuple of (device_type, arch).
    """
    if device_type == "cuda":
        major, minor = torch.cuda.get_device_capability(device_index)
        return device_type, major * 10 + minor
    return device_type, -1


@functools.lru_cache(maxsize=8)
def num_sms(device: torch.device) -> int:
    """
    Get the SM count for a `torch.device`.

    :param device: Torch device.

    :return num_sms: Number of streaming multiprocessors.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


def _current_arch_key() -> tuple[str, int]:
    """
    Get the arch key of the currently active device.

    :return key: Tuple of (device_type, arch).
    """
    device = utils.get_device()
    return _arch_key(device.type, device.index or 0)


def _infer_arch_key(args: tuple) -> tuple[str, int]:
    """
    Infer the arch key from the first tensor in `args`, else fall back to the
    current device.

    :param args: Runtime args tuple.

    :return key: Tuple of (device_type, arch).
    """
    for a in args:
        if torch.is_tensor(a):
            return _arch_key(a.device.type, a.device.index or 0)
    return _current_arch_key()


def cache_launch_config(fn):
    """
    Cache launch-config lookups by (arch, *args, **kwargs).

    :param fn: Launch-config function to wrap.

    :return wrapper: Wrapped function with cache.
    """
    cached = functools.lru_cache(maxsize=None)(
        lambda _arch, *args, **kwargs: fn(*args, **kwargs)
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return cached(_current_arch_key(), *args, **kwargs)

    return wrapper


def cache_grid_factory(fn, maxsize: int = 512):
    """
    Bounded cache for Triton grid-factory closures.

    :param fn: Grid-factory function to wrap.
    :param maxsize: Maximum number of cached entries.

    :return wrapper: Wrapped function.
    """
    return functools.lru_cache(maxsize=maxsize)(fn)


def _arg_spec(arg):
    """
    Build a lightweight per-argument fingerprint matching Triton's default
    specialization.

    :param arg: Runtime argument passed to a Triton kernel.

    :return spec: Hashable specialization tag.
    """
    if torch.is_tensor(arg):
        return arg.dtype, arg.device.type, arg.device.index, (arg.data_ptr() % 16) == 0
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, int):
        return int, arg == 1, (arg % 16) == 0
    if isinstance(arg, float):
        return float
    return arg


class _CachedLauncher:
    """
    Launcher returned by `CachedKernel.__getitem__`.
    """

    __slots__ = ("_kernel", "_grid")

    def __init__(self, kernel, grid):
        self._kernel = kernel
        self._grid = grid

    def __call__(
        self,
        *args,
        num_warps: int = 4,
        num_stages: int = 1,
        num_ctas: int = 1,
        **constexprs,
    ):
        grid = self._grid
        if callable(grid):
            grid = grid(constexprs)
        grid = tuple(grid)
        if len(grid) < 3:
            grid = grid + (1,) * (3 - len(grid))

        kernel = self._kernel
        key = (
            kernel.fn.__name__,
            _infer_arch_key(args),
            tuple(_arg_spec(a) for a in args),
            tuple(sorted(constexprs.items())),
            num_warps,
            num_stages,
            num_ctas,
        )
        compiled = _COMPILED_KERNEL_CACHE.get(key)
        if compiled is None:
            compiled = kernel.warmup(
                *args,
                grid=grid,
                num_warps=num_warps,
                num_stages=num_stages,
                num_ctas=num_ctas,
                **constexprs,
            )
            _COMPILED_KERNEL_CACHE[key] = compiled

        args_iter = iter(args)
        full_args = [
            constexprs[name] if name in constexprs else next(args_iter)
            for name in kernel.arg_names
        ]
        compiled[grid](*full_args)


class CachedKernel:
    """
    Wrap a `triton.jit`-decorated function so that each launch reuses a
    previously-compiled `CompiledKernel` when the specialization matches.
    """

    __slots__ = ("_kernel",)

    def __init__(self, kernel):
        self._kernel = kernel

    def __getitem__(self, grid):
        return _CachedLauncher(self._kernel, grid)

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def wrap_kernel(kernel):
    """
    Wrap a `triton.jit` kernel for compiled-kernel caching.

    :param kernel: Triton JIT kernel.

    :return wrapped: Wrapped kernel preserving the `kernel[grid](...)` syntax.
    """
    return CachedKernel(kernel)
