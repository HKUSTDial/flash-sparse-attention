import functools
import inspect
from collections import OrderedDict

import torch


_COMPILED_KERNEL_CACHE_MAXSIZE = 4096
_COMPILED_KERNEL_CACHE: OrderedDict = OrderedDict()
_LAUNCHER_CACHE_MAXSIZE = 1024


def _compiled_cache_get(key):
    compiled = _COMPILED_KERNEL_CACHE.get(key)
    if compiled is not None:
        _COMPILED_KERNEL_CACHE.move_to_end(key)
    return compiled


def _compiled_cache_put(key, compiled):
    _COMPILED_KERNEL_CACHE[key] = compiled
    _COMPILED_KERNEL_CACHE.move_to_end(key)
    if len(_COMPILED_KERNEL_CACHE) > _COMPILED_KERNEL_CACHE_MAXSIZE:
        _COMPILED_KERNEL_CACHE.popitem(last=False)


@functools.lru_cache(maxsize=8)
def get_device_num_sms(device: torch.device) -> int:
    """
    Get the SM count for a given device.

    :param device: Torch device.

    :return num_sms: Number of streaming multiprocessors.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=8)
def get_device_arch(device: torch.device) -> int:
    """
    Get the architecture for a given device.

    :param device: Torch device.

    :return arch: Architecture model as a number.
    """
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor
        return sm if sm >= 80 else -1
    if device.type in {"xpu", "mps", "cpu"}:
        return -1
    raise ValueError(f"Unsupported device: {device}")


def cache_launch_config(fn):
    """
    Cache launch-config lookups by (arch, *args, **kwargs).

    :param fn: Launch-config function to wrap.

    :return wrapper: Wrapped function with cache.
    """
    cached = functools.lru_cache(maxsize=None)(
        lambda _arch, *args, **kwargs: fn(*args, **kwargs)
    )
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        device = bound.arguments.get("device")
        arch = bound.arguments.get("arch")
        return cached((device.type, arch), *args, **kwargs)

    return wrapper


def cache_launch_grid(fn, maxsize: int = 512):
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


class CachedKernel:
    """
    Wrap a `triton.jit`-decorated function so that each launch reuses a
    previously-compiled `CompiledKernel` when the specialization matches.
    """

    __slots__ = ("_kernel", "_launcher_cache")

    def __init__(self, kernel):
        self._kernel = kernel
        self._launcher_cache = OrderedDict()

    @staticmethod
    def _normalize_static_grid(grid):
        grid = tuple(grid)
        if len(grid) < 3:
            grid = grid + (1,) * (3 - len(grid))
        return grid

    def __getitem__(self, grid):
        try:
            cache_key = grid if callable(grid) else self._normalize_static_grid(grid)
            cached_launcher = self._launcher_cache.get(cache_key)
            if cached_launcher is not None:
                self._launcher_cache.move_to_end(cache_key)
                return cached_launcher
        except TypeError:
            cache_key = None

        kernel = self._kernel
        kernel_fn_name = kernel.fn.__name__
        kernel_warmup = kernel.warmup
        kernel_arg_names = kernel.arg_names
        is_tensor = torch.is_tensor
        arg_spec = _arg_spec
        compiled_cache_get = _compiled_cache_get
        compiled_cache_put = _compiled_cache_put
        get_arch = get_device_arch

        static_grid = None
        if not callable(grid):
            static_grid = self._normalize_static_grid(grid)

        def launcher(
            *args,
            num_warps: int = 4,
            num_stages: int = 1,
            num_ctas: int = 1,
            **constexprs,
        ):
            launch_grid = static_grid
            if launch_grid is None:
                launch_grid = grid
                launch_grid = launch_grid(constexprs)
                launch_grid = CachedKernel._normalize_static_grid(launch_grid)

            device = next(a.device for a in args if is_tensor(a))
            constexpr_items = tuple(constexprs.items()) if constexprs else ()
            key = (
                kernel_fn_name,
                (device.type, get_arch(device)),
                tuple(arg_spec(a) for a in args),
                constexpr_items,
                num_warps,
                num_stages,
                num_ctas,
            )
            compiled = compiled_cache_get(key)
            if compiled is None:
                compiled = kernel_warmup(
                    *args,
                    grid=launch_grid,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    num_ctas=num_ctas,
                    **constexprs,
                )
                compiled_cache_put(key, compiled)

            if not constexprs:
                compiled[launch_grid](*args)
                return

            args_iter = iter(args)
            full_args = [
                constexprs[name] if name in constexprs else next(args_iter)
                for name in kernel_arg_names
            ]
            compiled[launch_grid](*full_args)

        if cache_key is not None:
            self._launcher_cache[cache_key] = launcher
            self._launcher_cache.move_to_end(cache_key)
            if len(self._launcher_cache) > _LAUNCHER_CACHE_MAXSIZE:
                self._launcher_cache.popitem(last=False)
        return launcher

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def wrap_kernel(kernel):
    """
    Wrap a `triton.jit` kernel for compiled-kernel caching.

    :param kernel: Triton JIT kernel.

    :return wrapped: Wrapped kernel preserving the `kernel[grid](...)` syntax.
    """
    return CachedKernel(kernel)
