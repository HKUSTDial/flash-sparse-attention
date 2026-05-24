import functools
import inspect
from collections import OrderedDict

import torch


_COMPILED_KERNEL_CACHE_MAXSIZE = 4096
_STATIC_BUFFER_POOL: dict[tuple, torch.Tensor] = {}


def get_static_buffer(shape, dtype, device, tag=""):
    key = (shape, dtype, device.type, device.index, tag)
    buf = _STATIC_BUFFER_POOL.get(key)
    if buf is not None and buf.shape == shape:
        return buf
    buf = torch.empty(shape, dtype=dtype, device=device)
    _STATIC_BUFFER_POOL[key] = buf
    return buf


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
    params = list(inspect.signature(fn).parameters.keys())
    device_idx = params.index("device") if "device" in params else None
    arch_idx = params.index("arch") if "arch" in params else None
    cached = functools.lru_cache(maxsize=None)(
        lambda _arch, *args, **kwargs: fn(*args, **kwargs)
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        device = (
            args[device_idx]
            if (device_idx is not None and device_idx < len(args))
            else kwargs.get("device")
        )
        arch = (
            args[arch_idx]
            if (arch_idx is not None and arch_idx < len(args))
            else kwargs.get("arch")
        )
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


class CachedKernel:
    """
    Wrap a `triton.jit`-decorated function so that each launch reuses a
    previously-compiled `CompiledKernel` when the specialization matches.
    """

    __slots__ = ("_kernel", "_cache")

    def __init__(self, kernel):
        self._kernel = kernel
        self._cache = OrderedDict()

    def __getitem__(self, grid):
        kernel = self._kernel
        kernel_warmup = kernel.warmup
        kernel_arg_names = kernel.arg_names
        cache = self._cache
        maxsize = _COMPILED_KERNEL_CACHE_MAXSIZE
        tensor_indices = None
        static_grid = None

        if not callable(grid):
            static_grid = tuple(grid)
            if len(static_grid) < 3:
                static_grid = static_grid + (1,) * (3 - len(static_grid))

        def launcher(
            *args,
            num_warps: int = 4,
            num_stages: int = 1,
            num_ctas: int = 1,
            **constexprs,
        ):
            nonlocal tensor_indices

            if static_grid is not None:
                launch_grid = static_grid
            else:
                launch_grid = tuple(grid(constexprs))
                if len(launch_grid) < 3:
                    launch_grid = launch_grid + (1,) * (3 - len(launch_grid))

            if tensor_indices is None:
                tensor_indices = tuple(
                    i for i, a in enumerate(args) if torch.is_tensor(a)
                )

            key = (
                tuple((args[i].dtype, args[i].device.type) for i in tensor_indices),
                tuple(constexprs.values()) if constexprs else (),
                num_warps,
                num_stages,
                num_ctas,
            )

            compiled = cache.get(key)
            if compiled is None:
                compiled = kernel_warmup(
                    *args,
                    grid=launch_grid,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    num_ctas=num_ctas,
                    **constexprs,
                )
                cache[key] = compiled
                if len(cache) > maxsize:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(key)

            runner = compiled[launch_grid]

            if not constexprs:
                runner(*args)
                return

            args_iter = iter(args)
            full_args = [
                constexprs[name] if name in constexprs else next(args_iter)
                for name in kernel_arg_names
            ]
            runner(*full_args)

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
