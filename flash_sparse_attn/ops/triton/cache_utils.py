import functools
import os

import torch

from flash_sparse_attn.ops.triton import utils


# Default cap for the grid-factory cache. Grid factories take per-call values
# like batch_size / seqlen_q, so their key space is unbounded in principle;
# a bounded LRU keeps memory flat under workload drift.
_GRID_FACTORY_CACHE_MAXSIZE = 512

# L4: compiled-kernel cache. Opt out via `FSA_TRITON_KERNEL_CACHE=0`.
# Requires Triton >= 3.6; relies on JITFunction.warmup returning a
# CompiledKernel and CompiledKernel.__getitem__(grid) returning a launcher.
_KERNEL_CACHE_ENABLED: bool = os.getenv("FSA_TRITON_KERNEL_CACHE", "1") != "0"
_COMPILED_KERNEL_CACHE: dict = {}

# Registry of all cached wrappers so `clear_all_caches` can flush them in bulk.
_REGISTERED_CACHES: list = []


def _register(cached) -> None:
    _REGISTERED_CACHES.append(cached)


@functools.lru_cache(maxsize=8)
def get_device_arch_key(device_type: str, device_index: int) -> tuple[str, int]:
    """
    Stable (device_type, arch) key for a specific device, safe for multi-GPU.
    """
    if device_type == "cuda":
        major, minor = torch.cuda.get_device_capability(device_index)
        sm = major * 10 + minor
        return device_type, sm if sm >= 80 else -1
    return device_type, -1


_register(get_device_arch_key)


@functools.lru_cache(maxsize=8)
def get_num_sms(device_type: str, device_index: int) -> int:
    """
    Cached SM count per device. Replaces repeated
    `torch.cuda.get_device_properties(...).multi_processor_count` calls on
    the hot launch path.
    """
    if device_type == "cuda":
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    raise NotImplementedError(
        f"get_num_sms not supported for device type: {device_type}"
    )


_register(get_num_sms)


def num_sms(device: torch.device) -> int:
    """Convenience wrapper accepting a `torch.device`."""
    idx = device.index if device.index is not None else 0
    return get_num_sms(device.type, idx)


def _current_arch_key() -> tuple[str, int]:
    """Arch key of the currently active device (cheap, cached)."""
    device = utils.get_device()
    idx = device.index if device.index is not None else 0
    return get_device_arch_key(device.type, idx)


def cache_launch_config(fn):
    """
    Cache launch-config lookups by (arch, *args, **kwargs).

    The wrapped function only depends on its args plus the active device arch
    (which it queries internally via `utils.get_arch`). Key space is bounded
    by the enumerable inputs (arch, tile sizes, a few flags), so an unbounded
    LRU is safe.
    """
    cached = functools.lru_cache(maxsize=None)(
        lambda _arch_key, *args, **kwargs: fn(*args, **kwargs)
    )

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return cached(_current_arch_key(), *args, **kwargs)

    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
    _register(wrapper)
    return wrapper


def cache_grid_factory(fn, maxsize: int = _GRID_FACTORY_CACHE_MAXSIZE):
    """
    Bounded cache for Triton grid-factory closures.

    The factory's inputs include per-call values (batch_size, seqlen, ...), so
    the cache is capped via LRU to avoid unbounded growth under varying
    shapes (common in training).
    """
    cached = functools.lru_cache(maxsize=maxsize)(fn)
    _register(cached)
    return cached


def clear_all_caches() -> None:
    """Clear every registered cache. Intended for tests and debugging."""
    for c in _REGISTERED_CACHES:
        c.cache_clear()
    _COMPILED_KERNEL_CACHE.clear()


# ---------------------------------------------------------------------------
# L4: compiled-kernel level caching (Triton >= 3.6).
#
# Host-side Triton launch overhead in Triton 3.x is dominated by the
# per-launch "specialization" pass in `JITFunction.run()`, which iterates every
# argument to build a cache key before the actual CUDA launch. For decode-like
# workloads (seqlen_q == 1) this can be a significant fraction of wall time.
#
# We bypass that by:
#   1. Calling `JITFunction.warmup(*args, grid=..., **kwargs)` once to obtain
#      the already-compiled `CompiledKernel` for a given specialization.
#   2. Stashing it in our own dict keyed by a lightweight fingerprint that
#      mirrors Triton's default specialization rules.
#   3. On subsequent launches, invoking `compiled[grid](*args)` directly,
#      skipping the arg-specialization loop inside Triton.
#
# The key must be *at least as discriminative* as Triton's own specialization
# key; otherwise a hit would return a compiled kernel built for different
# specialization and yield wrong results. `_arg_spec_class` below matches
# Triton's defaults: pointer div-by-16 for tensors, and (equal_1, div-by-16)
# for scalar ints. Non-int/float/bool args (e.g. None) are captured by value.
# ---------------------------------------------------------------------------


def _arg_spec_class(arg):
    """Cheap per-argument fingerprint matching Triton's default specialization."""
    if torch.is_tensor(arg):
        return (arg.dtype, (arg.data_ptr() % 16) == 0)
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, int):
        return (int, arg == 1, (arg % 16) == 0)
    if isinstance(arg, float):
        return float
    return arg  # None, str, and other hashable scalars


def _make_kernel_key(
    kernel_name: str,
    args: tuple,
    constexprs: tuple,
    num_warps: int,
    num_stages: int,
    num_ctas: int,
) -> tuple:
    return (
        kernel_name,
        _current_arch_key(),
        tuple(_arg_spec_class(a) for a in args),
        constexprs,
        num_warps,
        num_stages,
        num_ctas,
    )


class _CachedLauncher:
    """Launcher returned by `CachedKernel.__getitem__`. Defers caching to call."""

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
        if not _KERNEL_CACHE_ENABLED:
            self._kernel[self._grid](
                *args,
                num_warps=num_warps,
                num_stages=num_stages,
                num_ctas=num_ctas,
                **constexprs,
            )
            return

        # CompiledKernel expects a concrete tuple grid; resolve callables now.
        grid = self._grid
        if callable(grid):
            grid = grid(constexprs)
        if not isinstance(grid, tuple):
            grid = tuple(grid)
        # Pad to 3D, matching Triton's launcher expectations.
        if len(grid) < 3:
            grid = grid + (1,) * (3 - len(grid))

        constexpr_items = tuple(sorted(constexprs.items()))
        key = _make_kernel_key(
            self._kernel.fn.__name__,
            args,
            constexpr_items,
            num_warps,
            num_stages,
            num_ctas,
        )
        compiled = _COMPILED_KERNEL_CACHE.get(key)
        if compiled is None:
            compiled = self._kernel.warmup(
                *args,
                grid=grid,
                num_warps=num_warps,
                num_stages=num_stages,
                num_ctas=num_ctas,
                **constexprs,
            )
            _COMPILED_KERNEL_CACHE[key] = compiled
        compiled[grid](*args)


class CachedKernel:
    """
    Wraps a `triton.jit`-decorated function so that each launch reuses a
    previously-compiled `CompiledKernel` when the specialization matches.

    Usage (drop-in replacement; call sites are unchanged):

        _fwd_kernel = cache_utils.wrap_kernel(_fwd_kernel)
        ...
        _fwd_kernel[grid](*args, TILE_M=..., num_warps=..., num_stages=...)
    """

    __slots__ = ("_kernel",)

    def __init__(self, kernel):
        self._kernel = kernel

    def __getitem__(self, grid):
        return _CachedLauncher(self._kernel, grid)

    def __getattr__(self, name):
        # Forward unknown attrs (e.g. `.fn`, `.arg_names`, `.warmup`) to the
        # underlying JITFunction so code inspecting the kernel still works.
        return getattr(self._kernel, name)


def wrap_kernel(kernel):
    """Wrap a `triton.jit` kernel for L4 compiled-kernel caching."""
    return CachedKernel(kernel)
