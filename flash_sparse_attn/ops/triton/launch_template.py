"""
NOTE: Setting num_ctas=2 for the kernel can trigger Triton's PlanCTA assertion
Setting num_ctas=1 for now to avoid this issue, but we may want to revisit this in the future
"""

import functools
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from filelock import ReadWriteLock

import torch
import triton

from flash_sparse_attn.ops.triton import cache_utils


TRITON_CACHE_DIR: str = os.getenv(
    "FLASH_SPARSE_ATTENTION_LAUNCH_CONFIG_DIR",
    str(Path(__file__).resolve().parent / "launch_config"),
)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """
    Hash all Triton Python sources plus runtime ABI stamps into a short fingerprint.
    The fingerprint changes whenever any .py file under flash_sparse_attn/ops/triton is
    modified, the Python minor version changes, or the triton package version changes.

    :return: SHA-256 hex digest string.
    """
    triton_root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"triton={getattr(triton, '__version__', 'unknown')}".encode())
    for src in sorted(triton_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(triton_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)
    return h.hexdigest()


def _seqlen_bucket(seqlen: int) -> int:
    if seqlen <= 0:
        return 1
    return triton.next_power_of_2(seqlen)


def _sanitize_device_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


class LaunchConfigCache:
    """
    Persistent per-device cache for Triton kernel launch configurations.
    Stores tuned (TILE_M, TILE_N, num_warps, num_stages, num_ctas) keyed by
    kernel name, sequence length buckets, and dispatch flags. Cache is namespaced
    under a source fingerprint directory for automatic invalidation on code changes.
    """

    __slots__ = ("_memory", "_fingerprint")

    def __init__(self):
        self._memory: dict[str, dict[tuple, tuple]] = {}
        self._fingerprint: str = _compute_source_fingerprint()

    def _get_cache_dir(self) -> Path:
        cache_dir = Path(TRITON_CACHE_DIR) / self._fingerprint
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _json_path(self, device_name: str) -> Path:
        return self._get_cache_dir() / f"{_sanitize_device_name(device_name)}.json"

    def _lock(self, device_name: str) -> ReadWriteLock:
        lock_path = self._get_cache_dir() / f"{_sanitize_device_name(device_name)}"
        return ReadWriteLock(str(lock_path), timeout=15)

    def _read_unlocked(self, device_name: str) -> dict[tuple, tuple]:
        json_path = self._json_path(device_name)
        result: dict[tuple, tuple] = {}
        if not json_path.exists():
            return result
        try:
            data = json.loads(json_path.read_text())
            for entry in data:
                key = (
                    entry["kernel"],
                    entry["seqlen_q_bucket"],
                    entry["seqlen_k_bucket"],
                    entry["tile_k"],
                    entry["is_local"],
                    entry["qhead_per_kvhead"],
                    entry.get("is_causal", False),
                    entry.get("pack_gqa", False),
                    entry.get("is_quant", False),
                )
                result[key] = tuple(entry["config"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            result = {}
        return result

    def _write_unlocked(self, device_name: str, cache: dict[tuple, tuple]) -> None:
        json_path = self._json_path(device_name)
        data = []
        for (k_name, sq, sk, tk, local, qpk, causal, pgqa, quant), config in sorted(
            cache.items(), key=lambda x: x[0]
        ):
            data.append(
                {
                    "kernel": k_name,
                    "seqlen_q_bucket": sq,
                    "seqlen_k_bucket": sk,
                    "tile_k": tk,
                    "is_local": local,
                    "qhead_per_kvhead": qpk,
                    "is_causal": causal,
                    "pack_gqa": pgqa,
                    "is_quant": quant,
                    "config": list(config),
                }
            )

        fd, tmp_path = tempfile.mkstemp(
            dir=str(json_path.parent), suffix=".json.tmp", prefix=json_path.stem
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, str(json_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_device(self, device_name: str) -> dict[tuple, tuple]:
        if device_name in self._memory:
            return self._memory[device_name]
        with self._lock(device_name).read_lock():
            result = self._read_unlocked(device_name)
        self._memory[device_name] = result
        return result

    def get(
        self,
        device: torch.device,
        kernel_name: str,
        seqlen_q: int,
        seqlen_k: int,
        tile_k: int,
        is_local: bool = False,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        pack_gqa: bool = False,
        is_quant: bool = False,
    ) -> tuple[int, int, int, int, int] | None:
        """
        Load a cached launch config for the given kernel specialization.

        :param device: The device to run the kernel on
        :param kernel_name: Name of the Triton kernel
        :param seqlen_q: Sequence length of queries
        :param seqlen_k: Sequence length of keys
        :param tile_k: Tile size along the K dimension
        :param is_local: Whether local mask is applied
        :param qhead_per_kvhead: Ratio of query heads to key/value heads
        :param is_causal: Whether causal mask is applied
        :param pack_gqa: Whether GQA packing is enabled
        :param is_quant: Whether quantization is used

        :return launch_config: Tuple of (TILE_M, TILE_N, num_warps, num_stages, num_ctas) or None on cache miss
        """
        device_name = torch.cuda.get_device_name(device)
        cache = self._load_device(device_name)
        key = (
            kernel_name,
            _seqlen_bucket(seqlen_q),
            _seqlen_bucket(seqlen_k),
            tile_k,
            is_local,
            qhead_per_kvhead,
            is_causal,
            pack_gqa,
            is_quant,
        )
        return cache.get(key)

    def put(
        self,
        device: torch.device,
        kernel_name: str,
        seqlen_q: int,
        seqlen_k: int,
        tile_k: int,
        is_local: bool = False,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        pack_gqa: bool = False,
        is_quant: bool = False,
        *,
        config: tuple[int, int, int, int, int],
    ) -> None:
        """
        Store a launch config to the per-device cache.

        :param device: The device to run the kernel on
        :param kernel_name: Name of the Triton kernel
        :param seqlen_q: Sequence length of queries
        :param seqlen_k: Sequence length of keys
        :param tile_k: Tile size along the K dimension
        :param is_local: Whether local mask is applied
        :param qhead_per_kvhead: Ratio of query heads to key/value heads
        :param is_causal: Whether causal mask is applied
        :param pack_gqa: Whether GQA packing is enabled
        :param is_quant: Whether quantization is used
        :param config: Tuple of (TILE_M, TILE_N, num_warps, num_stages, num_ctas)
        """
        device_name = torch.cuda.get_device_name(device)
        key = (
            kernel_name,
            _seqlen_bucket(seqlen_q),
            _seqlen_bucket(seqlen_k),
            tile_k,
            is_local,
            qhead_per_kvhead,
            is_causal,
            pack_gqa,
            is_quant,
        )
        # Skip disk write if memory cache already has the same config
        mem = self._memory.get(device_name)
        if mem is not None and mem.get(key) == config:
            return
        with self._lock(device_name).write_lock():
            disk_cache = self._read_unlocked(device_name)
            disk_cache[key] = config
            self._write_unlocked(device_name, disk_cache)
        self._memory[device_name] = disk_cache

    def clear(self) -> None:
        self._memory.clear()

    def stats(self) -> dict:
        total_entries = sum(len(v) for v in self._memory.values())
        return {
            "devices": len(self._memory),
            "entries": total_entries,
            "fingerprint": self._fingerprint,
        }


_cache_instance: LaunchConfigCache | None = None


def get_launch_config_cache() -> LaunchConfigCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LaunchConfigCache()
    return _cache_instance


def load_launch_config(
    device: torch.device,
    kernel_name: str,
    seqlen_q: int,
    seqlen_k: int,
    tile_k: int,
    is_local: bool = False,
    qhead_per_kvhead: int = 1,
    is_causal: bool = False,
    pack_gqa: bool = False,
    is_quant: bool = False,
) -> tuple[int, int, int, int, int] | None:
    """
    Load cached launch config for a kernel specialization.

    :param device: The device to run the kernel on
    :param kernel_name: Name of the Triton kernel
    :param seqlen_q: Sequence length of queries
    :param seqlen_k: Sequence length of keys
    :param tile_k: Tile size along the K dimension
    :param is_local: Whether local mask is applied
    :param qhead_per_kvhead: Ratio of query heads to key/value heads
    :param is_causal: Whether causal mask is applied
    :param pack_gqa: Whether GQA packing is enabled
    :param is_quant: Whether quantization is used

    :return launch_config: Tuple of (TILE_M, TILE_N, num_warps, num_stages, num_ctas) or None on cache miss
    """
    return get_launch_config_cache().get(
        device,
        kernel_name,
        seqlen_q,
        seqlen_k,
        tile_k,
        is_local,
        qhead_per_kvhead,
        is_causal,
        pack_gqa,
        is_quant,
    )


def store_launch_config(
    device: torch.device,
    kernel_name: str,
    seqlen_q: int,
    seqlen_k: int,
    tile_k: int,
    *,
    config: tuple[int, int, int, int, int],
    is_local: bool = False,
    qhead_per_kvhead: int = 1,
    is_causal: bool = False,
    pack_gqa: bool = False,
    is_quant: bool = False,
) -> None:
    """
    Store a launch config to the per-device JSON cache.

    :param device: The device to run the kernel on
    :param kernel_name: Name of the Triton kernel
    :param seqlen_q: Sequence length of queries
    :param seqlen_k: Sequence length of keys
    :param tile_k: Tile size along the K dimension
    :param config: Tuple of (TILE_M, TILE_N, num_warps, num_stages, num_ctas)
    :param is_local: Whether local mask is applied
    :param qhead_per_kvhead: Ratio of query heads to key/value heads
    :param is_causal: Whether causal mask is applied
    :param pack_gqa: Whether GQA packing is enabled
    :param is_quant: Whether quantization is used
    """
    get_launch_config_cache().put(
        device,
        kernel_name,
        seqlen_q,
        seqlen_k,
        tile_k,
        is_local,
        qhead_per_kvhead,
        is_causal,
        pack_gqa,
        is_quant,
        config=config,
    )


def extract_best_config(autotuned_kernel) -> tuple[int, int, int, int, int] | None:
    """
    Extract best config from a Triton autotuned kernel after it has run.

    :param autotuned_kernel: A Triton autotuned kernel instance

    :return launch_config: Tuple of (TILE_M, TILE_N, num_warps, num_stages, num_ctas) or None if no best config found
    """
    raw = getattr(autotuned_kernel, "_autotuned", autotuned_kernel)
    best = getattr(raw, "best_config", None)
    if best is None:
        return None
    return (
        best.kwargs.get("TILE_M", 64),
        best.kwargs.get("TILE_N", 64),
        best.num_warps,
        best.num_stages,
        getattr(best, "num_ctas", 1),
    )


def get_fwd_combine_launch_config(
    tile_k: int,
    device: torch.device,
    arch: int,
) -> tuple[int, int, int, int]:
    """
    Get launch configuration for forward combine kernel based on input parameters and device architecture.

    :param tile_k: Tile size along the K dimension
    :param device: The device to run the kernel on
    :param arch: The architecture of the device

    :return launch_config: Tuple of (tile_m, num_warps, num_stages, num_ctas) for launching the kernel
    """
    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    if device.type == "cuda":
        if arch // 10 == 8:
            tile_m = 4 if tile_k % 128 == 0 else (8 if tile_k % 64 == 0 else 16)
            return (tile_m, 4, 1, 1)
        elif arch // 10 == 9:
            tile_m = 8 if tile_k % 128 == 0 else (16 if tile_k % 64 == 0 else 32)
            return (tile_m, 4, 1, 1)
        elif arch // 10 == 10:
            tile_m = 16 if tile_k % 128 == 0 else (32 if tile_k % 64 == 0 else 64)
            return (tile_m, 4, 1, 1)
        elif arch // 10 == 12:
            tile_m = 4 if tile_k % 128 == 0 else (8 if tile_k % 64 == 0 else 16)
            return (tile_m, 4, 1, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


get_fwd_combine_launch_config = cache_utils.cache_launch_config(
    get_fwd_combine_launch_config
)


def get_dec_combine_launch_config(
    tile_k: int,
    device: torch.device,
    arch: int,
) -> tuple[int, int, int]:
    """
    Get launch configuration for decode combine kernel based on input parameters and device architecture.

    :param tile_k: Tile size along the K dimension
    :param device: The device to run the kernel on
    :param arch: The architecture of the device

    :return launch_config: Tuple of (num_warps, num_stages, num_ctas) for launching the kernel
    """
    if arch == -1:
        raise NotImplementedError(f"Unsupported device: {device} with arch {arch}")

    if device.type == "cuda":
        if arch // 10 == 8:
            num_stages = max(min(8, 512 // tile_k), 1)
            return (4, num_stages, 1)
        elif arch // 10 == 9:
            num_stages = max(min(8, 512 // tile_k), 1)
            return (4, num_stages, 1)
        elif arch // 10 == 10:
            num_stages = max(min(16, 512 // tile_k), 1)
            return (4, num_stages, 1)
        elif arch // 10 == 12:
            num_stages = max(min(4, 512 // tile_k), 1)
            return (4, num_stages, 1)
        else:
            raise NotImplementedError(f"Unsupported CUDA architecture: {arch}")
    else:
        raise NotImplementedError(f"Unsupported device type: {device.type}")


get_dec_combine_launch_config = cache_utils.cache_launch_config(
    get_dec_combine_launch_config
)
