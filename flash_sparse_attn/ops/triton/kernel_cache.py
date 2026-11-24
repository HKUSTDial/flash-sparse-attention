import json
import os
import tempfile
from pathlib import Path

import torch
from filelock import ReadWriteLock


class LaunchConfigCache:
    """
    Persistent per-device cache for Triton kernel launch configurations.
    Stores tuned (TILE_M, TILE_N, num_warps, num_stages, num_ctas) keyed by
    kernel name, sequence length buckets, and dispatch flags. To invalidate,
    delete the cache directory or re-run with autotune=True.
    """

    __slots__ = ("_memory",)

    def __init__(self):
        self._memory: dict[str, dict[tuple, tuple]] = {}

    def _get_cache_dir(self) -> Path:
        cache_dir = Path(
            os.getenv(
                "FLASH_SPARSE_ATTENTION_LAUNCH_CONFIG_DIR",
                str(Path(__file__).resolve().parent / "launch_config"),
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _json_path(self, device_name: str) -> Path:
        device_name = device_name.replace(" ", "_").replace("/", "_")
        return self._get_cache_dir() / f"{device_name}.json"

    def _lock(self, device_name: str) -> ReadWriteLock:
        lock_path = self._get_cache_dir() / device_name.replace(" ", "_").replace("/", "_")
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
                    entry["qhead_per_kvhead"],
                    entry["pack_gqa"],
                    entry["is_causal"],
                    entry["is_local"],
                    entry["is_quant"],
                    entry["tile_k"],
                )
                result[key] = tuple(entry["config"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            result = {}
        return result

    def _write_unlocked(self, device_name: str, cache: dict[tuple, tuple]) -> None:
        json_path = self._json_path(device_name)
        data = []
        for (k_name, sq, sk, qpk, pgqa, causal, local, quant, tk), config in sorted(
            cache.items(), key=lambda x: repr(x[0])
        ):
            data.append(
                {
                    "config": list(config),
                    "kernel": k_name,
                    "seqlen_q_bucket": sq,
                    "seqlen_k_bucket": sk,
                    "qhead_per_kvhead": qpk,
                    "pack_gqa": pgqa,
                    "is_causal": causal,
                    "is_local": local,
                    "is_quant": quant,
                    "tile_k": tk,
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

    def load(
        self,
        device: torch.device,
        kernel_name: str,
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        is_causal: bool,
        is_local: bool,
        is_quant: bool,
        tile_k: int,
    ) -> tuple[int, int, int, int, int] | None:
        device_name = torch.cuda.get_device_name(device)
        cache = self._load_device(device_name)
        key = (
            kernel_name,
            seqlen_q,
            seqlen_k,
            qhead_per_kvhead,
            pack_gqa,
            is_causal,
            is_local,
            is_quant,
            tile_k,
        )
        return cache.get(key)

    def store(
        self,
        config: tuple[int, int, int, int, int],
        device: torch.device,
        kernel_name: str,
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        is_causal: bool,
        is_local: bool,
        is_quant: bool,
        tile_k: int,
    ) -> None:
        device_name = torch.cuda.get_device_name(device)
        key = (
            kernel_name,
            seqlen_q,
            seqlen_k,
            qhead_per_kvhead,
            pack_gqa,
            is_causal,
            is_local,
            is_quant,
            tile_k,
        )
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
        return {"devices": len(self._memory), "entries": total_entries}


_launch_config_cache = LaunchConfigCache()
