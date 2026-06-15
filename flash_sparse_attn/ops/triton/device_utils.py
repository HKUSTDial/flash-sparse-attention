import functools
from typing import Optional

import torch


SUPPORTED_DEVICE_TYPES = ("cuda", "musa")


def _get_backend(device_type: str):
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise RuntimeError(f"Unsupported Triton device type: {device_type}")
    backend = getattr(torch, device_type, None)
    if backend is None:
        raise RuntimeError(f"torch.{device_type} is not available")
    return backend


def is_available(device_type: str) -> bool:
    backend = getattr(torch, device_type, None)
    return backend is not None and backend.is_available()


def get_available_device() -> Optional[torch.device]:
    for device_type in SUPPORTED_DEVICE_TYPES:
        if is_available(device_type):
            return torch.device(device_type)
    return None


def get_backend(device: torch.device):
    return _get_backend(torch.device(device).type)


def normalize_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type not in SUPPORTED_DEVICE_TYPES:
        raise RuntimeError(
            f"Unsupported Triton device type: {device.type}. "
            f"Expected one of {SUPPORTED_DEVICE_TYPES}."
        )
    if device.index is None:
        device = torch.device(device.type, get_backend(device).current_device())
    return device


def get_device_name(device: torch.device) -> str:
    device = normalize_device(device)
    return get_backend(device).get_device_name(device)


def get_device_properties(device: torch.device):
    device = normalize_device(device)
    return get_backend(device).get_device_properties(device)


def get_device_num_sms(device: torch.device) -> int:
    properties = get_device_properties(device)
    return properties.multi_processor_count


def get_max_shared_memory(device: torch.device) -> int:
    properties = get_device_properties(device)
    shared_memory = getattr(properties, "shared_memory_per_block_optin", None)
    if shared_memory is None:
        shared_memory = getattr(properties, "shared_memory_per_block", None)
    if shared_memory is None:
        raise RuntimeError(
            f"Device properties for {device} do not expose shared memory capacity"
        )
    return shared_memory


@functools.lru_cache(maxsize=16)
def get_device_cache_key(device: torch.device) -> tuple[str, int, str]:
    device = normalize_device(device)
    return device.type, device.index, get_device_name(device)


def get_device_cache_name(device: torch.device) -> str:
    return get_device_cache_key(device)[2]


def manual_seed_all(seed: int) -> None:
    for device_type in SUPPORTED_DEVICE_TYPES:
        if is_available(device_type):
            _get_backend(device_type).manual_seed_all(seed)


def is_supported_device(device: torch.device) -> bool:
    return torch.device(device).type in SUPPORTED_DEVICE_TYPES


def set_triton_allocator(device: torch.device) -> None:
    import triton

    device = normalize_device(device)

    def alloc_fn(size: int, alignment: int, stream):
        del alignment, stream
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)
