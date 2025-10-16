# Copyright (c) 2025, Jingze Shi.

from typing import Optional

__version__ = "1.2.1"


# Import CUDA functions when available
try:
    from flash_dmattn.flash_dmattn_interface import flash_dmattn_func, flash_dmattn_varlen_func
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    flash_dmattn_func, flash_dmattn_varlen_func = None, None

# Import Triton functions when available
try:
    from flash_dmattn.flash_dmattn_triton import triton_dmattn_func
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton_dmattn_func = None

# Import Flex functions when available
try:
    from flash_dmattn.flash_dmattn_flex import flex_dmattn_func
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    flex_dmattn_func = None


def get_available_backends():
    """Return a list of available backends."""
    backends = []
    if CUDA_AVAILABLE:
        backends.append("cuda")
    if TRITON_AVAILABLE:
        backends.append("triton")
    if FLEX_AVAILABLE:
        backends.append("flex")
    return backends


def flash_dmattn_func_auto(backend: Optional[str] = None, **kwargs):
    """
    Flash Dynamic Mask Attention function with automatic backend selection.
    
    Args:
        backend (str, optional): Backend to use ('cuda', 'triton', 'flex'). 
            If None, will use the first available backend in order: cuda, triton, flex.
        **kwargs: Arguments to pass to the attention function.
    
    Returns:
        The attention function for the specified or auto-selected backend.
    """
    if backend is None:
        # Auto-select backend
        if CUDA_AVAILABLE:
            backend = "cuda"
        elif TRITON_AVAILABLE:
            backend = "triton"
        elif FLEX_AVAILABLE:
            backend = "flex"
        else:
            raise RuntimeError("No flash attention backend is available. Please install at least one of: triton, transformers, or build the CUDA extension.")
    
    if backend == "cuda":
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend is not available. Please build the CUDA extension.")
        return flash_dmattn_func
    
    elif backend == "triton":
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton backend is not available. Please install triton: pip install triton")
        return triton_dmattn_func
    
    elif backend == "flex":
        if not FLEX_AVAILABLE:
            raise RuntimeError("Flex backend is not available. Please install transformers: pip install transformers")
        return flex_dmattn_func
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Available backends: {get_available_backends()}")


__all__ = [
    "CUDA_AVAILABLE",
    "TRITON_AVAILABLE",
    "FLEX_AVAILABLE",
    "flash_dmattn_func",
    "flash_dmattn_varlen_func",
    "triton_dmattn_func",
    "flex_dmattn_func",
    "get_available_backends",
    "flash_dmattn_func_auto",
]
