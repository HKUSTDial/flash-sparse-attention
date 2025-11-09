# Copyright (c) 2025, Jingze Shi.

from typing import Optional

__version__ = "1.2.3"


# Import CUDA functions when available
try:
    from flash_sparse_attn.flash_sparse_attn_interface import flash_sparse_attn_func, flash_sparse_attn_varlen_func
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    flash_sparse_attn_func, flash_sparse_attn_varlen_func = None, None

# Import Triton functions when available
try:
    from flash_sparse_attn.flash_sparse_attn_triton import triton_sparse_attn_func
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton_sparse_attn_func = None

# Import Flex functions when available
try:
    from flash_sparse_attn.flash_sparse_attn_flex import flex_sparse_attn_func
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    flex_sparse_attn_func = None


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


def flash_sparse_attn_func_auto(backend: Optional[str] = None, **kwargs):
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
        return flash_sparse_attn_func
    
    elif backend == "triton":
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton backend is not available. Please install triton: pip install triton")
        return triton_sparse_attn_func
    
    elif backend == "flex":
        if not FLEX_AVAILABLE:
            raise RuntimeError("Flex backend is not available. Please install transformers: pip install transformers")
        return flex_sparse_attn_func
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Available backends: {get_available_backends()}")


__all__ = [
    "CUDA_AVAILABLE",
    "TRITON_AVAILABLE",
    "FLEX_AVAILABLE",
    "flash_sparse_attn_func",
    "flash_sparse_attn_varlen_func",
    "triton_sparse_attn_func",
    "flex_sparse_attn_func",
    "get_available_backends",
    "flash_sparse_attn_func_auto",
]
