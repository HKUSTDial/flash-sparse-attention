# Copyright (c) 2025, Jingze Shi.

__version__ = "2.0.1"

from flash_sparse_attn.ops.triton.interface import (
    flash_dense_attn_func,
    flash_dense_attn_with_kvcache_func,
    flash_dense_attn_varlen_func,
    flash_dense_attn_varlen_with_kvcache_func,
    flash_sparse_attn_func,
    flash_sparse_attn_with_kvcache_func,
    flash_sparse_attn_varlen_func,
    flash_sparse_attn_varlen_with_kvcache_func,
    flash_gated_attn_func,
    flash_gated_attn_with_kvcache_func,
    flash_gated_attn_varlen_func,
    flash_gated_attn_varlen_with_kvcache_func,
)


__all__ = [
    "flash_dense_attn_func",
    "flash_dense_attn_with_kvcache_func",
    "flash_dense_attn_varlen_func",
    "flash_dense_attn_varlen_with_kvcache_func",
    "flash_sparse_attn_func",
    "flash_sparse_attn_with_kvcache_func",
    "flash_sparse_attn_varlen_func",
    "flash_sparse_attn_varlen_with_kvcache_func",
    "flash_gated_attn_func",
    "flash_gated_attn_with_kvcache_func",
    "flash_gated_attn_varlen_func",
    "flash_gated_attn_varlen_with_kvcache_func",
]
