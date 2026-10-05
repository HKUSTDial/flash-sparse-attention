# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flash_sparse_attn.ops.gluon.launch_grid import get_bwd_postprocess_grid
from flash_sparse_attn.ops.gluon.kernel_repr import bwd_postprocess_repr
from flash_sparse_attn.ops.gluon.seqlen_info import (
    get_seqlen_info,
    offset_batch_Q,
    make_ptrs,
    make_seqlen_predicate,
)


@triton.heuristics(
    {
        "EVEN_M": lambda args: (
            not args["HAS_CU_SEQLENS_Q"]
            and not args["HAS_SEQUSED_Q"]
            and args["seqlen_q"] % args["TILE_M"] == 0
        ),
    }
)
@gluon.jit(repr=bwd_postprocess_repr)
def _bwd_postprocess_kernel(
    mdQaccum,
    mdQ,
    mCuSeqlensQ,
    mSeqUsedQ,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    seqlen_q,
    num_heads_q,
    scale,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    EVEN_M: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_K: gl.constexpr,
    PADDED_TILE_M: gl.constexpr,
    num_warps: gl.constexpr,
):
    warp_size: gl.constexpr = 32
    element_bits: gl.constexpr = mdQaccum.dtype.element_ty.primitive_bitwidth
    elements_per_thread: gl.constexpr = 128 // element_bits
    copy_atom_k: gl.constexpr = min(TILE_K, 1024 // element_bits)
    threads_per_row: gl.constexpr = copy_atom_k // elements_per_thread
    rows_per_warp: gl.constexpr = warp_size // threads_per_row

    copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    copy_row_layout: gl.constexpr = gl.SliceLayout(1, copy_layout)
    copy_col_layout: gl.constexpr = gl.SliceLayout(0, copy_layout)

    # Compute offsets for global memory copy
    copy_row_m_offs = gl.arange(0, TILE_M, copy_row_layout)
    copy_col_k_offs = gl.arange(0, TILE_K, copy_col_layout)

    # Create grid index
    m_block = gl.program_id(0)
    bh_idx = gl.program_id(1)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q

    # Get seqlen info for this batch
    offset_q, padded_offset_q, actual_seqlen_q = get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=mCuSeqlensQ,
        seqused=mSeqUsedQ,
        TILE_MN=PADDED_TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Initialize base pointers
    mdQaccum_base_ptr = offset_batch_Q(
        mdQaccum + head_idx * stride_dqah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqab,
        stride_dqam,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    mdQ_base_ptr = offset_batch_Q(
        mdQ + head_idx * stride_dqh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqb,
        stride_dqm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    mdQaccum_ptr = make_ptrs(
        mdQaccum_base_ptr,
        m_block,
        stride_dqam,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    mdQ_ptr = make_ptrs(
        mdQ_base_ptr,
        m_block,
        stride_dqm,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )

    # Load dQaccum
    if not EVEN_M:
        pdQaccum = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_row_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    rdQaccum = gl.load(
        mdQaccum_ptr,
        mask=pdQaccum if not EVEN_M else None,
        other=0.0 if not EVEN_M else None,
        cache_modifier=".cg",
    )

    # Scale dQaccum
    rdQ = rdQaccum * scale

    # Store dQ
    if not EVEN_M:
        pdQ = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_row_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    gl.store(mdQ_ptr, rdQ, mask=pdQ if not EVEN_M else None)


def _flash_attn_bwd_postprocess(
    dq_accum: torch.Tensor,
    dq: torch.Tensor,
    scale: float,
    head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: Optional[int] = None,
    padded_tile_m: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, _ = dq.shape
    else:
        total_q, num_heads_q, _ = dq.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else 32
    PADDED_TILE_M = padded_tile_m if padded_tile_m is not None else TILE_M
    num_warps = num_threads // 32 if num_threads is not None else 8

    # Get the grid for the kernel launch
    grid = get_bwd_postprocess_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
    )

    # Launch the kernel
    _bwd_postprocess_kernel[grid](
        dq_accum,
        dq,
        cu_seqlens_q,
        seqused_q,
        dq_accum.stride(0) if not is_varlen else 0,
        dq_accum.stride(-3),
        dq_accum.stride(-2),
        dq.stride(0) if not is_varlen else 0,
        dq.stride(-2),
        dq.stride(-3) if not is_varlen else dq.stride(0),
        seqlen_q,
        num_heads_q,
        scale,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        PADDED_TILE_M=PADDED_TILE_M,
        num_warps=num_warps,
    )
