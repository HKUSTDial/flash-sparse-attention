# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flash_sparse_attn.ops.gluon.assert_inputs import assert_fwd_combine_inputs
from flash_sparse_attn.ops.gluon.launch_grid import get_fwd_combine_grid
from flash_sparse_attn.ops.gluon.kernel_repr import fwd_combine_repr
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
@gluon.jit(repr=fwd_combine_repr)
def _fwd_combine_kernel(
    mOpartial,
    mLSEpartial,
    mO,
    mLSE,
    mCuSeqlensQ,
    mSeqUsedQ,
    stride_ops,
    stride_opb,
    stride_oph,
    stride_opm,
    stride_lps,
    stride_lpb,
    stride_lph,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    num_splits,
    seqlen_q,
    num_heads_q,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    EVEN_M: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    warp_size: gl.constexpr = 32
    elements_per_thread: gl.constexpr = (TILE_M * TILE_K) // (warp_size * num_warps)
    threads_per_row: gl.constexpr = TILE_K // elements_per_thread
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
        TILE_MN=TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Initialize base pointers
    mOpartial_base_ptr = offset_batch_Q(
        mOpartial + head_idx * stride_oph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_opb,
        stride_opm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mLSEpartial_base_ptr = offset_batch_Q(
        mLSEpartial + head_idx * stride_lph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lpb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mO_base_ptr = offset_batch_Q(
        mO + head_idx * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mLSE_base_ptr = offset_batch_Q(
        mLSE + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    mOpartial_ptr = make_ptrs(
        mOpartial_base_ptr,
        m_block,
        stride_opm,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    mLSEpartial_ptr = make_ptrs(
        mLSEpartial_base_ptr,
        m_block,
        1,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=1,
        SWAP_AB=False,
    )
    mO_ptr = make_ptrs(
        mO_base_ptr,
        m_block,
        stride_om,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    mLSE_ptr = make_ptrs(
        mLSE_base_ptr,
        m_block,
        1,
        copy_row_m_offs,
        copy_col_k_offs,
        TILE_K=1,
        SWAP_AB=False,
    )

    # Initialize accumulators
    row_max = gl.full([TILE_M], float("-inf"), gl.float32, copy_row_layout)
    row_sum = gl.zeros([TILE_M], gl.float32, copy_row_layout)
    acc_o = gl.zeros([TILE_M, TILE_K], gl.float32, copy_layout)

    # Combine split outputs
    for split_idx in range(num_splits):
        # Load partial LSE
        if not EVEN_M:
            pLSEpartial = make_seqlen_predicate(
                m_block,
                actual_seqlen_q,
                copy_row_m_offs,
                TILE_MN=TILE_M,
            )
        rLSEpartial_s = gl.load(
            mLSEpartial_ptr + split_idx * stride_lps,
            mask=pLSEpartial if not EVEN_M else None,
            other=float("-inf") if not EVEN_M else None,
        )

        # Load partial O
        if not EVEN_M:
            pOpartial = make_seqlen_predicate(
                m_block,
                actual_seqlen_q,
                copy_row_m_offs[:, None],
                TILE_MN=TILE_M,
            )
        rOpartial_s = gl.load(
            mOpartial_ptr + split_idx * stride_ops,
            mask=pOpartial if not EVEN_M else None,
            other=0.0 if not EVEN_M else None,
            cache_modifier=".cg",
        )

        # Compute normalized exponentials
        new_row_max = gl.maximum(rLSEpartial_s, row_max)
        old_scale = gl.where(
            row_max == float("-inf"), 0.0, gl.exp2(row_max - new_row_max)
        )
        exp_logic = gl.where(
            rLSEpartial_s == float("-inf"), 0.0, gl.exp2(rLSEpartial_s - new_row_max)
        )

        # Compute scaled O
        acc_o *= old_scale[:, None]
        acc_o += exp_logic[:, None] * rOpartial_s

        # Update row_sum and row_max
        row_sum = row_sum * old_scale + exp_logic
        row_max = new_row_max

    # Normalize O
    inv_sum = gl.where((row_sum == 0.0) | (row_sum != row_sum), 0.0, 1.0 / row_sum)
    acc_o *= inv_sum[:, None]

    # Store O
    if not EVEN_M:
        pO = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_row_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    gl.store(mO_ptr, acc_o, mask=pO if not EVEN_M else None)

    # Compute LSE
    ln2: gl.constexpr = 0.6931471805599453
    lse = gl.where(row_sum > 0.0, (row_max + gl.log2(row_sum)) * ln2, float("-inf"))

    # Store LSE
    if not EVEN_M:
        pLSE = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_row_m_offs,
            TILE_MN=TILE_M,
        )
    gl.store(mLSE_ptr, lse, mask=pLSE if not EVEN_M else None)


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
):
    is_varlen = cu_seqlens_q is not None
    num_splits = out_partial.shape[0]
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = out.shape
    else:
        total_q, num_heads_q, head_dim = out.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    # Assert the validity of inputs
    if not skip_checks:
        assert_fwd_combine_inputs(
            out_partial=out_partial,
            lse_partial=lse_partial,
            out=out,
            lse=lse,
            num_splits=num_splits,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
        )

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = (
        tile_m
        if tile_m is not None
        else max(
            min(
                4
                if TILE_K % 256 == 0
                else (8 if TILE_K % 128 == 0 else (16 if TILE_K % 64 == 0 else 32)),
                triton.next_power_of_2(seqlen_q),
            ),
            triton.cdiv(32, TILE_K),
        )
    )
    num_warps = (
        num_threads // 32
        if num_threads is not None
        else min(8, TILE_M, TILE_M * TILE_K // 32)
    )

    # Get the grid for the kernel launch
    grid = get_fwd_combine_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
    )

    # Launch the kernel
    _fwd_combine_kernel[grid](
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens_q,
        seqused_q,
        out_partial.stride(0),
        out_partial.stride(1) if not is_varlen else 0,
        out_partial.stride(-3),
        out_partial.stride(-2),
        lse_partial.stride(0),
        lse_partial.stride(1) if not is_varlen else 0,
        lse_partial.stride(-2),
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        out.stride(-3),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(-2),
        num_splits,
        seqlen_q,
        num_heads_q,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        num_warps=num_warps,
    )
