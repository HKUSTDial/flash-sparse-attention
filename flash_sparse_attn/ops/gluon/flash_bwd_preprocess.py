# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flash_sparse_attn.ops.gluon.launch_grid import get_bwd_preprocess_grid
from flash_sparse_attn.ops.gluon.kernel_repr import bwd_preprocess_repr
from flash_sparse_attn.ops.gluon.seqlen_info import (
    get_seqlen_info,
    offset_batch_Q,
    make_ptrs,
    make_seqlen_predicate,
)
from flash_sparse_attn.ops.gluon.softmax import check_inf


@triton.heuristics(
    {
        "EVEN_M": lambda args: (
            not args["HAS_CU_SEQLENS_Q"]
            and not args["HAS_SEQUSED_Q"]
            and args["seqlen_q"] % args["TILE_M"] == 0
        ),
    }
)
@gluon.jit(repr=bwd_preprocess_repr)
def _bwd_preprocess_kernel(
    Out,
    dO,
    dPsum,
    LSE,
    LSELog2,
    dQaccum,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_l2b,
    stride_l2h,
    stride_l2m,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    cu_seqlens_q,
    seqused_q,
    seqlen_q,
    num_heads_q,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    EVEN_M: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_K: gl.constexpr,
    PADDED_TILE_M: gl.constexpr,
    num_warps: gl.constexpr,
):
    warp_size: gl.constexpr = 32
    element_bits: gl.constexpr = Out.dtype.element_ty.primitive_bitwidth
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

    zero_elements_per_thread: gl.constexpr = 4
    copy_zero_atom_k: gl.constexpr = min(TILE_K, 128)
    zero_threads_per_row: gl.constexpr = copy_zero_atom_k // zero_elements_per_thread
    zero_rows_per_warp: gl.constexpr = warp_size // zero_threads_per_row
    copy_zero_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, zero_elements_per_thread],
        threads_per_warp=[zero_rows_per_warp, zero_threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    copy_zero_row_layout: gl.constexpr = gl.SliceLayout(1, copy_zero_layout)
    copy_zero_col_layout: gl.constexpr = gl.SliceLayout(0, copy_zero_layout)

    # Compute offsets for global memory copy
    copy_raw_m_offs = gl.arange(0, TILE_M, copy_row_layout)
    copy_col_k_offs = gl.arange(0, TILE_K, copy_col_layout)
    copy_zero_row_m_offs = gl.arange(0, TILE_M, copy_zero_row_layout)
    copy_zero_col_k_offs = gl.arange(0, TILE_K, copy_zero_col_layout)

    # Create grid index
    m_block = gl.program_id(0)
    bh_idx = gl.program_id(1)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q

    # Get seqlen info for this batch
    offset_q, padded_offset_q, actual_seqlen_q = get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        TILE_MN=PADDED_TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Initialize base pointers
    mO_base_ptr = offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mdO_base_ptr = offset_batch_Q(
        dO + head_idx * stride_doh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dob,
        stride_dom,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mdPsum_base_ptr = offset_batch_Q(
        dPsum + head_idx * stride_ph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_pb,
        stride_pm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    mLSE_base_ptr = offset_batch_Q(
        LSE + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        stride_lm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    mLSElog2_base_ptr = offset_batch_Q(
        LSELog2 + head_idx * stride_l2h,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_l2b,
        stride_l2m,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    mdQaccum_base_ptr = offset_batch_Q(
        dQaccum + head_idx * stride_dqah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqab,
        stride_dqam,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )

    # Create pointers
    mO_ptr = make_ptrs(
        mO_base_ptr,
        m_block,
        stride_om,
        copy_raw_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    mdO_ptr = make_ptrs(
        mdO_base_ptr,
        m_block,
        stride_dom,
        copy_raw_m_offs,
        copy_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    mdPsum_ptr = make_ptrs(
        mdPsum_base_ptr,
        m_block,
        stride_pm,
        copy_raw_m_offs,
        copy_col_k_offs,
        TILE_K=1,
        SWAP_AB=False,
    )
    mLSE_ptr = make_ptrs(
        mLSE_base_ptr,
        m_block,
        stride_lm,
        copy_raw_m_offs,
        copy_col_k_offs,
        TILE_K=1,
        SWAP_AB=False,
    )
    mLSElog2_ptr = make_ptrs(
        mLSElog2_base_ptr,
        m_block,
        stride_l2m,
        copy_raw_m_offs,
        copy_col_k_offs,
        TILE_K=1,
        SWAP_AB=False,
    )
    mdQaccum_ptr = make_ptrs(
        mdQaccum_base_ptr,
        m_block,
        stride_dqam,
        copy_zero_row_m_offs,
        copy_zero_col_k_offs,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )

    # Load O
    if not EVEN_M:
        pO = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_raw_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    rO = gl.load(
        mO_ptr,
        mask=pO if not EVEN_M else None,
        other=0.0 if not EVEN_M else None,
    ).to(gl.float32)

    # Load dO
    if not EVEN_M:
        pdO = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_raw_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    rdO = gl.load(
        mdO_ptr,
        mask=pdO if not EVEN_M else None,
        other=0.0 if not EVEN_M else None,
    ).to(gl.float32)

    # Compute dPsum
    rdPsum = gl.sum(rO * rdO, axis=1)

    # Store dPsum
    if not EVEN_M:
        pdPsum = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_raw_m_offs,
            TILE_MN=TILE_M,
        )
    gl.store(mdPsum_ptr, rdPsum, mask=pdPsum if not EVEN_M else None)

    # Load LSE
    if not EVEN_M:
        pLSE = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_raw_m_offs,
            TILE_MN=TILE_M,
        )
    rLSE = gl.load(
        mLSE_ptr,
        mask=pLSE if not EVEN_M else None,
        other=0.0 if not EVEN_M else None,
    )

    # Compute LSELog2
    log2_e: gl.constexpr = 1.4426950408889634
    rLSElog2 = gl.where(rLSE == rLSE, rLSE * log2_e, 1e6)
    rLSElog2 = check_inf(rLSElog2)

    # Store LSELog2
    if not EVEN_M:
        pLSElog2 = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_raw_m_offs,
            TILE_MN=TILE_M,
        )
    gl.store(mLSElog2_ptr, rLSElog2, mask=pLSElog2 if not EVEN_M else None)

    # Create zero tensor
    zero = gl.zeros([TILE_M, TILE_K], gl.float32, copy_zero_layout)

    # Store dQaccum
    if not EVEN_M:
        pdQaccum = make_seqlen_predicate(
            m_block,
            actual_seqlen_q,
            copy_zero_row_m_offs[:, None],
            TILE_MN=TILE_M,
        )
    gl.store(mdQaccum_ptr, zero, mask=pdQaccum if not EVEN_M else None)


def _flash_attn_bwd_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    dpsum: torch.Tensor,
    lse: torch.Tensor,
    lse_log2: torch.Tensor,
    dq_accum: torch.Tensor,
    head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: Optional[int] = None,
    padded_tile_m: Optional[int] = None,
    num_threads: Optional[int] = None,
):
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, _ = out.shape
    else:
        total_q, num_heads_q, _ = out.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else 64
    PADDED_TILE_M = padded_tile_m if padded_tile_m is not None else TILE_M
    num_warps = num_threads // 32 if num_threads is not None else 8

    # Get the grid for the kernel launch
    grid = get_bwd_preprocess_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
    )

    # Launch the kernel
    _bwd_preprocess_kernel[grid](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        out.stride(-3),
        dout.stride(0) if not is_varlen else 0,
        dout.stride(-2),
        dout.stride(-3),
        dpsum.stride(0) if not is_varlen else 0,
        dpsum.stride(-2) if not is_varlen else dpsum.stride(0),
        dpsum.stride(-1),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(-2) if not is_varlen else lse.stride(0),
        lse.stride(-1),
        lse_log2.stride(0) if not is_varlen else 0,
        lse_log2.stride(-2) if not is_varlen else lse_log2.stride(0),
        lse_log2.stride(-1),
        dq_accum.stride(0) if not is_varlen else 0,
        dq_accum.stride(-3),
        dq_accum.stride(-2),
        cu_seqlens_q,
        seqused_q,
        seqlen_q,
        num_heads_q,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        PADDED_TILE_M=PADDED_TILE_M,
        num_warps=num_warps,
    )
