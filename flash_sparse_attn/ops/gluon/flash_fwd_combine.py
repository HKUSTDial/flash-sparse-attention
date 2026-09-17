# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flash_sparse_attn.ops.gluon.assert_inputs import assert_fwd_combine_inputs
from flash_sparse_attn.ops.gluon.seqlen_info import (
    get_seqlen_info,
    offset_batch_Q,
    make_ptrs,
)
from flash_sparse_attn.ops.gluon.kernel_repr import fwd_combine_repr
from flash_sparse_attn.ops.gluon.launch_grid import get_fwd_combine_grid


@triton.heuristics(
    {
        "EVEN_M": lambda args: not args["HAS_CU_SEQLENS_Q"]
        and not args["HAS_SEQUSED_Q"]
        and args["seqlen_q"] % args["TILE_M"] == 0,
    }
)
@gluon.jit(repr=fwd_combine_repr)
def _fwd_combine_kernel(
    Out_partial,
    Lse_partial,
    Out,
    Lse,
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
    cu_seqlens_q,
    seqused_q,
    num_splits,
    seqlen_q,
    num_heads_q,
    TILE_M: gl.constexpr,
    TILE_K: gl.constexpr,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    EVEN_M: gl.constexpr,
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
    copy_offs_m = gl.arange(0, TILE_M, copy_row_layout)
    copy_offs_k = gl.arange(0, TILE_K, copy_col_layout)

    # Create grid index
    m_block = gl.program_id(0)
    bh_idx = gl.program_id(1)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q

    # Get seqlen info for this batch
    offset_q, actual_seqlen_q = get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Compute predicates for global memory copy
    if not EVEN_M:
        copy_m_rows = m_block * TILE_M + copy_offs_m
        predicate_copy_m = copy_m_rows < actual_seqlen_q

    # Initialize base pointers
    out_partial_base = offset_batch_Q(
        Out_partial + head_idx * stride_oph,
        batch_idx,
        offset_q,
        0,
        stride_opb,
        stride_opm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_partial_base = offset_batch_Q(
        Lse_partial + head_idx * stride_lph,
        batch_idx,
        offset_q,
        0,
        stride_lpb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    out_base = offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        0,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_base = offset_batch_Q(
        Lse + head_idx * stride_lh,
        batch_idx,
        offset_q,
        0,
        stride_lb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    out_partial_ptrs = make_ptrs(
        out_partial_base,
        m_block,
        stride_opm,
        copy_offs_m,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    lse_partial_ptrs = make_ptrs(
        lse_partial_base,
        m_block,
        1,
        copy_offs_m,
        copy_offs_k,
        TILE_K=1,
        SWAP_AB=False,
    )
    out_ptrs = make_ptrs(
        out_base,
        m_block,
        stride_om,
        copy_offs_m,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    lse_out_ptrs = make_ptrs(
        lse_base,
        m_block,
        1,
        copy_offs_m,
        copy_offs_k,
        TILE_K=1,
        SWAP_AB=False,
    )

    # Initialize accumulators
    e_sum = gl.zeros([TILE_M], gl.float32, copy_row_layout)
    e_max = gl.full([TILE_M], float("-inf"), gl.float32, copy_row_layout)
    acc_o = gl.zeros([TILE_M, TILE_K], gl.float32, copy_layout)

    # Combine split outputs
    for split_idx in range(num_splits):
        # Load partial LSE
        lse_s = gl.load(
            lse_partial_ptrs + split_idx * stride_lps,
            mask=predicate_copy_m if not EVEN_M else None,
            other=float("-inf") if not EVEN_M else None,
        )

        # Load partial outputs
        partial_o = gl.load(
            out_partial_ptrs + split_idx * stride_ops,
            mask=predicate_copy_m[:, None] if not EVEN_M else None,
            other=0.0 if not EVEN_M else None,
            cache_modifier=".cg",
        )

        # Compute normalized exponentials
        new_e_max = gl.maximum(lse_s, e_max)
        old_scale = gl.where(e_max == float("-inf"), 0.0, gl.exp2(e_max - new_e_max))
        exp_logic = gl.where(lse_s == float("-inf"), 0.0, gl.exp2(lse_s - new_e_max))

        # Compute scaled outputs
        acc_o *= old_scale[:, None]
        acc_o += exp_logic[:, None] * partial_o

        # Update e_sum and e_max
        e_sum = e_sum * old_scale + exp_logic
        e_max = new_e_max

    # Normalize output
    inv_sum = gl.where((e_sum == 0.0) | (e_sum != e_sum), 0.0, 1.0 / e_sum)
    acc_o *= inv_sum[:, None]

    # Store output
    gl.store(out_ptrs, acc_o, mask=predicate_copy_m[:, None] if not EVEN_M else None)

    # Compute LSE
    ln2: gl.constexpr = 0.6931471805599453
    final_lse = gl.where(e_sum > 0.0, (e_max + gl.log2(e_sum)) * ln2, float("-inf"))

    # Store LSE
    gl.store(lse_out_ptrs, final_lse, mask=predicate_copy_m if not EVEN_M else None)


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
):
    is_varlen = cu_seqlens_q is not None
    num_splits = out_partial.shape[0]
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = out_partial.shape[1:]
    else:
        total_q, num_heads_q, head_dim = out_partial.shape[1:]
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = total_q

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
    TILE_M = min(
        4
        if TILE_K % 256 == 0
        else (8 if TILE_K % 128 == 0 else (16 if TILE_K % 64 == 0 else 32)),
        triton.next_power_of_2(seqlen_q),
    )
    TILE_M = max(TILE_M, triton.cdiv(32, TILE_K))
    num_warps = min(8, TILE_M, TILE_M * TILE_K // 32)

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
        out_partial.stride(0),
        out_partial.stride(1) if not is_varlen else 0,
        out_partial.stride(-2),
        out_partial.stride(-3),
        lse_partial.stride(0),
        lse_partial.stride(1) if not is_varlen else 0,
        lse_partial.stride(-2),
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        out.stride(-3) if not is_varlen else out.stride(0),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(-2) if not is_varlen else lse.stride(0),
        cu_seqlens_q,
        seqused_q,
        num_splits,
        seqlen_q,
        num_heads_q,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        num_warps=num_warps,
    )
