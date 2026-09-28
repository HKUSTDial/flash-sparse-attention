# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from flash_sparse_attn.ops.gluon.assert_inputs import assert_bwd_combine_inputs
from flash_sparse_attn.ops.gluon.launch_grid import get_bwd_combine_grid
from flash_sparse_attn.ops.gluon.kernel_repr import bwd_combine_repr
from flash_sparse_attn.ops.gluon.seqlen_info import (
    get_seqlen_info,
    offset_batch_K,
    make_ptrs,
)


@triton.heuristics(
    {
        "EVEN_N": lambda args: (
            not args["HAS_CU_SEQLENS_K"]
            and not args["HAS_SEQUSED_K"]
            and args["seqlen_k"] % args["TILE_N"] == 0
        ),
    }
)
@gluon.jit(repr=bwd_combine_repr)
def _bwd_combine_kernel(
    dK_partial,
    dV_partial,
    dK,
    dV,
    stride_dkps,
    stride_dkpb,
    stride_dkph,
    stride_dkpn,
    stride_dvps,
    stride_dvpb,
    stride_dvph,
    stride_dvpn,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    cu_seqlens_k,
    seqused_k,
    num_splits,
    seqlen_k,
    num_heads_kv,
    HAS_CU_SEQLENS_K: gl.constexpr,
    HAS_SEQUSED_K: gl.constexpr,
    EVEN_N: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    warp_size: gl.constexpr = 32
    elements_per_thread: gl.constexpr = (TILE_N * TILE_K) // (warp_size * num_warps)
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
    copy_offs_n = gl.arange(0, TILE_N, copy_row_layout)
    copy_offs_k = gl.arange(0, TILE_K, copy_col_layout)

    # Create grid index
    n_block = gl.program_id(0)
    bh_idx = gl.program_id(1)
    batch_idx = bh_idx // num_heads_kv
    head_idx = bh_idx - batch_idx * num_heads_kv

    # Get seqlen info for this batch
    offset_k, actual_seqlen_k = get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_k,
        cu_seqlens=cu_seqlens_k,
        seqused=seqused_k,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
        HAS_SEQUSED=HAS_SEQUSED_K,
    )

    # Initialize base pointers
    dk_partial_base = offset_batch_K(
        dK_partial + head_idx * stride_dkph,
        batch_idx,
        offset_k,
        0,
        stride_dkpb,
        stride_dkpn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    dv_partial_base = offset_batch_K(
        dV_partial + head_idx * stride_dvph,
        batch_idx,
        offset_k,
        0,
        stride_dvpb,
        stride_dvpn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    dk_base = offset_batch_K(
        dK + head_idx * stride_dkh,
        batch_idx,
        offset_k,
        0,
        stride_dkb,
        stride_dkn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    dv_base = offset_batch_K(
        dV + head_idx * stride_dvh,
        batch_idx,
        offset_k,
        0,
        stride_dvb,
        stride_dvn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )

    # Create pointers
    dk_partial_ptrs = make_ptrs(
        dk_partial_base,
        n_block,
        stride_dkpn,
        copy_offs_n,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    dv_partial_ptrs = make_ptrs(
        dv_partial_base,
        n_block,
        stride_dvpn,
        copy_offs_n,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    dk_ptrs = make_ptrs(
        dk_base,
        n_block,
        stride_dkn,
        copy_offs_n,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )
    dv_ptrs = make_ptrs(
        dv_base,
        n_block,
        stride_dvn,
        copy_offs_n,
        copy_offs_k,
        TILE_K=TILE_K,
        SWAP_AB=False,
    )

    # Initialize accumulators
    acc_dk = gl.zeros([TILE_N, TILE_K], gl.float32, copy_layout)
    acc_dv = gl.zeros([TILE_N, TILE_K], gl.float32, copy_layout)

    # Compute predicates for global memory copy
    if not EVEN_N:
        copy_n_rows = n_block * TILE_N + copy_offs_n
        predicate_copy_n = copy_n_rows < actual_seqlen_k

    # Combine split gradients
    for split_idx in range(num_splits):
        # Load partial dKaccum
        acc_dk_s = gl.load(
            dk_partial_ptrs + split_idx * stride_dkps,
            mask=predicate_copy_n[:, None] if not EVEN_N else None,
            other=0.0 if not EVEN_N else None,
            cache_modifier=".cg",
        )

        # Compute dKaccum
        acc_dk += acc_dk_s

        # Load partial dVaccum
        acc_dv_s = gl.load(
            dv_partial_ptrs + split_idx * stride_dvps,
            mask=predicate_copy_n[:, None] if not EVEN_N else None,
            other=0.0 if not EVEN_N else None,
            cache_modifier=".cg",
        )

        # Compute dVaccum
        acc_dv += acc_dv_s

    # Store dK
    gl.store(dk_ptrs, acc_dk, mask=predicate_copy_n[:, None] if not EVEN_N else None)

    # Store dV
    gl.store(dv_ptrs, acc_dv, mask=predicate_copy_n[:, None] if not EVEN_N else None)


def _flash_attn_bwd_combine(
    dk_partial: torch.Tensor,
    dv_partial: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    tile_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
) -> None:
    is_varlen = cu_seqlens_k is not None
    num_splits = dk_partial.shape[0]
    if not is_varlen:
        batch_size, seqlen_k, num_heads_kv, head_dim = dk_partial.shape[1:]
    else:
        total_k, num_heads_kv, head_dim = dk_partial.shape[1:]
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    # Assert the validity of inputs
    if not skip_checks:
        assert_bwd_combine_inputs(
            dk_partial=dk_partial,
            dv_partial=dv_partial,
            dk=dk,
            dv=dv,
            num_splits=num_splits,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
        )

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_N = (
        tile_n
        if tile_n is not None
        else max(
            min(
                4
                if TILE_K % 256 == 0
                else (8 if TILE_K % 128 == 0 else (16 if TILE_K % 64 == 0 else 32)),
                triton.next_power_of_2(seqlen_k),
            ),
            triton.cdiv(32, TILE_K),
        )
    )
    num_warps = (
        num_threads // 32
        if num_threads is not None
        else min(8, TILE_N, TILE_N * TILE_K // 32)
    )

    # Get the grid for the kernel launch
    grid = get_bwd_combine_grid(
        batch_size=batch_size,
        seqlen_k=seqlen_k,
        num_heads_kv=num_heads_kv,
    )

    # Launch the kernel
    _bwd_combine_kernel[grid](
        dk_partial,
        dv_partial,
        dk,
        dv,
        dk_partial.stride(0),
        dk_partial.stride(1) if not is_varlen else 0,
        dk_partial.stride(-2),
        dk_partial.stride(-3),
        dv_partial.stride(0),
        dv_partial.stride(1) if not is_varlen else 0,
        dv_partial.stride(-2),
        dv_partial.stride(-3),
        dk.stride(0) if not is_varlen else 0,
        dk.stride(-2),
        dk.stride(-3),
        dv.stride(0) if not is_varlen else 0,
        dv.stride(-2),
        dv.stride(-3),
        cu_seqlens_k,
        seqused_k,
        num_splits,
        seqlen_k,
        num_heads_kv,
        HAS_CU_SEQLENS_K=is_varlen,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
    )
