from typing import Optional

import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    utils,
    launch_grid,
    kernel_repr,
    seqlen_info,
)


@triton.jit(repr=kernel_repr.bwd_combine_repr)
def _bwd_combine_kernel(
    dKpartial,
    dVpartial,
    dK,
    dV,
    CuSeqlensK,
    SeqUsedK,
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
    num_splits,
    seqlen_k,
    num_heads_kv,
    head_dim,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    PADDED_TILE_N: tl.constexpr,
):
    # Create grid index
    n_block = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // num_heads_kv
    head_idx = bh_idx - batch_idx * num_heads_kv

    # Get seqlen info for this batch
    offset_k, padded_offset_k, actual_seqlen_k = seqlen_info.get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_k,
        cu_seqlens=CuSeqlensK,
        seqused=SeqUsedK,
        TILE_MN=PADDED_TILE_N,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
        HAS_SEQUSED=HAS_SEQUSED_K,
    )

    # Initialize base pointers
    dk_partial_base_ptr = seqlen_info.offset_batch_K(
        dKpartial + head_idx * stride_dkph,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dkpb,
        stride_dkpn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=True,
    )
    dv_partial_base_ptr = seqlen_info.offset_batch_K(
        dVpartial + head_idx * stride_dvph,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dvpb,
        stride_dvpn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=True,
    )
    dk_base_ptr = seqlen_info.offset_batch_K(
        dK + head_idx * stride_dkh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dkb,
        stride_dkn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    dv_base_ptr = seqlen_info.offset_batch_K(
        dV + head_idx * stride_dvh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dvb,
        stride_dvn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )

    # Create pointers
    dk_partial_ptr = tl.make_tensor_descriptor(
        base=dk_partial_base_ptr,
        shape=[num_splits, actual_seqlen_k, head_dim],
        strides=[stride_dkps, stride_dkpn, 1],
        block_shape=[1, TILE_N, TILE_K],
    )
    dv_partial_ptr = tl.make_tensor_descriptor(
        base=dv_partial_base_ptr,
        shape=[num_splits, actual_seqlen_k, head_dim],
        strides=[stride_dvps, stride_dvpn, 1],
        block_shape=[1, TILE_N, TILE_K],
    )
    dk_ptr = tl.make_tensor_descriptor(
        base=dk_base_ptr,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_dkn, 1],
        block_shape=[TILE_N, TILE_K],
    )
    dv_ptr = tl.make_tensor_descriptor(
        base=dv_base_ptr,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_dvn, 1],
        block_shape=[TILE_N, TILE_K],
    )

    # Initialize accumulators
    acc_dk = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
    acc_dv = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)

    # Combine split gradients
    for split_idx in range(num_splits):
        # Load partial dk
        dk_partial_s = dk_partial_ptr.load([split_idx, n_block * TILE_N, 0]).reshape([TILE_N, TILE_K])

        # Compute dk
        acc_dk += dk_partial_s

        # Load partial dv
        dv_partial_s = dv_partial_ptr.load([split_idx, n_block * TILE_N, 0]).reshape([TILE_N, TILE_K])

        # Compute dv
        acc_dv += dv_partial_s

    # Store dK
    dk_ptr.store([n_block * TILE_N, 0], acc_dk.to(dK.dtype.element_ty))

    # Store dV
    dv_ptr.store([n_block * TILE_N, 0], acc_dv.to(dV.dtype.element_ty))


def _flash_attn_bwd_combine(
    dk_partial: torch.Tensor,
    dv_partial: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    tile_n: Optional[int] = None,
    padded_tile_n: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> None:
    is_varlen = cu_seqlens_k is not None
    num_splits = dk_partial.shape[0]
    if not is_varlen:
        batch_size, seqlen_k, num_heads_kv, head_dim = dk.shape
    else:
        total_k, num_heads_kv, head_dim = dk.shape
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

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
    PADDED_TILE_N = padded_tile_n if padded_tile_n is not None else TILE_N
    num_warps = num_warps if num_warps is not None else 8

    # Get the grid for the kernel launch
    grid = launch_grid.get_bwd_combine_grid(
        batch_size=batch_size,
        seqlen_k=seqlen_k,
        num_heads_kv=num_heads_kv,
    )

    triton.set_allocator(utils.alloc_fn)

    # Launch the kernel
    _bwd_combine_kernel[grid](
        dk_partial,
        dv_partial,
        dk,
        dv,
        cu_seqlens_k,
        seqused_k,
        dk_partial.stride(0),
        dk_partial.stride(1) if not is_varlen else 0,
        dk_partial.stride(-3),
        dk_partial.stride(-2),
        dv_partial.stride(0),
        dv_partial.stride(1) if not is_varlen else 0,
        dv_partial.stride(-3),
        dv_partial.stride(-2),
        dk.stride(0) if not is_varlen else 0,
        dk.stride(-2),
        dk.stride(-3),
        dv.stride(0) if not is_varlen else 0,
        dv.stride(-2),
        dv.stride(-3),
        num_splits,
        seqlen_k,
        num_heads_kv,
        head_dim,
        HAS_CU_SEQLENS_K=is_varlen,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        PADDED_TILE_N=PADDED_TILE_N,
        num_warps=num_warps,
    )
