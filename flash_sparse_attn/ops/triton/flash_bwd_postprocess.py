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


@triton.jit(repr=kernel_repr.bwd_postprocess_repr)
def _bwd_postprocess_kernel(
    dQaccum,
    dQ,
    CuSeqlensQ,
    SeqUsedQ,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    seqlen_q,
    num_heads_q,
    head_dim,
    scale,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    PADDED_TILE_M: tl.constexpr,
):
    # Create grid index
    m_block = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q

    # Get seqlen info for this batch
    offset_q, padded_offset_q, actual_seqlen_q = seqlen_info.get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=CuSeqlensQ,
        seqused=SeqUsedQ,
        TILE_MN=PADDED_TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Initialize base pointers
    dq_accum_base_ptr = seqlen_info.offset_batch_Q(
        dQaccum + head_idx * stride_dqah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqab,
        stride_dqam,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    dq_base_ptr = seqlen_info.offset_batch_Q(
        dQ + head_idx * stride_dqh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqb,
        stride_dqm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    dq_accum_ptr = tl.make_tensor_descriptor(
        base=dq_accum_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dqam, 1],
        block_shape=[TILE_M, TILE_K],
    )
    dq_ptr = tl.make_tensor_descriptor(
        base=dq_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dqm, 1],
        block_shape=[TILE_M, TILE_K],
    )

    # Load dQaccum
    dq_accum = dq_accum_ptr.load([m_block * TILE_M, 0])

    # Scale dQaccum
    dq = dq_accum * scale

    # Store dQ
    dq_ptr.store([m_block * TILE_M, 0], dq.to(dQ.dtype.element_ty))


def _flash_attn_bwd_postprocess(
    dq_accum: torch.Tensor,
    dq: torch.Tensor,
    scale: float,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: int = 128,
    padded_tile_m: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = dq.shape
    else:
        _, num_heads_q, head_dim = dq.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else 32
    PADDED_TILE_M = padded_tile_m if padded_tile_m is not None else TILE_M
    num_warps = num_warps if num_warps is not None else 8

    # Get the grid for the kernel launch
    grid = launch_grid.get_bwd_postprocess_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
    )

    triton.set_allocator(utils.alloc_fn)

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
        head_dim,
        scale,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        PADDED_TILE_M=PADDED_TILE_M,
        num_warps=num_warps,
    )

    return dq
