from typing import Optional

import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import launch_grid, seqlen_info


@triton.jit
def _bwd_postprocess_kernel(
    dQaccum,
    dQ,
    dAaccum,
    dA,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_daab,
    stride_daah,
    stride_dab,
    stride_dah,
    cu_seqlens_q,
    seqused_q,
    seqlen_q,
    head_dim,
    head_dim_rounded,
    scale,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_DA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # Get seqlen info for this batch
    offset_q, actual_seqlen_q = seqlen_info.get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )
    padded_offset_q = (offset_q + batch_idx * TILE_M) // TILE_M * TILE_M

    # Initialize base pointers
    dq_accum_base = seqlen_info.offset_batch_Q(
        dQaccum + head_idx * stride_dqah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dqab,
        stride_dqam,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    dq_base = seqlen_info.offset_batch_Q(
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
    dq_accum_ptrs = tl.make_block_ptr(
        base=dq_accum_base,
        shape=(actual_seqlen_q, head_dim_rounded),
        strides=(stride_dqam, 1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )
    dq_ptrs = tl.make_block_ptr(
        base=dq_base,
        shape=(actual_seqlen_q, head_dim),
        strides=(stride_dqm, 1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    # Advance dq_accum pointers
    dq_accum_ptrs = tl.advance(dq_accum_ptrs, (m_block * TILE_M, 0))

    # Load accumulators
    acc_dq = tl.load(dq_accum_ptrs, boundary_check=(0, 1))

    # Scale dq
    dq = (acc_dq * scale).to(dQ.dtype.element_ty)

    # Advance dq pointers
    dq_ptrs = tl.advance(dq_ptrs, (m_block * TILE_M, 0))

    # Store dq
    tl.store(dq_ptrs, dq, boundary_check=(0, 1))

    if HAS_DA:
        da_accum_base = seqlen_info.offset_batch_Q(
            dAaccum + head_idx * stride_daah,
            batch_idx,
            offset_q,
            padded_offset_q,
            stride_daab,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        da_base = seqlen_info.offset_batch_Q(
            dA + head_idx * stride_dah,
            batch_idx,
            offset_q,
            padded_offset_q,
            stride_dab,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )

        da_accum_ptrs = tl.make_block_ptr(
            base=da_accum_base,
            shape=(actual_seqlen_q,),
            strides=(1,),
            offsets=(0,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        da_ptrs = tl.make_block_ptr(
            base=da_base,
            shape=(actual_seqlen_q,),
            strides=(1,),
            offsets=(0,),
            block_shape=(TILE_M,),
            order=(0,),
        )

        # Advance da_accum pointers
        da_accum_ptrs = tl.advance(da_accum_ptrs, (m_block * TILE_M,))

        # Load da accumulators
        acc_da = tl.load(da_accum_ptrs, boundary_check=(0,))

        # Scale da
        da = (acc_da * scale).to(dA.dtype.element_ty)

        # Advance da pointers
        da_ptrs = tl.advance(da_ptrs, (m_block * TILE_M,))

        # Store da
        tl.store(da_ptrs, da, boundary_check=(0,))


def _flash_attn_bwd_postprocess(
    dq_accum: torch.Tensor,
    dq: torch.Tensor,
    scale: float,
    head_dim_rounded: int,
    da_accum: Optional[torch.Tensor] = None,
    da: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: int = 128,
    tile_k: int = 128,
) -> torch.Tensor:
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = dq.shape
    else:
        _, num_heads_q, head_dim = dq.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q
    has_da = da_accum is not None and da is not None

    grid = launch_grid.get_bwd_postprocess_grid(batch_size, seqlen_q, num_heads_q)

    _bwd_postprocess_kernel[grid](
        dq_accum,
        dq,
        da_accum if has_da else dq_accum,
        da if has_da else dq,
        dq_accum.stride(0) if not is_varlen else 0,
        dq_accum.stride(1) if not is_varlen else dq_accum.stride(0),
        head_dim_rounded,
        dq.stride(0) if not is_varlen else 0,
        dq.stride(-2),
        dq.stride(-3) if not is_varlen else dq.stride(0),
        da_accum.stride(0) if (has_da and not is_varlen) else 0,
        da_accum.stride(1)
        if (has_da and not is_varlen)
        else (da_accum.stride(0) if has_da else 0),
        da.stride(0) if (has_da and not is_varlen) else 0,
        da.stride(-2) if has_da else 0,
        cu_seqlens_q,
        seqused_q,
        seqlen_q,
        head_dim,
        head_dim_rounded,
        scale,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_DA=has_da,
        TILE_M=tile_m,
        TILE_K=tile_k,
        num_warps=4,
        num_stages=1,
        num_ctas=1,
    )

    return dq
