from typing import Optional

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    utils,
    cache_utils,
    launch_grid,
    seqlen_info,
    activations,
    kernel_repr,
)


@triton.jit(repr=kernel_repr.bwd_preprocess_repr)
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
    head_dim,
    head_dim_rounded,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
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
    o_base = seqlen_info.offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    do_base = seqlen_info.offset_batch_Q(
        dO + head_idx * stride_doh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dob,
        stride_dom,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    dpsum_base = seqlen_info.offset_batch_Q(
        dPsum + head_idx * stride_ph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_pb,
        stride_pm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    lse_base = seqlen_info.offset_batch_Q(
        LSE + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        stride_lm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_log2_base = seqlen_info.offset_batch_Q(
        LSELog2 + head_idx * stride_l2h,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_l2b,
        stride_l2m,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
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

    # Create descriptors
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_om, 1],
        block_shape=[TILE_M, TILE_K],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do_base,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dom, 1],
        block_shape=[TILE_M, TILE_K],
    )
    dpsum_desc = tl.make_tensor_descriptor(
        base=dpsum_base,
        shape=[actual_seqlen_q],
        strides=[stride_pm],
        block_shape=[TILE_M],
    )
    lse_desc = tl.make_tensor_descriptor(
        base=lse_base,
        shape=[actual_seqlen_q],
        strides=[stride_lm],
        block_shape=[TILE_M],
    )
    lse_log2_desc = tl.make_tensor_descriptor(
        base=lse_log2_base,
        shape=[actual_seqlen_q],
        strides=[stride_l2m],
        block_shape=[TILE_M],
    )
    dq_accum_desc = tl.make_tensor_descriptor(
        base=dq_accum_base,
        shape=[actual_seqlen_q, head_dim_rounded],
        strides=[stride_dqam, 1],
        block_shape=[TILE_M, TILE_K],
    )

    # Initialize accumulators
    acc_dq = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load o tile
    o_tile = o_desc.load([m_block * TILE_M, 0]).to(tl.float32)

    # Load do tile
    do_tile = do_desc.load([m_block * TILE_M, 0]).to(tl.float32)

    # Compute dpsum
    dpsum = tl.sum(o_tile * do_tile, axis=1)

    # Store dpsum
    dpsum_desc.store([m_block * TILE_M], dpsum)

    # Load lse tile
    lse_tile = lse_desc.load([m_block * TILE_M])

    # Compute lse_log2
    lse_log2 = tl.where(lse_tile == lse_tile, lse_tile * math.log2(math.e), 1e6)
    lse_log2 = activations.check_inf(lse_log2)

    # Store dq_accum
    dq_accum_desc.store([m_block * TILE_M, 0], acc_dq)

    # Store lse_log2
    lse_log2_desc.store([m_block * TILE_M], lse_log2)


_bwd_preprocess_kernel = cache_utils.wrap_kernel(_bwd_preprocess_kernel)


def _flash_attn_bwd_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    dpsum: torch.Tensor,
    lse: torch.Tensor,
    lse_log2: torch.Tensor,
    dq_accum: torch.Tensor,
    head_dim_rounded: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    tile_m: int = 128,
    tile_k: int = 128,
) -> torch.Tensor:
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = out.shape
    else:
        _, num_heads_q, head_dim = out.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q

    grid = launch_grid.get_bwd_preprocess_grid(batch_size, seqlen_q, num_heads_q)

    triton.set_allocator(utils.alloc_fn)

    _bwd_preprocess_kernel[grid](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        out.stride(-3) if not is_varlen else out.stride(0),
        dout.stride(0) if not is_varlen else 0,
        dout.stride(-2),
        dout.stride(-3) if not is_varlen else dout.stride(0),
        dpsum.stride(0) if not is_varlen else 0,
        dpsum.stride(1) if not is_varlen else dpsum.stride(0),
        dpsum.stride(-1),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(1) if not is_varlen else lse.stride(0),
        lse.stride(-1),
        lse_log2.stride(0) if not is_varlen else 0,
        lse_log2.stride(1) if not is_varlen else lse_log2.stride(0),
        lse_log2.stride(-1),
        dq_accum.stride(0) if not is_varlen else 0,
        dq_accum.stride(1) if not is_varlen else dq_accum.stride(0),
        head_dim_rounded,
        cu_seqlens_q,
        seqused_q,
        seqlen_q,
        head_dim,
        head_dim_rounded,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=tile_m,
        TILE_K=tile_k,
        num_warps=4,
        num_stages=1,
        num_ctas=1,
    )

    return dpsum
