from typing import Optional

import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    utils,
    launch_grid,
    kernel_repr,
    seqlen_info,
    softmax,
)


@triton.jit(repr=kernel_repr.bwd_preprocess_repr)
def _bwd_preprocess_kernel(
    Out,
    dO,
    dPsum,
    LSE,
    LSELog2,
    dQaccum,
    CuSeqlensQ,
    SeqUsedQ,
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
    seqlen_q,
    num_heads_q,
    head_dim,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    PADDED_TILE_M: tl.constexpr,
):
    # Compute offsets
    offs_m = tl.arange(0, TILE_M)

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
    o_base_ptr = seqlen_info.offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    do_base_ptr = seqlen_info.offset_batch_Q(
        dO + head_idx * stride_doh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dob,
        stride_dom,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    dpsum_base_ptr = seqlen_info.offset_batch_Q(
        dPsum + head_idx * stride_ph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_pb,
        stride_pm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    lse_base_ptr = seqlen_info.offset_batch_Q(
        LSE + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        stride_lm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_log2_base_ptr = seqlen_info.offset_batch_Q(
        LSELog2 + head_idx * stride_l2h,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_l2b,
        stride_l2m,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
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

    # Create pointers
    o_ptr = tl.make_tensor_descriptor(
        base=o_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_om, 1],
        block_shape=[TILE_M, TILE_K],
    )
    do_ptr = tl.make_tensor_descriptor(
        base=do_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dom, 1],
        block_shape=[TILE_M, TILE_K],
    )
    lse_ptr = seqlen_info.make_ptrs(
        lse_base_ptr,
        m_block,
        stride_lm,
        offs_m,
        None,
        TILE_K=1,
        SWAP_AB=False,
    )
    dpsum_ptr = seqlen_info.make_ptrs(
        dpsum_base_ptr,
        m_block,
        stride_pm,
        offs_m,
        None,
        TILE_K=1,
        SWAP_AB=False,
    )
    lse_log2_ptr = seqlen_info.make_ptrs(
        lse_log2_base_ptr,
        m_block,
        stride_l2m,
        offs_m,
        None,
        TILE_K=1,
        SWAP_AB=False,
    )
    dq_accum_ptr = tl.make_tensor_descriptor(
        base=dq_accum_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dqam, 1],
        block_shape=[TILE_M, TILE_K],
    )

    # Load o tile
    o_tile = o_ptr.load([m_block * TILE_M, 0]).to(tl.float32)

    # Load do tile
    do_tile = do_ptr.load([m_block * TILE_M, 0]).to(tl.float32)

    # Compute dpsum
    dpsum = tl.sum(o_tile * do_tile, axis=1)

    # Store dpsum
    tl.store(dpsum_ptr, dpsum, mask=(m_block * TILE_M + offs_m) < actual_seqlen_q)

    # Load lse tile
    lse_tile = tl.load(
        lse_ptr,
        mask=(m_block * TILE_M + offs_m) < actual_seqlen_q,
        other=0.0,
    )

    # Compute lse_log2
    log2_e: tl.constexpr = 1.4426950408889634
    lse_log2 = tl.where(lse_tile == lse_tile, lse_tile * log2_e, 1e6)
    lse_log2 = softmax.check_neginf(lse_log2)

    # Store lse_log2
    tl.store(lse_log2_ptr, lse_log2, mask=(m_block * TILE_M + offs_m) < actual_seqlen_q)

    # Create zero tensor
    zero = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Store dq_accum
    dq_accum_ptr.store([m_block * TILE_M, 0], zero)


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
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    is_varlen = cu_seqlens_q is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = out.shape
    else:
        _, num_heads_q, head_dim = out.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else 64
    PADDED_TILE_M = padded_tile_m if padded_tile_m is not None else TILE_M
    num_warps = num_warps if num_warps is not None else 8

    # Get the grid for the kernel launch
    grid = launch_grid.get_bwd_preprocess_grid(
        batch_size=batch_size,
        seqlen_q=triton.cdiv(seqlen_q, PADDED_TILE_M) * PADDED_TILE_M,
        num_heads_q=num_heads_q,
    )

    triton.set_allocator(utils.alloc_fn)

    # Launch the kernel
    _bwd_preprocess_kernel[grid](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        seqused_q,
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
        seqlen_q,
        num_heads_q,
        head_dim,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        PADDED_TILE_M=PADDED_TILE_M,
        num_warps=num_warps,
    )
