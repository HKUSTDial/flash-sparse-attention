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


@triton.jit(repr=kernel_repr.fwd_combine_repr)
def _fwd_combine_kernel(
    Out_partial,
    Lse_partial,
    Out,
    Lse,
    CuSeqlensQ,
    SeqUsedQ,
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
    head_dim,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
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
        TILE_MN=TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

    # Initialize base pointers
    out_partial_base_ptr = seqlen_info.offset_batch_Q(
        Out_partial + head_idx * stride_oph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_opb,
        stride_opm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_partial_base_ptr = seqlen_info.offset_batch_Q(
        Lse_partial + head_idx * stride_lph,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lpb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    out_base_ptr = seqlen_info.offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_base_ptr = seqlen_info.offset_batch_Q(
        Lse + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    out_partial_ptr = tl.make_tensor_descriptor(
        base=out_partial_base_ptr,
        shape=[num_splits, actual_seqlen_q, head_dim],
        strides=[stride_ops, stride_opm, 1],
        block_shape=[1, TILE_M, TILE_K],
    )
    out_ptr = tl.make_tensor_descriptor(
        base=out_base_ptr,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_om, 1],
        block_shape=[TILE_M, TILE_K],
    )
    lse_partial_ptr = seqlen_info.make_ptrs(
        lse_partial_base_ptr,
        m_block,
        1,
        offs_m,
        None,
        TILE_K=1,
        SWAP_AB=False,
    )
    lse_ptr = seqlen_info.make_ptrs(
        lse_base_ptr,
        m_block,
        1,
        offs_m,
        None,
        TILE_K=1,
        SWAP_AB=False,
    )

    # Initialize accumulators
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Combine split outputs
    for split_idx in tl.range(0, num_splits):
        # Load partial LSE
        lse_partial_s = tl.load(
            lse_partial_ptr + split_idx * stride_lps,
            mask=(m_block * TILE_M + offs_m) < actual_seqlen_q,
            other=float("-inf"),
        )

        # Load partial O
        o_partial_s = out_partial_ptr.load([split_idx, m_block * TILE_M, 0]).reshape(
            [TILE_M, TILE_K]
        )

        # Compute normalized exponentials
        new_row_max = tl.maximum(lse_partial_s, row_max)
        old_scale = tl.where(row_max == float("-inf"), 0.0, tl.exp2(row_max - new_row_max))
        exp_logic = tl.where(
            lse_partial_s == float("-inf"), 0.0, tl.exp2(lse_partial_s - new_row_max)
        )

        # Compute scaled O
        acc_o *= old_scale[:, None]
        acc_o += exp_logic[:, None] * o_partial_s

        # Update row_sum and row_max
        row_sum = row_sum * old_scale + exp_logic
        row_max = new_row_max

    # Normalize O
    inv_sum = tl.where((row_sum == 0.0) | (row_sum != row_sum), 0.0, 1.0 / row_sum)
    acc_o *= inv_sum[:, None]

    # Store O
    out_ptr.store([m_block * TILE_M, 0], acc_o.to(Out.dtype.element_ty))

    # Compute LSE
    # ln2 = math.log(2.0)
    ln2 = 0.6931471805599453
    lse = tl.where(row_sum > 0.0, (row_max + tl.log2(row_sum)) * ln2, float("-inf"))

    # Store LSE
    tl.store(lse_ptr, lse, mask=(m_block * TILE_M + offs_m) < actual_seqlen_q)


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    tile_m: Optional[int] = None,
    num_warps: Optional[int] = None,
):
    is_varlen = cu_seqlens_q is not None
    num_splits = out_partial.shape[0]
    if not is_varlen:
        batch_size, num_heads_q, seqlen_q, head_dim = out_partial.shape[1:]
    else:
        num_heads_q, total_q, head_dim = out_partial.shape[1:]
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = total_q

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
    num_warps = num_warps if num_warps is not None else 4

    # Get the grid for the kernel launch
    grid = launch_grid.get_fwd_combine_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
    )

    triton.set_allocator(utils.alloc_fn)

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
        out.stride(-3) if not is_varlen else out.stride(0),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(-2) if not is_varlen else lse.stride(0),
        num_splits,
        seqlen_q,
        num_heads_q,
        head_dim,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        num_warps=num_warps,
    )
