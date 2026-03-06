import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import utils, seqlen_info, launch_template


@triton.jit
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
    head_dim,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_block = tl.program_id(1)
    bh_idx = tl.program_id(2)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)

    # Get seqlen info for this batch
    offset_q, actual_seqlen_q = seqlen_info.get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=seqlen_q,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )
    mask_m = offs_m < actual_seqlen_q

    # Initialize base pointers
    out_part_base = seqlen_info.offset_batch_Q(
        Out_partial + head_idx * stride_oph,
        batch_idx,
        offset_q,
        0,
        stride_opb,
        stride_opm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_part_base = seqlen_info.offset_batch_Q(
        Lse_partial + head_idx * stride_lph,
        batch_idx,
        offset_q,
        0,
        stride_lpb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    out_base = seqlen_info.offset_batch_Q(
        Out + head_idx * stride_oh,
        batch_idx,
        offset_q,
        0,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create pointers
    out_part_ptrs = tl.make_block_ptr(
        base=out_part_base,
        shape=(num_splits, actual_seqlen_q, head_dim),
        strides=(stride_ops, stride_opm, 1),
        offsets=(0, m_block * TILE_M, k_block * TILE_K),
        block_shape=(1, TILE_M, TILE_K),
        order=(2, 1, 0),
    )
    lse_part_ptrs = tl.make_block_ptr(
        base=lse_part_base,
        shape=(num_splits, actual_seqlen_q),
        strides=(stride_lps, 1),
        offsets=(0, m_block * TILE_M),
        block_shape=(1, TILE_M),
        order=(1, 0),
    )
    out_ptrs = tl.make_block_ptr(
        base=out_base,
        shape=(actual_seqlen_q, head_dim),
        strides=(stride_om, 1),
        offsets=(m_block * TILE_M, k_block * TILE_K),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    # Initialize accumulators
    lse_vals = tl.full((MAX_SPLITS, TILE_M), float("-inf"), dtype=tl.float32)
    max_lse = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Compute max across splits
    for s in tl.static_range(MAX_SPLITS):
        if s < num_splits:
            lse_s = tl.sum(tl.load(lse_part_ptrs, boundary_check=(0, 1)), axis=0)
            # boundary_check pads OOB with 0, fixup to -inf for correct softmax
            lse_s = tl.where(mask_m, lse_s, float("-inf"))
            lse_part_ptrs = tl.advance(lse_part_ptrs, (1, 0))
        else:
            lse_s = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
        lse_vals = tl.where(
            (tl.arange(0, MAX_SPLITS) == s)[:, None],
            lse_s[None, :],
            lse_vals,
        )
        max_lse = tl.maximum(max_lse, lse_s)

    # if all -inf, use 0 to avoid nan in exp
    max_lse = tl.where(max_lse == float("-inf"), 0.0, max_lse)

    # Compute normalized scales
    sum_exp = tl.zeros((TILE_M,), dtype=tl.float32)
    for s in tl.static_range(MAX_SPLITS):
        lse_s = tl.sum(
            tl.where(
                (tl.arange(0, MAX_SPLITS) == s)[:, None],
                lse_vals,
                tl.full((MAX_SPLITS, TILE_M), 0.0, dtype=tl.float32),
            ),
            axis=0,
        )
        exp_s = tl.where(s < num_splits, tl.exp(lse_s - max_lse), 0.0)
        sum_exp += exp_s

    inv_sum = tl.where((sum_exp == 0.0) | (sum_exp != sum_exp), 0.0, 1.0 / sum_exp)

    for s in tl.static_range(MAX_SPLITS):
        if s < num_splits:
            lse_s = tl.sum(
                tl.where(
                    (tl.arange(0, MAX_SPLITS) == s)[:, None],
                    lse_vals,
                    tl.full((MAX_SPLITS, TILE_M), 0.0, dtype=tl.float32),
                ),
                axis=0,
            )
            scale = tl.exp(lse_s - max_lse) * inv_sum
            o_s = tl.sum(tl.load(out_part_ptrs, boundary_check=(0, 1, 2)), axis=0)
            out_part_ptrs = tl.advance(out_part_ptrs, (1, 0, 0))
            acc_o += scale[:, None] * o_s

    # Store output
    tl.store(
        out_ptrs,
        acc_o.to(Out.dtype.element_ty),
        boundary_check=(0, 1),
    )
    # Store LSE
    # Only from the first k_block to avoid duplicates
    if k_block == 0:
        lse_base = seqlen_info.offset_batch_Q(
            Lse + head_idx * stride_lh,
            batch_idx,
            offset_q,
            0,
            stride_lb,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_base,
            shape=(actual_seqlen_q,),
            strides=(1,),
            offsets=(m_block * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        lse = tl.where(
            sum_exp > 0.0,
            max_lse + tl.log(sum_exp),
            float("-inf"),
        )
        tl.store(lse_ptrs, lse, boundary_check=(0,))


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
):
    is_varlen = cu_seqlens_q is not None
    num_splits = out_partial.shape[0]
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = out_partial.shape[1:]
    else:
        total_q, num_heads_q, head_dim = out_partial.shape[1:]
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = total_q
    MAX_SPLITS = 1 << max(int(math.ceil(math.log2(max(num_splits, 1)))), 1)

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, num_warps, num_stages, num_ctas = (
        launch_template.get_fwd_combine_launch_config(
            tile_k=TILE_K,
        )
    )

    grid = utils.get_fwd_combine_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        head_dim=head_dim,
    )

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
        head_dim,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HAS_CU_SEQLENS_Q=cu_seqlens_q is not None,
        HAS_SEQUSED_Q=seqused_q is not None,
        MAX_SPLITS=MAX_SPLITS,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )
