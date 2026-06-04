import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    cache_utils,
    launch_template,
    launch_grid,
    seqlen_info,
    kernel_repr,
)


@triton.jit(repr=kernel_repr.dec_combine_repr)
def _dec_combine_kernel(
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
    stride_lpm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    stride_lm,
    cu_seqlens_q,
    seqused_q,
    num_splits,
    num_heads_q,
    head_dim,
    TILE_K: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    batch_idx = bh_idx // num_heads_q
    head_idx = bh_idx - batch_idx * num_heads_q

    # Get seqlen info for this batch
    offset_q, _ = seqlen_info.get_seqlen_info(
        batch_idx=batch_idx,
        seqlen_static=1,
        cu_seqlens=cu_seqlens_q,
        seqused=seqused_q,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )

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
        stride_lpm,
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
    lse_base = seqlen_info.offset_batch_Q(
        Lse + head_idx * stride_lh,
        batch_idx,
        offset_q,
        0,
        stride_lb,
        stride_lm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # Create descriptors
    out_part_desc = tl.make_tensor_descriptor(
        base=out_part_base,
        shape=[num_splits, head_dim],
        strides=[stride_ops, 1],
        block_shape=[1, TILE_K],
    )
    out_desc = tl.make_tensor_descriptor(
        base=out_base,
        shape=[head_dim],
        strides=[1],
        block_shape=[TILE_K],
    )

    # Initialize accumulators
    e_sum = tl.full((), 0.0, dtype=tl.float32)
    e_max = tl.full((), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((TILE_K,), dtype=tl.float32)

    # Combine split outputs
    for s in tl.range(0, num_splits):
        # Load partial LSE
        lse_s = tl.load(lse_part_base + s * stride_lps).to(tl.float32)

        # Compute normalized exponentials
        new_e_max = tl.maximum(lse_s, e_max)
        old_scale = tl.exp2(e_max - new_e_max)
        exp_logic = tl.exp2(lse_s - new_e_max)

        # Load partial outputs
        o_s = tl.sum(out_part_desc.load([s, 0]), axis=0)

        # Compute scaled outputs
        acc_o *= old_scale
        acc_o += exp_logic * o_s

        # Update e_sum and e_max
        e_sum = e_sum * old_scale + exp_logic
        e_max = new_e_max

    # Normalize output
    inv_sum = tl.where((e_sum == 0.0) | (e_sum != e_sum), 0.0, 1.0 / e_sum)
    acc_o *= inv_sum

    # Store output
    out_desc.store([0], acc_o.to(Out.dtype.element_ty))

    # Compute LSE
    # ln2 = math.log(2.0)
    ln2 = 0.6931471805599453
    lse = tl.where(e_sum > 0.0, (e_max + tl.log2(e_sum)) * ln2, float("-inf"))

    # Store LSE
    tl.store(lse_base, lse)


_dec_combine_kernel = cache_utils.wrap_kernel(_dec_combine_kernel)


def _flash_attn_dec_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
):
    device = out.device
    arch = cache_utils.get_device_arch(device)
    is_varlen = cu_seqlens_q is not None
    num_splits = out_partial.shape[0]
    if not is_varlen:
        batch_size, num_heads_q, head_dim = out_partial.shape[1:]
    else:
        _, num_heads_q, head_dim = out_partial.shape[1:]
        batch_size = cu_seqlens_q.shape[0] - 1

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    num_warps, num_stages, num_ctas = launch_template.get_dec_combine_launch_config(
        tile_k=TILE_K,
        device=device,
        arch=arch,
    )

    grid = launch_grid.get_dec_combine_grid(
        batch_size=batch_size,
        num_heads_q=num_heads_q,
    )

    _dec_combine_kernel[grid](
        out_partial,
        lse_partial,
        out,
        lse,
        out_partial.stride(0),
        out_partial.stride(1) if not is_varlen else 0,
        out_partial.stride(-2),
        1 if not is_varlen else out_partial.stride(1),
        lse_partial.stride(0),
        lse_partial.stride(1) if not is_varlen else 0,
        lse_partial.stride(-1),
        1 if not is_varlen else lse_partial.stride(1),
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        1 if not is_varlen else out.stride(0),
        lse.stride(0) if not is_varlen else 0,
        lse.stride(-1),
        1 if not is_varlen else lse.stride(0),
        cu_seqlens_q,
        seqused_q,
        num_splits,
        num_heads_q,
        head_dim,
        TILE_K=TILE_K,
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_SEQUSED_Q=seqused_q is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )
