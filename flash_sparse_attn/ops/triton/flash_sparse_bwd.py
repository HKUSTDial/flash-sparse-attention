from typing import Optional, Tuple

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    launch_template,
    launch_grid,
    seqlen_info,
    block_info,
    mask,
    flash_bwd_preprocess,
    flash_bwd_postprocess,
)


@triton.jit
def _bwd_inner_sparse_base_kernel(
    acc_dk,
    acc_dv,
    block_max,
    k_tile,
    v_tile,
    q_ptrs,
    do_ptrs,
    dq_accum_ptrs,
    lse_ptrs,
    dpsum_ptrs,
    softmax_scale_log2,
    softmax_threshold_log2,
    m_block,
    n_block,
    actual_seqlen_q,
    actual_seqlen_k,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    # Load query tile
    q_tile = tl.load(q_ptrs, boundary_check=(0, 1))

    # Advance query pointers
    q_ptrs = tl.advance(q_ptrs, (0, TILE_M))

    # Compute attention scores
    acc_s = tl.dot(k_tile, q_tile)

    if IS_MASK:
        # Apply mask
        acc_s = mask.apply_mask(
            acc_s=acc_s,
            m_block=m_block,
            n_block=n_block,
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            MASK_SEQLEN=True,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            QHEADS_PER_KVHEAD_PACKGQA=1,
            SWAP_AB=True,
        )

    # Compute current block max
    block_max_curr = tl.max(acc_s)

    # Update skip condition based on threshold
    block_max_diff_log2 = (block_max_curr - block_max) * softmax_scale_log2
    skip_softmax = block_max_diff_log2 < softmax_threshold_log2

    if not skip_softmax:
        # Update block max
        block_max = tl.maximum(block_max_curr, block_max)

        # Load LSE
        lse_log2 = tl.load(lse_ptrs, boundary_check=(0,))

        # Advance LSE pointers
        lse_ptrs = tl.advance(lse_ptrs, (TILE_M,))

        # Compute attention weights
        p = tl.math.exp2(acc_s * softmax_scale_log2 - lse_log2[None, :]).to(
            q_tile.dtype
        )

        # Load output gradients tile
        do_tile = tl.load(do_ptrs, boundary_check=(0, 1))

        # Advance output gradients pointers
        do_ptrs = tl.advance(do_ptrs, (TILE_M, 0))

        # Compute value gradients
        acc_dv += tl.dot(p, do_tile)

        # Compute attention weight gradients
        acc_dp = tl.dot(v_tile, tl.trans(do_tile))

        # Load dpsum
        dpsum = tl.load(dpsum_ptrs, boundary_check=(0,))

        # Advance dpsum pointers
        dpsum_ptrs = tl.advance(dpsum_ptrs, (TILE_M,))

        # Compute attention score gradients
        ds = p * (acc_dp - dpsum[None, :]).to(q_tile.dtype)

        # Compute query gradients
        dq = tl.dot(tl.trans(ds), k_tile)

        # Store query gradients
        tl.atomic_add(dq_accum_ptrs, dq, sem="relaxed")

        # Compute key gradients
        acc_dk += tl.dot(ds, tl.trans(q_tile))
    else:
        # Advance LSE pointers
        lse_ptrs = tl.advance(lse_ptrs, (TILE_M,))

        # Advance output gradients pointers
        do_ptrs = tl.advance(do_ptrs, (TILE_M, 0))

        # Advance dpsum pointers
        dpsum_ptrs = tl.advance(dpsum_ptrs, (TILE_M,))

    return acc_dk, acc_dv, block_max, q_ptrs, do_ptrs, lse_ptrs, dpsum_ptrs


@triton.jit
def _bwd_sparse_base_kernel(
    Q,
    K,
    V,
    dO,
    LSELog2,
    dPsum,
    dQaccum,
    dK,
    dV,
    softmax_scale,
    softmax_scale_log2,
    softmax_threshold,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_lb,
    stride_lh,
    stride_ll,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    QHEADS_PER_KVHEAD: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    n_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    head_kv_idx = head_idx // QHEADS_PER_KVHEAD

    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)
    offs_kb = tl.arange(0, TILE_K)

    # Get seqlen info for this batch
    (
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        actual_seqlen_q,
        actual_seqlen_k,
    ) = seqlen_info.get_seqlen_info_qk(
        batch_idx=batch_idx,
        seqlen_q_static=seqlen_q,
        seqlen_k_static=seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Early exit if no n_blocks to process
    if n_block * TILE_N >= actual_seqlen_k:
        return

    # Initialize base pointers
    q_base = seqlen_info.offset_batch_Q(
        Q + head_idx * stride_qh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_qb,
        stride_qm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    k_base = seqlen_info.offset_batch_K(
        K + head_kv_idx * stride_kh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_kb,
        stride_kn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    v_base = seqlen_info.offset_batch_K(
        V + head_kv_idx * stride_vh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_vb,
        stride_vn,
        HAS_CU_SEQLENS_K,
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
    lse_base = seqlen_info.offset_batch_Q(
        LSELog2 + head_idx * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        stride_ll,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
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
    dk_base = seqlen_info.offset_batch_K(
        dK + head_kv_idx * stride_dkh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dkb,
        stride_dkn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    dv_base = seqlen_info.offset_batch_K(
        dV + head_kv_idx * stride_dvh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_dvb,
        stride_dvn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )

    # Compute m_block range for this n_block
    m_block_min, m_block_max = block_info.get_m_block_min_max(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        n_block=n_block,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
    )
    m_block_min_no_mask = block_info.get_m_block_min_causal_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        n_block=n_block,
        m_block_min=m_block_min,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
    )
    m_block_max_no_mask = block_info.get_m_block_max_before_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        n_block=n_block,
        m_block_max=m_block_max,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_LOCAL=IS_LOCAL,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
    )

    # Create pointers
    k_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(actual_seqlen_k, head_dim),
        strides=(stride_kn, 1),
        offsets=(n_block * TILE_N, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(actual_seqlen_k, head_dim),
        strides=(stride_vn, 1),
        offsets=(n_block * TILE_N, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
    )
    if QHEADS_PER_KVHEAD > 1:
        dk_ptrs = seqlen_info.make_ptrs(
            base_ptrs=dk_base,
            mn_block=n_block,
            stride_seq=stride_dkn,
            TILE_MN=TILE_N,
            TILE_K=TILE_K,
            SWAP_AB=False,
        )
        dv_ptrs = seqlen_info.make_ptrs(
            base_ptrs=dv_base,
            mn_block=n_block,
            stride_seq=stride_dvn,
            TILE_MN=TILE_N,
            TILE_K=TILE_K,
            SWAP_AB=False,
        )
    else:
        dk_ptrs = tl.make_block_ptr(
            base=dk_base,
            shape=(actual_seqlen_k, head_dim),
            strides=(stride_dkn, 1),
            offsets=(n_block * TILE_N, 0),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0),
        )
        dv_ptrs = tl.make_block_ptr(
            base=dv_base,
            shape=(actual_seqlen_k, head_dim),
            strides=(stride_dvn, 1),
            offsets=(n_block * TILE_N, 0),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0),
        )

    # Load K tile
    k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

    # Load V tile
    v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

    # Initialize accumulators
    block_max = tl.full((), float("-inf"), dtype=tl.float32)
    acc_dk = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dv = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)

    # Process m_blocks with masking
    if IS_CAUSAL or IS_LOCAL:
        q_ptrs = tl.make_block_ptr(
            base=q_base,
            shape=(head_dim, actual_seqlen_q),
            strides=(1, stride_qm),
            offsets=(0, m_block_min * TILE_M),
            block_shape=(TILE_K, TILE_M),
            order=(0, 1),
        )
        do_ptrs = tl.make_block_ptr(
            base=do_base,
            shape=(actual_seqlen_q, head_dim),
            strides=(stride_dom, 1),
            offsets=(m_block_min * TILE_M, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_base,
            shape=(actual_seqlen_q,),
            strides=(stride_ll,),
            offsets=(m_block_min * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        dpsum_ptrs = tl.make_block_ptr(
            base=dpsum_base,
            shape=(actual_seqlen_q,),
            strides=(stride_pm,),
            offsets=(m_block_min * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        for m_block in tl.range(m_block_min, m_block_min_no_mask):
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEADS_PER_KVHEAD_PACKGQA=1,
            )
            dq_accum_ptrs = seqlen_info.make_ptrs(
                base_ptrs=dq_accum_base,
                mn_block=m_block,
                stride_seq=stride_dqam,
                TILE_MN=TILE_M,
                TILE_K=TILE_K,
                SWAP_AB=False,
            )

            acc_dk, acc_dv, block_max, q_ptrs, do_ptrs, lse_ptrs, dpsum_ptrs = (
                _bwd_inner_sparse_base_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    IS_MASK=True,
                    MASK_CAUSAL=IS_CAUSAL,
                    MASK_LOCAL=IS_LOCAL,
                )
            )

    # Process m_blocks without masking
    if m_block_min_no_mask < m_block_max_no_mask:
        q_ptrs = tl.make_block_ptr(
            base=q_base,
            shape=(head_dim, actual_seqlen_q),
            strides=(1, stride_qm),
            offsets=(0, m_block_min_no_mask * TILE_M),
            block_shape=(TILE_K, TILE_M),
            order=(0, 1),
        )
        do_ptrs = tl.make_block_ptr(
            base=do_base,
            shape=(actual_seqlen_q, head_dim),
            strides=(stride_dom, 1),
            offsets=(m_block_min_no_mask * TILE_M, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_base,
            shape=(actual_seqlen_q,),
            strides=(stride_ll,),
            offsets=(m_block_min_no_mask * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        dpsum_ptrs = tl.make_block_ptr(
            base=dpsum_base,
            shape=(actual_seqlen_q,),
            strides=(stride_pm,),
            offsets=(m_block_min_no_mask * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        for m_block in tl.range(m_block_min_no_mask, m_block_max_no_mask):
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEADS_PER_KVHEAD_PACKGQA=1,
            )
            dq_accum_ptrs = seqlen_info.make_ptrs(
                base_ptrs=dq_accum_base,
                mn_block=m_block,
                stride_seq=stride_dqam,
                TILE_MN=TILE_M,
                TILE_K=TILE_K,
                SWAP_AB=False,
            )

            acc_dk, acc_dv, block_max, q_ptrs, do_ptrs, lse_ptrs, dpsum_ptrs = (
                _bwd_inner_sparse_base_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                )
            )

    # Process m_blocks with masking
    if IS_LOCAL and m_block_max_no_mask < m_block_max:
        q_ptrs = tl.make_block_ptr(
            base=q_base,
            shape=(head_dim, actual_seqlen_q),
            strides=(1, stride_qm),
            offsets=(0, m_block_max_no_mask * TILE_M),
            block_shape=(TILE_K, TILE_M),
            order=(0, 1),
        )
        do_ptrs = tl.make_block_ptr(
            base=do_base,
            shape=(actual_seqlen_q, head_dim),
            strides=(stride_dom, 1),
            offsets=(m_block_max_no_mask * TILE_M, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_base,
            shape=(actual_seqlen_q,),
            strides=(stride_ll,),
            offsets=(m_block_max_no_mask * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        dpsum_ptrs = tl.make_block_ptr(
            base=dpsum_base,
            shape=(actual_seqlen_q,),
            strides=(stride_pm,),
            offsets=(m_block_max_no_mask * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        for m_block in tl.range(m_block_max_no_mask, m_block_max):
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEADS_PER_KVHEAD_PACKGQA=1,
            )
            dq_accum_ptrs = seqlen_info.make_ptrs(
                base_ptrs=dq_accum_base,
                mn_block=m_block,
                stride_seq=stride_dqam,
                TILE_MN=TILE_M,
                TILE_K=TILE_K,
                SWAP_AB=False,
            )

            acc_dk, acc_dv, block_max, q_ptrs, do_ptrs, lse_ptrs, dpsum_ptrs = (
                _bwd_inner_sparse_base_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    IS_MASK=True,
                    MASK_CAUSAL=IS_CAUSAL,
                    MASK_LOCAL=IS_LOCAL,
                )
            )

    # Store value gradients
    if QHEADS_PER_KVHEAD > 1:
        tl.atomic_add(
            dv_ptrs,
            acc_dv,
            mask=(offs_n[:, None] < actual_seqlen_k) & (offs_kb[None, :] < head_dim),
            sem="relaxed",
        )
    else:
        tl.store(dv_ptrs, acc_dv, boundary_check=(0, 1))

    # Scale key gradients
    acc_dk = acc_dk * softmax_scale

    # Store key gradients
    if QHEADS_PER_KVHEAD > 1:
        tl.atomic_add(
            dk_ptrs,
            acc_dk,
            mask=(offs_n[:, None] < actual_seqlen_k) & (offs_kb[None, :] < head_dim),
            sem="relaxed",
        )
    else:
        tl.store(dk_ptrs, acc_dk, boundary_check=(0, 1))


def _flash_sparse_attn_base_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    window_size: Tuple[int, int] = (None, None),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    qhead_per_kvhead = num_heads_q // num_heads_kv

    assert_inputs.assert_bwd_inputs(
        query,
        key,
        value,
        out,
        dout,
        lse,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_bwd_sparse_launch_config(
            tile_k=TILE_K,
        )
    )

    seqlen_q_rounded = int(math.ceil(seqlen_q / TILE_M) * TILE_M)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    dq = torch.empty_like(query)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    lse_log2 = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded * head_dim_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dk_accum = torch.zeros(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        batch_size,
        seqlen_k,
        num_heads_kv,
        head_dim,
        dtype=torch.float32,
        device=query.device,
    )

    flash_bwd_preprocess._flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    grid = launch_grid.get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
    )

    _bwd_sparse_base_kernel[grid](
        query,
        key,
        value,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        softmax_scale,
        softmax_scale_log2,
        softmax_threshold,
        query.stride(0),
        query.stride(-2),
        query.stride(-3),
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        dout.stride(0),
        dout.stride(-2),
        dout.stride(-3),
        lse_log2.stride(0),
        lse_log2.stride(1),
        lse_log2.stride(2),
        dpsum.stride(0),
        dpsum.stride(1),
        dpsum.stride(2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        head_dim_rounded,
        dk_accum.stride(0),
        dk_accum.stride(-2),
        dk_accum.stride(-3),
        dv_accum.stride(0),
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        None,
        None,
        None,
        None,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    flash_bwd_postprocess._flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv


def _flash_sparse_attn_varlen_base_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q, num_heads_q, head_dim = query.shape
    total_k, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    qhead_per_kvhead = num_heads_q // num_heads_kv

    assert_inputs.assert_bwd_inputs(
        query,
        key,
        value,
        out,
        dout,
        lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_bwd_sparse_launch_config(
            tile_k=TILE_K,
        )
    )

    total_q_rounded_padded = int(
        math.ceil((total_q + batch_size * TILE_M) / TILE_M) * TILE_M
    )
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    dq = torch.empty_like(query)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    lse_log2 = torch.empty(
        num_heads_q,
        total_q_rounded_padded,
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        num_heads_q,
        total_q_rounded_padded,
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        num_heads_q,
        total_q_rounded_padded * head_dim_rounded,
        dtype=torch.float32,
        device=query.device,
    )
    dk_accum = torch.zeros(
        total_k,
        num_heads_kv,
        head_dim,
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        total_k,
        num_heads_kv,
        head_dim,
        dtype=torch.float32,
        device=query.device,
    )

    flash_bwd_preprocess._flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim_rounded=head_dim_rounded,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    grid = launch_grid.get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
    )

    _bwd_sparse_base_kernel[grid](
        query,
        key,
        value,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        softmax_scale,
        softmax_scale_log2,
        softmax_threshold,
        0,
        query.stride(-2),
        query.stride(0),
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        0,
        dout.stride(-2),
        dout.stride(0),
        0,
        lse_log2.stride(0),
        lse_log2.stride(1),
        0,
        dpsum.stride(0),
        dpsum.stride(1),
        0,
        dq_accum.stride(0),
        head_dim_rounded,
        0,
        dk_accum.stride(-2),
        dk_accum.stride(0),
        0,
        dv_accum.stride(-2),
        dv_accum.stride(0),
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    flash_bwd_postprocess._flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv
