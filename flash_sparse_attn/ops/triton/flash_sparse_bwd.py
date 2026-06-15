from typing import Optional, Tuple

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    utils,
    device_utils,
    cache_utils,
    launch_template,
    launch_grid,
    seqlen_info,
    block_info,
    activations,
    mask,
    flash_bwd_preprocess,
    flash_bwd_postprocess,
    kernel_repr,
    autotuner,
)


@triton.jit
def _bwd_inner_sparse_kernel(
    acc_dk,
    acc_dv,
    block_max,
    k_tile,
    v_tile,
    q_desc,
    do_desc,
    dq_accum_desc,
    lse_desc,
    dpsum_desc,
    softmax_scale_log2,
    q_scale,
    softmax_threshold_log2,
    m_block,
    n_block,
    actual_seqlen_q,
    actual_seqlen_k,
    window_size_left,
    window_size_right,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    # Load query tile
    q_tile = q_desc.load([m_block * TILE_M, 0])

    # Rescale query
    q_tile = (q_tile * q_scale).to(q_scale.dtype)

    # Compute attention scores
    acc_s = tl.dot(k_tile, q_tile.T)

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask.apply_mask(
            acc_s=acc_s,
            m_block=m_block,
            n_block=n_block,
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            MASK_SEQLEN=True,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            QHEAD_PER_KVHEAD_PACKGQA=1,
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
        lse_log2 = lse_desc.load([m_block * TILE_M])

        # Compute attention weights
        p = activations.exp2(acc_s * softmax_scale_log2 - lse_log2[None, :]).to(
            q_tile.dtype
        )

        # Load output gradients tile
        do_tile = do_desc.load([m_block * TILE_M, 0])

        # Compute value gradients
        acc_dv += tl.dot(p, do_tile)

        # Compute attention weight gradients
        acc_dp = tl.dot(v_tile, tl.trans(do_tile))

        # Load dpsum
        dpsum = dpsum_desc.load([m_block * TILE_M])

        # Compute attention score gradients
        ds = p * (acc_dp - dpsum[None, :]).to(q_tile.dtype)

        # Compute query gradients
        dq = tl.dot(tl.trans(ds), k_tile)

        # Store query gradients
        dq_accum_desc.atomic_add([m_block * TILE_M, 0], dq)

        # Compute key gradients
        acc_dk += tl.dot(ds, q_tile)

    return acc_dk, acc_dv, block_max


@triton.jit(repr=kernel_repr.bwd_sparse_repr)
def _bwd_sparse_kernel(
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
    query_scale,
    key_scale,
    value_scale,
    softmax_threshold,
    window_sizes,
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
    stride_dks,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvs,
    stride_wh,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_splits,
    seqlen_q,
    seqlen_k,
    head_dim,
    SEQLEN_Q_CACHE: tl.constexpr,
    SEQLEN_K_CACHE: tl.constexpr,
    QHEAD_PER_KVHEAD: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    n_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_split_idx = tl.program_id(2)
    if IS_SPLIT_QO:
        batch_idx = batch_split_idx // num_splits
        split_idx = batch_split_idx - batch_idx * num_splits
    else:
        batch_idx = batch_split_idx
        split_idx = 0
    head_kv_idx = head_idx // QHEAD_PER_KVHEAD

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

    # For split QO, offset key and value gradients base pointers by split_idx
    if IS_SPLIT_QO:
        dk_base += split_idx * stride_dks
        dv_base += split_idx * stride_dvs

    # Load window sizes
    if IS_LOCAL:
        window_size_left = tl.load(window_sizes + head_kv_idx * stride_wh)
        window_size_right = tl.load(window_sizes + head_kv_idx * stride_wh + 1)
    else:
        window_size_left = 0
        window_size_right = 0

    # Compute m_block range for this n_block
    m_block_min, m_block_max, m_block_window_min, m_block_window_max = (
        block_info.get_m_block_min_max(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            n_block=n_block,
            split_idx=split_idx,
            num_splits=num_splits,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_CAUSAL=IS_CAUSAL,
            IS_LOCAL=IS_LOCAL,
            IS_SPLIT_QO=IS_SPLIT_QO,
        )
    )
    m_block_min_no_mask = block_info.get_m_block_min_causal_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        n_block=n_block,
        m_block_min=m_block_min,
        window_size_right=0,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=IS_CAUSAL or IS_LOCAL,
        IS_LOCAL=False,
    )

    # Clamp to split's range so the no-mask loop stays within bounds
    if IS_SPLIT_QO:
        m_block_min_no_mask = tl.minimum(m_block_min_no_mask, m_block_max)

    # Create pointers or descriptors
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_qm, 1],
        block_shape=[TILE_M, TILE_K],
    )
    k_desc = tl.make_tensor_descriptor(
        base=k_base,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_kn, 1],
        block_shape=[TILE_N, TILE_K],
    )
    v_desc = tl.make_tensor_descriptor(
        base=v_base,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_vn, 1],
        block_shape=[TILE_N, TILE_K],
    )
    do_desc = tl.make_tensor_descriptor(
        base=do_base,
        shape=[actual_seqlen_q, head_dim],
        strides=[stride_dom, 1],
        block_shape=[TILE_M, TILE_K],
    )
    lse_desc = tl.make_tensor_descriptor(
        base=lse_base,
        shape=[actual_seqlen_q],
        strides=[stride_ll],
        block_shape=[TILE_M],
    )
    dpsum_desc = tl.make_tensor_descriptor(
        base=dpsum_base,
        shape=[actual_seqlen_q],
        strides=[stride_pm],
        block_shape=[TILE_M],
    )
    dq_accum_desc = tl.make_tensor_descriptor(
        base=dq_accum_base,
        shape=[actual_seqlen_q, stride_dqam],
        strides=[stride_dqam, 1],
        block_shape=[TILE_M, TILE_K],
    )
    dk_desc = tl.make_tensor_descriptor(
        base=dk_base,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_dkn, 1],
        block_shape=[TILE_N, TILE_K],
    )
    dv_desc = tl.make_tensor_descriptor(
        base=dv_base,
        shape=[actual_seqlen_k, head_dim],
        strides=[stride_dvn, 1],
        block_shape=[TILE_N, TILE_K],
    )

    # Load query scale
    q_scale = tl.load(query_scale)

    # Load key scale
    k_scale = tl.load(key_scale)

    # Load value scale
    v_scale = tl.load(value_scale)

    # Load key tile
    k_tile = k_desc.load([n_block * TILE_N, 0])

    # Rescale key
    k_tile = (k_tile * k_scale).to(k_scale.dtype)

    # Load value tile
    v_tile = v_desc.load([n_block * TILE_N, 0])

    # Rescale value
    v_tile = (v_tile * v_scale).to(v_scale.dtype)

    # Initialize accumulators
    block_max = tl.full((), float("-inf"), dtype=tl.float32)
    acc_dk = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dv = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)

    # Process m_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for m_block in tl.range(m_block_min, m_block_min_no_mask):
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
            )

            acc_dk, acc_dv, block_max = _bwd_inner_sparse_kernel(
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                block_max=block_max,
                k_tile=k_tile,
                v_tile=v_tile,
                q_desc=q_desc,
                do_desc=do_desc,
                dq_accum_desc=dq_accum_desc,
                lse_desc=lse_desc,
                dpsum_desc=dpsum_desc,
                softmax_scale_log2=softmax_scale_log2,
                q_scale=q_scale,
                softmax_threshold_log2=softmax_threshold_log2,
                m_block=m_block,
                n_block=n_block,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                IS_MASK=True,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=True if IS_LOCAL else False,
            )

    # Process m_blocks without masking
    if not IS_LOCAL and m_block_min_no_mask < m_block_max:
        for m_block in tl.range(m_block_min_no_mask, m_block_max):
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
            )

            acc_dk, acc_dv, block_max = _bwd_inner_sparse_kernel(
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                block_max=block_max,
                k_tile=k_tile,
                v_tile=v_tile,
                q_desc=q_desc,
                do_desc=do_desc,
                dq_accum_desc=dq_accum_desc,
                lse_desc=lse_desc,
                dpsum_desc=dpsum_desc,
                softmax_scale_log2=softmax_scale_log2,
                q_scale=q_scale,
                softmax_threshold_log2=softmax_threshold_log2,
                m_block=m_block,
                n_block=n_block,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                IS_MASK=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
            )

    if IS_LOCAL:
        # Compute m_block range for this n_block
        m_block_window_min = tl.maximum(m_block_window_min, m_block_min_no_mask)
        m_block_window_max = tl.minimum(m_block_window_max, m_block_max)
        m_block_window_min_no_mask = block_info.get_m_block_min_causal_local_mask(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            n_block=n_block,
            m_block_min=m_block_window_min,
            window_size_right=window_size_right,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_CAUSAL=False,
            IS_LOCAL=True,
        )
        m_block_window_min_no_mask = tl.maximum(
            m_block_window_min_no_mask, m_block_window_min
        )
        m_block_window_max_no_mask = block_info.get_m_block_max_before_local_mask(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            n_block=n_block,
            m_block_max=m_block_window_max,
            window_size_left=window_size_left,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_LOCAL=True,
        )
        m_block_window_max_no_mask = tl.maximum(
            m_block_window_max_no_mask, m_block_window_min_no_mask
        )

        # Process m_blocks with local right masking
        if m_block_window_min < m_block_window_min_no_mask:
            for m_block in tl.range(m_block_window_min, m_block_window_min_no_mask):
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                acc_dk, acc_dv, block_max = _bwd_inner_sparse_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_desc=q_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                )

        # Process m_blocks without masking
        if m_block_window_min_no_mask < m_block_window_max_no_mask:
            for m_block in tl.range(
                m_block_window_min_no_mask, m_block_window_max_no_mask
            ):
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                acc_dk, acc_dv, block_max = _bwd_inner_sparse_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_desc=q_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                )

        # Process m_blocks with local left masking
        if m_block_window_max_no_mask < m_block_window_max:
            for m_block in tl.range(m_block_window_max_no_mask, m_block_window_max):
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                acc_dk, acc_dv, block_max = _bwd_inner_sparse_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_desc=q_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                )

    # Store value gradients
    if QHEAD_PER_KVHEAD > 1:
        dv_desc.atomic_add([n_block * TILE_N, 0], acc_dv)
    else:
        dv_desc.store([n_block * TILE_N, 0], acc_dv)

    # Scale key gradients
    acc_dk = acc_dk * softmax_scale

    # Store key gradients
    if QHEAD_PER_KVHEAD > 1:
        dk_desc.atomic_add([n_block * TILE_N, 0], acc_dk)
    else:
        dk_desc.store([n_block * TILE_N, 0], acc_dk)


_bwd_sparse_kernel = cache_utils.wrap_kernel(_bwd_sparse_kernel)


_bwd_sparse_kernel_autotuned = None


def _get_autotuned_kernel():
    global _bwd_sparse_kernel_autotuned
    if _bwd_sparse_kernel_autotuned is None:
        jit_kernel = _bwd_sparse_kernel._kernel
        autotuned = autotuner.make_bwd_sparse_autotuned_kernel(jit_kernel)
        _bwd_sparse_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _bwd_sparse_kernel_autotuned


def _flash_sparse_attn_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: float = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else head_dim / seqlen_k
    )
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 2), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="bwd_sparse",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_sparse_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_k,
            seqlen_k=seqlen_q,
            num_SMs=num_SMs,
            TILE_M=TILE_N,
            TILE_N=TILE_M,
        )
        if is_split_qo
        else 1
    )

    seqlen_q_rounded = int(math.ceil(seqlen_q / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    dq = torch.empty_like(query, dtype=query_scale.dtype)
    dk = torch.empty_like(key, dtype=key_scale.dtype)
    dv = torch.empty_like(value, dtype=value_scale.dtype)
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
        (num_splits, batch_size, seqlen_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (batch_size, seqlen_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        (num_splits, batch_size, seqlen_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (batch_size, seqlen_k, num_heads_kv, head_dim),
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
        num_splits=num_splits,
    )

    device_utils.set_triton_allocator(device)

    kernel[grid](
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
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        window_sizes,
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
        dk_accum.stride(-4) if is_split_qo and num_splits > 1 else dk_accum.stride(0),
        dk_accum.stride(-2),
        dk_accum.stride(-3),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dv_accum.stride(-4) if is_split_qo and num_splits > 1 else dv_accum.stride(0),
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0),
        None,
        None,
        None,
        None,
        num_splits,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=seqlen_q // 1024,
        SEQLEN_K_CACHE=seqlen_k // 1024,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_sparse",
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_quant=is_quant,
            )

    flash_bwd_postprocess._flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv


def _flash_sparse_attn_varlen_backward(
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
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: float = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    total_q, num_heads_q, head_dim = query.shape
    total_k, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else head_dim / seqlen_k
    )
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 2), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="bwd_sparse",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_sparse_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_k,
            seqlen_k=seqlen_q,
            num_SMs=num_SMs,
            TILE_M=TILE_N,
            TILE_N=TILE_M,
        )
        if is_split_qo
        else 1
    )

    total_q_rounded_padded = int(math.ceil((total_q + batch_size * 128) / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    dq = torch.empty_like(query, dtype=query_scale.dtype)
    dk = torch.empty_like(key, dtype=key_scale.dtype)
    dv = torch.empty_like(value, dtype=value_scale.dtype)
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
        (num_splits, total_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (total_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        (num_splits, total_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (total_k, num_heads_kv, head_dim),
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
        num_splits=num_splits,
    )

    device_utils.set_triton_allocator(device)

    kernel[grid](
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
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        window_sizes,
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
        dk_accum.stride(-3),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        0,
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0),
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        num_splits,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=seqlen_q // 1024,
        SEQLEN_K_CACHE=seqlen_k // 1024,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_sparse",
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_quant=is_quant,
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

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv
