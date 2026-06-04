from typing import Optional, Tuple

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    utils,
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
def _bwd_inner_gated_kernel(
    acc_dk,
    acc_dv,
    acc_dd,
    block_max,
    k_tile,
    v_tile,
    d_tile,
    q_desc,
    a_desc,
    do_desc,
    dq_accum_desc,
    da_accum_desc,
    lse_desc,
    dpsum_desc,
    d_max,
    d_min,
    gate_max,
    softmax_scale_log2,
    q_scale,
    softmax_threshold_log2,
    gate_threshold_log2,
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
    IS_LOGSIGMOID_GATE: tl.constexpr,
):
    skip_gate = False
    skip_softmax = False

    # Load alpha tile
    a_tile = a_desc.load([m_block * TILE_M]).to(tl.float32)
    a_max = tl.max(a_tile)
    a_min = tl.min(a_tile)

    # Check if any gates are active for current tile
    gate_max, skip_gate = activations.online_gate(
        a_max,
        a_min,
        d_max,
        d_min,
        gate_max,
        scale_log2=softmax_scale_log2,
        gate_threshold_log2=gate_threshold_log2,
    )

    if not skip_gate:
        # Compute attention gates
        acc_s = d_tile[:, None] * a_tile[None, :]

        # Compute scaling factor for gated attention score gradients
        if IS_LOGSIGMOID_GATE:
            ds_scale = tl.sigmoid(-acc_s)
        else:
            ds_scale = 1.0

        # Load query tile
        q_tile = q_desc.load([m_block * TILE_M, 0])

        # Rescale query
        q_tile = (q_tile * q_scale).to(q_scale.dtype)

        if IS_LOGSIGMOID_GATE:
            acc_s = activations.log_sigmoid(acc_s, FASTMATH=False)

        # Compute attention scores
        acc_s += tl.dot(k_tile, q_tile.T)

        if IS_MASK:
            # Apply mask
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

            # Compute alpha gradients
            da = tl.sum(ds * ds_scale * d_tile[:, None], axis=0)

            # Store alpha gradients
            da_accum_desc.atomic_add([m_block * TILE_M], da)

            # Compute delta gradients
            acc_dd += tl.sum(ds * ds_scale * a_tile[None, :], axis=1)

    return (
        acc_dk,
        acc_dv,
        acc_dd,
        block_max,
        gate_max,
    )


@triton.jit(repr=kernel_repr.bwd_gated_repr)
def _bwd_gated_kernel(
    Q,
    K,
    V,
    A,
    D,
    dO,
    LSELog2,
    dPsum,
    dQaccum,
    dK,
    dV,
    dA,
    dD,
    softmax_scale,
    softmax_scale_log2,
    query_scale,
    key_scale,
    value_scale,
    window_sizes,
    softmax_threshold,
    gate_threshold,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ab,
    stride_ah,
    stride_am,
    stride_db,
    stride_dh,
    stride_dn,
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
    stride_dab,
    stride_dah,
    stride_dam,
    stride_ddb,
    stride_ddh,
    stride_ddn,
    stride_dks,
    stride_dvs,
    stride_dds,
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
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
    IS_ADAPT_GATE: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
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
    a_base = seqlen_info.offset_batch_Q(
        A + head_idx * stride_ah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ab,
        stride_am,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    d_base = seqlen_info.offset_batch_K(
        D + head_kv_idx * stride_dh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_db,
        stride_dn,
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
    da_base = seqlen_info.offset_batch_Q(
        dA + head_idx * stride_dah,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_dab,
        stride_dam,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=True,
    )
    dd_base = seqlen_info.offset_batch_K(
        dD + head_kv_idx * stride_ddh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_ddb,
        stride_ddn,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )

    # For split QO, offset key, value and delta gradients base pointers by split_idx
    if IS_SPLIT_QO:
        dk_base += split_idx * stride_dks
        dv_base += split_idx * stride_dvs
        dd_base += split_idx * stride_dds

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
    a_desc = tl.make_tensor_descriptor(
        base=a_base,
        shape=[actual_seqlen_q],
        strides=[stride_am],
        block_shape=[TILE_M],
    )
    d_desc = tl.make_tensor_descriptor(
        base=d_base,
        shape=[actual_seqlen_k],
        strides=[stride_dn],
        block_shape=[TILE_N],
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
    da_accum_desc = tl.make_tensor_descriptor(
        base=da_base,
        shape=[actual_seqlen_q],
        strides=[stride_dam],
        block_shape=[TILE_M],
    )
    dd_desc = tl.make_tensor_descriptor(
        base=dd_base,
        shape=[actual_seqlen_k],
        strides=[stride_ddn],
        block_shape=[TILE_N],
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
    gate_max = tl.full((), float("-inf"), dtype=tl.float32)
    block_max = tl.full((), float("-inf"), dtype=tl.float32)
    acc_dk = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dv = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dd = tl.zeros((TILE_N,), dtype=tl.float32)

    # Load delta tile
    d_tile = d_desc.load([n_block * TILE_N]).to(tl.float32)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    # Process m_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for m_block in tl.range(m_block_min, m_block_min_no_mask):
            gate_threshold_log2 = seqlen_info.get_gate_threshold(
                gate_threshold=gate_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
                IS_ADAPT_GATE=IS_ADAPT_GATE,
            )
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
            )

            (
                acc_dk,
                acc_dv,
                acc_dd,
                block_max,
                gate_max,
            ) = _bwd_inner_gated_kernel(
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                acc_dd=acc_dd,
                block_max=block_max,
                k_tile=k_tile,
                v_tile=v_tile,
                d_tile=d_tile,
                q_desc=q_desc,
                a_desc=a_desc,
                do_desc=do_desc,
                dq_accum_desc=dq_accum_desc,
                da_accum_desc=da_accum_desc,
                lse_desc=lse_desc,
                dpsum_desc=dpsum_desc,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                softmax_scale_log2=softmax_scale_log2,
                q_scale=q_scale,
                softmax_threshold_log2=softmax_threshold_log2,
                gate_threshold_log2=gate_threshold_log2,
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
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
            )

    # Process m_blocks without masking
    if not IS_LOCAL and m_block_min_no_mask < m_block_max:
        for m_block in tl.range(m_block_min_no_mask, m_block_max):
            gate_threshold_log2 = seqlen_info.get_gate_threshold(
                gate_threshold=gate_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
                IS_ADAPT_GATE=IS_ADAPT_GATE,
            )
            softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                softmax_threshold=softmax_threshold,
                m_block=m_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                IS_CAUSAL=IS_CAUSAL,
                TILE_M=TILE_M,
                QHEAD_PER_KVHEAD_PACKGQA=1,
            )

            (
                acc_dk,
                acc_dv,
                acc_dd,
                block_max,
                gate_max,
            ) = _bwd_inner_gated_kernel(
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                acc_dd=acc_dd,
                block_max=block_max,
                k_tile=k_tile,
                v_tile=v_tile,
                d_tile=d_tile,
                q_desc=q_desc,
                a_desc=a_desc,
                do_desc=do_desc,
                dq_accum_desc=dq_accum_desc,
                da_accum_desc=da_accum_desc,
                lse_desc=lse_desc,
                dpsum_desc=dpsum_desc,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                softmax_scale_log2=softmax_scale_log2,
                q_scale=q_scale,
                softmax_threshold_log2=softmax_threshold_log2,
                gate_threshold_log2=gate_threshold_log2,
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
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
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
                gate_threshold_log2 = seqlen_info.get_gate_threshold(
                    gate_threshold=gate_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                    IS_ADAPT_GATE=IS_ADAPT_GATE,
                )
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    block_max,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_desc=q_desc,
                    a_desc=a_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    da_accum_desc=da_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    gate_threshold_log2=gate_threshold_log2,
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
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                )

        # Process m_blocks without masking
        if m_block_window_min_no_mask < m_block_window_max_no_mask:
            for m_block in tl.range(
                m_block_window_min_no_mask, m_block_window_max_no_mask
            ):
                gate_threshold_log2 = seqlen_info.get_gate_threshold(
                    gate_threshold=gate_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                    IS_ADAPT_GATE=IS_ADAPT_GATE,
                )
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    block_max,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_desc=q_desc,
                    a_desc=a_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    da_accum_desc=da_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    gate_threshold_log2=gate_threshold_log2,
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
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                )

        # Process m_blocks with local left masking
        if m_block_window_max_no_mask < m_block_window_max:
            for m_block in tl.range(m_block_window_max_no_mask, m_block_window_max):
                gate_threshold_log2 = seqlen_info.get_gate_threshold(
                    gate_threshold=gate_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                    IS_ADAPT_GATE=IS_ADAPT_GATE,
                )
                softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
                    softmax_threshold=softmax_threshold,
                    m_block=m_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    IS_CAUSAL=IS_CAUSAL,
                    TILE_M=TILE_M,
                    QHEAD_PER_KVHEAD_PACKGQA=1,
                )

                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    block_max,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    block_max=block_max,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_desc=q_desc,
                    a_desc=a_desc,
                    do_desc=do_desc,
                    dq_accum_desc=dq_accum_desc,
                    da_accum_desc=da_accum_desc,
                    lse_desc=lse_desc,
                    dpsum_desc=dpsum_desc,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
                    softmax_scale_log2=softmax_scale_log2,
                    q_scale=q_scale,
                    softmax_threshold_log2=softmax_threshold_log2,
                    gate_threshold_log2=gate_threshold_log2,
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
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                )

    # Store value gradients
    if QHEAD_PER_KVHEAD > 1:
        dv_desc.atomic_add([n_block * TILE_N, 0], acc_dv)
    else:
        dv_desc.store([n_block * TILE_N, 0], acc_dv)

    # Scale delta gradients
    acc_dd = acc_dd * softmax_scale

    # Store delta gradients
    if QHEAD_PER_KVHEAD > 1:
        dd_desc.atomic_add([n_block * TILE_N], acc_dd)
    else:
        dd_desc.store([n_block * TILE_N], acc_dd)

    # Scale key gradients
    acc_dk = acc_dk * softmax_scale

    # Store key gradients
    if QHEAD_PER_KVHEAD > 1:
        dk_desc.atomic_add([n_block * TILE_N, 0], acc_dk)
    else:
        dk_desc.store([n_block * TILE_N, 0], acc_dk)


_bwd_gated_kernel = cache_utils.wrap_kernel(_bwd_gated_kernel)


_bwd_gated_kernel_autotuned = None


def _get_autotuned_kernel():
    global _bwd_gated_kernel_autotuned
    if _bwd_gated_kernel_autotuned is None:
        jit_kernel = _bwd_gated_kernel._kernel
        autotuned = autotuner.make_bwd_gated_autotuned_kernel(jit_kernel)
        _bwd_gated_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _bwd_gated_kernel_autotuned


def _flash_gated_attn_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
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
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else head_dim / seqlen_k
    )
    gate_threshold = (
        gate_threshold if gate_threshold is not None else head_dim / seqlen_k
    )
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
            alpha=alpha,
            delta=delta,
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
        kernel_name="bwd_gated",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_gated_kernel
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
    da = torch.empty_like(alpha)
    dd = torch.empty_like(delta)
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
    da_accum = torch.zeros(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dd_accum = torch.zeros(
        (num_splits, batch_size, num_heads_kv, seqlen_k)
        if is_split_qo and num_splits > 1
        else (batch_size, num_heads_kv, seqlen_k),
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

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        da_accum,
        dd_accum,
        softmax_scale,
        softmax_scale_log2,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
        query.stride(0),
        query.stride(-2),
        query.stride(-3),
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        alpha.stride(0),
        alpha.stride(-2),
        alpha.stride(-1),
        delta.stride(0),
        delta.stride(-2),
        delta.stride(-1),
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
        dv_accum.stride(-4) if is_split_qo and num_splits > 1 else dv_accum.stride(0),
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        da_accum.stride(0),
        da_accum.stride(-2),
        da_accum.stride(-1),
        dd_accum.stride(-3) if is_split_qo and num_splits > 1 else dd_accum.stride(0),
        dd_accum.stride(-2),
        dd_accum.stride(-1),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dd_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
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
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_gated",
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
        da_accum=da_accum,
        da=da,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)
        dd_accum = dd_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)
    dd.copy_(dd_accum)

    return dq, dk, dv, da, dd


def _flash_gated_attn_varlen_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
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
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else head_dim / seqlen_k
    )
    gate_threshold = (
        gate_threshold if gate_threshold is not None else head_dim / seqlen_k
    )
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
            alpha=alpha,
            delta=delta,
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
        kernel_name="bwd_gated",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_gated_kernel
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
    da = torch.empty_like(alpha)
    dd = torch.empty_like(delta)
    lse_log2 = torch.empty(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        (num_heads_q, total_q_rounded_padded * head_dim_rounded),
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
    da_accum = torch.zeros(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dd_accum = torch.zeros(
        (num_splits, num_heads_kv, total_k)
        if is_split_qo and num_splits > 1
        else (num_heads_kv, total_k),
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

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        da_accum,
        dd_accum,
        softmax_scale,
        softmax_scale_log2,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
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
        alpha.stride(-2),
        alpha.stride(-1),
        0,
        delta.stride(-2),
        delta.stride(-1),
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
        0,
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        0,
        da_accum.stride(0),
        da_accum.stride(-1),
        0,
        dd_accum.stride(-2),
        dd_accum.stride(-1),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dd_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
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
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_gated",
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
        da_accum=da_accum,
        da=da,
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
        dd_accum = dd_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)
    dd.copy_(dd_accum)

    return dq, dk, dv, da, dd
