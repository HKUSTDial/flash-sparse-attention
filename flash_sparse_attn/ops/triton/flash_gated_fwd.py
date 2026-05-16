from typing import Tuple, Optional

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
    flash_dec_combine,
    kernel_repr,
    autotuner,
)


@triton.jit
def _fwd_inner_gated_kernel(
    skip_gate_curr,
    acc_s,
    acc_o,
    q_tile,
    a_tile,
    k_ptrs,
    v_ptrs,
    d_ptrs,
    a_max,
    a_min,
    gate_max,
    block_max,
    row_max,
    row_sum,
    softmax_scale_log2,
    gate_threshold_log2,
    softmax_threshold_log2,
    m_block,
    n_block,
    n_block_min,
    actual_seqlen_q,
    actual_seqlen_k,
    window_size_left,
    window_size_right,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
    CHECK_INF: tl.constexpr,
):
    # Advance delta pointers
    d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

    # Load next delta tile
    d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(tl.float32)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    skip_gate_next = True
    if not skip_gate_curr:
        # Advance key pointers
        k_ptrs = tl.advance(k_ptrs, (0, -TILE_N))

        if n_block > n_block_min:
            # Check if any gates are active for next tile
            gate_max, skip_gate_next = activations.online_gate(
                a_max,
                a_min,
                d_max,
                d_min,
                gate_max,
                scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
            )

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
                QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                SWAP_AB=False,
            )

        # Apply online softmax
        p, block_max, row_max, row_sum, row_scale, skip_softmax = (
            activations.online_sparse_softmax(
                acc_s=acc_s,
                block_max=block_max,
                row_max=row_max,
                row_sum=row_sum,
                scale_log2=softmax_scale_log2,
                softmax_threshold_log2=softmax_threshold_log2,
                CHECK_INF=CHECK_INF,
            )
        )

        if not skip_softmax:
            # Load value tile
            v_tile = tl.load(v_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

            # Rescale output accumulator
            acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)

            # Update output accumulator
            acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

        # Advance value pointers
        v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))
    else:
        # Advance key and value pointers
        k_ptrs = tl.advance(k_ptrs, (0, -TILE_N))
        v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

        if n_block > n_block_min:
            # Check if any gates are active for next tile
            gate_max, skip_gate_next = activations.online_gate(
                a_max,
                a_min,
                d_max,
                d_min,
                gate_max,
                scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
            )

    if not skip_gate_next:
        # Compute attention gates for next tile
        acc_s = a_tile[:, None] * d_tile[None, :]

        if IS_LOGSIGMOID_GATE:
            acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

        # Load next key tile
        k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

        # Compute attention scores for next tile
        acc_s += tl.dot(q_tile, k_tile)

    return (
        skip_gate_next,
        acc_s,
        acc_o,
        k_ptrs,
        v_ptrs,
        d_ptrs,
        gate_max,
        block_max,
        row_max,
        row_sum,
    )


@triton.jit(repr=kernel_repr.fwd_gated_repr)
def _fwd_gated_kernel(
    Q,
    K,
    V,
    A,
    D,
    Out,
    Lse,
    softmax_scale_log2,
    softmax_threshold,
    gate_threshold,
    query_scale,
    key_scale,
    value_scale,
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
    stride_ab,
    stride_ah,
    stride_am,
    stride_db,
    stride_dh,
    stride_dn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_os,
    stride_lb,
    stride_lh,
    stride_ls,
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
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    PACK_GQA: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
    IS_ADAPT_GATE: tl.constexpr,
):
    m_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_split_idx = tl.program_id(2)
    if IS_SPLIT_KV:
        batch_idx = batch_split_idx // num_splits
        split_idx = batch_split_idx - batch_idx * num_splits
    else:
        batch_idx = batch_split_idx
        split_idx = 0
    if PACK_GQA:
        head_kv_idx = head_idx
    else:
        head_kv_idx = head_idx // QHEAD_PER_KVHEAD

    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
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

    # Initialize base pointers
    q_base = seqlen_info.offset_batch_Q(
        Q + head_idx * stride_qh if not PACK_GQA else Q,
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
        A + head_idx * stride_ah if not PACK_GQA else A,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ab,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    d_base = seqlen_info.offset_batch_K(
        D + head_kv_idx * stride_dh,
        batch_idx,
        offset_k,
        padded_offset_k,
        stride_db,
        1,
        HAS_CU_SEQLENS_K,
        USE_PADDED=False,
    )
    out_base = seqlen_info.offset_batch_Q(
        Out + head_idx * stride_oh if not PACK_GQA else Out,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_base = seqlen_info.offset_batch_Q(
        Lse + head_idx * stride_lh if not PACK_GQA else Lse,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        1,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # For split KV, offset output and LSE base pointers by split_idx
    if IS_SPLIT_KV:
        out_base += split_idx * stride_os
        lse_base += split_idx * stride_ls

    # Load window sizes
    if IS_LOCAL:
        window_size_left = tl.load(window_sizes + head_kv_idx * stride_wh)
        window_size_right = tl.load(window_sizes + head_kv_idx * stride_wh + 1)
    else:
        window_size_left = 0
        window_size_right = 0

    # Compute n_block range for this m_block
    n_block_min, n_block_max, n_block_window_min, n_block_window_max = (
        block_info.get_n_block_min_max(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            m_block=m_block,
            split_idx=split_idx,
            num_splits=num_splits,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_CAUSAL=IS_CAUSAL,
            IS_LOCAL=IS_LOCAL,
            IS_SPLIT_KV=IS_SPLIT_KV,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
    )
    n_block_max_no_mask = block_info.get_n_block_min_causal_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=m_block,
        n_block_min=n_block_min,
        window_size_right=0,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_LOCAL=False,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
    )

    # Clamp to split's range so the no-mask loop stays within bounds
    if IS_SPLIT_KV:
        n_block_max_no_mask = tl.minimum(n_block_max_no_mask, n_block_max)

    # Create pointers
    if not PACK_GQA:
        lse_ptrs = tl.make_block_ptr(
            base=lse_base,
            shape=(actual_seqlen_q,),
            strides=(1,),
            offsets=(m_block * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
        out_ptrs = tl.make_block_ptr(
            base=out_base,
            shape=(actual_seqlen_q, head_dim),
            strides=(stride_om, 1),
            offsets=(m_block * TILE_M, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
    else:
        lse_ptrs = seqlen_info.make_pack_gqa_ptrs(
            lse_base,
            m_block,
            head_idx,
            stride_lh,
            1,
            TILE_M=TILE_M,
            TILE_K=1,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
        out_ptrs = seqlen_info.make_pack_gqa_ptrs(
            out_base,
            m_block,
            head_idx,
            stride_oh,
            stride_om,
            TILE_M=TILE_M,
            TILE_K=TILE_K,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )

    # Early exit if no n_blocks to process
    if n_block_min >= n_block_max:
        # Write LSE as -inf for proper handling
        lse_tile = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
        if PACK_GQA:
            tl.store(
                lse_ptrs,
                lse_tile,
                mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
                cache_modifier=".wb",
            )
        else:
            tl.store(lse_ptrs, lse_tile, boundary_check=(0,), cache_modifier=".wb")

        # Write output as zero for proper handling
        o_tile = tl.zeros((TILE_M, TILE_K), dtype=Out.dtype.element_ty)
        if PACK_GQA:
            tl.store(
                out_ptrs,
                o_tile,
                mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
                & (offs_kb < head_dim)[None, :],
                cache_modifier=".wb",
            )
        else:
            tl.store(out_ptrs, o_tile, boundary_check=(0, 1), cache_modifier=".wb")
        return

    if not PACK_GQA:
        q_ptrs = tl.make_block_ptr(
            base=q_base,
            shape=(actual_seqlen_q, head_dim),
            strides=(stride_qm, 1),
            offsets=(m_block * TILE_M, 0),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0),
        )
        a_ptrs = tl.make_block_ptr(
            base=a_base,
            shape=(actual_seqlen_q,),
            strides=(stride_am,),
            offsets=(m_block * TILE_M,),
            block_shape=(TILE_M,),
            order=(0,),
        )
    else:
        q_ptrs = seqlen_info.make_pack_gqa_ptrs(
            q_base,
            m_block,
            head_idx,
            stride_qh,
            stride_qm,
            TILE_M=TILE_M,
            TILE_K=TILE_K,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
        a_ptrs = seqlen_info.make_pack_gqa_ptrs(
            a_base,
            m_block,
            head_idx,
            stride_ah,
            stride_am,
            TILE_M=TILE_M,
            TILE_K=1,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
    k_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(head_dim, actual_seqlen_k),
        strides=(1, stride_kn),
        offsets=(0, (n_block_max - 1) * TILE_N),
        block_shape=(TILE_K, TILE_N),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(actual_seqlen_k, head_dim),
        strides=(stride_vn, 1),
        offsets=((n_block_max - 1) * TILE_N, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_base,
        shape=(actual_seqlen_k,),
        strides=(stride_dn,),
        offsets=((n_block_max - 1) * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,),
    )

    # Get gate threshold
    gate_threshold_log2 = seqlen_info.get_gate_threshold(
        gate_threshold=gate_threshold,
        m_block=m_block,
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        IS_CAUSAL=IS_CAUSAL,
        TILE_M=TILE_M,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        IS_ADAPT_GATE=IS_ADAPT_GATE,
    )

    # Get softmax threshold
    softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
        softmax_threshold=softmax_threshold,
        m_block=m_block,
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        IS_CAUSAL=IS_CAUSAL,
        TILE_M=TILE_M,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
    )

    # Load query scale
    q_scale = tl.load(query_scale)

    # Load key scale
    k_scale = tl.load(key_scale)

    # Rescale softmax scale
    softmax_scale_log2 = softmax_scale_log2 * q_scale * k_scale

    # Load query tile
    if PACK_GQA:
        q_tile = tl.load(
            q_ptrs,
            mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
            & (offs_kb < head_dim)[None, :],
            other=0.0,
            cache_modifier=".ca",
        )
    else:
        q_tile = tl.load(q_ptrs, boundary_check=(0, 1), cache_modifier=".ca")

    # Load key tile
    k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

    # Initialize accumulators
    gate_max = tl.full((), float("-inf"), dtype=tl.float32)
    block_max = tl.full((), float("-inf"), dtype=tl.float32)
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load alpha tile
    if PACK_GQA:
        a_tile = tl.load(
            a_ptrs,
            mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.float32)
    else:
        a_tile = tl.load(a_ptrs, boundary_check=(0,), cache_modifier=".ca").to(
            tl.float32
        )
    a_max = tl.max(a_tile)
    a_min = tl.min(a_tile)

    # Load delta tile
    d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(tl.float32)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    # Check if any gates are active for current tile
    gate_max, skip_gate_curr = activations.online_gate(
        a_max,
        a_min,
        d_max,
        d_min,
        gate_max,
        scale_log2=softmax_scale_log2,
        gate_threshold_log2=gate_threshold_log2,
    )

    # Initialize skip_gate_curr to False for the first iteration
    skip_gate_curr = False

    # Compute attention gates for first tile
    acc_s = a_tile[:, None] * d_tile[None, :]
    if IS_LOGSIGMOID_GATE:
        acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

    # Compute attention scores for first tile
    acc_s += tl.dot(q_tile, k_tile)

    # Process n_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for n_block in tl.range(n_block_max - 1, n_block_max_no_mask - 1, -1):
            (
                skip_gate_curr,
                acc_s,
                acc_o,
                k_ptrs,
                v_ptrs,
                d_ptrs,
                gate_max,
                block_max,
                row_max,
                row_sum,
            ) = _fwd_inner_gated_kernel(
                skip_gate_curr=skip_gate_curr,
                acc_s=acc_s,
                acc_o=acc_o,
                q_tile=q_tile,
                a_tile=a_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                d_ptrs=d_ptrs,
                a_max=a_max,
                a_min=a_min,
                gate_max=gate_max,
                block_max=block_max,
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
                softmax_threshold_log2=softmax_threshold_log2,
                m_block=m_block,
                n_block=n_block,
                n_block_min=n_block_max_no_mask,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                IS_MASK=True,
                MASK_CAUSAL=True,
                MASK_LOCAL=True if IS_LOCAL else False,
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                CHECK_INF=True,
            )
    else:
        # First iteration with seqlen masking
        n_block = n_block_max - 1

        (
            skip_gate_curr,
            acc_s,
            acc_o,
            k_ptrs,
            v_ptrs,
            d_ptrs,
            gate_max,
            block_max,
            row_max,
            row_sum,
        ) = _fwd_inner_gated_kernel(
            skip_gate_curr=skip_gate_curr,
            acc_s=acc_s,
            acc_o=acc_o,
            q_tile=q_tile,
            a_tile=a_tile,
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
            d_ptrs=d_ptrs,
            a_max=a_max,
            a_min=a_min,
            gate_max=gate_max,
            block_max=block_max,
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
            gate_threshold_log2=gate_threshold_log2,
            softmax_threshold_log2=softmax_threshold_log2,
            m_block=m_block,
            n_block=n_block,
            n_block_min=n_block,
            actual_seqlen_q=actual_seqlen_q,
            actual_seqlen_k=actual_seqlen_k,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
            IS_MASK=True,
            MASK_CAUSAL=False,
            MASK_LOCAL=False,
            IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
            CHECK_INF=True,
        )

        n_block_max_no_mask = n_block_max - 1

    # Process n_blocks without masking
    if not IS_LOCAL and n_block_max_no_mask > n_block_min:
        k_ptrs = tl.make_block_ptr(
            base=k_base,
            shape=(head_dim, actual_seqlen_k),
            strides=(1, stride_kn),
            offsets=(0, (n_block_max_no_mask - 1) * TILE_N),
            block_shape=(TILE_K, TILE_N),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_base,
            shape=(actual_seqlen_k, head_dim),
            strides=(stride_vn, 1),
            offsets=((n_block_max_no_mask - 1) * TILE_N, 0),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0),
        )
        d_ptrs = tl.make_block_ptr(
            base=d_base,
            shape=(actual_seqlen_k,),
            strides=(stride_dn,),
            offsets=((n_block_max_no_mask - 1) * TILE_N,),
            block_shape=(TILE_N,),
            order=(0,),
        )

        # Load key tile
        k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

        # Load delta tile
        d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(
            tl.float32
        )
        d_max = tl.max(d_tile)
        d_min = tl.min(d_tile)

        # Check if any gates are active for current tile
        gate_max, skip_gate_curr = activations.online_gate(
            a_max,
            a_min,
            d_max,
            d_min,
            gate_max,
            scale_log2=softmax_scale_log2,
            gate_threshold_log2=gate_threshold_log2,
        )

        # Compute attention gates
        acc_s = a_tile[:, None] * d_tile[None, :]
        if IS_LOGSIGMOID_GATE:
            acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

        # Compute attention scores
        acc_s += tl.dot(q_tile, k_tile)

        for n_block in tl.range(n_block_max_no_mask - 1, n_block_min - 1, -1):
            (
                skip_gate_curr,
                acc_s,
                acc_o,
                k_ptrs,
                v_ptrs,
                d_ptrs,
                gate_max,
                block_max,
                row_max,
                row_sum,
            ) = _fwd_inner_gated_kernel(
                skip_gate_curr=skip_gate_curr,
                acc_s=acc_s,
                acc_o=acc_o,
                q_tile=q_tile,
                a_tile=a_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                d_ptrs=d_ptrs,
                a_max=a_max,
                a_min=a_min,
                gate_max=gate_max,
                block_max=block_max,
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
                softmax_threshold_log2=softmax_threshold_log2,
                m_block=m_block,
                n_block=n_block,
                n_block_min=n_block_min,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                IS_MASK=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                CHECK_INF=False,
            )

    if IS_LOCAL:
        # Compute n_block range for this m_block
        n_block_window_min = tl.maximum(n_block_window_min, n_block_min)
        n_block_window_max = tl.minimum(n_block_window_max, n_block_max_no_mask)
        n_block_window_max_no_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            m_block=m_block,
            n_block_min=n_block_window_min,
            window_size_right=window_size_right,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_LOCAL=True,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
        n_block_window_min_no_mask = block_info.get_n_block_min_before_local_mask(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            m_block=m_block,
            n_block_min=n_block_window_min,
            window_size_left=window_size_left,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_LOCAL=True,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
        n_block_window_min_no_mask = tl.minimum(
            n_block_window_min_no_mask, n_block_window_max_no_mask
        )

        # Clamp window no-mask boundaries to the split's assigned range
        if IS_SPLIT_KV:
            n_block_window_max_no_mask = tl.maximum(
                tl.minimum(n_block_window_max_no_mask, n_block_window_max),
                n_block_window_min,
            )
            n_block_window_min_no_mask = tl.maximum(
                tl.minimum(n_block_window_min_no_mask, n_block_window_max),
                n_block_window_min,
            )

        # Process n_blocks with local right masking
        if n_block_window_max > n_block_window_max_no_mask:
            k_ptrs = tl.make_block_ptr(
                base=k_base,
                shape=(head_dim, actual_seqlen_k),
                strides=(1, stride_kn),
                offsets=(0, (n_block_window_max - 1) * TILE_N),
                block_shape=(TILE_K, TILE_N),
                order=(0, 1),
            )
            v_ptrs = tl.make_block_ptr(
                base=v_base,
                shape=(actual_seqlen_k, head_dim),
                strides=(stride_vn, 1),
                offsets=((n_block_window_max - 1) * TILE_N, 0),
                block_shape=(TILE_N, TILE_K),
                order=(1, 0),
            )
            d_ptrs = tl.make_block_ptr(
                base=d_base,
                shape=(actual_seqlen_k,),
                strides=(stride_dn,),
                offsets=((n_block_window_max - 1) * TILE_N,),
                block_shape=(TILE_N,),
                order=(0,),
            )

            # Load key tile
            k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

            # Load delta tile
            d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(
                tl.float32
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = activations.online_gate(
                a_max,
                a_min,
                d_max,
                d_min,
                gate_max,
                scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile)

            for n_block in tl.range(
                n_block_window_max - 1, n_block_window_max_no_mask - 1, -1
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    k_ptrs,
                    v_ptrs,
                    d_ptrs,
                    gate_max,
                    block_max,
                    row_max,
                    row_sum,
                ) = _fwd_inner_gated_kernel(
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    gate_threshold_log2=gate_threshold_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    n_block_min=n_block_window_max_no_mask,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                    CHECK_INF=True,
                )

        # Process n_blocks without masking
        if n_block_window_max_no_mask > n_block_window_min_no_mask:
            k_ptrs = tl.make_block_ptr(
                base=k_base,
                shape=(head_dim, actual_seqlen_k),
                strides=(1, stride_kn),
                offsets=(0, (n_block_window_max_no_mask - 1) * TILE_N),
                block_shape=(TILE_K, TILE_N),
                order=(0, 1),
            )
            v_ptrs = tl.make_block_ptr(
                base=v_base,
                shape=(actual_seqlen_k, head_dim),
                strides=(stride_vn, 1),
                offsets=((n_block_window_max_no_mask - 1) * TILE_N, 0),
                block_shape=(TILE_N, TILE_K),
                order=(1, 0),
            )
            d_ptrs = tl.make_block_ptr(
                base=d_base,
                shape=(actual_seqlen_k,),
                strides=(stride_dn,),
                offsets=((n_block_window_max_no_mask - 1) * TILE_N,),
                block_shape=(TILE_N,),
                order=(0,),
            )

            # Load key tile
            k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

            # Load delta tile
            d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(
                tl.float32
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = activations.online_gate(
                a_max,
                a_min,
                d_max,
                d_min,
                gate_max,
                scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile)

            for n_block in tl.range(
                n_block_window_max_no_mask - 1, n_block_window_min_no_mask - 1, -1
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    k_ptrs,
                    v_ptrs,
                    d_ptrs,
                    gate_max,
                    block_max,
                    row_max,
                    row_sum,
                ) = _fwd_inner_gated_kernel(
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    gate_threshold_log2=gate_threshold_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    n_block_min=n_block_window_min_no_mask,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                    CHECK_INF=False,
                )

        # Process n_blocks with local left masking
        if n_block_window_min_no_mask > n_block_window_min:
            k_ptrs = tl.make_block_ptr(
                base=k_base,
                shape=(head_dim, actual_seqlen_k),
                strides=(1, stride_kn),
                offsets=(0, (n_block_window_min_no_mask - 1) * TILE_N),
                block_shape=(TILE_K, TILE_N),
                order=(0, 1),
            )
            v_ptrs = tl.make_block_ptr(
                base=v_base,
                shape=(actual_seqlen_k, head_dim),
                strides=(stride_vn, 1),
                offsets=((n_block_window_min_no_mask - 1) * TILE_N, 0),
                block_shape=(TILE_N, TILE_K),
                order=(1, 0),
            )
            d_ptrs = tl.make_block_ptr(
                base=d_base,
                shape=(actual_seqlen_k,),
                strides=(stride_dn,),
                offsets=((n_block_window_min_no_mask - 1) * TILE_N,),
                block_shape=(TILE_N,),
                order=(0,),
            )

            # Load key tile
            k_tile = tl.load(k_ptrs, boundary_check=(0, 1), cache_modifier=".cg")

            # Load delta tile
            d_tile = tl.load(d_ptrs, boundary_check=(0,), cache_modifier=".cg").to(
                tl.float32
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = activations.online_gate(
                a_max,
                a_min,
                d_max,
                d_min,
                gate_max,
                scale_log2=softmax_scale_log2,
                gate_threshold_log2=gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = activations.log_sigmoid(acc_s, FASTMATH=True)

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile)

            for n_block in tl.range(
                n_block_window_min_no_mask - 1, n_block_window_min - 1, -1
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    k_ptrs,
                    v_ptrs,
                    d_ptrs,
                    gate_max,
                    block_max,
                    row_max,
                    row_sum,
                ) = _fwd_inner_gated_kernel(
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    gate_threshold_log2=gate_threshold_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    n_block_min=n_block_window_min,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                    CHECK_INF=True,
                )

    # Load value scale
    v_scale = tl.load(value_scale)

    # Finalize softmax
    row_scale, lse_tile = activations.finalize(
        row_max=row_max,
        row_sum=row_sum,
        scale_log2=softmax_scale_log2,
        final_scale=v_scale,
        IS_LOG2=IS_SPLIT_KV,
        CHECK_NAN=True,
    )

    # Store LSE
    if PACK_GQA:
        tl.store(
            lse_ptrs,
            lse_tile,
            mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
            cache_modifier=".wb",
        )
    else:
        tl.store(lse_ptrs, lse_tile, boundary_check=(0,), cache_modifier=".wb")

    # Finalize rescale
    acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)

    # Store output
    # When IS_SPLIT_KV, store float32 partial results.
    # Otherwise, convert back to input dtype.
    if not IS_SPLIT_KV:
        acc_o = acc_o.to(Out.dtype.element_ty)
    if PACK_GQA:
        tl.store(
            out_ptrs,
            acc_o,
            mask=((offs_m // QHEAD_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
            & (offs_kb < head_dim)[None, :],
            cache_modifier=".wb",
        )
    else:
        tl.store(out_ptrs, acc_o, boundary_check=(0, 1), cache_modifier=".wb")


_fwd_gated_kernel = cache_utils.wrap_kernel(_fwd_gated_kernel)


_fwd_gated_kernel_autotuned = None


def _get_autotuned_kernel():
    global _fwd_gated_kernel_autotuned
    if _fwd_gated_kernel_autotuned is None:
        jit_kernel = _fwd_gated_kernel._kernel
        autotuned = autotuner.make_fwd_gated_autotuned_kernel(jit_kernel)
        _fwd_gated_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _fwd_gated_kernel_autotuned


def _flash_gated_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = False,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    device = query.device
    arch = cache_utils.get_device_arch(device)
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    gate_threshold = gate_threshold or head_dim / seqlen_k
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    else:
        window_sizes = torch.zeros((num_heads_kv, 2), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            alpha=alpha,
            delta=delta,
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

    if is_autotune:
        kernel = _get_autotuned_kernel()
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None
    else:
        kernel = _fwd_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
            launch_template.get_fwd_gated_launch_config(
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                tile_k=TILE_K,
                device=device,
                arch=arch,
            )
        )

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
        )
        if is_split_kv
        else 1
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = out if out is not None else torch.zeros_like(query, dtype=out_dtype)
    lse = (
        lse
        if lse is not None
        else torch.zeros(
            (batch_size, num_heads_q, seqlen_q),
            device=query.device,
            dtype=torch.float32,
        )
    )

    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, batch_size, seqlen_q, num_heads_q, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        lse_partial = torch.empty(
            (num_splits, batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        softmax_scale_log2,
        softmax_threshold,
        gate_threshold,
        query_scale,
        key_scale,
        value_scale,
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
        alpha.stride(0),
        alpha.stride(-1),
        alpha.stride(-2),
        delta.stride(0),
        delta.stride(-1),
        delta.stride(-2),
        out.stride(0) if not is_split_kv else out_partial.stride(1),
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(-3) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        lse.stride(0) if not is_split_kv else lse_partial.stride(1),
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
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
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_split_kv:
        flash_dec_combine._flash_attn_dec_combine(
            out_partial,
            lse_partial,
            out,
            lse,
        )

    return out, lse, softmax_scale, softmax_threshold, gate_threshold


def _flash_gated_attn_varlen_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    device = query.device
    arch = cache_utils.get_device_arch(device)
    num_SMs = cache_utils.get_device_num_sms(device)
    total_seqlen_q, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    gate_threshold = gate_threshold or head_dim / seqlen_k
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    else:
        window_sizes = torch.zeros((num_heads_kv, 2), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            alpha=alpha,
            delta=delta,
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

    if is_autotune:
        kernel = _get_autotuned_kernel()
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None
    else:
        kernel = _fwd_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
            launch_template.get_fwd_gated_launch_config(
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                tile_k=TILE_K,
                device=device,
                arch=arch,
            )
        )

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
        )
        if is_split_kv
        else 1
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = out if out is not None else torch.zeros_like(query, dtype=out_dtype)
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (num_heads_q, total_seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )
    )

    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, total_seqlen_q, num_heads_q, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        lse_partial = torch.empty(
            (num_splits, num_heads_q, total_seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        softmax_scale_log2,
        softmax_threshold,
        gate_threshold,
        query_scale,
        key_scale,
        value_scale,
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
        alpha.stride(-1),
        alpha.stride(0),
        0,
        delta.stride(-1),
        delta.stride(0),
        0,
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(0) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        0,
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
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
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_split_kv:
        flash_dec_combine._flash_attn_dec_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
        )

    return out, lse, softmax_scale, softmax_threshold, gate_threshold
