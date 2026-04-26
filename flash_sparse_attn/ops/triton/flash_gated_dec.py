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
)


@triton.jit
def _dec_inner_gated_base_kernel(
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
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    QHEADS_PER_KVHEAD_PACKGQA: tl.constexpr,
    IS_MASK: tl.constexpr,
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
                MASK_SEQLEN=True,
                MASK_CAUSAL=False,
                MASK_LOCAL=MASK_LOCAL,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
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


@triton.jit
def _dec_gated_base_kernel(
    Q,
    K,
    V,
    A,
    D,
    Out,
    Lse,
    softmax_scale_log2,
    softmax_threshold_log2,
    gate_threshold_log2,
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
    stride_lm,
    stride_ls,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_splits,
    seqlen_q,
    seqlen_k,
    head_dim,
    QHEADS_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
    IS_ADAPT_GATE: tl.constexpr,
):
    head_idx = tl.program_id(0)
    batch_split_idx = tl.program_id(1)
    batch_idx = batch_split_idx // num_splits
    split_idx = batch_split_idx - batch_idx * num_splits
    head_kv_idx = head_idx

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
        Q + head_idx * QHEADS_PER_KVHEAD_PACKGQA * stride_qh,
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
        A + head_idx * QHEADS_PER_KVHEAD_PACKGQA * stride_ah,
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
    out_base = seqlen_info.offset_batch_Q(
        Out + head_idx * QHEADS_PER_KVHEAD_PACKGQA * stride_oh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_ob,
        stride_om,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )
    lse_base = seqlen_info.offset_batch_Q(
        Lse + head_idx * QHEADS_PER_KVHEAD_PACKGQA * stride_lh,
        batch_idx,
        offset_q,
        padded_offset_q,
        stride_lb,
        stride_lm,
        HAS_CU_SEQLENS_Q,
        USE_PADDED=False,
    )

    # For split KV, offset output and LSE base pointers by split_idx
    out_base += split_idx * stride_os
    lse_base += split_idx * stride_ls

    # Compute n_block range for this m_block
    n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=0,
        split_idx=split_idx,
        num_splits=num_splits,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=False,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=True,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )
    n_block_min_no_mask = block_info.get_n_block_min_before_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=0,
        n_block_min=n_block_min,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_LOCAL=IS_LOCAL,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )

    # Clamp to split's range so the no-mask loop stays within bounds
    n_block_min_no_mask = tl.maximum(n_block_min_no_mask, n_block_min)

    # Create pointers
    lse_ptrs = tl.make_block_ptr(
        base=lse_base,
        shape=(actual_seqlen_q,),
        strides=(stride_lh,),
        offsets=(0,),
        block_shape=(TILE_M,),
        order=(0,),
    )
    out_ptrs = tl.make_block_ptr(
        base=out_base,
        shape=(actual_seqlen_q, head_dim),
        strides=(stride_oh, 1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    # Early exit if no n_blocks to process
    if n_block_min >= n_block_max:
        # Write LSE as -inf for proper handling
        lse_tile = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
        tl.store(lse_ptrs, lse_tile, boundary_check=(0,), cache_modifier=".wb")

        # Write output as zero for proper handling
        o_tile = tl.zeros((TILE_M, TILE_K), dtype=Out.dtype.element_ty)
        tl.store(out_ptrs, o_tile, boundary_check=(0, 1), cache_modifier=".wb")
        return

    q_ptrs = tl.make_block_ptr(
        base=q_base,
        shape=(actual_seqlen_q, head_dim),
        strides=(stride_qh, 1),
        offsets=(0, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )
    a_ptrs = tl.make_block_ptr(
        base=a_base,
        shape=(actual_seqlen_q,),
        strides=(stride_ah,),
        offsets=(0,),
        block_shape=(TILE_M,),
        order=(0,),
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

    # Load query tile
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
    a_tile = tl.load(a_ptrs, boundary_check=(0,), cache_modifier=".ca").to(tl.float32)
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

    # Process n_blocks with masking
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
    ) = _dec_inner_gated_base_kernel(
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
        m_block=0,
        n_block=n_block,
        n_block_min=n_block,
        actual_seqlen_q=actual_seqlen_q,
        actual_seqlen_k=actual_seqlen_k,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        IS_MASK=True,
        MASK_LOCAL=False,
        IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
        CHECK_INF=True,
    )

    n_block_max_no_mask = n_block_max - 1
    n_block_min_no_mask = tl.minimum(n_block_min_no_mask, n_block_max_no_mask)

    # Process n_blocks without masking
    if n_block_max_no_mask > n_block_min_no_mask:
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

        for n_block in tl.range(n_block_max_no_mask - 1, n_block_min_no_mask - 1, -1):
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
            ) = _dec_inner_gated_base_kernel(
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
                m_block=0,
                n_block=n_block,
                n_block_min=n_block_min_no_mask,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                IS_MASK=False,
                MASK_LOCAL=False,
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                CHECK_INF=False,
            )

    # Process n_blocks with masking
    if IS_LOCAL and n_block_min_no_mask > n_block_min:
        k_ptrs = tl.make_block_ptr(
            base=k_base,
            shape=(head_dim, actual_seqlen_k),
            strides=(1, stride_kn),
            offsets=(0, (n_block_min_no_mask - 1) * TILE_N),
            block_shape=(TILE_K, TILE_N),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_base,
            shape=(actual_seqlen_k, head_dim),
            strides=(stride_vn, 1),
            offsets=((n_block_min_no_mask - 1) * TILE_N, 0),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0),
        )
        d_ptrs = tl.make_block_ptr(
            base=d_base,
            shape=(actual_seqlen_k,),
            strides=(stride_dn,),
            offsets=((n_block_min_no_mask - 1) * TILE_N,),
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

        for n_block in tl.range(n_block_min_no_mask - 1, n_block_min - 1, -1):
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
            ) = _dec_inner_gated_base_kernel(
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
                m_block=0,
                n_block=n_block,
                n_block_min=n_block_min,
                actual_seqlen_q=actual_seqlen_q,
                actual_seqlen_k=actual_seqlen_k,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                IS_MASK=True,
                MASK_LOCAL=True,
                IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
                CHECK_INF=True,
            )

    # Finalize softmax
    row_scale, lse_tile = activations.finalize(
        row_max=row_max,
        row_sum=row_sum,
        scale_log2=softmax_scale_log2,
        final_scale=1.0,
        IS_LOG2=True,
    )

    # Store LSE
    tl.store(lse_ptrs, lse_tile, boundary_check=(0,), cache_modifier=".wb")

    # Final rescale
    acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)

    # Store output
    tl.store(out_ptrs, acc_o, boundary_check=(0, 1), cache_modifier=".wb")


_dec_gated_base_kernel = cache_utils.wrap_kernel(_dec_gated_base_kernel)


def _flash_gated_attn_base_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[int, int] = (None, None),
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    arch = cache_utils.get_device_arch(device)
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    gate_threshold = gate_threshold or head_dim / seqlen_k
    softmax_threshold_log2 = math.log2(softmax_threshold)
    gate_threshold_log2 = math.log2(gate_threshold)
    qheads_per_kvhead = num_heads_q // num_heads_kv

    assert_inputs.assert_dec_inputs(
        query,
        key,
        value,
        alpha=alpha,
        delta=delta,
        cu_seqlens_k=None,
        seqused_k=None,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        device=device,
        arch=arch,
    )
    assert_inputs.assert_dec_outputs(
        out=out,
        lse=lse,
        dtype=query.dtype,
        device=device,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_dec_gated_launch_config(
            qheads_per_kvhead=qheads_per_kvhead,
            tile_k=TILE_K,
            device=device,
            arch=arch,
        )
    )

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qheads_per_kvhead,
        seqlen_k=seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
    )

    out = out if out is not None else torch.empty_like(query)
    lse = (
        lse
        if lse is not None
        else torch.empty((batch_size, num_heads_q), dtype=torch.float32, device=device)
    )

    out_partial = torch.empty(
        (num_splits, batch_size, num_heads_q, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    lse_partial = torch.empty(
        (num_splits, batch_size, num_heads_q),
        dtype=torch.float32,
        device=query.device,
    )

    grid = launch_grid.get_dec_grid(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        num_splits=num_splits,
    )

    _dec_gated_base_kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out_partial,
        lse_partial,
        softmax_scale_log2,
        softmax_threshold_log2,
        gate_threshold_log2,
        query.stride(0),
        query.stride(-2),
        1,
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        alpha.stride(0),
        alpha.stride(-1),
        1,
        delta.stride(0),
        delta.stride(-2),
        delta.stride(-1),
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        None,
        None,
        None,
        None,
        num_splits,
        seqlen_q=qheads_per_kvhead,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=qheads_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    flash_dec_combine._flash_attn_dec_combine(
        out_partial,
        lse_partial,
        out,
        lse,
    )

    return out, lse


def _flash_gated_attn_varlen_base_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[int, int] = (None, None),
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    arch = cache_utils.get_device_arch(device)
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    seqlen_k = max_seqlen_k
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    gate_threshold = gate_threshold or head_dim / seqlen_k
    softmax_threshold_log2 = math.log2(softmax_threshold)
    gate_threshold_log2 = math.log2(gate_threshold)
    qheads_per_kvhead = num_heads_q // num_heads_kv

    assert_inputs.assert_dec_inputs(
        query,
        key,
        value,
        alpha=alpha,
        delta=delta,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        device=device,
        arch=arch,
    )
    assert_inputs.assert_dec_outputs(
        out=out,
        lse=lse,
        dtype=query.dtype,
        device=device,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_dec_gated_launch_config(
            qheads_per_kvhead=qheads_per_kvhead,
            tile_k=TILE_K,
            device=device,
            arch=arch,
        )
    )

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qheads_per_kvhead,
        seqlen_k=seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
    )

    out = out if out is not None else torch.empty_like(query)
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (batch_size, num_heads_q),
            dtype=torch.float32,
            device=device,
        )
    )

    out_partial = torch.empty(
        (num_splits, batch_size, num_heads_q, head_dim),
        dtype=torch.float32,
        device=device,
    )
    lse_partial = torch.empty(
        (num_splits, batch_size, num_heads_q),
        dtype=torch.float32,
        device=device,
    )

    grid = launch_grid.get_dec_grid(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        num_splits=num_splits,
    )

    _dec_gated_base_kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out_partial,
        lse_partial,
        softmax_scale_log2,
        softmax_threshold_log2,
        gate_threshold_log2,
        query.stride(0),
        query.stride(-2),
        1,
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        0,
        alpha.stride(0),
        1,
        0,
        delta.stride(-2),
        1,
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        None,
        cu_seqlens_k,
        None,
        seqused_k,
        num_splits,
        seqlen_q=qheads_per_kvhead,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=qheads_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    flash_dec_combine._flash_attn_dec_combine(
        out_partial,
        lse_partial,
        out,
        lse,
    )

    return out, lse
