from typing import Optional, Tuple
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    activations,
    utils,
    seqlen_info,
    block_info,
    mask,
)


fwd_base_autotune_configs = utils.get_fwd_base_autotune_configs(True)


@triton.autotune(
    configs=fwd_base_autotune_configs,
    key=utils.FWD_BASE_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.jit
def _fwd_sparse_base_kernel(
    Q,
    K,
    V,
    A,
    D,
    Out,
    Lse,
    softmax_scale,
    gate_scale,
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
    stride_db,
    stride_dh,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    qhead_per_kvhead,
    seqlen_q,
    seqlen_k,
    head_dim,
    QHEADS_PER_KVHEAD_PACKGQA: tl.constexpr,
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
    PACK_GQA: tl.constexpr,
):
    m_block = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_kb = tl.arange(0, TILE_K)

    if PACK_GQA:
        head_kv_idx = head_idx
    else:
        head_kv_idx = head_idx // qhead_per_kvhead

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

    # Compute n_block range for this m_block
    n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=m_block,
        split_idx=0,
        num_splits=1,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=False,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )

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
            stride_lb,
            TILE_M=TILE_M,
            TILE_K=1,
            QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        )
        out_ptrs = seqlen_info.make_pack_gqa_ptrs(
            out_base,
            m_block,
            head_idx,
            stride_oh,
            stride_om,
            TILE_M=TILE_M,
            TILE_K=TILE_K,
            QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        )

    # Early exit if no n_blocks to process
    if n_block_min >= n_block_max:
        # Write LSE as -inf for proper handling
        lse_tile = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
        if PACK_GQA:
            tl.store(
                lse_ptrs,
                lse_tile,
                mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
            )
        else:
            tl.store(lse_ptrs, lse_tile, boundary_check=(0,))

        # We can't get dtype of query for output here, so we initialize output to zero
        # # Write output as zero for proper handling
        # if PACK_GQA:
        #     tl.store(
        #         out_ptrs,
        #         o_tile,
        #         mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
        #         & (offs_kb < head_dim)[None, :],
        #     )
        # else:
        #     tl.store(out_ptrs, o_tile, boundary_check=(0, 1))
        return

    # Create query and alpha pointers
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
            strides=(1,),
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
            QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        )
        a_ptrs = seqlen_info.make_pack_gqa_ptrs(
            a_base,
            m_block,
            head_idx,
            stride_ah,
            stride_ab,
            TILE_M=TILE_M,
            TILE_K=1,
            QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        )
    k_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(actual_seqlen_k, head_dim),
        strides=(stride_kn, 1),
        offsets=((n_block_max - 1) * TILE_N, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
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
        strides=(1,),
        offsets=((n_block_max - 1) * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,),
    )

    # Get attention gate threshold
    g_thr = seqlen_info.get_gate_threshold(
        gate_scale=gate_scale,
        m_block=m_block,
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        IS_CAUSAL=IS_CAUSAL,
        TILE_M=TILE_M,
        QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        SWAP_AB=False,
    )
    g_thr_min = tl.min(g_thr)

    # Load query tile
    if PACK_GQA:
        q_tile = tl.load(
            q_ptrs,
            mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
            & (offs_kb < head_dim)[None, :],
            other=0.0,
        )
    else:
        q_tile = tl.load(q_ptrs, boundary_check=(0, 1))

    # Scale query
    q_tile = (q_tile * softmax_scale).to(q_tile.dtype)

    # Initialize accumulators
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load alpha tile
    if PACK_GQA:
        a_tile = tl.load(
            a_ptrs,
            mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
            other=0.0,
        )
    else:
        a_tile = tl.load(a_ptrs, boundary_check=(0,))
    a_max = tl.max(a_tile)
    a_min = tl.min(a_tile)

    # Load delta tile
    d_tile = tl.load(d_ptrs, boundary_check=(0,))

    # Compute attention gates
    g = a_tile[:, None] * d_tile[None, :]
    gate_mask = g >= g_thr

    # Check if any gates are active for current tile
    active_curr = tl.reduce_or(gate_mask, axis=None)

    # Load key tile
    k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

    # Process n_blocks with masking
    if IS_CAUSAL or IS_LOCAL:
        n_block_min_causal_local = block_info.get_n_block_min_causal_local_mask(
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            m_block=m_block,
            n_block_min=n_block_min,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            IS_LOCAL=IS_LOCAL,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
        )

        for n_block in range(n_block_max - 1, n_block_min_causal_local - 1, -1):
            if active_curr:
                # Load value tile
                v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

                # Compute attention scores
                acc_s = activations.log_sigmoid(g, gate_mask)
                acc_s += tl.dot(q_tile, tl.trans(k_tile))

                # Advance key and delta pointers
                k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
                d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

                active_next = False
                if n_block > n_block_min:
                    # Load next delta tile
                    d_tile = tl.load(d_ptrs, boundary_check=(0,))

                    d_max = tl.max(d_tile)
                    d_min = tl.min(d_tile)
                    maybe_active = activations.gate_skip(
                        a_max, a_min, d_max, d_min, g_thr_min
                    )
                    if maybe_active:
                        # Compute next attention gates
                        g = a_tile[:, None] * d_tile[None, :]
                        gate_mask = g >= g_thr

                        # Check if any gates are active for next tile
                        active_next = tl.reduce_or(gate_mask, axis=None)

                        if active_next:
                            # Load next key tile
                            k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

                # Apply mask
                acc_s = mask.apply_mask(
                    acc_s=acc_s,
                    m_block=m_block,
                    n_block=n_block,
                    seqlen_q=actual_seqlen_q,
                    seqlen_k=actual_seqlen_k,
                    MASK_SEQLEN=True,
                    MASK_CAUSAL=IS_CAUSAL,
                    MASK_LOCAL=IS_LOCAL,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                    SWAP_AB=False,
                )

                # Apply online softmax
                p, row_max, row_sum, row_scale = activations.online_softmax(
                    acc_s=acc_s,
                    row_max=row_max,
                    row_sum=row_sum,
                    CHECK_INF=True,
                )

                # Rescale output accumulator
                acc_o = activations.rescale_o(acc_o, row_scale)

                # Update output accumulator
                acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

                # Advance value pointers
                v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

                # Update active status for next iteration
                active_curr = active_next
            else:
                # Advance key, value, and delta pointers
                k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
                v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))
                d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

                active_next = False
                if n_block > n_block_min:
                    # Load next delta tile
                    d_tile = tl.load(d_ptrs, boundary_check=(0,))

                    d_max = tl.max(d_tile)
                    d_min = tl.min(d_tile)
                    maybe_active = activations.gate_skip(
                        a_max, a_min, d_max, d_min, g_thr_min
                    )
                    if maybe_active:
                        # Compute next attention gates
                        g = a_tile[:, None] * d_tile[None, :]
                        gate_mask = g >= g_thr

                        # Check if any gates are active for next tile
                        active_next = tl.reduce_or(gate_mask, axis=None)

                        if active_next:
                            # Load next key tile
                            k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

                # Update active status for next iteration
                active_curr = active_next

        n_block_max_no_mask = n_block_min_causal_local
    else:
        # First iteration with seqlen masking
        n_block = n_block_max - 1

        if active_curr:
            # Load value tile
            v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

            # Compute attention scores
            acc_s = activations.log_sigmoid(g, gate_mask)
            acc_s += tl.dot(q_tile, tl.trans(k_tile))

            # Advance key and delta pointers
            k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
            d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

            active_next = False
            if n_block > n_block_min:
                # Load next delta tile
                d_tile = tl.load(d_ptrs, boundary_check=(0,))

                d_max = tl.max(d_tile)
                d_min = tl.min(d_tile)
                maybe_active = activations.gate_skip(
                    a_max, a_min, d_max, d_min, g_thr_min
                )
                if maybe_active:
                    # Compute next attention gates
                    g = a_tile[:, None] * d_tile[None, :]
                    gate_mask = g >= g_thr

                    # Check if any gates are active for next tile
                    active_next = tl.reduce_or(gate_mask, axis=None)

                    if active_next:
                        # Load next key tile
                        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

            # Apply seqlen mask only
            acc_s = mask.apply_mask(
                acc_s=acc_s,
                m_block=m_block,
                n_block=n_block,
                seqlen_q=actual_seqlen_q,
                seqlen_k=actual_seqlen_k,
                MASK_SEQLEN=True,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                SWAP_AB=False,
            )

            # Apply online softmax
            p, row_max, row_sum, row_scale = activations.online_softmax(
                acc_s=acc_s,
                row_max=row_max,
                row_sum=row_sum,
                CHECK_INF=True,
            )

            # Rescale output accumulator
            acc_o = activations.rescale_o(acc_o, row_scale)

            # Update output accumulator
            acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

            # Advance value pointers
            v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

            # Update active status for next iteration
            active_curr = active_next
        else:
            # Advance key, value, and delta pointers
            k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
            v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))
            d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

            active_next = False
            if n_block > n_block_min:
                # Load next delta tile
                d_tile = tl.load(d_ptrs, boundary_check=(0,))

                d_max = tl.max(d_tile)
                d_min = tl.min(d_tile)
                maybe_active = activations.gate_skip(
                    a_max, a_min, d_max, d_min, g_thr_min
                )
                if maybe_active:
                    # Compute next attention gates
                    g = a_tile[:, None] * d_tile[None, :]
                    gate_mask = g >= g_thr

                    # Check if any gates are active for next tile
                    active_next = tl.reduce_or(gate_mask, axis=None)

                    if active_next:
                        # Load next key tile
                        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

            # Update active status for next iteration
            active_curr = active_next

        n_block_max_no_mask = n_block_max - 1

    # Recreate block pointers for the non-masking loop to provide
    # a fresh tl.make_block_ptr origin, breaking the SSA def-use chain
    # from the masking loop's data-dependent phi nodes that the
    # TritonRewriteTensorPointer pass cannot trace through.
    k_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(actual_seqlen_k, head_dim),
        strides=(stride_kn, 1),
        offsets=((n_block_max_no_mask - 1) * TILE_N, 0),
        block_shape=(TILE_N, TILE_K),
        order=(1, 0),
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
        strides=(1,),
        offsets=((n_block_max_no_mask - 1) * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,),
    )

    # Process n_blocks without masking
    for n_block in range(n_block_max_no_mask - 1, n_block_min - 1, -1):
        if active_curr:
            # Load value tile
            v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

            # Compute attention scores
            acc_s = activations.log_sigmoid(g, gate_mask)
            acc_s += tl.dot(q_tile, tl.trans(k_tile))

            # Advance key and delta pointers
            k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
            d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

            active_next = False
            if n_block > n_block_min:
                # Load next delta tile
                d_tile = tl.load(d_ptrs, boundary_check=(0,))

                d_max = tl.max(d_tile)
                d_min = tl.min(d_tile)
                maybe_active = activations.gate_skip(
                    a_max, a_min, d_max, d_min, g_thr_min
                )
                if maybe_active:
                    # Compute next attention gates
                    g = a_tile[:, None] * d_tile[None, :]
                    gate_mask = g >= g_thr

                    # Check if any gates are active for next tile
                    active_next = tl.reduce_or(gate_mask, axis=None)

                    if active_next:
                        # Load next key tile
                        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

            # Apply online softmax
            p, row_max, row_sum, row_scale = activations.online_softmax(
                acc_s=acc_s,
                row_max=row_max,
                row_sum=row_sum,
                CHECK_INF=False,
            )

            # Rescale output accumulator
            acc_o = activations.rescale_o(acc_o, row_scale)

            # Update output accumulator
            acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

            # Advance value pointers
            v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

            # Update active status for next iteration
            active_curr = active_next
        else:
            # Advance key, value, and delta pointers
            k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
            v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))
            d_ptrs = tl.advance(d_ptrs, (-TILE_N,))

            active_next = False
            if n_block > n_block_min:
                # Load next delta tile
                d_tile = tl.load(d_ptrs, boundary_check=(0,))

                d_max = tl.max(d_tile)
                d_min = tl.min(d_tile)
                maybe_active = activations.gate_skip(
                    a_max, a_min, d_max, d_min, g_thr_min
                )
                if maybe_active:
                    # Compute next attention gates
                    g = a_tile[:, None] * d_tile[None, :]
                    gate_mask = g >= g_thr

                    # Check if any gates are active for next tile
                    active_next = tl.reduce_or(gate_mask, axis=None)

                    if active_next:
                        # Load next key tile
                        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

            # Update active status for next iteration
            active_curr = active_next

    # Finalize softmax
    o_scale, lse_tile = activations.finalize(
        row_max=row_max,
        row_sum=row_sum,
        final_scale=1.0,
    )
    acc_o = activations.rescale_o(acc_o, o_scale)

    # Store LSE
    if PACK_GQA:
        tl.store(
            lse_ptrs,
            lse_tile,
            mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q),
        )
    else:
        tl.store(lse_ptrs, lse_tile, boundary_check=(0,))
    # Store output
    if PACK_GQA:
        tl.store(
            out_ptrs,
            acc_o.to(q_tile.dtype),
            mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
            & (offs_kb < head_dim)[None, :],
        )
    else:
        tl.store(out_ptrs, acc_o.to(q_tile.dtype), boundary_check=(0, 1))


def _flash_sparse_attn_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float,
    gate_scale: float,
    is_causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    pack_gqa: bool = False,
):
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape

    is_local = window_size[0] is not None or window_size[1] is not None
    if is_local:
        window_size_left, window_size_right = window_size
    else:
        window_size_left, window_size_right = None, None

    utils.assert_fwd_sparse_base_inputs(
        query,
        key,
        value,
        alpha,
        delta,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        gate_scale=gate_scale,
    )

    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    gate_scale = gate_scale or 1.0 / (seqlen_k + 1)

    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads_q, seqlen_q),
        device=query.device,
        dtype=torch.float32,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    grid = utils.get_fwd_base_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
    )

    _fwd_sparse_base_kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out,
        lse,
        softmax_scale,
        gate_scale,
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
        alpha.stride(1),
        delta.stride(0),
        delta.stride(1),
        out.stride(0),
        out.stride(-2),
        out.stride(-3),
        lse.stride(0),
        lse.stride(1),
        None,
        None,
        None,
        None,
        num_heads_q // num_heads_kv,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=(num_heads_q // num_heads_kv) if pack_gqa else 1,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
    )

    return out, lse, softmax_scale, gate_scale


def _flash_sparse_attn_varlen_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    gate_scale: float,
    is_causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    pack_gqa: bool = False,
):
    total_seqlen_q, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k

    is_local = window_size[0] is not None or window_size[1] is not None
    if is_local:
        window_size_left, window_size_right = window_size
    else:
        window_size_left, window_size_right = None, None

    utils.assert_fwd_sparse_base_inputs(
        query,
        key,
        value,
        alpha,
        delta,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        gate_scale=gate_scale,
    )

    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    gate_scale = gate_scale or 1.0 / (seqlen_k + 1)

    out = torch.empty_like(query)
    lse = torch.empty(
        (total_seqlen_q, num_heads_q), device=query.device, dtype=torch.float32
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    grid = utils.get_fwd_base_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
    )

    _fwd_sparse_base_kernel[grid](
        query,
        key,
        value,
        alpha,
        delta,
        out,
        lse,
        softmax_scale,
        gate_scale,
        0,
        query.stride(-2),
        query.stride(0),
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        alpha.stride(0),
        alpha.stride(1),
        delta.stride(0),
        delta.stride(1),
        0,
        out.stride(-2),
        out.stride(0),
        lse.stride(0),
        lse.stride(1),
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        num_heads_q // num_heads_kv,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=(num_heads_q // num_heads_kv) if pack_gqa else 1,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
    )

    return out, lse, softmax_scale, gate_scale
