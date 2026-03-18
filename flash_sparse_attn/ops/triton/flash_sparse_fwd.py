from typing import Tuple

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    utils,
    launch_template,
    launch_grid,
    seqlen_info,
    block_info,
    activations,
    mask,
    flash_fwd_combine,
)


@triton.jit
def _fwd_inner_sparse_base_kernel(
    q_tile,
    k_tile,
    k_ptrs,
    v_ptrs,
    acc_o,
    block_max,
    row_max,
    row_sum,
    softmax_scale_log2,
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
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    CHECK_INF: tl.constexpr,
):
    # Compute attention scores
    acc_s = tl.dot(q_tile, k_tile)

    # Advance key pointer
    k_ptrs = tl.advance(k_ptrs, (0, -TILE_N))
    if n_block > n_block_min:
        # Load next key tile
        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

    if IS_MASK:
        # Apply mask to attention scores
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
            QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
            SWAP_AB=False,
        )

    # Apply online softmax
    p, block_max, row_max, row_sum, row_scale, skip_softmax = (
        activations.online_softmax(
            acc_s=acc_s,
            block_max=block_max,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=softmax_scale_log2,
            softmax_threshold_log2=softmax_threshold_log2,
            CHECK_INF=CHECK_INF,
            RESCALE_THRESHOLD=0.0,
        )
    )

    if not skip_softmax:
        # Load value tile
        v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

        # Rescale output accumulator
        acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)

        # Update output accumulator
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

    # Advance value pointer
    v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

    return k_tile, k_ptrs, v_ptrs, acc_o, block_max, row_max, row_sum


@triton.jit
def _fwd_base_sparse_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_os,
    stride_lb,
    stride_lh,
    stride_ls,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    qhead_per_kvhead,
    num_splits,
    seqlen_q,
    seqlen_k,
    head_dim,
    QHEADS_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
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
        head_kv_idx = head_idx // qhead_per_kvhead

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

    # Compute n_block range for this m_block
    n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=m_block,
        split_idx=split_idx,
        num_splits=num_splits,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=IS_SPLIT_KV,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
        QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )
    n_block_min_no_mask = block_info.get_n_block_min_before_local_mask(
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        m_block=m_block,
        n_block_min=n_block_min,
        TILE_N=TILE_N,
        TILE_M=TILE_M,
        IS_LOCAL=IS_LOCAL,
        WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
        QHEAD_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )
    n_block_max_no_mask = block_info.get_n_block_min_causal_local_mask(
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

    # Clamp to split's range so the no-mask loop stays within bounds
    if IS_SPLIT_KV:
        n_block_min_no_mask = tl.maximum(n_block_min_no_mask, n_block_min)
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

        # Write output as zero for proper handling
        o_tile = tl.zeros((TILE_M, TILE_K), dtype=Out.dtype.element_ty)
        if PACK_GQA:
            tl.store(
                out_ptrs,
                o_tile,
                mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
                & (offs_kb < head_dim)[None, :],
            )
        else:
            tl.store(out_ptrs, o_tile, boundary_check=(0, 1))
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

    # Get softmax threshold
    softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
        softmax_threshold=softmax_threshold,
        m_block=m_block,
        seqlen_q=actual_seqlen_q,
        seqlen_k=actual_seqlen_k,
        IS_CAUSAL=IS_CAUSAL,
        TILE_M=TILE_M,
        QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
    )

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

    # Initialize accumulators
    block_max = tl.full((), float("-inf"), dtype=tl.float32)
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load key tile
    k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

    # Process n_blocks with masking
    if IS_CAUSAL or IS_LOCAL:
        for n_block in tl.range(n_block_max - 1, n_block_max_no_mask - 1, -1):
            k_tile, k_ptrs, v_ptrs, acc_o, block_max, row_max, row_sum = (
                _fwd_inner_sparse_base_kernel(
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    n_block_min=n_block_max_no_mask,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                    IS_MASK=True,
                    MASK_CAUSAL=IS_CAUSAL,
                    MASK_LOCAL=IS_LOCAL,
                    CHECK_INF=True,
                )
            )
    else:
        # First iteration with seqlen masking
        n_block = n_block_max - 1

        k_tile, k_ptrs, v_ptrs, acc_o, block_max, row_max, row_sum = (
            _fwd_inner_sparse_base_kernel(
                q_tile=q_tile,
                k_tile=k_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                acc_o=acc_o,
                block_max=block_max,
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
                softmax_threshold_log2=softmax_threshold_log2,
                m_block=m_block,
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
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                CHECK_INF=True,
            )
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
        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))
        for n_block in tl.range(n_block_max_no_mask - 1, n_block_min_no_mask - 1, -1):
            k_tile, k_ptrs, v_ptrs, acc_o, block_max, row_max, row_sum = (
                _fwd_inner_sparse_base_kernel(
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
                    n_block=n_block,
                    n_block_min=n_block_min_no_mask,
                    actual_seqlen_q=actual_seqlen_q,
                    actual_seqlen_k=actual_seqlen_k,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
                    WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
                    QHEADS_PER_KVHEAD_PACKGQA=QHEADS_PER_KVHEAD_PACKGQA,
                    IS_MASK=IS_LOCAL,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    CHECK_INF=IS_LOCAL,
                )
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
        k_tile = tl.load(k_ptrs, boundary_check=(0, 1))
        for n_block in tl.range(n_block_min_no_mask - 1, n_block_min - 1, -1):
            k_tile, k_ptrs, v_ptrs, acc_o, block_max, row_max, row_sum = (
                _fwd_inner_sparse_base_kernel(
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    block_max=block_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_threshold_log2=softmax_threshold_log2,
                    m_block=m_block,
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
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    CHECK_INF=True,
                )
            )

    # Finalize softmax
    row_scale, lse_tile = activations.finalize(
        row_max=row_max,
        row_sum=row_sum,
        scale_log2=softmax_scale_log2,
        final_scale=1.0,
    )
    acc_o = activations.rescale_o(acc_o, row_scale, LAZY_RESCALE=False)

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
    # When IS_SPLIT_KV, store float32 partial results.
    # Otherwise, convert back to input dtype.
    if not IS_SPLIT_KV:
        acc_o = acc_o.to(q_tile.dtype)
    if PACK_GQA:
        tl.store(
            out_ptrs,
            acc_o,
            mask=((offs_m // QHEADS_PER_KVHEAD_PACKGQA) < actual_seqlen_q)[:, None]
            & (offs_kb < head_dim)[None, :],
        )
    else:
        tl.store(out_ptrs, acc_o, boundary_check=(0, 1))


def _flash_sparse_attn_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    window_size: Tuple[int, int] = (None, None),
    pack_gqa: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    num_SMs = torch.cuda.get_device_properties(query.device).multi_processor_count
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    is_split_kv = seqlen_q == 1 and seqlen_q != seqlen_k
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    qheads_per_kvhead = num_heads_q // num_heads_kv
    qheads_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1

    assert_inputs.assert_fwd_inputs(
        query,
        key,
        value,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_fwd_sparse_launch_config(
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            qheads_per_kvhead=qheads_per_kvhead,
            tile_k=TILE_K,
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

    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads_q, seqlen_q),
        dtype=torch.float32,
        device=query.device,
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

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    _fwd_base_sparse_kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
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
        out.stride(0) if not is_split_kv else out_partial.stride(1),
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(-3) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        lse.stride(0) if not is_split_kv else lse_partial.stride(1),
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
        None,
        None,
        None,
        None,
        qheads_per_kvhead,
        num_splits,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=qheads_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_split_kv:
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
        )

    return out, lse, softmax_scale, softmax_threshold


def _flash_sparse_attn_varlen_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    window_size: Tuple[int, int] = (None, None),
    pack_gqa: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    num_SMs = torch.cuda.get_device_properties(query.device).multi_processor_count
    total_seqlen_q, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    is_split_kv = seqlen_q == 1 and seqlen_q != seqlen_k
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    softmax_threshold = softmax_threshold or head_dim / seqlen_k
    qheads_per_kvhead = num_heads_q // num_heads_kv
    qheads_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1

    assert_inputs.assert_fwd_inputs(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_fwd_sparse_launch_config(
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            qheads_per_kvhead=qheads_per_kvhead,
            tile_k=TILE_K,
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

    out = torch.empty_like(query)
    lse = torch.empty(
        (num_heads_q, total_seqlen_q),
        dtype=torch.float32,
        device=query.device,
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

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    _fwd_base_sparse_kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
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
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(0) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        0,
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        qheads_per_kvhead,
        num_splits,
        seqlen_q,
        seqlen_k,
        head_dim,
        QHEADS_PER_KVHEAD_PACKGQA=qheads_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_split_kv:
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
        )

    return out, lse, softmax_scale, softmax_threshold
