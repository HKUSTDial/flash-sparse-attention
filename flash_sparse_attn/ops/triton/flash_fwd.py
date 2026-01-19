from typing import Optional, Tuple
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import seqlen_info, block_info, mask, softmax


@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 128, "TILE_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"TILE_M": 128, "TILE_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"TILE_M": 64, "TILE_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"TILE_M": 128, "TILE_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"TILE_M": 128, "TILE_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"TILE_M": 64, "TILE_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["IS_CAUSAL", "IS_LOCAL", "TILE_K"],
)
@triton.jit
def _fwd_base_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    softmax_scale,
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
        return

    # Create query pointers
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
            # Load value tile
            v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

            # Compute attention scores
            acc_s = tl.dot(q_tile, tl.trans(k_tile))

            # Advance key pointers
            k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
            if n_block > n_block_min:
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
            p, row_max, row_sum, row_scale = softmax.online_softmax(
                acc_s=acc_s,
                row_max=row_max,
                row_sum=row_sum,
                CHECK_INF=True,
            )

            # Rescale output accumulator
            acc_o = softmax.rescale_o(acc_o, row_scale)

            # Update output accumulator
            acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

            # Advance value pointers
            v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

        n_block_max_no_mask = n_block_min_causal_local
    else:
        # First iteration with seqlen masking
        n_block = n_block_max - 1

        # Load value tile
        v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

        # Compute attention scores
        acc_s = tl.dot(q_tile, tl.trans(k_tile))

        # Advance key pointers
        k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
        if n_block > n_block_min:
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
        p, row_max, row_sum, row_scale = softmax.online_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            CHECK_INF=True,
        )

        # Rescale output accumulator
        acc_o = softmax.rescale_o(acc_o, row_scale)

        # Update output accumulator
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

        # Advance value pointers
        v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

        n_block_max_no_mask = n_block_max - 1

    # Process n_blocks without masking
    for n_block in range(n_block_max_no_mask - 1, n_block_min - 1, -1):
        # Load value tile
        v_tile = tl.load(v_ptrs, boundary_check=(0, 1))

        # Compute attention scores
        acc_s = tl.dot(q_tile, tl.trans(k_tile))

        # Advance key pointers
        k_ptrs = tl.advance(k_ptrs, (-TILE_N, 0))
        if n_block > n_block_min:
            # Load next key tile
            k_tile = tl.load(k_ptrs, boundary_check=(0, 1))

        # Apply online softmax
        p, row_max, row_sum, row_scale = softmax.online_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            CHECK_INF=False,
        )

        # Rescale output accumulator
        acc_o = softmax.rescale_o(acc_o, row_scale)

        # Update output accumulator
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

        # Advance value pointers
        v_ptrs = tl.advance(v_ptrs, (-TILE_N, 0))

    # Finalize softmax
    o_scale, lse_tile = softmax.finalize(
        row_max=row_max,
        row_sum=row_sum,
        final_scale=1.0,
    )
    acc_o = softmax.rescale_o(acc_o, o_scale)

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


def _flash_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    is_causal: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    pack_gqa: bool = False,
):
    is_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None
    if not is_varlen:
        batch_size, seqlen_q, num_heads_q, head_dim = query.shape
        _, seqlen_k, num_heads_kv, _ = key.shape
    else:
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

    assert query.is_cuda and key.is_cuda and value.is_cuda, (
        "All inputs must be on CUDA device"
    )
    assert query.dtype in [torch.float16, torch.bfloat16], (
        "Input dtype must be float16 or bfloat16"
    )
    assert query.dtype == key.dtype == value.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    if is_varlen:
        assert (
            cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        ), "cu_seqlens_q and cu_seqlens_k must be of int32"

    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)

    out = torch.zeros_like(query)
    if not is_varlen:
        lse = torch.empty(
            (batch_size, num_heads_q, seqlen_q),
            device=query.device,
            dtype=torch.float32,
        )
    else:
        lse = torch.empty(
            (total_seqlen_q, num_heads_q), device=query.device, dtype=torch.float32
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    def grid(META):
        return (
            triton.cdiv(
                seqlen_q * (num_heads_q // num_heads_kv) if pack_gqa else seqlen_q,
                META["TILE_M"],
            ),
            num_heads_kv if pack_gqa else num_heads_q,
            batch_size,
        )

    _fwd_base_kernel[grid](
        query,
        key,
        value,
        out,
        lse,
        softmax_scale,
        query.stride(0) if not is_varlen else 0,
        query.stride(-2),
        query.stride(-3) if not is_varlen else query.stride(0),
        key.stride(0) if not is_varlen else 0,
        key.stride(-2),
        key.stride(-3) if not is_varlen else key.stride(0),
        value.stride(0) if not is_varlen else 0,
        value.stride(-2),
        value.stride(-3) if not is_varlen else value.stride(0),
        out.stride(0) if not is_varlen else 0,
        out.stride(-2),
        out.stride(-3) if not is_varlen else out.stride(0),
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
        HAS_CU_SEQLENS_Q=is_varlen,
        HAS_CU_SEQLENS_K=is_varlen,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        PACK_GQA=pack_gqa,
    )
    return out, lse, softmax_scale
