import triton
import triton.language as tl


@triton.jit
def get_seqlen_info(
    batch_idx,
    seqlen_static,
    cu_seqlens,
    seqused,
    HAS_CU_SEQLENS: tl.constexpr,
    HAS_SEQUSED: tl.constexpr,
):
    """
    Get offset and seqlen for a given batch index.

    :param batch_idx: Index of the batch.
    :param seqlen_static: Static sequence length if cu_seqlens is not provided.
    :param cu_seqlens: Cumulative sequence lengths tensor.
    :param seqused: Actual sequence lengths tensor.
    :param HAS_CU_SEQLENS: Boolean flag indicating if cu_seqlens is provided.
    :param HAS_SEQUSED: Boolean flag indicating if seqused is provided.

    :return offset: Offset for the given batch index.
    :return seqlen: Sequence length for the given batch index.
    """
    if HAS_CU_SEQLENS:
        offset = tl.load(cu_seqlens + batch_idx)
        if HAS_SEQUSED:
            seqlen = tl.load(seqused + batch_idx)
        else:
            seqlen = tl.load(cu_seqlens + batch_idx + 1) - offset
    else:
        offset = 0
        if HAS_SEQUSED:
            seqlen = tl.load(seqused + batch_idx)
        else:
            seqlen = seqlen_static
    return offset, seqlen


@triton.jit
def get_seqlen_info_qk(
    batch_idx,
    seqlen_q_static,
    seqlen_k_static,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    """
    Get offset, padded_offset, and seqlen for both Q and K.

    :param batch_idx: Index of the batch.
    :param seqlen_q_static: Static sequence length for Q if cu_seqlens_q is not provided.
    :param seqlen_k_static: Static sequence length for K if cu_seqlens_k is not provided.
    :param cu_seqlens_q: Cumulative sequence lengths tensor for Q.
    :param cu_seqlens_k: Cumulative sequence lengths tensor for K.
    :param seqused_q: Actual sequence lengths tensor for Q.
    :param seqused_k: Actual sequence lengths tensor for K.
    :param TILE_M: Tile size for Q.
    :param TILE_N: Tile size for K.
    :param HAS_CU_SEQLENS_Q: Boolean flag indicating if cu_seqlens_q is provided.
    :param HAS_CU_SEQLENS_K: Boolean flag indicating if cu_seqlens_k is provided.
    :param HAS_SEQUSED_Q: Boolean flag indicating if seqused_q is provided.
    :param HAS_SEQUSED_K: Boolean flag indicating if seqused_k is provided.

    :return offset_q: Offset for Q for the given batch index.
    :return offset_k: Offset for K for the given batch index.
    :return padded_offset_q: Padded offset for Q aligned to TILE_M.
    :return padded_offset_k: Padded offset for K aligned to TILE_N.
    :return seqlen_q: Sequence length for Q for the given batch index.
    :return seqlen_k: Sequence length for K for the given batch index.
    """

    # Q offset and seqlen
    if HAS_CU_SEQLENS_Q:
        offset_q = tl.load(cu_seqlens_q + batch_idx)
        padded_offset_q = (offset_q + batch_idx * TILE_M) // TILE_M * TILE_M
        if HAS_SEQUSED_Q:
            seqlen_q = tl.load(seqused_q + batch_idx)
        else:
            seqlen_q = tl.load(cu_seqlens_q + batch_idx + 1) - offset_q
    else:
        offset_q = 0
        padded_offset_q = 0
        if HAS_SEQUSED_Q:
            seqlen_q = tl.load(seqused_q + batch_idx)
        else:
            seqlen_q = seqlen_q_static

    # K offset and seqlen
    if HAS_CU_SEQLENS_K:
        offset_k = tl.load(cu_seqlens_k + batch_idx)
        padded_offset_k = (offset_k + batch_idx * TILE_N) // TILE_N * TILE_N
        if HAS_SEQUSED_K:
            seqlen_k = tl.load(seqused_k + batch_idx)
        else:
            seqlen_k = tl.load(cu_seqlens_k + batch_idx + 1) - offset_k
    else:
        offset_k = 0
        padded_offset_k = 0
        if HAS_SEQUSED_K:
            seqlen_k = tl.load(seqused_k + batch_idx)
        else:
            seqlen_k = seqlen_k_static

    return offset_q, offset_k, padded_offset_q, padded_offset_k, seqlen_q, seqlen_k


@triton.jit
def offset_batch_Q(
    base_ptr,
    batch_idx,
    offset,
    padded_offset,
    stride_batch,
    stride_seq,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_PADDED: tl.constexpr,
):
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    else:
        return base_ptr + batch_idx * stride_batch


@triton.jit
def offset_batch_K(
    base_ptr,
    batch_idx,
    offset,
    padded_offset,
    stride_batch,
    stride_seq,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_PADDED: tl.constexpr,
):
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    else:
        return base_ptr + batch_idx * stride_batch
