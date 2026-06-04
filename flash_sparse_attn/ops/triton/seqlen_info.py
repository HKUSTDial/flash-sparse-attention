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
def get_softmax_threshold(
    softmax_threshold,
    m_block,
    seqlen_q,
    seqlen_k,
    IS_CAUSAL: tl.constexpr,
    TILE_M: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    """
    Compute the softmax threshold for a given block.

    :param softmax_threshold: Threshold value normalized by full key length.
    :param m_block: Current block index along the M dimension.
    :param seqlen_q: Sequence length of the query.
    :param seqlen_k: Sequence length of the key.
    :param IS_CAUSAL: Boolean flag indicating if the attention is causal.
    :param TILE_M: Tile size along the M dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.

    :return softmax_threshold_log2: Lower-bound scalar softmax threshold in log2-domain for the given block.
    """
    if IS_CAUSAL:
        offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
        q_idx = offs_m
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx = q_idx // QHEAD_PER_KVHEAD_PACKGQA
        causal_offset = seqlen_k - seqlen_q
        s_thr = tl.min(softmax_threshold * (q_idx + causal_offset + 1.0) / seqlen_k)
    else:
        s_thr = tl.full((), softmax_threshold, dtype=tl.float32)
    softmax_threshold_log2 = tl.log2(s_thr)
    return softmax_threshold_log2


@triton.jit
def get_gate_threshold(
    gate_threshold,
    m_block,
    seqlen_q,
    seqlen_k,
    IS_CAUSAL: tl.constexpr,
    TILE_M: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    IS_ADAPT_GATE: tl.constexpr,
):
    """
    Compute the gate threshold for a given block.

    :param gate_threshold: Threshold value for the gate.
    :param m_block: Current block index along the M dimension.
    :param seqlen_q: Sequence length of the query.
    :param seqlen_k: Sequence length of the key.
    :param IS_CAUSAL: Boolean flag indicating if the attention is causal.
    :param TILE_M: Tile size along the M dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.
    :param IS_ADAPT_GATE: Boolean flag indicating if self-adaptive gate threshold is enabled.
    :return gate_threshold_log2: Lower-bound scalar gate threshold in log2-domain for the given block.
    """
    if IS_CAUSAL and IS_ADAPT_GATE:
        offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
        q_idx = offs_m
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx = q_idx // QHEAD_PER_KVHEAD_PACKGQA
        causal_offset = seqlen_k - seqlen_q
        g_thr = tl.min(gate_threshold * (q_idx + causal_offset + 1.0) / seqlen_k)
    else:
        g_thr = tl.full((), gate_threshold, dtype=tl.float32)
    gate_threshold_log2 = tl.log2(g_thr)
    return gate_threshold_log2


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


@triton.jit
def make_ptrs(
    base_ptrs,
    mn_block,
    stride_seq,
    TILE_MN: tl.constexpr,
    TILE_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    offs_mn = mn_block * TILE_MN + tl.arange(0, TILE_MN)
    if TILE_K > 1:
        offs_k = tl.arange(0, TILE_K)
        if SWAP_AB:
            ptrs = base_ptrs + offs_mn[None, :] * stride_seq + offs_k[:, None]
        else:
            ptrs = base_ptrs + offs_mn[:, None] * stride_seq + offs_k[None, :]
    else:
        ptrs = base_ptrs + offs_mn * stride_seq
    return ptrs


@triton.jit
def make_pack_gqa_ptrs(
    base_ptrs,
    m_block,
    head_idx,
    stride_head,
    stride_seq,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    m_idx = offs_m // QHEAD_PER_KVHEAD_PACKGQA
    q_head_offset = offs_m - m_idx * QHEAD_PER_KVHEAD_PACKGQA
    q_head = head_idx * QHEAD_PER_KVHEAD_PACKGQA + q_head_offset
    if TILE_K > 1:
        offs_k = tl.arange(0, TILE_K)
        ptrs = (
            base_ptrs
            + m_idx[:, None] * stride_seq
            + q_head[:, None] * stride_head
            + offs_k[None, :]
        )
    else:
        ptrs = base_ptrs + m_idx * stride_seq + q_head * stride_head
    return ptrs
