from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_seqlen_info(
    batch_idx,
    seqlen_static,
    cu_seqlens,
    seqused,
    HAS_CU_SEQLENS: gl.constexpr,
    HAS_SEQUSED: gl.constexpr,
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
        offset = gl.load(cu_seqlens + batch_idx)
        if HAS_SEQUSED:
            seqlen = gl.load(seqused + batch_idx)
        else:
            seqlen = gl.load(cu_seqlens + batch_idx + 1) - offset
    else:
        offset = gl.to_tensor(0)
        seqlen = gl.load(seqused + batch_idx) if HAS_SEQUSED else seqlen_static
    return offset, seqlen


@gluon.jit
def get_seqlen_info_qk(
    batch_idx,
    seqlen_q_static,
    seqlen_k_static,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_CU_SEQLENS_K: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    HAS_SEQUSED_K: gl.constexpr,
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
    offset_q, seqlen_q = get_seqlen_info(
        batch_idx,
        seqlen_q_static,
        cu_seqlens_q,
        seqused_q,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )
    offset_k, seqlen_k = get_seqlen_info(
        batch_idx,
        seqlen_k_static,
        cu_seqlens_k,
        seqused_k,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
        HAS_SEQUSED=HAS_SEQUSED_K,
    )
    padded_offset_q = (
        (offset_q + batch_idx * TILE_M) // TILE_M * TILE_M
        if HAS_CU_SEQLENS_Q
        else gl.to_tensor(0)
    )
    padded_offset_k = (
        (offset_k + batch_idx * TILE_N) // TILE_N * TILE_N
        if HAS_CU_SEQLENS_K
        else gl.to_tensor(0)
    )
    return offset_q, offset_k, padded_offset_q, padded_offset_k, seqlen_q, seqlen_k


@gluon.jit
def get_softmax_threshold(
    softmax_threshold,
    m_block,
    seqlen_q,
    seqlen_k,
    row_offsets,
    IS_CAUSAL: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
):
    """
    Compute the softmax threshold for a given block.

    :param softmax_threshold: Dimensionless multiple of uniform attention.
    :param m_block: Current block index along the M dimension.
    :param seqlen_q: Sequence length of the query.
    :param seqlen_k: Sequence length of the key.
    :param row_offsets: Row offsets for the current block.
    :param IS_CAUSAL: Boolean flag indicating if the attention is causal.
    :param TILE_M: Tile size along the M dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.

    :return softmax_threshold_log2: softmax threshold of shape [TILE_M] in log2-domain for the given block.
    """
    if IS_CAUSAL:
        q_idx = m_block * row_offsets.shape[0] + row_offsets
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx //= QHEAD_PER_KVHEAD_PACKGQA
        visible_len = (q_idx + seqlen_k - seqlen_q + 1).to(gl.float32)
    else:
        visible_len = row_offsets * 0.0 + seqlen_k
    threshold = gl.maximum(gl.minimum(softmax_threshold / visible_len, 1.0), 0.0)
    return gl.log2(threshold)


@gluon.jit
def get_gate_threshold(
    gate_threshold,
    m_block,
    seqlen_q,
    seqlen_k,
    row_offsets,
    IS_CAUSAL: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    IS_ADAPT_GATE: gl.constexpr,
):
    """
    Compute the gate threshold for a given block.

    :param gate_threshold: Threshold value for the gate.
    :param m_block: Current block index along the M dimension.
    :param seqlen_q: Sequence length of the query.
    :param seqlen_k: Sequence length of the key.
    :param row_offsets: Row offsets for the current block.
    :param IS_CAUSAL: Boolean flag indicating if the attention is causal.
    :param TILE_M: Tile size along the M dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.
    :param IS_ADAPT_GATE: Boolean flag indicating if self-adaptive gate threshold is enabled.

    :return gate_threshold_log2: Lower-bound scalar gate threshold in log2-domain for the given block.
    """
    if IS_CAUSAL and IS_ADAPT_GATE:
        q_idx = m_block * row_offsets.shape[0] + row_offsets
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx //= QHEAD_PER_KVHEAD_PACKGQA
        gate_threshold = gl.min(
            gate_threshold * (q_idx + seqlen_k - seqlen_q + 1.0) / seqlen_k,
            axis=0,
        )
    return gl.log2(gate_threshold)


@gluon.jit
def offset_batch_Q(
    base_ptr,
    batch_idx,
    offset,
    padded_offset,
    stride_batch,
    stride_seq,
    HAS_CU_SEQLENS: gl.constexpr,
    USE_PADDED: gl.constexpr,
):
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    return base_ptr + batch_idx * stride_batch


@gluon.jit
def offset_batch_K(
    base_ptr,
    batch_idx,
    offset,
    padded_offset,
    stride_batch,
    stride_seq,
    HAS_CU_SEQLENS: gl.constexpr,
    USE_PADDED: gl.constexpr,
):
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    return base_ptr + batch_idx * stride_batch


@gluon.jit
def make_ptrs(
    base_ptrs,
    mn_block,
    stride_seq,
    offs_mn,
    offs_k,
    TILE_K: gl.constexpr,
    SWAP_AB: gl.constexpr,
):
    offs_mn = mn_block * offs_mn.shape[0] + offs_mn
    if TILE_K == 1:
        return base_ptrs + offs_mn * stride_seq
    else:
        if SWAP_AB:
            return base_ptrs + offs_mn[None, :] * stride_seq + offs_k[:, None]
        else:
            return base_ptrs + offs_mn[:, None] * stride_seq + offs_k[None, :]


@gluon.jit
def make_pack_gqa_ptrs(
    base_ptrs,
    m_block,
    head_idx,
    stride_head,
    stride_seq,
    offs_m,
    offs_k,
    TILE_K: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
):
    offs_m = m_block * offs_m.shape[0] + offs_m
    m_idx = offs_m // QHEAD_PER_KVHEAD_PACKGQA
    q_head = (
        head_idx * QHEAD_PER_KVHEAD_PACKGQA + offs_m - m_idx * QHEAD_PER_KVHEAD_PACKGQA
    )
    if TILE_K == 1:
        return base_ptrs + m_idx * stride_seq + q_head * stride_head
    else:
        return (
            base_ptrs
            + m_idx[:, None] * stride_seq
            + q_head[:, None] * stride_head
            + offs_k[None, :]
        )
