# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_seqlen_info(
    batch_idx: gl.tensor,
    seqlen_static: gl.tensor,
    cu_seqlens: gl.tensor,
    seqused: gl.tensor,
    HAS_CU_SEQLENS: gl.constexpr,
    HAS_SEQUSED: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor]:
    """
    Get offset and seqlen for a given batch index.

    :param batch_idx: index of the batch
    :type batch_idx: tensor
    :param seqlen_static: static sequence length if cu_seqlens is not provided
    :type seqlen_static: tensor
    :param cu_seqlens: cumulative sequence lengths tensor
    :type cu_seqlens: tensor
    :param seqused: actual sequence lengths tensor
    :type seqused: tensor
    :param HAS_CU_SEQLENS: boolean flag indicating if cu_seqlens is provided
    :type HAS_CU_SEQLENS: bool
    :param HAS_SEQUSED: boolean flag indicating if seqused is provided
    :type HAS_SEQUSED: bool

    :return offset: offset for the given batch index
    :return seqlen: sequence length for the given batch index
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
    batch_idx: gl.tensor,
    seqlen_q_static: gl.tensor,
    seqlen_k_static: gl.tensor,
    cu_seqlens_q: gl.tensor,
    cu_seqlens_k: gl.tensor,
    seqused_q: gl.tensor,
    seqused_k: gl.tensor,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_CU_SEQLENS_K: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    HAS_SEQUSED_K: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    """
    Get offset, padded_offset, and seqlen for both Q and K.

    :param batch_idx: index of the batch
    :type batch_idx: tensor
    :param seqlen_q_static: static sequence length for Q if cu_seqlens_q is not provided
    :type seqlen_q_static: tensor
    :param seqlen_k_static: static sequence length for K if cu_seqlens_k is not provided
    :type seqlen_k_static: tensor
    :param cu_seqlens_q: cumulative sequence lengths tensor for Q
    :type cu_seqlens_q: tensor
    :param cu_seqlens_k: cumulative sequence lengths tensor for K
    :type cu_seqlens_k: tensor
    :param seqused_q: actual sequence lengths tensor for Q
    :type seqused_q: tensor
    :param seqused_k: actual sequence lengths tensor for K
    :type seqused_k: tensor
    :param TILE_M: tile size for Q
    :type TILE_M: int
    :param TILE_N: tile size for K
    :type TILE_N: int
    :param HAS_CU_SEQLENS_Q: boolean flag indicating if cu_seqlens_q is provided
    :type HAS_CU_SEQLENS_Q: bool
    :param HAS_CU_SEQLENS_K: boolean flag indicating if cu_seqlens_k is provided
    :type HAS_CU_SEQLENS_K: bool
    :param HAS_SEQUSED_Q: boolean flag indicating if seqused_q is provided
    :type HAS_SEQUSED_Q: bool
    :param HAS_SEQUSED_K: boolean flag indicating if seqused_k is provided
    :type HAS_SEQUSED_K: bool

    :return offset_q: offset for Q for the given batch index
    :return offset_k: offset for K for the given batch index
    :return padded_offset_q: padded offset for Q aligned to TILE_M
    :return padded_offset_k: padded offset for K aligned to TILE_N
    :return seqlen_q: sequence length for Q for the given batch index
    :return seqlen_k: sequence length for K for the given batch index
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
    softmax_threshold: gl.tensor,
    m_block: gl.tensor,
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    row_offsets: gl.tensor,
    IS_CAUSAL: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
) -> gl.tensor:
    """
    Compute the softmax threshold for a given block.

    :param softmax_threshold: dimensionless multiple of uniform attention
    :type softmax_threshold: tensor
    :param m_block: current block index along the M dimension
    :type m_block: tensor
    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param row_offsets: row offsets for the current block
    :type row_offsets: tensor
    :param IS_CAUSAL: boolean flag indicating if the attention is causal
    :type IS_CAUSAL: bool
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int

    :return softmax_threshold_log2: softmax threshold of shape [TILE_M] in log2-domain for the given block
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
def offset_batch_Q(
    base_ptr: gl.tensor,
    batch_idx: gl.tensor,
    offset: gl.tensor,
    padded_offset: gl.tensor,
    stride_batch: gl.tensor,
    stride_seq: gl.tensor,
    HAS_CU_SEQLENS: gl.constexpr,
    USE_PADDED: gl.constexpr,
) -> gl.tensor:
    """
    Offset a Q-like base pointer to the selected batch or packed sequence.

    :param base_ptr: base pointer before applying the batch or sequence offset
    :type base_ptr: tensor
    :param batch_idx: index of the batch
    :type batch_idx: tensor
    :param offset: unpadded token offset for the selected packed sequence
    :type offset: tensor
    :param padded_offset: token offset aligned to TILE_M
    :type padded_offset: tensor
    :param stride_batch: stride between batches
    :type stride_batch: tensor
    :param stride_seq: stride between sequence positions
    :type stride_seq: tensor
    :param HAS_CU_SEQLENS: boolean flag indicating if packed sequence offsets are provided
    :type HAS_CU_SEQLENS: bool
    :param USE_PADDED: boolean flag indicating if the padded sequence offset should be used
    :type USE_PADDED: bool

    :return base_ptr: base pointer offset to the selected batch or packed sequence
    """
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    return base_ptr + batch_idx * stride_batch


@gluon.jit
def offset_batch_K(
    base_ptr: gl.tensor,
    batch_idx: gl.tensor,
    offset: gl.tensor,
    padded_offset: gl.tensor,
    stride_batch: gl.tensor,
    stride_seq: gl.tensor,
    HAS_CU_SEQLENS: gl.constexpr,
    USE_PADDED: gl.constexpr,
) -> gl.tensor:
    """
    Offset a K-like base pointer to the selected batch or packed sequence.

    :param base_ptr: base pointer before applying the batch or sequence offset
    :type base_ptr: tensor
    :param batch_idx: index of the batch
    :type batch_idx: tensor
    :param offset: unpadded token offset for the selected packed sequence
    :type offset: tensor
    :param padded_offset: token offset aligned to TILE_N
    :type padded_offset: tensor
    :param stride_batch: stride between batches
    :type stride_batch: tensor
    :param stride_seq: stride between sequence positions
    :type stride_seq: tensor
    :param HAS_CU_SEQLENS: boolean flag indicating if packed sequence offsets are provided
    :type HAS_CU_SEQLENS: bool
    :param USE_PADDED: boolean flag indicating if the padded sequence offset should be used
    :type USE_PADDED: bool

    :return base_ptr: base pointer offset to the selected batch or packed sequence
    """
    if HAS_CU_SEQLENS:
        actual_offset = padded_offset if USE_PADDED else offset
        return base_ptr + actual_offset * stride_seq
    return base_ptr + batch_idx * stride_batch


@gluon.jit
def make_ptrs(
    base_ptr: gl.tensor,
    mn_block: gl.tensor,
    stride_seq: gl.tensor,
    offs_mn: gl.tensor,
    offs_k: gl.tensor,
    TILE_K: gl.constexpr,
    SWAP_AB: gl.constexpr,
) -> gl.tensor:
    """
    Construct pointers for a sequence tile.

    :param base_ptr: base pointer for the selected batch and head
    :type base_ptr: tensor
    :param mn_block: current block index along the sequence dimension
    :type mn_block: tensor
    :param stride_seq: stride between sequence positions
    :type stride_seq: tensor
    :param offs_mn: lane offsets along the M or N dimension
    :type offs_mn: tensor
    :param offs_k: lane offsets along the K dimension
    :type offs_k: tensor
    :param TILE_K: tile size along the K dimension
    :type TILE_K: int
    :param SWAP_AB: boolean flag indicating if the sequence and K dimensions are swapped
    :type SWAP_AB: bool

    :return ptrs: pointer tensor for the requested tile
    """
    offs_mn = mn_block * offs_mn.shape[0] + offs_mn
    if TILE_K == 1:
        return base_ptr + offs_mn * stride_seq
    else:
        if SWAP_AB:
            return base_ptr + offs_mn[None, :] * stride_seq + offs_k[:, None]
        else:
            return base_ptr + offs_mn[:, None] * stride_seq + offs_k[None, :]


@gluon.jit
def make_pack_gqa_ptrs(
    base_ptr: gl.tensor,
    m_block: gl.tensor,
    head_idx: gl.tensor,
    stride_head: gl.tensor,
    stride_seq: gl.tensor,
    offs_m: gl.tensor,
    offs_k: gl.tensor,
    TILE_K: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
) -> gl.tensor:
    """
    Construct pointers for a packed-GQA tile.

    :param base_ptr: base pointer for the selected batch
    :type base_ptr: tensor
    :param m_block: current block index along the packed M dimension
    :type m_block: tensor
    :param head_idx: index of the current KV head
    :type head_idx: tensor
    :param stride_head: stride between query heads
    :type stride_head: tensor
    :param stride_seq: stride between sequence positions
    :type stride_seq: tensor
    :param offs_m: lane offsets along the packed M dimension
    :type offs_m: tensor
    :param offs_k: lane offsets along the K dimension
    :type offs_k: tensor
    :param TILE_K: tile size along the K dimension
    :type TILE_K: int
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int

    :return ptrs: pointer tensor for the requested packed-GQA tile
    """
    offs_m = m_block * offs_m.shape[0] + offs_m
    m_idx = offs_m // QHEAD_PER_KVHEAD_PACKGQA
    q_head = (
        head_idx * QHEAD_PER_KVHEAD_PACKGQA + offs_m - m_idx * QHEAD_PER_KVHEAD_PACKGQA
    )
    if TILE_K == 1:
        return base_ptr + m_idx * stride_seq + q_head * stride_head
    else:
        return (
            base_ptr
            + m_idx[:, None] * stride_seq
            + q_head[:, None] * stride_head
            + offs_k[None, :]
        )
