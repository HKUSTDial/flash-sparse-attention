# Copyright (c) 2026, Jingze Shi.
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def get_seqlen_info(
    batch_idx: tl.tensor,
    seqlen_static: tl.tensor,
    cu_seqlens: tl.tensor,
    seqused: tl.tensor,
    TILE_MN: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr = False,
    HAS_SEQUSED: tl.constexpr = False,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
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
    :param TILE_MN: tile size along the M or N dimension
    :type TILE_MN: int
    :param HAS_CU_SEQLENS: boolean flag indicating if cu_seqlens is provided
    :type HAS_CU_SEQLENS: bool
    :param HAS_SEQUSED: boolean flag indicating if seqused is provided
    :type HAS_SEQUSED: bool

    :return offset: offset for the given batch index
    :return padded_offset: offset aligned to TILE_MN for the given batch index
    :return seqlen: sequence length for the given batch index
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
    padded_offset = (
        (offset + batch_idx * TILE_MN) // TILE_MN * TILE_MN if HAS_CU_SEQLENS else tl.to_tensor(0)
    )
    return offset, padded_offset, seqlen


@triton.jit
def get_seqlen_info_qk(
    batch_idx: tl.tensor,
    seqlen_q_static: tl.tensor,
    seqlen_k_static: tl.tensor,
    cu_seqlens_q: tl.tensor,
    cu_seqlens_k: tl.tensor,
    seqused_q: tl.tensor,
    seqused_k: tl.tensor,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr = False,
    HAS_CU_SEQLENS_K: tl.constexpr = False,
    HAS_SEQUSED_Q: tl.constexpr = False,
    HAS_SEQUSED_K: tl.constexpr = False,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
    offset_q, padded_offset_q, seqlen_q = get_seqlen_info(
        batch_idx,
        seqlen_q_static,
        cu_seqlens_q,
        seqused_q,
        TILE_MN=TILE_M,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
        HAS_SEQUSED=HAS_SEQUSED_Q,
    )
    offset_k, padded_offset_k, seqlen_k = get_seqlen_info(
        batch_idx,
        seqlen_k_static,
        cu_seqlens_k,
        seqused_k,
        TILE_MN=TILE_N,
        HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
        HAS_SEQUSED=HAS_SEQUSED_K,
    )
    return offset_q, offset_k, padded_offset_q, padded_offset_k, seqlen_q, seqlen_k


@triton.jit
def get_softmax_threshold(
    softmax_threshold: tl.tensor,
    m_block: tl.tensor,
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    row_offsets: tl.tensor,
    IS_CAUSAL: tl.constexpr = False,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr = 1,
    IS_LOG2: tl.constexpr = True,
    FAST_MATH: tl.constexpr = True,
):
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
    :param IS_LOG2: boolean flag indicating if the threshold should be returned in log2-domain
    :type IS_LOG2: bool
    :param FAST_MATH: use approximate FTZ log2
    :type FAST_MATH: bool

    :return softmax_threshold: softmax threshold of shape [TILE_M]
    """
    if IS_CAUSAL:
        q_idx = m_block * row_offsets.shape[0] + row_offsets
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx //= QHEAD_PER_KVHEAD_PACKGQA
        visible_len = (q_idx + seqlen_k - seqlen_q + 1).to(tl.float32)
    else:
        visible_len = row_offsets * 0.0 + seqlen_k
    threshold = tl.maximum(tl.minimum(softmax_threshold / visible_len, 1.0), 0.0)
    if not IS_LOG2:
        return threshold
    if FAST_MATH:
        return libdevice.fast_log2f(threshold)
    else:
        return tl.log2(threshold)


@triton.jit
def offset_batch_Q(
    base_ptr: tl.tensor,
    batch_idx: tl.tensor,
    offset: tl.tensor,
    padded_offset: tl.tensor,
    stride_batch: tl.tensor,
    stride_seq: tl.tensor,
    HAS_CU_SEQLENS: tl.constexpr = False,
    USE_PADDED: tl.constexpr = False,
) -> tl.tensor:
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
    else:
        return base_ptr + batch_idx * stride_batch


@triton.jit
def offset_batch_K(
    base_ptr: tl.tensor,
    batch_idx: tl.tensor,
    offset: tl.tensor,
    padded_offset: tl.tensor,
    stride_batch: tl.tensor,
    stride_seq: tl.tensor,
    HAS_CU_SEQLENS: tl.constexpr = False,
    USE_PADDED: tl.constexpr = False,
) -> tl.tensor:
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
    else:
        return base_ptr + batch_idx * stride_batch


@triton.jit
def make_ptrs(
    base_ptr: tl.tensor,
    mn_block: tl.tensor,
    stride_seq: tl.tensor,
    offs_mn: tl.tensor,
    offs_k: tl.tensor,
    TILE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
) -> tl.tensor:
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


@triton.jit
def make_pack_gqa_ptrs(
    base_ptr: tl.tensor,
    m_block: tl.tensor,
    head_idx: tl.tensor,
    stride_head: tl.tensor,
    stride_seq: tl.tensor,
    offs_m: tl.tensor,
    offs_k: tl.tensor,
    TILE_K: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr = 1,
) -> tl.tensor:
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
    q_head = head_idx * QHEAD_PER_KVHEAD_PACKGQA + offs_m - m_idx * QHEAD_PER_KVHEAD_PACKGQA
    if TILE_K == 1:
        return base_ptr + m_idx * stride_seq + q_head * stride_head
    else:
        return (
            base_ptr + m_idx[:, None] * stride_seq + q_head[:, None] * stride_head + offs_k[None, :]
        )
