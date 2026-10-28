# Copyright (c) 2026, Jingze Shi.
import triton
import triton.language as tl


@triton.jit
def apply_mask(
    acc_s: tl.tensor,
    m_block: tl.tensor,
    n_block: tl.tensor,
    offs_m: tl.tensor,
    offs_n: tl.tensor,
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    window_size_sink: tl.tensor,
    window_size_left: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    MASK_SEQLEN: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    MASK_SINK: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    SWAP_AB: tl.constexpr,
) -> tl.tensor:
    """
    Apply seqlen, causal, and local masks to the attention scores.

    :param acc_s: attention scores of shape [BLOCK_M, BLOCK_N]
    :type acc_s: tensor
    :param m_block: current block index along the M dimension
    :type m_block: tensor
    :param n_block: current block index along the N dimension
    :type n_block: tensor
    :param offs_m: lane offsets for the M dimension
    :type offs_m: tensor
    :param offs_n: lane offsets for the N dimension
    :type offs_n: tensor
    :param seqlen_q: The sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: The sequence length of the key
    :type seqlen_k: tensor
    :param window_size_sink: prefix-sink token count
    :type window_size_sink: tensor
    :param window_size_left: distant local band token count
    :type window_size_left: tensor
    :param window_size_right: gap token count after the near-diagonal window before the distant band
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param MASK_SEQLEN: boolean flag indicating if seqlen masking should be applied
    :type MASK_SEQLEN: bool
    :param MASK_CAUSAL: boolean flag indicating if causal masking should be applied
    :type MASK_CAUSAL: bool
    :param MASK_LOCAL: boolean flag indicating if local masking should be applied
    :type MASK_LOCAL: bool
    :param MASK_SINK: boolean flag indicating if sink masking should be applied
    :type MASK_SINK: bool
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int
    :param SWAP_AB: boolean flag indicating if query and key dimensions are swapped
    :type SWAP_AB: bool

    :return acc_s: masked attention scores of shape [BLOCK_M, BLOCK_N]
    """
    if SWAP_AB:
        tl.static_assert(QHEAD_PER_KVHEAD_PACKGQA == 1, "SWAP_AB with PackGQA > 1 is not supported")
        offs_m = m_block * TILE_M + offs_m
        offs_n = n_block * TILE_N + offs_n
        q_idx = offs_m[None, :]
        k_idx = offs_n[:, None]
    else:
        offs_m = m_block * TILE_M + offs_m
        offs_n = n_block * TILE_N + offs_n
        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx //= QHEAD_PER_KVHEAD_PACKGQA

    if MASK_SEQLEN:
        valid = (q_idx < seqlen_q) & (k_idx < seqlen_k)
    else:
        valid = q_idx == q_idx  # constant True avoid creating additional layout

    if MASK_CAUSAL or MASK_LOCAL or MASK_SINK:
        near = q_idx + seqlen_k - seqlen_q - k_idx
        if MASK_LOCAL or MASK_SINK:
            allowed = near < near  # constant False avoid creating additional layout
            if MASK_LOCAL:
                allowed |= (near >= 0) & (near < window_size_near)
                allowed |= (near >= window_size_near + window_size_right) & (
                    near < window_size_near + window_size_right + window_size_left
                )
            if MASK_SINK:
                allowed |= (k_idx < window_size_sink) & (near >= 0)
            valid &= allowed
        elif MASK_CAUSAL:
            valid &= near >= 0
    return tl.where(valid, acc_s, float("-inf"))
