import triton
import triton.language as tl


@triton.jit
def apply_mask(
    acc_s,
    m_block,
    n_block,
    seqlen_q,
    seqlen_k,
    window_size_left,
    window_size_right,
    MASK_SEQLEN: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """
    Apply seqlen, causal, and local masks to the attention scores.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param m_block: Current block index along the M dimension.
    :param n_block: Current block index along the N dimension.
    :param seqlen_q: The sequence length of the query.
    :param seqlen_k: The sequence length of the key.
    :param window_size_left: Left window size for local masking.
    :param window_size_right: Right window size for local masking.
    :param MASK_SEQLEN: Boolean flag indicating if seqlen masking should be applied.
    :param MASK_CAUSAL: Boolean flag indicating if causal masking should be applied.
    :param MASK_LOCAL: Boolean flag indicating if local masking should be applied.
    :param TILE_M: Tile size along the M dimension.
    :param TILE_N: Tile size along the N dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.
    :param SWAP_AB: Boolean flag indicating if query and key dimensions are swapped.

    :return acc_s: Masked attention scores tensor of shape [BLOCK_M, BLOCK_N].
    """
    tl.static_assert(
        not (MASK_CAUSAL and MASK_LOCAL),
        "MASK_CAUSAL and MASK_LOCAL cannot be both True",
    )
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)

    if SWAP_AB:
        tl.static_assert(
            QHEAD_PER_KVHEAD_PACKGQA == 1, "SWAP_AB with PACKGQA > 1 not supported"
        )
        q_idx = offs_m[None, :]
        k_idx = offs_n[:, None]
    else:
        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            q_idx = q_idx // QHEAD_PER_KVHEAD_PACKGQA

    if MASK_SEQLEN:
        acc_s = tl.where(
            (k_idx < seqlen_k) & (q_idx < seqlen_q),
            acc_s,
            float("-inf"),
        )

    if MASK_CAUSAL or MASK_LOCAL:
        causal_offset = seqlen_k - seqlen_q
        dist = q_idx + causal_offset - k_idx

        if MASK_CAUSAL:
            acc_s = tl.where(
                dist >= 0,
                acc_s,
                float("-inf"),
            )
        else:
            acc_s = tl.where(
                (dist >= window_size_right) & (dist <= window_size_left),
                acc_s,
                float("-inf"),
            )

    return acc_s
