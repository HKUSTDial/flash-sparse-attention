from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def apply_mask(
    acc_s,
    m_block,
    n_block,
    offs_m,
    offs_n,
    seqlen_q,
    seqlen_k,
    window_size_sink,
    window_size_left,
    window_size_right,
    window_size_near,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr,
    MASK_SINK: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    SWAP_AB: gl.constexpr,
):
    """
    Apply seqlen, causal, and local masks to the attention scores.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param m_block: Current block index along the M dimension.
    :param n_block: Current block index along the N dimension.
    :param offs_m: Lane offsets for the M dimension.
    :param offs_n: Lane offsets for the N dimension.
    :param seqlen_q: The sequence length of the query.
    :param seqlen_k: The sequence length of the key.
    :param window_size_sink: Prefix sink token count.
    :param window_size_left: Distant local band token count.
    :param window_size_right: Gap token count after the near-diagonal window before the distant band.
    :param window_size_near: Near-diagonal local token count.
    :param MASK_SEQLEN: Boolean flag indicating if seqlen masking should be applied.
    :param MASK_CAUSAL: Boolean flag indicating if causal masking should be applied.
    :param MASK_LOCAL: Boolean flag indicating if local masking should be applied.
    :param MASK_SINK: Boolean flag indicating if sink masking should be applied.
    :param TILE_M: Tile size along the M dimension.
    :param TILE_N: Tile size along the N dimension.
    :param QHEAD_PER_KVHEAD_PACKGQA: Ratio of query heads to key/value heads for packed GQA.
    :param SWAP_AB: Boolean flag indicating if query and key dimensions are swapped.

    :return acc_s: Masked attention scores tensor of shape [BLOCK_M, BLOCK_N].
    """
    if SWAP_AB:
        gl.static_assert(
            QHEAD_PER_KVHEAD_PACKGQA == 1, "SWAP_AB with PackGQA > 1 is not supported"
        )
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
    return gl.where(valid, acc_s, float("-inf"))
