import triton
import triton.language as tl


@triton.jit
def get_n_block_min_max(
    seqlen_q,
    seqlen_k,
    m_block,
    split_idx,
    num_splits,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    n_block_max = tl.cdiv(seqlen_k, TILE_N)
    if IS_CAUSAL or (IS_LOCAL and WINDOW_SIZE_RIGHT is not None):
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_right = n_idx if IS_CAUSAL else n_idx + WINDOW_SIZE_RIGHT
        n_block_max = min(n_block_max, tl.cdiv(n_idx_right, TILE_N))
    n_block_min = 0
    if IS_LOCAL and WINDOW_SIZE_LEFT is not None:
        m_idx_min = m_block * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_left = n_idx - WINDOW_SIZE_LEFT
        n_block_min = max(n_idx_left // TILE_N, 0)
    if IS_SPLIT_KV:
        num_n_blocks_per_split = (
            0
            if n_block_max <= n_block_min
            else (n_block_max - n_block_min + num_splits - 1) // num_splits
        )
        n_block_min = n_block_min + split_idx * num_n_blocks_per_split
        n_block_max = min(n_block_min + num_n_blocks_per_split, n_block_max)
    return n_block_min, n_block_max


@triton.jit
def get_m_block_min_max(
    seqlen_q,
    seqlen_k,
    n_block,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
):
    m_block_max = tl.cdiv(seqlen_q, TILE_M)
    m_block_min = 0
    if IS_CAUSAL or (IS_LOCAL and WINDOW_SIZE_RIGHT is not None):
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_idx_right = m_idx if IS_CAUSAL else m_idx - WINDOW_SIZE_RIGHT
        m_block_min = max(m_block_min, m_idx_right // TILE_M)
    if IS_LOCAL and WINDOW_SIZE_LEFT is not None:
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_left = m_idx + WINDOW_SIZE_LEFT
        m_block_max = min(m_block_max, tl.cdiv(m_idx_left, TILE_M))
    return m_block_min, m_block_max


@triton.jit
def get_n_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    m_idx_min = m_block * TILE_M
    if QHEAD_PER_KVHEAD_PACKGQA > 1:
        m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
    n_idx = m_idx_min + seqlen_k - seqlen_q
    n_idx_right = (
        n_idx
        if (not IS_LOCAL or WINDOW_SIZE_RIGHT is None)
        else n_idx + WINDOW_SIZE_RIGHT
    )
    return max(n_block_min, n_idx_right // TILE_N)


@triton.jit
def get_n_block_min_before_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    if not IS_LOCAL or WINDOW_SIZE_LEFT is None:
        return n_block_min
    else:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_left = n_idx - WINDOW_SIZE_LEFT
        return max(n_block_min, tl.cdiv(n_idx_left, TILE_N))
