import triton
import triton.language as tl


@triton.jit
def get_n_block_min_max(
    seqlen_q,
    seqlen_k,
    m_block,
    split_idx,
    num_splits,
    window_size_left,
    window_size_right,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    n_block_max = tl.cdiv(seqlen_k, TILE_N)
    n_block_min = 0
    n_block_window_max = n_block_max
    n_block_window_min = n_block_min
    if IS_CAUSAL or IS_LOCAL:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = tl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_block_max = tl.minimum(n_block_max, tl.cdiv(n_idx, TILE_N))
        if IS_LOCAL:
            n_idx_right = n_idx - window_size_right
            n_block_window_max = tl.minimum(
                n_block_window_max, tl.cdiv(n_idx_right, TILE_N)
            )
    if IS_LOCAL:
        m_idx_min = m_block * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_left = n_idx - window_size_left
        n_block_window_min = tl.maximum(n_idx_left // TILE_N, 0)
    if IS_SPLIT_KV:
        if IS_LOCAL:
            n_block_min = tl.maximum(n_block_min, n_block_window_min)
            n_block_max_with_diag = n_block_max
            n_block_max = tl.maximum(n_block_window_max, n_block_min)
        total_n_blocks = tl.maximum(n_block_max - n_block_min, 0)
        base = total_n_blocks // num_splits
        extra = total_n_blocks % num_splits
        n_block_min_new = n_block_min + tl.where(
            split_idx < extra,
            split_idx * (base + 1),
            extra * (base + 1) + (split_idx - extra) * base,
        )
        n_block_count = tl.where(split_idx < extra, base + 1, base)
        n_block_max = tl.minimum(n_block_min_new + n_block_count, n_block_max)
        n_block_min = n_block_min_new
        if IS_LOCAL:
            n_block_max = tl.where(
                split_idx >= num_splits - 1,
                n_block_max_with_diag,
                n_block_max,
            )
    return n_block_min, n_block_max, n_block_window_min, n_block_window_max


@triton.jit
def get_m_block_min_max(
    seqlen_q,
    seqlen_k,
    n_block,
    window_size_left,
    window_size_right,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    m_block_max = tl.cdiv(seqlen_q, TILE_M)
    m_block_min = 0
    m_block_window_max = m_block_max
    m_block_window_min = m_block_min
    if IS_CAUSAL or IS_LOCAL:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_block_min = tl.maximum(m_block_min, m_idx // TILE_M)
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_right
            m_block_window_min = tl.maximum(m_block_window_min, m_idx_right // TILE_M)
    if IS_LOCAL:
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_left = m_idx + window_size_left
        m_block_window_max = tl.minimum(m_block_window_max, tl.cdiv(m_idx_left, TILE_M))
    return m_block_min, m_block_max, m_block_window_min, m_block_window_max


@triton.jit
def get_n_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    window_size_right,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    m_idx_min = m_block * TILE_M
    if QHEAD_PER_KVHEAD_PACKGQA > 1:
        m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
    n_idx = m_idx_min + seqlen_k - seqlen_q
    n_idx_right = n_idx if not IS_LOCAL else n_idx - window_size_right
    return tl.maximum(n_block_min, n_idx_right // TILE_N)


@triton.jit
def get_n_block_min_before_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    window_size_left,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    if not IS_LOCAL:
        return n_block_min
    else:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = tl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_left = n_idx - window_size_left
        return tl.maximum(n_block_min, tl.cdiv(n_idx_left, TILE_N))


@triton.jit
def get_m_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    n_block,
    m_block_min,
    window_size_right,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if not IS_CAUSAL and not IS_LOCAL:
        return m_block_min
    else:
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_right = m_idx if IS_CAUSAL else m_idx + window_size_right
        return tl.maximum(m_block_min, tl.cdiv(m_idx_right, TILE_M))


@triton.jit
def get_m_block_max_before_local_mask(
    seqlen_q,
    seqlen_k,
    n_block,
    m_block_max,
    window_size_left,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if not IS_LOCAL:
        return m_block_max
    else:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_idx_left = m_idx + window_size_left
        return tl.minimum(m_block_max, m_idx_left // TILE_M)
