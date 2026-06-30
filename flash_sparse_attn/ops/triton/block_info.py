import triton
import triton.language as tl


@triton.jit
def get_n_block_min_max(
    seqlen_q,
    seqlen_k,
    m_block,
    split_idx,
    window_size_sink,
    window_size_left,
    window_size_right,
    window_size_dist,
    NUM_SPLITS: tl.constexpr,
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
    n_block_sink_min = 0
    n_block_sink_max = 0
    if IS_CAUSAL or IS_LOCAL:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = tl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_block_max = tl.minimum(n_block_max, tl.cdiv(n_idx, TILE_N))
        if IS_LOCAL:
            n_idx_right = n_idx - window_size_dist - window_size_right
            n_block_window_max = tl.minimum(
                n_block_window_max, tl.maximum(tl.cdiv(n_idx_right, TILE_N), 0)
            )
    if IS_LOCAL:
        n_block_sink_max = tl.minimum(
            tl.cdiv(window_size_sink, TILE_N),
            tl.maximum(tl.cdiv(n_idx, TILE_N), 0),
        )
        n_block_sink_exclude_max = tl.cdiv(window_size_sink, TILE_N)
        m_idx_min = m_block * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_dist = n_idx - window_size_dist
        n_block_min = tl.maximum(n_idx_dist // TILE_N, 0)
        n_block_min = tl.maximum(n_block_min, n_block_sink_exclude_max)
        n_idx_left = n_idx - window_size_dist - window_size_right - window_size_left
        n_block_window_min = tl.maximum(n_idx_left // TILE_N, 0)
        n_block_window_min = tl.maximum(n_block_window_min, n_block_sink_exclude_max)
    if IS_SPLIT_KV:
        if IS_LOCAL:
            n_block_diag_min = n_block_min
            n_block_diag_max = n_block_max
            n_block_window_max = tl.maximum(n_block_window_max, n_block_window_min)
            total_n_blocks = tl.maximum(n_block_window_max - n_block_window_min, 0)
            base = total_n_blocks // NUM_SPLITS
            extra = total_n_blocks % NUM_SPLITS
            n_block_window_min_new = n_block_window_min + tl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            n_block_count = tl.where(split_idx < extra, base + 1, base)
            n_block_window_max = tl.minimum(
                n_block_window_min_new + n_block_count, n_block_window_max
            )
            n_block_window_min = n_block_window_min_new
            n_block_sink_max = tl.where(split_idx == 0, n_block_sink_max, 0)
            n_block_non_diag_max = tl.maximum(n_block_window_max, n_block_sink_max)
            n_block_max = tl.where(
                split_idx >= NUM_SPLITS - 1,
                n_block_diag_max,
                n_block_non_diag_max,
            )
            n_block_min = tl.where(
                split_idx >= NUM_SPLITS - 1,
                n_block_diag_min,
                n_block_non_diag_max,
            )
        else:
            total_n_blocks = tl.maximum(n_block_max - n_block_min, 0)
            base = total_n_blocks // NUM_SPLITS
            extra = total_n_blocks % NUM_SPLITS
            n_block_min_new = n_block_min + tl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            n_block_count = tl.where(split_idx < extra, base + 1, base)
            n_block_max = tl.minimum(n_block_min_new + n_block_count, n_block_max)
            n_block_min = n_block_min_new
            n_block_sink_max = tl.where(split_idx == 0, n_block_sink_max, 0)
    return (
        n_block_min,
        n_block_max,
        n_block_window_min,
        n_block_window_max,
        n_block_sink_min,
        n_block_sink_max,
    )


@triton.jit
def get_m_block_min_max(
    seqlen_q,
    seqlen_k,
    n_block,
    split_idx,
    window_size_sink,
    window_size_left,
    window_size_right,
    window_size_dist,
    NUM_SPLITS: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
):
    m_block_max = tl.cdiv(seqlen_q, TILE_M)
    m_block_min = 0
    m_block_window_max = m_block_max
    m_block_window_min = m_block_min
    m_block_sink_min = 0
    m_block_sink_max = 0
    if IS_CAUSAL or IS_LOCAL:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_block_min = tl.maximum(m_block_min, m_idx // TILE_M)
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_dist + window_size_right
            m_block_window_min = tl.maximum(m_block_window_min, m_idx_right // TILE_M)
    if IS_LOCAL:
        n_block_sink_exclude_max = tl.cdiv(window_size_sink, TILE_N)
        is_sink_block = n_block < n_block_sink_exclude_max
        n_idx_min = n_block * TILE_N
        m_idx_sink = n_idx_min + seqlen_q - seqlen_k
        m_block_sink_min = tl.maximum(m_idx_sink // TILE_M, 0)
        m_block_sink_max = tl.where(is_sink_block, tl.cdiv(seqlen_q, TILE_M), 0)
        m_block_sink_min = tl.where(is_sink_block, m_block_sink_min, 0)
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_dist = m_idx + window_size_dist
        m_block_max = tl.minimum(m_block_max, tl.cdiv(m_idx_dist, TILE_M))
        m_idx_left = m_idx + window_size_dist + window_size_right + window_size_left
        m_block_window_max = tl.minimum(m_block_window_max, tl.cdiv(m_idx_left, TILE_M))
        m_block_min = tl.where(is_sink_block, 0, m_block_min)
        m_block_max = tl.where(is_sink_block, 0, m_block_max)
        m_block_window_min = tl.where(is_sink_block, 0, m_block_window_min)
        m_block_window_max = tl.where(is_sink_block, 0, m_block_window_max)
    if IS_SPLIT_QO:
        if IS_LOCAL:
            n_idx_max = (n_block + 1) * TILE_N
            m_idx = n_idx_max + seqlen_q - seqlen_k
            m_block_min_no_mask = tl.maximum(m_block_min, tl.cdiv(m_idx, TILE_M))
            m_block_window_min = tl.maximum(m_block_window_min, m_block_min_no_mask)
            win_blocks = tl.maximum(m_block_window_max - m_block_window_min, 0)
            base = win_blocks // NUM_SPLITS
            extra = win_blocks % NUM_SPLITS
            win_off = tl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            win_cnt = tl.where(split_idx < extra, base + 1, base)
            m_block_window_min = m_block_window_min + win_off
            m_block_window_max = tl.minimum(
                m_block_window_min + win_cnt, m_block_window_max
            )
            # First split keeps diagonal, others skip it
            m_block_min = tl.where(split_idx == 0, m_block_min, m_block_max)
            m_block_sink_max = tl.where(split_idx == 0, m_block_sink_max, 0)
            m_block_sink_min = tl.where(split_idx == 0, m_block_sink_min, 0)
        else:
            total_m_blocks = tl.maximum(m_block_max - m_block_min, 0)
            base = total_m_blocks // NUM_SPLITS
            extra = total_m_blocks % NUM_SPLITS
            m_block_min_new = m_block_min + tl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            m_block_count = tl.where(split_idx < extra, base + 1, base)
            m_block_max = tl.minimum(m_block_min_new + m_block_count, m_block_max)
            m_block_min = m_block_min_new
    return (
        m_block_min,
        m_block_max,
        m_block_window_min,
        m_block_window_max,
        m_block_sink_min,
        m_block_sink_max,
    )


@triton.jit
def get_n_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    window_size_right,
    window_size_dist,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
):
    m_idx_min = m_block * TILE_M
    if QHEAD_PER_KVHEAD_PACKGQA > 1:
        m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
    n_idx = m_idx_min + seqlen_k - seqlen_q
    n_idx_right = (
        n_idx if not IS_LOCAL else n_idx - window_size_dist - window_size_right
    )
    return tl.maximum(n_block_min, n_idx_right // TILE_N)


@triton.jit
def get_n_block_min_before_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    window_size_left,
    window_size_right,
    window_size_dist,
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
        n_idx_left = n_idx - window_size_dist - window_size_right - window_size_left
        return tl.maximum(n_block_min, tl.cdiv(n_idx_left, TILE_N))


@triton.jit
def get_m_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    n_block,
    m_block_min,
    window_size_right,
    window_size_dist,
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
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_dist + window_size_right
        else:
            m_idx_right = m_idx
        return tl.maximum(m_block_min, tl.cdiv(m_idx_right, TILE_M))


@triton.jit
def get_m_block_max_before_local_mask(
    seqlen_q,
    seqlen_k,
    n_block,
    m_block_max,
    window_size_left,
    window_size_right,
    window_size_dist,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if not IS_LOCAL:
        return m_block_max
    else:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_idx_left = m_idx + window_size_dist + window_size_right + window_size_left
        return tl.minimum(m_block_max, m_idx_left // TILE_M)


@triton.jit
def get_topk_n_block_min_max(
    split_idx,
    NUM_SPLITS: tl.constexpr,
    TOPK_SEQLEN_K: tl.constexpr,
    TILE_N: tl.constexpr,
):
    total_n_blocks = tl.cdiv(TOPK_SEQLEN_K, TILE_N)
    base = total_n_blocks // NUM_SPLITS
    extra = total_n_blocks % NUM_SPLITS
    n_block_min = tl.where(
        split_idx < extra,
        split_idx * (base + 1),
        extra * (base + 1) + (split_idx - extra) * base,
    )
    n_block_count = tl.where(split_idx < extra, base + 1, base)
    n_block_max = tl.minimum(n_block_min + n_block_count, total_n_blocks)
    n_block_window_min = 0
    n_block_window_max = 0
    n_block_sink_min = 0
    n_block_sink_max = 0
    return (
        n_block_min,
        n_block_max,
        n_block_window_min,
        n_block_window_max,
        n_block_sink_min,
        n_block_sink_max,
    )
