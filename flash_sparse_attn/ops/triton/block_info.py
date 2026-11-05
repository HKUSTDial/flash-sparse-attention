# Copyright (c) 2026, Jingze Shi.
import triton
import triton.language as tl


@triton.jit
def get_n_block_min_max(
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    m_block: tl.tensor,
    split_idx: tl.tensor,
    window_size_sink: tl.tensor,
    window_size_left: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    NUM_SPLITS: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Compute the key block ranges processed by a forward query block.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param m_block: current block index along the M dimension
    :type m_block: tensor
    :param split_idx: index of the current KV split
    :type split_idx: tensor
    :param window_size_sink: prefix-sink token count
    :type window_size_sink: tensor
    :param window_size_left: distant local band token count
    :type window_size_left: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param NUM_SPLITS: number of KV splits
    :type NUM_SPLITS: int
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_CAUSAL: boolean flag indicating if the attention is causal
    :type IS_CAUSAL: bool
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool
    :param IS_SPLIT_KV: boolean flag indicating if the KV range is split
    :type IS_SPLIT_KV: bool
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int

    :return n_block_min: minimum N block in the near-diagonal range
    :return n_block_max: maximum N block in the near-diagonal range
    :return n_block_window_min: minimum N block in the distant local-window range
    :return n_block_window_max: maximum N block in the distant local-window range
    :return n_block_sink_min: minimum N block in the sink range
    :return n_block_sink_max: maximum N block in the sink range
    """
    n_block_max = tl.cdiv(seqlen_k, TILE_N)
    n_block_min = tl.to_tensor(0)
    n_block_window_max = n_block_max
    n_block_window_min = n_block_min
    n_block_sink_min = tl.to_tensor(0)
    n_block_sink_max = tl.to_tensor(0)
    if IS_CAUSAL or IS_LOCAL:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = tl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_block_max = tl.minimum(n_block_max, tl.maximum(tl.cdiv(n_idx, TILE_N), 0))
        if IS_LOCAL:
            n_idx_right = n_idx - window_size_near - window_size_right
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
        n_idx_near = n_idx - window_size_near
        n_block_min = tl.maximum(n_idx_near // TILE_N, 0)
        n_block_min = tl.maximum(n_block_min, n_block_sink_exclude_max)
        n_idx_left = n_idx - window_size_near - window_size_right - window_size_left
        n_block_window_min = tl.maximum(n_idx_left // TILE_N, 0)
        n_block_window_min = tl.maximum(n_block_window_min, n_block_sink_exclude_max)
    if IS_SPLIT_KV:
        if IS_LOCAL:
            n_block_diag_min = tl.where(
                seqlen_q == 1,
                tl.maximum(tl.cdiv(tl.maximum(n_idx_near + 1, 0), TILE_N), 0),
                n_block_min,
            )
            n_block_diag_min = tl.maximum(n_block_diag_min, n_block_sink_exclude_max)
            n_block_diag_max = n_block_max
            n_block_window_max = tl.maximum(n_block_window_max, n_block_window_min)
            n_block_diag_min = tl.maximum(n_block_diag_min, n_block_window_max)
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
    n_block_min = tl.minimum(n_block_min, n_block_max)
    if IS_LOCAL:
        n_block_window_max = tl.maximum(
            tl.minimum(n_block_window_max, n_block_min),
            n_block_window_min,
        )
        n_block_sink_max = tl.maximum(n_block_sink_max, n_block_sink_min)
        n_block_max = tl.maximum(
            tl.where(n_block_min >= n_block_max, 0, n_block_max),
            tl.maximum(
                tl.where(
                    n_block_window_min < n_block_window_max,
                    n_block_window_max,
                    0,
                ),
                tl.where(n_block_sink_min < n_block_sink_max, n_block_sink_max, 0),
            ),
        )
        n_block_min = tl.where(
            n_block_min >= n_block_max, tl.maximum(n_block_max - 1, 0), n_block_min
        )
        n_block_window_max = tl.maximum(
            n_block_window_min, tl.minimum(n_block_window_max, n_block_max - 1)
        )
        n_block_sink_max = tl.maximum(
            n_block_sink_min, tl.minimum(n_block_sink_max, n_block_max - 1)
        )
    else:
        n_block_window_min = 0
        n_block_window_max = 0
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
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    n_block: tl.tensor,
    split_idx: tl.tensor,
    window_size_sink: tl.tensor,
    window_size_left: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    NUM_SPLITS: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Compute the query block ranges processed by a backward key block.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param n_block: current block index along the N dimension
    :type n_block: tensor
    :param split_idx: index of the current QO split
    :type split_idx: tensor
    :param window_size_sink: prefix-sink token count
    :type window_size_sink: tensor
    :param window_size_left: distant local band token count
    :type window_size_left: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param NUM_SPLITS: number of QO splits
    :type NUM_SPLITS: int
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_CAUSAL: boolean flag indicating if the attention is causal
    :type IS_CAUSAL: bool
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool
    :param IS_SPLIT_QO: boolean flag indicating if the QO range is split
    :type IS_SPLIT_QO: bool

    :return m_block_min: minimum M block in the near-diagonal range
    :return m_block_max: maximum M block in the near-diagonal range
    :return m_block_window_min: minimum M block in the distant local-window range
    :return m_block_window_max: maximum M block in the distant local-window range
    :return m_block_sink_min: minimum M block in the sink range
    :return m_block_sink_max: maximum M block in the sink range
    """
    m_block_max = tl.cdiv(seqlen_q, TILE_M)
    m_block_min = tl.to_tensor(0)
    m_block_window_max = m_block_max
    m_block_window_min = m_block_min
    m_block_sink_min = tl.to_tensor(0)
    m_block_sink_max = tl.to_tensor(0)
    if IS_CAUSAL or IS_LOCAL:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_block_min = tl.maximum(m_block_min, m_idx // TILE_M)
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_near + window_size_right
            m_block_window_min = tl.maximum(m_block_window_min, m_idx_right // TILE_M)
    if IS_LOCAL:
        n_block_sink_exclude_max = tl.cdiv(window_size_sink, TILE_N)
        is_sink_block = n_block < n_block_sink_exclude_max
        n_idx_min = n_block * TILE_N
        m_idx_sink = n_idx_min + seqlen_q - seqlen_k
        m_block_sink_min = tl.maximum(m_idx_sink // TILE_M, 0)
        m_block_sink_max = tl.where(is_sink_block, tl.cdiv(seqlen_q, TILE_M) - 1, 0)
        m_block_sink_min = tl.where(is_sink_block, m_block_sink_min, 0)
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_near = m_idx + window_size_near
        m_block_max = tl.minimum(m_block_max, tl.maximum(tl.cdiv(m_idx_near, TILE_M), 0))
        m_idx_left = m_idx + window_size_near + window_size_right + window_size_left
        m_block_window_max = tl.minimum(
            m_block_window_max, tl.maximum(tl.cdiv(m_idx_left, TILE_M), 0)
        )
        m_block_min = tl.where(is_sink_block, m_block_sink_max, m_block_min)
        m_block_max = tl.where(is_sink_block, m_block_sink_max + 1, m_block_max)
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
            m_block_window_max = tl.minimum(m_block_window_min + win_cnt, m_block_window_max)
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
    m_block_min = tl.minimum(m_block_min, m_block_max)
    if IS_LOCAL:
        m_block_window_min = tl.minimum(
            tl.maximum(m_block_window_min, m_block_max),
            m_block_window_max,
        )
        m_block_sink_min = tl.minimum(m_block_sink_min, m_block_sink_max)
        m_block_max = tl.maximum(
            m_block_max,
            tl.where(m_block_window_min < m_block_window_max, m_block_window_max, 0),
        )
        m_block_window_max = tl.maximum(
            m_block_window_min, tl.minimum(m_block_window_max, m_block_max - 1)
        )
    else:
        m_block_window_min = 0
        m_block_window_max = 0
    return (
        m_block_min,
        m_block_max,
        m_block_window_min,
        m_block_window_max,
        m_block_sink_min,
        m_block_sink_max,
    )


@triton.jit
def get_n_block_max_no_causal_local_mask(
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    m_block: tl.tensor,
    n_block_min: tl.tensor,
    n_block_max: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
) -> tl.tensor:
    """
    Compute the maximum N block not requiring causal or local masking.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param m_block: current block index along the M dimension
    :type m_block: tensor
    :param n_block_min: minimum N block in the current range
    :type n_block_min: tensor
    :param n_block_max: maximum N block in the current range
    :type n_block_max: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_CAUSAL: boolean flag indicating if the attention is causal
    :type IS_CAUSAL: bool
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int

    :return: maximum N block that does not require causal or local masking
    """
    if not IS_CAUSAL and not IS_LOCAL:
        return n_block_max
    else:
        m_idx_min = m_block * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_right = n_idx if not IS_LOCAL else n_idx - window_size_near - window_size_right
        return tl.minimum(
            tl.maximum(n_block_min, n_idx_right // TILE_N),
            n_block_max,
        )


@triton.jit
def get_n_block_min_before_local_mask(
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    m_block: tl.tensor,
    n_block_min: tl.tensor,
    n_block_max: tl.tensor,
    window_size_left: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
) -> tl.tensor:
    """
    Compute the minimum N block not requiring local masking.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param m_block: current block index along the M dimension
    :type m_block: tensor
    :param n_block_min: minimum N block in the current local range
    :type n_block_min: tensor
    :param n_block_max: maximum N block in the current local range
    :type n_block_max: tensor
    :param window_size_left: distant local band token count
    :type window_size_left: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool
    :param QHEAD_PER_KVHEAD_PACKGQA: ratio of query heads to key/value heads for packed GQA
    :type QHEAD_PER_KVHEAD_PACKGQA: int

    :return: minimum N block that does not require local masking
    """
    if not IS_LOCAL:
        return n_block_min
    else:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = tl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = tl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_left = n_idx - window_size_near - window_size_right - window_size_left
        return tl.minimum(
            tl.maximum(n_block_min, tl.cdiv(n_idx_left, TILE_N)),
            n_block_max,
        )


@triton.jit
def get_m_block_min_no_causal_local_mask(
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    n_block: tl.tensor,
    m_block_min: tl.tensor,
    m_block_max: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
) -> tl.tensor:
    """
    Compute the minimum M block not requiring causal or local masking.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param n_block: current block index along the N dimension
    :type n_block: tensor
    :param m_block_min: minimum M block in the current range
    :type m_block_min: tensor
    :param m_block_max: maximum M block in the current range
    :type m_block_max: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_CAUSAL: boolean flag indicating if the attention is causal
    :type IS_CAUSAL: bool
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool

    :return: minimum M block that does not require causal or local masking
    """
    if not IS_CAUSAL and not IS_LOCAL:
        return m_block_min
    else:
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_near + window_size_right
        else:
            m_idx_right = m_idx
        return tl.minimum(
            tl.maximum(m_block_min, tl.cdiv(m_idx_right, TILE_M)),
            m_block_max,
        )


@triton.jit
def get_m_block_max_before_local_mask(
    seqlen_q: tl.tensor,
    seqlen_k: tl.tensor,
    n_block: tl.tensor,
    m_block_min: tl.tensor,
    m_block_max: tl.tensor,
    window_size_left: tl.tensor,
    window_size_right: tl.tensor,
    window_size_near: tl.tensor,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    IS_LOCAL: tl.constexpr,
) -> tl.tensor:
    """
    Compute the maximum M block not requiring local masking.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param n_block: current block index along the N dimension
    :type n_block: tensor
    :param m_block_min: minimum M block in the current local range
    :type m_block_min: tensor
    :param m_block_max: maximum M block in the current local range
    :type m_block_max: tensor
    :param window_size_left: distant local band token count
    :type window_size_left: tensor
    :param window_size_right: gap token count after the near-diagonal window
    :type window_size_right: tensor
    :param window_size_near: near-diagonal local token count
    :type window_size_near: tensor
    :param TILE_N: tile size along the N dimension
    :type TILE_N: int
    :param TILE_M: tile size along the M dimension
    :type TILE_M: int
    :param IS_LOCAL: boolean flag indicating if local attention is enabled
    :type IS_LOCAL: bool

    :return: maximum M block that does not require local masking
    """
    if not IS_LOCAL:
        return m_block_max
    else:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_idx_left = m_idx + window_size_near + window_size_right + window_size_left
        return tl.maximum(
            tl.minimum(m_idx_left // TILE_M, m_block_max),
            m_block_min,
        )
