# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_n_block_min_max(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    m_block: gl.tensor,
    split_idx: gl.tensor,
    window_size_sink: gl.tensor,
    window_size_left: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    NUM_SPLITS: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    IS_SPLIT_KV: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
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
    n_block_max = gl.cdiv(seqlen_k, TILE_N)
    n_block_min = gl.to_tensor(0)
    n_block_window_max = n_block_max
    n_block_window_min = n_block_min
    n_block_sink_min = gl.to_tensor(0)
    n_block_sink_max = gl.to_tensor(0)
    if IS_CAUSAL or IS_LOCAL:
        m_idx_max = (m_block + 1) * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_max = gl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = gl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_block_max = gl.minimum(n_block_max, gl.cdiv(n_idx, TILE_N))
        if IS_LOCAL:
            n_idx_right = n_idx - window_size_near - window_size_right
            n_block_window_max = gl.minimum(
                n_block_window_max, gl.maximum(gl.cdiv(n_idx_right, TILE_N), 0)
            )
    if IS_LOCAL:
        n_block_sink_max = gl.minimum(
            gl.cdiv(window_size_sink, TILE_N),
            gl.maximum(gl.cdiv(n_idx, TILE_N), 0),
        )
        n_block_sink_exclude_max = gl.cdiv(window_size_sink, TILE_N)
        m_idx_min = m_block * TILE_M
        if QHEAD_PER_KVHEAD_PACKGQA > 1:
            m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_near = n_idx - window_size_near
        n_block_min = gl.maximum(n_idx_near // TILE_N, 0)
        n_block_min = gl.maximum(n_block_min, n_block_sink_exclude_max)
        n_idx_left = n_idx - window_size_near - window_size_right - window_size_left
        n_block_window_min = gl.maximum(n_idx_left // TILE_N, 0)
        n_block_window_min = gl.maximum(n_block_window_min, n_block_sink_exclude_max)
    if IS_SPLIT_KV:
        if IS_LOCAL:
            n_block_diag_min = gl.where(
                seqlen_q == 1,
                gl.maximum(gl.cdiv(gl.maximum(n_idx_near + 1, 0), TILE_N), 0),
                n_block_min,
            )
            n_block_diag_min = gl.maximum(n_block_diag_min, n_block_sink_exclude_max)
            n_block_diag_max = n_block_max
            n_block_window_max = gl.maximum(n_block_window_max, n_block_window_min)
            total_n_blocks = gl.maximum(n_block_window_max - n_block_window_min, 0)
            base = total_n_blocks // NUM_SPLITS
            extra = total_n_blocks % NUM_SPLITS
            n_block_window_min_new = n_block_window_min + gl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            n_block_count = gl.where(split_idx < extra, base + 1, base)
            n_block_window_max = gl.minimum(
                n_block_window_min_new + n_block_count, n_block_window_max
            )
            n_block_window_min = n_block_window_min_new
            n_block_sink_max = gl.where(split_idx == 0, n_block_sink_max, 0)
            n_block_non_diag_max = gl.maximum(n_block_window_max, n_block_sink_max)
            n_block_max = gl.where(
                split_idx >= NUM_SPLITS - 1,
                n_block_diag_max,
                n_block_non_diag_max,
            )
            n_block_min = gl.where(
                split_idx >= NUM_SPLITS - 1,
                n_block_diag_min,
                n_block_non_diag_max,
            )
        else:
            total_n_blocks = gl.maximum(n_block_max - n_block_min, 0)
            base = total_n_blocks // NUM_SPLITS
            extra = total_n_blocks % NUM_SPLITS
            n_block_min_new = n_block_min + gl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            n_block_count = gl.where(split_idx < extra, base + 1, base)
            n_block_max = gl.minimum(n_block_min_new + n_block_count, n_block_max)
            n_block_min = n_block_min_new
            n_block_sink_max = gl.where(split_idx == 0, n_block_sink_max, 0)
    n_block_min = gl.minimum(n_block_min, n_block_max)
    if IS_LOCAL:
        n_block_window_max = gl.maximum(
            gl.minimum(n_block_window_max, n_block_min),
            n_block_window_min,
        )
        n_block_sink_max = gl.maximum(n_block_sink_max, n_block_sink_min)
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


@gluon.jit
def get_m_block_min_max(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    n_block: gl.tensor,
    split_idx: gl.tensor,
    window_size_sink: gl.tensor,
    window_size_left: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    NUM_SPLITS: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    IS_SPLIT_QO: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
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
    m_block_max = gl.cdiv(seqlen_q, TILE_M)
    m_block_min = gl.to_tensor(0)
    m_block_window_max = m_block_max
    m_block_window_min = m_block_min
    m_block_sink_min = gl.to_tensor(0)
    m_block_sink_max = gl.to_tensor(0)
    if IS_CAUSAL or IS_LOCAL:
        n_idx_min = n_block * TILE_N
        m_idx = n_idx_min + seqlen_q - seqlen_k
        m_block_min = gl.maximum(m_block_min, m_idx // TILE_M)
        if IS_LOCAL:
            m_idx_right = m_idx + window_size_near + window_size_right
            m_block_window_min = gl.maximum(m_block_window_min, m_idx_right // TILE_M)
    if IS_LOCAL:
        n_block_sink_exclude_max = gl.cdiv(window_size_sink, TILE_N)
        is_sink_block = n_block < n_block_sink_exclude_max
        n_idx_min = n_block * TILE_N
        m_idx_sink = n_idx_min + seqlen_q - seqlen_k
        m_block_sink_min = gl.maximum(m_idx_sink // TILE_M, 0)
        m_block_sink_max = gl.where(is_sink_block, gl.cdiv(seqlen_q, TILE_M), 0)
        m_block_sink_min = gl.where(is_sink_block, m_block_sink_min, 0)
        n_idx_max = (n_block + 1) * TILE_N
        m_idx = n_idx_max + seqlen_q - seqlen_k
        m_idx_near = m_idx + window_size_near
        m_block_max = gl.minimum(m_block_max, gl.cdiv(m_idx_near, TILE_M))
        m_idx_left = m_idx + window_size_near + window_size_right + window_size_left
        m_block_window_max = gl.minimum(m_block_window_max, gl.cdiv(m_idx_left, TILE_M))
        m_block_min = gl.where(is_sink_block, 0, m_block_min)
        m_block_max = gl.where(is_sink_block, 0, m_block_max)
        m_block_window_min = gl.where(is_sink_block, 0, m_block_window_min)
        m_block_window_max = gl.where(is_sink_block, 0, m_block_window_max)
    if IS_SPLIT_QO:
        if IS_LOCAL:
            n_idx_max = (n_block + 1) * TILE_N
            m_idx = n_idx_max + seqlen_q - seqlen_k
            m_block_min_no_mask = gl.maximum(m_block_min, gl.cdiv(m_idx, TILE_M))
            m_block_window_min = gl.maximum(m_block_window_min, m_block_min_no_mask)
            win_blocks = gl.maximum(m_block_window_max - m_block_window_min, 0)
            base = win_blocks // NUM_SPLITS
            extra = win_blocks % NUM_SPLITS
            win_off = gl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            win_cnt = gl.where(split_idx < extra, base + 1, base)
            m_block_window_min = m_block_window_min + win_off
            m_block_window_max = gl.minimum(
                m_block_window_min + win_cnt, m_block_window_max
            )
            # First split keeps diagonal, others skip it
            m_block_min = gl.where(split_idx == 0, m_block_min, m_block_max)
            m_block_sink_max = gl.where(split_idx == 0, m_block_sink_max, 0)
            m_block_sink_min = gl.where(split_idx == 0, m_block_sink_min, 0)
        else:
            total_m_blocks = gl.maximum(m_block_max - m_block_min, 0)
            base = total_m_blocks // NUM_SPLITS
            extra = total_m_blocks % NUM_SPLITS
            m_block_min_new = m_block_min + gl.where(
                split_idx < extra,
                split_idx * (base + 1),
                extra * (base + 1) + (split_idx - extra) * base,
            )
            m_block_count = gl.where(split_idx < extra, base + 1, base)
            m_block_max = gl.minimum(m_block_min_new + m_block_count, m_block_max)
            m_block_min = m_block_min_new
    m_block_min = gl.minimum(m_block_min, m_block_max)
    if IS_LOCAL:
        m_block_window_min = gl.minimum(
            gl.maximum(m_block_window_min, m_block_max),
            m_block_window_max,
        )
        m_block_sink_min = gl.minimum(m_block_sink_min, m_block_sink_max)
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


@gluon.jit
def get_n_block_max_no_causal_local_mask(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    m_block: gl.tensor,
    n_block_min: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
) -> gl.tensor:
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

    :return: maximum N block that does not require causal or local masking
    """
    m_idx_min = m_block * TILE_M
    if QHEAD_PER_KVHEAD_PACKGQA > 1:
        m_idx_min = m_idx_min // QHEAD_PER_KVHEAD_PACKGQA
    n_idx = m_idx_min + seqlen_k - seqlen_q
    n_idx_right = (
        n_idx if not IS_LOCAL else n_idx - window_size_near - window_size_right
    )
    return gl.maximum(n_block_min, n_idx_right // TILE_N)


@gluon.jit
def get_n_block_min_before_local_mask(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    m_block: gl.tensor,
    n_block_min: gl.tensor,
    window_size_left: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
) -> gl.tensor:
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
            m_idx_max = gl.cdiv(m_idx_max, QHEAD_PER_KVHEAD_PACKGQA)
        m_idx_max = gl.minimum(m_idx_max, seqlen_q)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_left = n_idx - window_size_near - window_size_right - window_size_left
        return gl.maximum(n_block_min, gl.cdiv(n_idx_left, TILE_N))


@gluon.jit
def get_m_block_min_no_causal_local_mask(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    n_block: gl.tensor,
    m_block_min: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
) -> gl.tensor:
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
        return gl.maximum(m_block_min, gl.cdiv(m_idx_right, TILE_M))


@gluon.jit
def get_m_block_max_before_local_mask(
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    n_block: gl.tensor,
    m_block_max: gl.tensor,
    window_size_left: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_LOCAL: gl.constexpr,
) -> gl.tensor:
    """
    Compute the maximum M block not requiring local masking.

    :param seqlen_q: sequence length of the query
    :type seqlen_q: tensor
    :param seqlen_k: sequence length of the key
    :type seqlen_k: tensor
    :param n_block: current block index along the N dimension
    :type n_block: tensor
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
        return gl.minimum(m_block_max, m_idx_left // TILE_M)
