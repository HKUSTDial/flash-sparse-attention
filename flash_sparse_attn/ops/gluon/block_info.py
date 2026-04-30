"""
SM100 (Blackwell) block-level N/M range helpers for attention tile scheduling.

Gluon equivalents of the Triton block_info helpers (ops/triton/block_info.py),
stripped of split_kv and pack_gqa branches that are not used in the Gluon path.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_n_block_min_max(
    seqlen_q,
    seqlen_k,
    m_block,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    WINDOW_SIZE_LEFT: gl.constexpr,
    WINDOW_SIZE_RIGHT: gl.constexpr,
):
    n_block_max = gl.cdiv(seqlen_k, TILE_N)
    if IS_CAUSAL or (IS_LOCAL and WINDOW_SIZE_RIGHT is not None):
        m_idx_max = (m_block + 1) * TILE_M
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_right = n_idx if IS_CAUSAL else n_idx + WINDOW_SIZE_RIGHT
        n_block_max = gl.minimum(n_block_max, gl.cdiv(n_idx_right, TILE_N))
    n_block_min = gl.to_tensor(0)
    if IS_LOCAL and WINDOW_SIZE_LEFT is not None:
        m_idx_min = m_block * TILE_M
        n_idx = m_idx_min + seqlen_k - seqlen_q
        n_idx_left = n_idx - WINDOW_SIZE_LEFT
        n_block_min = gl.maximum(n_idx_left // TILE_N, gl.to_tensor(0))
    return n_block_min, n_block_max


@gluon.jit
def get_n_block_min_causal_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    WINDOW_SIZE_RIGHT: gl.constexpr,
):
    m_idx_min = m_block * TILE_M
    n_idx = m_idx_min + seqlen_k - seqlen_q
    n_idx_right = (
        n_idx
        if (not IS_LOCAL or WINDOW_SIZE_RIGHT is None)
        else n_idx + WINDOW_SIZE_RIGHT
    )
    return gl.maximum(n_block_min, n_idx_right // TILE_N)


@gluon.jit
def get_n_block_min_before_local_mask(
    seqlen_q,
    seqlen_k,
    m_block,
    n_block_min,
    TILE_N: gl.constexpr,
    TILE_M: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    WINDOW_SIZE_LEFT: gl.constexpr,
):
    if not IS_LOCAL or WINDOW_SIZE_LEFT is None:
        return n_block_min
    else:
        m_idx_max = (m_block + 1) * TILE_M
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_left = n_idx - WINDOW_SIZE_LEFT
        return gl.maximum(n_block_min, gl.cdiv(n_idx_left, TILE_N))
