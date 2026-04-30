"""
SM100 (Blackwell) attention mask using bitmask trick for R2P layout.

Unified apply_mask with MASK_CAUSAL / MASK_LOCAL flags,
matching the Triton mask.apply_mask interface.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _mask_scalar_right(acc_s, col_limit_right, s, i):
    col_lim_right_s = col_limit_right - s
    col_lim_right_cur = max(col_lim_right_s, 0)
    mask = -1 << col_lim_right_cur
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, acc_s, -float("inf"))


@gluon.jit
def _mask_scalar_left(acc_s, col_limit_left, s, i):
    col_lim_left_s = col_limit_left - s
    col_lim_left_cur = min(max(col_lim_left_s, 0), 16)
    mask = (1 << col_lim_left_cur) - 1
    mask_i_bit = (mask & (1 << i)) == 0
    return gl.where(mask_i_bit, acc_s, -float("inf"))


@gluon.jit
def apply_mask(
    acc_s,
    offs_m,
    start_n,
    seqlen_q,
    seqlen_k,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr,
    WINDOW_SIZE_LEFT: gl.constexpr,
    WINDOW_SIZE_RIGHT: gl.constexpr,
):
    offs_n = gl.arange(0, acc_s.shape[1])[None, :]
    s = offs_n & ~0xF
    i = offs_n & 0xF
    causal_offset = seqlen_k - seqlen_q

    if MASK_SEQLEN:
        col_limit_right = seqlen_k - start_n
        acc_s = gl.map_elementwise(_mask_scalar_right, acc_s, col_limit_right, s, i)

    if MASK_CAUSAL:
        col_limit_right = (offs_m + causal_offset - start_n + 1)[:, None]
        acc_s = gl.map_elementwise(_mask_scalar_right, acc_s, col_limit_right, s, i)

    if MASK_LOCAL:
        if WINDOW_SIZE_RIGHT is not None:
            col_limit_right = (
                offs_m + causal_offset + WINDOW_SIZE_RIGHT - start_n + 1
            )[:, None]
            acc_s = gl.map_elementwise(_mask_scalar_right, acc_s, col_limit_right, s, i)
        if WINDOW_SIZE_LEFT is not None:
            col_limit_left = (offs_m + causal_offset - WINDOW_SIZE_LEFT - start_n)[
                :, None
            ]
            acc_s = gl.map_elementwise(_mask_scalar_left, acc_s, col_limit_left, s, i)

    return acc_s
