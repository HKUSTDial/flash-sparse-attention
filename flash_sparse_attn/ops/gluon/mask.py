# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.constexpr_function
def get_mask_atom_stride(layout):
    assert isinstance(layout, gl.NVMMADistributedLayout), (
        "Packed masks require an NVIDIA MMA layout"
    )
    assert layout.version[0] in (2, 3), "Packed masks require MMAv2 or MMAv3"
    if layout.version[0] == 3:
        assert layout.warps_per_cta[1] == 1, (
            "Packed MMAv3 masks require warp groups along M"
        )
        return 2, 8
    return 2, 8 * layout.warps_per_cta[1]


@gluon.jit
def make_prefix_mask(
    col_limit,
    col_base,
    mask_atom: gl.constexpr,
    mask_stride: gl.constexpr,
):
    return gl.inline_asm_elementwise(
        f"""{{
        .reg .s32 offset, rank, remainder, shift;
        sub.s32 offset, $1, $2;
        shr.s32 rank, offset, {mask_stride.bit_length() - 1};
        shl.b32 rank, rank, {mask_atom.bit_length() - 1};
        and.b32 remainder, offset, {mask_stride - 1};
        min.s32 remainder, remainder, {mask_atom};
        add.s32 rank, rank, remainder;
        sub.s32 shift, 32, rank;
        max.s32 shift, shift, 0;
        shr.b32 $0, 0xffffffff, shift;
        }}""",
        "=r,r,r",
        [col_limit, col_base],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def make_interval_mask(
    col_limit_min,
    col_limit_max,
    col_base,
    mask_atom: gl.constexpr,
    mask_stride: gl.constexpr,
):
    return make_prefix_mask(
        col_limit_max, col_base, mask_atom, mask_stride
    ) & ~make_prefix_mask(col_limit_min, col_base, mask_atom, mask_stride)


@gluon.jit
def apply_mask(
    acc_S: gl.tensor,
    m_block: gl.tensor,
    n_block: gl.tensor,
    offs_m: gl.tensor,
    offs_n: gl.tensor,
    seqlen_q: gl.tensor,
    seqlen_k: gl.tensor,
    window_size_sink: gl.tensor,
    window_size_left: gl.tensor,
    window_size_right: gl.tensor,
    window_size_near: gl.tensor,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr,
    MASK_SINK: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    SWAP_AB: gl.constexpr,
) -> gl.tensor:
    """
    Apply seqlen, causal, and local masks to the attention scores.

    :param acc_S: attention scores of shape [BLOCK_M, BLOCK_N]
    :type acc_S: tensor
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

    :return acc_S: masked attention scores of shape [BLOCK_M, BLOCK_N]
    """
    gl.static_assert(
        not SWAP_AB or QHEAD_PER_KVHEAD_PACKGQA == 1,
        "SWAP_AB with PackGQA > 1 is not supported",
    )
    if not (MASK_SEQLEN or MASK_CAUSAL or MASK_LOCAL or MASK_SINK):
        return acc_S

    mask_atom: gl.constexpr = get_mask_atom_stride(acc_S.type.layout)[0]
    mask_stride: gl.constexpr = get_mask_atom_stride(acc_S.type.layout)[1]
    mask_cols: gl.constexpr = mask_stride * (32 // mask_atom)
    m_block_start = m_block * TILE_M
    q_base = m_block_start // QHEAD_PER_KVHEAD_PACKGQA
    k_base = n_block * TILE_N
    offs_m = (
        m_block_start % QHEAD_PER_KVHEAD_PACKGQA + offs_m
    ) // QHEAD_PER_KVHEAD_PACKGQA

    if SWAP_AB:
        q_idx = offs_m[None, :]
        k_idx = offs_n[:, None]
        offs_col = q_idx
        col_limit = seqlen_q - q_base
        row_valid = k_idx < seqlen_k - k_base
        col_limit_causal = (k_base - q_base - (seqlen_k - seqlen_q)) + k_idx
    else:
        q_idx = offs_m[:, None]
        k_idx = offs_n[None, :]
        offs_col = k_idx
        col_limit = seqlen_k - k_base
        row_valid = q_idx < seqlen_q - q_base
        col_limit_causal = (q_base + (seqlen_k - seqlen_q) - k_base) + q_idx + 1

    col_base = (
        offs_col % mask_stride // mask_atom * mask_atom
        + offs_col // mask_cols * mask_cols
    )

    if MASK_LOCAL or MASK_SINK:
        mask = gl.full(acc_S.shape, 0, gl.uint32, acc_S.type.layout)
        if MASK_LOCAL:
            if SWAP_AB:
                col_limit_min = col_limit_causal
                col_limit_max = col_limit_min + window_size_near
                mask |= make_interval_mask(
                    col_limit_min, col_limit_max, col_base, mask_atom, mask_stride
                )
                col_limit_min = col_limit_max + window_size_right
                col_limit_max = col_limit_min + window_size_left
            else:
                col_limit_max = col_limit_causal
                col_limit_min = col_limit_max - window_size_near
                mask |= make_interval_mask(
                    col_limit_min, col_limit_max, col_base, mask_atom, mask_stride
                )
                col_limit_max = col_limit_min - window_size_right
                col_limit_min = col_limit_max - window_size_left
            mask |= make_interval_mask(
                col_limit_min, col_limit_max, col_base, mask_atom, mask_stride
            )
        if MASK_SINK:
            if SWAP_AB:
                mask |= gl.where(
                    k_idx < window_size_sink - k_base,
                    ~make_prefix_mask(
                        col_limit_causal, col_base, mask_atom, mask_stride
                    ),
                    0,
                )
            else:
                mask |= make_prefix_mask(
                    gl.minimum(col_limit_causal, window_size_sink - k_base),
                    col_base,
                    mask_atom,
                    mask_stride,
                )
    elif MASK_CAUSAL:
        mask = make_prefix_mask(col_limit_causal, col_base, mask_atom, mask_stride)
        if SWAP_AB:
            mask = ~mask
    else:
        mask = gl.full(acc_S.shape, 0xFFFFFFFF, gl.uint32, acc_S.type.layout)

    if MASK_SEQLEN:
        mask &= make_prefix_mask(col_limit, col_base, mask_atom, mask_stride)
        mask = gl.where(row_valid, mask, 0)

    mask_bit = (
        1 << ((offs_col % mask_cols) // mask_stride * mask_atom + offs_col % mask_atom)
    ).to(gl.uint32)
    # TODO: pack=8 is slower yet to be optimized
    return gl.inline_asm_elementwise(
        """{
            .reg .pred valid;
            .reg .b32 bit;
            and.b32 bit, $2, $3;
            setp.ne.u32 valid, bit, 0;
            selp.f32 $0, $1, 0fFF800000, valid;
        }""",
        "=f,f,r,r",
        [acc_S, mask, mask_bit],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
