import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    float2,
    mbarrier,
)
from triton.experimental.gluon.language.nvidia.blackwell.float2 import Float2Tensor

from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from flash_sparse_attn.ops.gluon.utils import (
    borrow_s_as_p,
    borrow_s_as_row_scale,
    borrow_s_for_finalize,
    get_split_n_layout,
    split_n,
    compute_and_store_exp2,
    subtiled_s_load,
)
from flash_sparse_attn.ops.gluon.mask import apply_mask
from flash_sparse_attn.ops.gluon.scheduling import ProgramScheduler


@triton.jit
def check_inf(x):
    return tl.where(x == float("-inf"), 0.0, x)


@triton.jit
def online_sparse_softmax(
    acc_s,
    block_max,
    row_max,
    row_sum,
    scale_log2,
    softmax_threshold_log2,
    CHECK_INF: tl.constexpr,
):
    """
    Apply online sparse softmax to acc_s, and update block_max, row_max and row_sum.

    :param acc_s: Attention scores tensor of shape [TILE_M, TILE_N].
    :param block_max: Running block-wise maximum scalar, init to -inf.
    :param row_max: Current maximum values per row of shape [TILE_M], init to -inf.
    :param row_sum: Current sum values per row of shape [TILE_M], init to 0.
    :param scale_log2: Log2 of the scaling factor to be applied to acc_s.
    :param softmax_threshold_log2: Threshold in log2-domain for block-level skip. If > -inf and block max is below threshold relative to running max, skip softmax update.
    :param CHECK_INF: Boolean flag indicating if -inf row_max should be clamped to 0.

    :return p: Softmax probabilities tensor of shape [TILE_M, TILE_N].
    :return block_max_new: Updated block-wise maximum scalar.
    :return row_max_new: Updated maximum values per row of shape [TILE_M].
    :return row_sum_new: Updated sum values per row of shape [TILE_M].
    :return row_scale: Scaling factors per row of shape [TILE_M].
    :return skip_softmax: Boolean indicating whether this block was skipped.
    """
    # Compute current block max
    block_max_curr = tl.max(acc_s)

    # Update skip condition based on threshold
    block_max_diff_log2 = (block_max_curr - block_max) * scale_log2
    skip_softmax = block_max_diff_log2 < softmax_threshold_log2

    # Return zero attention weights
    if skip_softmax:
        p = acc_s * 0.0
        block_max_new = block_max
        row_max_new = row_max
        row_sum_new = row_sum
        row_scale = row_max * 0.0 + 1.0
    else:
        # Compute current row max
        row_max_curr = tl.max(acc_s, axis=1)

        # Update block max
        block_max_new = tl.maximum(block_max_curr, block_max)

        # Update row max
        row_max_new = tl.maximum(row_max_curr, row_max)

        # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
        if CHECK_INF:
            row_max_new = check_inf(row_max_new)

        # Compute scaled differences to new row max
        acc_scale_log2 = (row_max - row_max_new) * scale_log2

        # Compute row scale
        row_scale = tl.exp2(acc_scale_log2)

        # Compute attention weights
        p = tl.exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

        # Update row sum
        row_sum_cur = tl.sum(p, axis=1)
        row_sum_new = row_sum * row_scale + row_sum_cur

    return p, block_max_new, row_max_new, row_sum_new, row_scale, skip_softmax


@gluon.jit
def finalize(
    config,
    prog,
    s_tmem,
    Lse,
    scale_consumer,
    o_smem_producer,
    o_tmem_consumer,
    final_scale,
    CHECK_NAN: gl.constexpr,
    tile_id: gl.constexpr = 0,
):
    """
    Finalize online softmax: scale O by final_scale/row_sum, store to SMEM, write LSE.

    :param config: AttentionConfig with layout and shape info.
    :param prog: AttentionProgram with m_block and head_idx.
    :param s_tmem: Shared TMEM buffer holding row_max and row_sum.
    :param Lse: Pointer to LSE output buffer.
    :param scale_consumer: Scale mbarrier channel consumer.
    :param o_smem_producer: O SMEM channel producer.
    :param o_tmem_consumer: O TMEM channel consumer.
    :param final_scale: Scaling factor applied to output (1.0 for fp16/bf16, descale for fp8).
    :param CHECK_NAN: If True, guard against row_sum==0 (fully masked rows).
    :param tile_id: Tile index for split-M offset (0 or 1).

    :return: Updated (scale_consumer, o_smem_producer, o_tmem_consumer).
    """
    row_scale_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    _, scale_bar, scale_consumer = scale_consumer.acquire()
    row_max_tmem, row_sum_tmem = borrow_s_for_finalize(config, s_tmem)
    row_max = row_max_tmem.load(config.row_scale_tmem_layout).reshape([config.SPLIT_M])
    row_max = gl.convert_layout(row_max, row_scale_layout)
    row_sum = row_sum_tmem.load(config.row_scale_tmem_layout).reshape([config.SPLIT_M])
    row_sum = gl.convert_layout(row_sum, row_scale_layout)
    mbarrier.arrive(scale_bar, count=1)

    if CHECK_NAN:
        row_sum = gl.where(row_sum == 0.0, 1.0, row_sum)

    o_smem_raw, o_smem_bar, o_smem_producer = o_smem_producer.acquire()
    o_tmem, o_bar, o_tmem_consumer = o_tmem_consumer.acquire()

    o_smem = o_smem_raw.reshape([o_smem_raw.shape[-2], o_smem_raw.shape[-1]])

    contigDimSize: gl.constexpr = (
        o_smem.type.layout.swizzle_byte_width
        * 8
        // o_smem.type.element_ty.primitive_bitwidth
    )
    if o_smem.type.shape[1] // config.SPLIT_K_FACTOR >= contigDimSize:
        SPLIT_N_FACTOR: gl.constexpr = config.SPLIT_K_FACTOR
    else:
        SPLIT_N_FACTOR: gl.constexpr = 1
    gl.static_assert(
        o_smem.type.shape[1] // SPLIT_N_FACTOR >= contigDimSize,
        "Block shape is too small for the swizzle byte size in NVMMA Shared Layout",
    )
    SPLIT_N: gl.constexpr = o_smem.type.shape[1] // SPLIT_N_FACTOR

    scale = float2.pack(
        (final_scale / row_sum)[:, None].broadcast_to(config.o_tmem_shape[0], SPLIT_N),
        axis=1,
    )
    for i in gl.static_range(SPLIT_N_FACTOR):
        o_tmem_slice = o_tmem.slice(i * SPLIT_N, SPLIT_N)
        o = float2.pack(o_tmem_slice.load(config.o_splitn_layout), axis=1)
        o = o * scale
        o_smem.slice(i * SPLIT_N, SPLIT_N, dim=1).store(
            float2.unpack(o, axis=1).to(config.dtype)
        )

    fence_async_shared()
    mbarrier.arrive(o_smem_bar, count=1)
    mbarrier.arrive(o_bar, count=1)

    row_max += gl.log2(row_sum)
    coalesced: gl.constexpr = gl.BlockedLayout([1], [32], [config.num_warps], [0])
    offs_m = prog.m_block * config.TILE_M + tile_id * config.SPLIT_M
    offs_m += gl.arange(0, config.SPLIT_M, coalesced)
    lse_ptrs = Lse + prog.batch_head_idx * config.seqlen_q + offs_m
    gl.store(lse_ptrs, gl.convert_layout(row_max, coalesced))

    return scale_consumer, o_smem_producer, o_tmem_consumer


@gluon.jit
def rescale_o(config, s_tmem, scale_consumer, o_tmem_consumer):
    """
    Rescale output accumulator by row_scale loaded from TMEM.

    Acquires O TMEM and scale mbarrier channels, loads row_scale from shared TMEM,
    then applies element-wise scaling to each O split in TMEM.

    :param config: AttentionConfig with layout and shape info.
    :param s_tmem: Shared TMEM buffer holding row_scale.
    :param scale_consumer: Scale mbarrier channel consumer.
    :param o_tmem_consumer: O TMEM channel consumer.

    :return: Updated (scale_consumer, o_tmem_consumer).
    """
    row_scale_layout: gl.constexpr = gl.SliceLayout(1, config.o_splitn_layout)

    o_tmem, o_bar, o_tmem_consumer = o_tmem_consumer.acquire()

    _, scale_bar, scale_consumer = scale_consumer.acquire()
    row_scale = borrow_s_as_row_scale(config, s_tmem).load(config.row_scale_tmem_layout)
    mbarrier.arrive(scale_bar, count=1)
    row_scale = gl.convert_layout(row_scale.reshape([config.SPLIT_M]), row_scale_layout)

    row_scale = float2.pack(
        row_scale[:, None].broadcast_to(config.o_tmem_shape[0], config.SPLIT_K), axis=1
    )
    for i in gl.static_range(config.SPLIT_K_FACTOR):
        o_tmem_slice = o_tmem.slice(i * config.SPLIT_K, config.SPLIT_K)
        o = float2.pack(o_tmem_slice.load(config.o_splitn_layout), axis=1)
        o = o * row_scale
        o_tmem_slice.store(float2.unpack(o, axis=1))
    mbarrier.arrive(o_bar, count=1)
    return scale_consumer, o_tmem_consumer


@triton.jit
def log_sigmoid(x, FASTMATH: tl.constexpr):
    """
    Compute log-sigmoid of x.

    :param x: Input tensor of shape [TILE_M, TILE_N].
    :param FASTMATH: Boolean flag indicating if the fast approximation should be used.

    :return: Tensor of shape [TILE_M, TILE_N] containing log-sigmoid values.
    """
    if FASTMATH:
        xc = tl.minimum(tl.abs(x), 4.0)
        xc2 = xc * xc
        out = tl.minimum(x, 0.0) - 0.05674870 * xc2 + 0.37664706 * xc - 0.65169323
        return out
    else:
        out = tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))
        return out


@triton.jit
def online_gate(
    a_max,
    a_min,
    d_max,
    d_min,
    gate_max,
    scale_log2,
    gate_threshold_log2,
):
    """
    Determine whether to skip gate computation for the current tile based on the maximum possible gate value.

    :param a_max: Maximum value of the Alpha tile.
    :param a_min: Minimum value of the Alpha tile.
    :param d_max: Maximum value of the Delta tile.
    :param d_min: Minimum value of the Delta tile.
    :param gate_max: Maximum value of the gate.
    :param scale_log2: Log2 of the scaling factor applied to the gate max.
    :param gate_threshold_log2: Threshold in log2-domain for gate-level skip.

    :return gate_max_new: Updated gate max value after considering current tile.
    :return skip_gate: Boolean indicating whether to skip the gate computation for this tile.
    """
    gate_max_curr = tl.maximum(
        tl.maximum(a_max * d_max, a_max * d_min),
        tl.maximum(a_min * d_max, a_min * d_min),
    )
    gate_max_diff_log2 = (gate_max_curr - gate_max) * scale_log2
    skip_gate = gate_max_diff_log2 < gate_threshold_log2
    if skip_gate:
        gate_max_new = gate_max
    else:
        gate_max_new = tl.maximum(gate_max_curr, gate_max)
    return gate_max_new, skip_gate


# ===-----------------------------------------------------------------------===#
# Gluon SM100: Online Softmax (Dense)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def online_softmax_inner(
    config,
    s_consumer,
    scale_producer,
    exp_turnstile,
    scale_bar,
    offs_m,
    row_max,
    row_sum,
    n_start,
    n_end,
    IS_MASK: gl.constexpr,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr,
    CHECK_INF: gl.constexpr,
    use_tmem_red: gl.constexpr,
):
    """
    Inner N-dimension loop of Gluon online softmax for one mask segment.

    Iterates over N blocks in reverse order, loading QK scores from TMEM,
    optionally applying masks, computing row-wise max/scale/exp2/sum updates,
    and synchronizing with rescale and MMA partitions via mbarrier.

    :param config: AttentionConfig with tile shapes, scale, and layout info.
    :param s_consumer: S TMEM channel consumer for QK score buffers.
    :param scale_producer: Scale mbarrier channel producer for row_scale writeback.
    :param exp_turnstile: Exp2 turnstile mbarrier for pipelining (may be producer or consumer).
    :param scale_bar: Current scale mbarrier token.
    :param offs_m: Row offsets tensor of shape [SPLIT_M].
    :param row_max: Running per-row max of shape [SPLIT_M], init to -inf.
    :param row_sum: Running per-row sum as Float2Tensor of shape [SPLIT_M], init to 0.
    :param n_start: Start offset in N dimension (element offset, not block index).
    :param n_end: End offset in N dimension (element offset, not block index).
    :param IS_MASK: Whether to apply any mask in this segment.
    :param MASK_SEQLEN: Whether to apply sequence length boundary mask.
    :param MASK_CAUSAL: Whether to apply causal (upper-triangular) mask.
    :param MASK_LOCAL: Whether to apply sliding window mask.
    :param CHECK_INF: Whether to clamp -inf row_max to 0 (needed for masked segments).
    :param use_tmem_red: Whether to use TMEM hardware reduction for per-row max.

    :return: (row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile).
    """
    num_blocks: gl.constexpr = (n_end - n_start) // config.TILE_N
    for i in range(num_blocks):
        start_n = n_end - (1 + i) * config.TILE_N
        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        acc_s, acc_s_max = subtiled_s_load(config, s_tmem, use_tmem_red)

        if IS_MASK:
            acc_s = apply_mask(
                acc_s,
                offs_m,
                start_n,
                config.seqlen_q,
                config.seqlen_k,
                MASK_SEQLEN=MASK_SEQLEN,
                MASK_CAUSAL=MASK_CAUSAL,
                MASK_LOCAL=MASK_LOCAL,
                WINDOW_SIZE_LEFT=config.WINDOW_SIZE_LEFT,
                WINDOW_SIZE_RIGHT=config.WINDOW_SIZE_RIGHT,
            )

        if use_tmem_red:
            acc_s_max = gl.convert_layout(acc_s_max, row_max.type.layout)
            row_max_new = gl.maximum(row_max, acc_s_max * config.softmax_scale_log2)
        else:
            row_max_new = gl.maximum(
                row_max, gl.max(acc_s, 1) * config.softmax_scale_log2
            )
        if CHECK_INF:
            row_max_new = gl.where(row_max_new == -float("inf"), 0.0, row_max_new)
        row_scale = gl.exp2(row_max - row_max_new)

        row_scale_tmem = borrow_s_as_row_scale(config, s_tmem)
        row_scale_tmem.store(
            gl.convert_layout(row_scale.expand_dims(1), config.row_scale_tmem_layout)
        )
        mbarrier.arrive(scale_bar, count=1)

        neg_row_max_scaled = float2.pack(
            -row_max_new[:, None].broadcast_to(acc_s.shape), axis=1
        )
        acc_s = float2.pack(acc_s, axis=1)
        acc_s = float2.fma(
            acc_s,
            float2.full_like(acc_s, config.softmax_scale_log2),
            neg_row_max_scaled,
        )
        acc_s = float2.unpack(acc_s, axis=1)

        if config.use_exp2_turnstile:
            _, exp_bar, exp_turnstile = exp_turnstile.acquire()

        p_tmem = borrow_s_as_p(config, s_tmem)
        p = compute_and_store_exp2(config, acc_s, p_tmem)

        mbarrier.arrive(s_bar, count=1)
        _, scale_bar, scale_producer = scale_producer.acquire()

        if config.use_exp2_turnstile:
            mbarrier.arrive(exp_bar, count=1)

        row_sum_cur = float2.pack2(*split_n(p)).sum(axis=1)
        row_sum_cur = Float2Tensor(
            gl.convert_layout(
                row_sum_cur.value, row_sum.value.type.layout, assert_trivial=True
            )
        )
        row_scale = gl.convert_layout(
            row_scale, row_sum.value.type.layout, assert_trivial=True
        )
        row_sum = float2.fma(row_sum, float2.pack2(row_scale, row_scale), row_sum_cur)
        row_max = row_max_new

    return row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile


@gluon.jit
def online_softmax(
    tile_id: gl.constexpr,
    config,
    s_chnl,
    corr_chnl,
    exp_turnstile,
    use_tmem_red: gl.constexpr,
):
    """
    Tile-level Gluon online softmax for one SPLIT_M half of the M dimension.

    Iterates over program tiles, computes 3-segment masked softmax (right-mask,
    no-mask, left-mask) via online_softmax_inner, then writes final row_max and
    row_sum to TMEM for the correction/finalize partition.

    :param tile_id: Which SPLIT_M half (0 or 1).
    :param config: AttentionConfig with tile shapes, mask flags, and layout info.
    :param s_chnl: S channel carrying QK score TMEM buffers from MMA partition.
    :param corr_chnl: Correction channel for row_scale / row_max / row_sum writeback.
    :param exp_turnstile: Exp2 turnstile for pipelining between softmax0 and softmax1.
    :param use_tmem_red: Whether to use TMEM hardware reduction for per-row max.
    """
    s_slice_dim1: gl.constexpr = gl.SliceLayout(1, config.s_instr_layout)
    sum_layout: gl.constexpr = get_split_n_layout(config.s_instr_layout)

    s_consumer = s_chnl.create_consumer()
    scale_producer = corr_chnl.create_producer()
    _, scale_bar, scale_producer = scale_producer.acquire()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        offs_m = prog.m_block * config.TILE_M
        offs_m += gl.arange(tile_id * config.SPLIT_M, (1 + tile_id) * config.SPLIT_M)

        row_max = gl.full([config.SPLIT_M], -float("inf"), gl.float32, s_slice_dim1)
        row_sum = gl.full(
            [config.SPLIT_M], 0.0, gl.float32, gl.SliceLayout(1, sum_layout)
        )
        row_sum = float2.pack2(row_sum, row_sum)

        n_block_min, n_block_min_no_mask, n_block_max_no_mask, n_block_max = (
            prog.get_seg_bounds()
        )

        if config.IS_CAUSAL or config.IS_LOCAL:
            row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile = (
                online_softmax_inner(
                    config,
                    s_consumer,
                    scale_producer,
                    exp_turnstile,
                    scale_bar,
                    offs_m,
                    row_max,
                    row_sum,
                    n_start=n_block_max_no_mask,
                    n_end=n_block_max,
                    IS_MASK=True,
                    MASK_SEQLEN=True,
                    MASK_CAUSAL=config.IS_CAUSAL,
                    MASK_LOCAL=config.IS_LOCAL,
                    CHECK_INF=True,
                    use_tmem_red=use_tmem_red,
                )
            )
        else:
            row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile = (
                online_softmax_inner(
                    config,
                    s_consumer,
                    scale_producer,
                    exp_turnstile,
                    scale_bar,
                    offs_m,
                    row_max,
                    row_sum,
                    n_start=n_block_max - config.TILE_N,
                    n_end=n_block_max,
                    IS_MASK=True,
                    MASK_SEQLEN=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    CHECK_INF=True,
                    use_tmem_red=use_tmem_red,
                )
            )
            n_block_max_no_mask = n_block_max - config.TILE_N
            n_block_min_no_mask = gl.minimum(n_block_min_no_mask, n_block_max_no_mask)

        row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile = (
            online_softmax_inner(
                config,
                s_consumer,
                scale_producer,
                exp_turnstile,
                scale_bar,
                offs_m,
                row_max,
                row_sum,
                n_start=n_block_min_no_mask,
                n_end=n_block_max_no_mask,
                IS_MASK=False,
                MASK_SEQLEN=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                CHECK_INF=config.IS_LOCAL,
                use_tmem_red=use_tmem_red,
            )
        )

        if config.IS_LOCAL:
            row_max, row_sum, scale_bar, s_consumer, scale_producer, exp_turnstile = (
                online_softmax_inner(
                    config,
                    s_consumer,
                    scale_producer,
                    exp_turnstile,
                    scale_bar,
                    offs_m,
                    row_max,
                    row_sum,
                    n_start=n_block_min,
                    n_end=n_block_min_no_mask,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    CHECK_INF=True,
                    use_tmem_red=use_tmem_red,
                )
            )

        row_sum0, row_sum1 = float2.unpack2(row_sum)
        row_sum = row_sum0 + row_sum1

        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        row_max_tmem, row_sum_tmem = borrow_s_for_finalize(config, s_tmem)
        row_max_tmem.store(
            gl.convert_layout(row_max.expand_dims(1), config.row_scale_tmem_layout)
        )
        row_sum_tmem.store(
            gl.convert_layout(row_sum.expand_dims(1), config.row_scale_tmem_layout)
        )

        mbarrier.arrive(scale_bar, count=1)
        _, scale_bar, scale_producer = scale_producer.acquire()

        mbarrier.arrive(s_bar, count=1)
