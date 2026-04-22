import triton
import triton.language as tl


@triton.jit
def check_inf(x):
    return tl.where(x == float("-inf"), 0.0, x)


@triton.jit
def online_softmax(
    acc_s,
    row_max,
    row_sum,
    scale_log2,
    CHECK_INF: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    """
    Apply online softmax to acc_s, and update row_max and row_sum.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param row_max: Current maximum values per row of shape [BLOCK_M], init to -inf.
    :param row_sum: Current sum values per row of shape [BLOCK_M], init to 0.
    :param scale_log2: Log2 of the scaling factor to be applied to acc_s.
    :param CHECK_INF: Boolean flag indicating if -inf row_max should be clamped to 0.
    :param RESCALE_THRESHOLD: Threshold for rescaling to avoid underflow. If <= 0, rescaling is disabled.

    :return p: Softmax probabilities tensor of shape [BLOCK_M, BLOCK_N].
    :return row_max_new: Updated maximum values per row of shape [BLOCK_M].
    :return row_sum_new: Updated sum values per row of shape [BLOCK_M].
    :return row_scale: Scaling factors per row of shape [BLOCK_M].
    """
    # Compute current row max
    row_max_curr = tl.max(acc_s, axis=1)

    # Update row max
    row_max_new = tl.maximum(row_max_curr, row_max)

    # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
    if CHECK_INF:
        row_max_new = check_inf(row_max_new)

    # Compute scaled differences to new row max
    acc_scale_log2 = (row_max - row_max_new) * scale_log2

    # Compute row scale
    if RESCALE_THRESHOLD > 0.0:
        # Triton can only skip computation at block granularity
        if tl.min(acc_scale_log2) < -RESCALE_THRESHOLD:
            row_scale = tl.exp2(acc_scale_log2)
        else:
            row_max_new = row_max
            row_scale = acc_scale_log2 * 0.0 + 1.0
    else:
        row_scale = tl.exp2(acc_scale_log2)

    # Compute attention weights
    p = tl.exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

    # Update row sum
    row_sum_cur = tl.sum(p, axis=1)
    row_sum_new = row_sum * row_scale + row_sum_cur

    return p, row_max_new, row_sum_new, row_scale


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

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param block_max: Running block-wise maximum scalar, init to -inf.
    :param row_max: Current maximum values per row of shape [BLOCK_M], init to -inf.
    :param row_sum: Current sum values per row of shape [BLOCK_M], init to 0.
    :param scale_log2: Log2 of the scaling factor to be applied to acc_s.
    :param softmax_threshold_log2: Threshold in log2-domain for block-level skip. If > -inf and block max is below threshold relative to running max, skip softmax update.
    :param CHECK_INF: Boolean flag indicating if -inf row_max should be clamped to 0.

    :return p: Softmax probabilities tensor of shape [BLOCK_M, BLOCK_N].
    :return block_max_new: Updated block-wise maximum scalar.
    :return row_max_new: Updated maximum values per row of shape [BLOCK_M].
    :return row_sum_new: Updated sum values per row of shape [BLOCK_M].
    :return row_scale: Scaling factors per row of shape [BLOCK_M].
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


@triton.jit
def finalize(
    row_max,
    row_sum,
    scale_log2,
    final_scale,
    IS_LOG2: tl.constexpr,
):
    """
    Finalize online softmax by computing output scale and logsumexp.

    :param row_max: Final maximum values per row of shape [BLOCK_M].
    :param row_sum: Final sum values per row of shape [BLOCK_M].
    :param final_scale: Scaling factor to be applied to the output.
    :param IS_LOG2: Boolean flag indicating if the returned logsumexp should be in log2-space.

    :return row_scale: Final scaling factors per row of shape [BLOCK_M].
    :return lse: Logsumexp values per row of shape [BLOCK_M].
    """
    # if row_sum is zero or nan, set it to 1 to avoid division by zero
    acc_o_is_zero_or_nan = (row_sum == 0.0) | (row_sum != row_sum)
    row_scale = tl.where(acc_o_is_zero_or_nan, 1.0, 1.0 / row_sum) * final_scale
    lse = tl.where(
        acc_o_is_zero_or_nan,
        float("-inf"),
        row_max * scale_log2 + tl.log2(row_sum),
    )
    if not IS_LOG2:
        # ln2 = math.log(2.0)
        ln2 = 0.6931471805599453
        lse *= ln2
    return row_scale, lse


@triton.jit
def rescale_o(
    acc_o,
    row_scale,
    LAZY_RESCALE: tl.constexpr,
):
    """
    Rescale output accumulator by row_scale.

    :param acc_o: Output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    :param row_scale: Scaling factors per row of shape [BLOCK_M].
    :param LAZY_RESCALE: Boolean flag indicating if rescaling should be applied lazily.

    :return: Rescaled output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    """
    if LAZY_RESCALE:
        # On some hardware, the check can cause severe performance regression
        if tl.min(row_scale) != 1.0:
            acc_o = acc_o * row_scale[:, None]
    else:
        acc_o = acc_o * row_scale[:, None]
    return acc_o


@triton.jit
def log_sigmoid(x, FASTMATH: tl.constexpr):
    """
    Compute log-sigmoid of x.

    :param x: Input tensor of shape [BLOCK_M, BLOCK_N].
    :param FASTMATH: Boolean flag indicating if the fast approximation should be used.

    :return: Tensor of shape [BLOCK_M, BLOCK_N] containing log-sigmoid values.
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
