from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def check_inf(x):
    return gl.maximum(x, -1e6)


@gluon.jit
def exp2(x):
    """
    Compute 2^x.

    :param x: Input tensor of shape [BLOCK_M, BLOCK_N].

    :return: Tensor of shape [BLOCK_M, BLOCK_N] containing 2^x values.
    """
    return gl.exp2(x)


@gluon.jit
def exp(x):
    """
    Compute e^x.

    :param x: Input tensor of shape [BLOCK_M, BLOCK_N].

    :return: Tensor of shape [BLOCK_M, BLOCK_N] containing e^x values.
    """
    log2_e: gl.constexpr = 1.4426950408889634
    return exp2(x * log2_e)


@gluon.jit
def online_softmax(
    acc_s,
    row_max,
    row_sum,
    scale_log2,
    CHECK_INF: gl.constexpr,
):
    """
    Apply online softmax to acc_s, and update row_max and row_sum.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param row_max: Current maximum values per row of shape [BLOCK_M], init to -inf.
    :param row_sum: Current sum values per row of shape [BLOCK_M], init to 0.
    :param scale_log2: Log2 of the scaling factor to be applied to acc_s.
    :param CHECK_INF: Boolean flag indicating if -inf row_max should be clamped to 0.

    :return p: Softmax probabilities tensor of shape [BLOCK_M, BLOCK_N].
    :return row_max_new: Updated maximum values per row of shape [BLOCK_M].
    :return row_sum_new: Updated sum values per row of shape [BLOCK_M].
    :return row_scale: Scaling factors per row of shape [BLOCK_M].
    """
    # Compute current row max
    row_max_curr = gl.max(acc_s, axis=1)

    # Update row max
    row_max_new = gl.maximum(row_max_curr, row_max)

    # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
    if CHECK_INF:
        row_max_new = check_inf(row_max_new)

    # Compute scaled differences to new row max
    acc_scale_log2 = (row_max - row_max_new) * scale_log2

    # Compute row scale
    row_scale = exp2(acc_scale_log2)

    # Compute attention weights
    p = exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

    # Update row sum
    row_sum_new = row_sum * row_scale + gl.sum(p, axis=1)

    return p, row_max_new, row_sum_new, row_scale


@gluon.jit
def online_sparse_softmax(
    acc_s,
    row_max,
    row_sum,
    scale_log2,
    softmax_threshold_log2,
    CHECK_INF: gl.constexpr,
):
    """
    Apply online sparse softmax to acc_s, and update row_max and row_sum.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param row_max: Current maximum values per row of shape [BLOCK_M], init to -inf.
    :param row_sum: Current sum values per row of shape [BLOCK_M], init to 0.
    :param scale_log2: Log2 of the scaling factor to be applied to acc_s.
    :param softmax_threshold_log2: Threshold in log2-domain for block-level skip.
    :param CHECK_INF: Boolean flag indicating if -inf row_max should be clamped to 0.

    :return p: Softmax probabilities tensor of shape [BLOCK_M, BLOCK_N].
    :return row_max_new: Updated maximum values per row of shape [BLOCK_M].
    :return row_sum_new: Updated sum values per row of shape [BLOCK_M].
    :return row_scale: Scaling factors per row of shape [BLOCK_M].
    :return skip_softmax: Boolean indicating whether this block was skipped.
    """
    # Compute current row max
    row_max_curr = gl.max(acc_s, axis=1)

    # Compute scaled differences to new row max
    row_max_diff_log2 = (row_max_curr - row_max) * scale_log2

    # Compute approximate final probability with current row max diff and row sum
    row_max_diff_log2 -= gl.log2(row_sum)

    # Update skip condition based on threshold
    skip_softmax = gl.max(row_max_diff_log2 - softmax_threshold_log2) < 0.0

    if skip_softmax:
        # Return zero attention weights
        p = gl.zeros(acc_s.shape, acc_s.dtype, layout=acc_s.type.layout)
        row_max_new = row_max
        row_sum_new = row_sum
        row_scale = gl.full(
            row_max.shape, 1.0, row_max.dtype, layout=row_max.type.layout
        )
    else:
        # Update row max
        row_max_new = gl.maximum(row_max_curr, row_max)

        # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
        if CHECK_INF:
            row_max_new = check_inf(row_max_new)

        # Compute scaled differences to new row max
        acc_scale_log2 = (row_max - row_max_new) * scale_log2

        # Compute row scale
        row_scale = exp2(acc_scale_log2)

        # Compute attention weights
        p = exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

        # Update row sum
        row_sum_new = row_sum * row_scale + gl.sum(p, axis=1)

    return p, row_max_new, row_sum_new, row_scale, skip_softmax


@gluon.jit
def finalize(
    row_max,
    row_sum,
    scale_log2,
    final_scale,
    IS_LOG2: gl.constexpr,
    CHECK_NAN: gl.constexpr,
):
    """
    Finalize online softmax by computing output scale and logsumexp.

    :param row_max: Final maximum values per row of shape [BLOCK_M].
    :param row_sum: Final sum values per row of shape [BLOCK_M].
    :param final_scale: Scaling factor to be applied to the output.
    :param IS_LOG2: Boolean flag indicating if the returned logsumexp should be in log2-space.
    :param CHECK_NAN: Boolean flag indicating if nan values in row_sum should be checked and set to 1 to avoid returning nan.

    :return row_scale: Final scaling factors per row of shape [BLOCK_M].
    :return lse: Logsumexp values per row of shape [BLOCK_M].
    """
    # if row_sum is zero or nan, set it to 1 to avoid division by zero
    if CHECK_NAN:
        invalid = (row_sum == 0.0) | (row_sum != row_sum)
        row_scale = gl.where(invalid, 1.0, final_scale / row_sum)
        lse = gl.where(
            invalid,
            float("-inf"),
            row_max * scale_log2 + gl.log2(row_sum),
        )
    else:
        row_scale = final_scale / row_sum
        lse = row_max * scale_log2 + gl.log2(row_sum)
    if not IS_LOG2:
        ln2: gl.constexpr = 0.6931471805599453
        lse *= ln2
    return row_scale, lse


@gluon.jit
def rescale_o(acc_o, row_scale):
    """
    Rescale output accumulator by row_scale.

    :param acc_o: Output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    :param row_scale: Scaling factors per row of shape [BLOCK_M].

    :return: Rescaled output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    """
    acc_o *= row_scale[:, None]
    return acc_o


@gluon.jit
def sigmoid(x):
    """
    Compute sigmoid of x.

    :param x: Input tensor of shape [BLOCK_M, BLOCK_N].

    :return: Tensor of shape [BLOCK_M, BLOCK_N] containing sigmoid values.
    """
    return 1.0 / (1.0 + exp(-x))


@gluon.jit
def log_sigmoid(x, FASTMATH: gl.constexpr):
    """
    Compute log-sigmoid of x.

    :param x: Input tensor of shape [BLOCK_M, BLOCK_N].
    :param FASTMATH: Boolean flag indicating if the fast approximation should be used.

    :return: Tensor of shape [BLOCK_M, BLOCK_N] containing log-sigmoid values.
    """
    if FASTMATH:
        xc = gl.minimum(gl.abs(x), 4.0)
        return gl.minimum(x, 0.0) - 0.05674870 * xc * xc + 0.37664706 * xc - 0.65169323
    return gl.minimum(x, 0.0) - gl.log(1.0 + exp(-gl.abs(x)))


@gluon.jit
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
    gate_max_curr = gl.maximum(
        gl.maximum(a_max * d_max, a_max * d_min),
        gl.maximum(a_min * d_max, a_min * d_min),
    )
    skip_gate = (gate_max_curr - gate_max) * scale_log2 < gate_threshold_log2
    gate_max_new = gate_max if skip_gate else gl.maximum(gate_max_curr, gate_max)
    return gate_max_new, skip_gate
