# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def check_inf(
    x: gl.tensor,
) -> gl.tensor:
    """
    Clamp -inf values in the input tensor to -1e6 to avoid NaN issues in subsequent computations.

    :param x: input tensor
    :type x: tensor

    :return: tensor with -inf values clamp values below -1e6
    """
    return gl.maximum(x, -1e6)


@gluon.jit
def exp2(
    x: gl.tensor,
) -> gl.tensor:
    """
    Compute 2^x.

    :param x: input tensor
    :type x: tensor

    :return: tensor containing 2^x values
    """
    return gl.exp2(x)


@gluon.jit
def exp(
    x: gl.tensor,
) -> gl.tensor:
    """
    Compute e^x.

    :param x: input tensor
    :type x: tensor

    :return: tensor containing e^x values
    """
    log2_e: gl.constexpr = 1.4426950408889634
    return exp2(x * log2_e)


@gluon.jit
def online_softmax(
    acc_s: gl.tensor,
    row_max: gl.tensor,
    row_sum: gl.tensor,
    scale_log2: gl.tensor,
    CHECK_INF: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    """
    Apply online softmax to acc_s, and update row_max and row_sum.

    :param acc_s: attention scores of shape [TILE_M, TILE_N]
    :type acc_s: tensor
    :param row_max: running maximum values per row of shape [TILE_M], init to -inf
    :type row_max: tensor
    :param row_sum: running sum values per row of shape [TILE_M], init to 0
    :type row_sum: tensor
    :param scale_log2: log2 of the scaling factor to be applied to acc_s
    :type scale_log2: tensor
    :param CHECK_INF: boolean flag indicating if -inf row_max should be clamp values below -1e6
    :type CHECK_INF: bool

    :return p: online softmax probabilities of shape [TILE_M, TILE_N]
    :return row_max_new: updated running maximum values per row of shape [TILE_M]
    :return row_sum_new: updated running sum values per row of shape [TILE_M]
    :return row_scale: scaling factors per row of shape [TILE_M]
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
    acc_s: gl.tensor,
    row_max: gl.tensor,
    row_sum: gl.tensor,
    scale_log2: gl.tensor,
    softmax_threshold_log2: gl.tensor,
    CHECK_INF: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
    """
    Apply online sparse softmax to acc_s, and update row_max and row_sum.

    :param acc_s: attention scores of shape [TILE_M, TILE_N]
    :type acc_s: tensor
    :param row_max: running maximum values per row of shape [TILE_M], init to -inf
    :type row_max: tensor
    :param row_sum: running sum values per row of shape [TILE_M], init to 0
    :type row_sum: tensor
    :param scale_log2: log2 of the scaling factor to be applied to acc_s
    :type scale_log2: tensor
    :param softmax_threshold_log2: threshold in log2-domain for block-level skip
    :type softmax_threshold_log2: tensor
    :param CHECK_INF: boolean flag indicating if -inf row_max should be clamp values below -1e6
    :type CHECK_INF: bool

    :return p: online softmax probabilities of shape [TILE_M, TILE_N]
    :return row_max_new: updated running maximum values per row of shape [TILE_M]
    :return row_sum_new: updated running sum values per row of shape [TILE_M]
    :return row_scale: scaling factors per row of shape [TILE_M]
    :return skip_softmax: boolean indicating whether this block was skipped
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
    row_max: gl.tensor,
    row_sum: gl.tensor,
    scale_log2: gl.tensor,
    final_scale: gl.tensor,
    IS_LOG2: gl.constexpr,
    CHECK_NAN: gl.constexpr,
) -> tuple[gl.tensor, gl.tensor]:
    """
    Finalize online softmax by computing output scale and logsumexp.

    :param row_max: final maximum values per row of shape [TILE_M]
    :type row_max: tensor
    :param row_sum: final sum values per row of shape [TILE_M]
    :type row_sum: tensor
    :param scale_log2: log2 of the scaling factor applied to acc_s
    :type scale_log2: tensor
    :param final_scale: scaling factor to be applied to the output
    :type final_scale: tensor
    :param IS_LOG2: boolean flag indicating if the returned logsumexp should be in log2-space
    :type IS_LOG2: bool
    :param CHECK_NAN: boolean flag indicating if nan values in row_sum should be checked and set to 1 to avoid returning nan
    :type CHECK_NAN: bool

    :return row_scale: final scaling factors per row of shape [TILE_M]
    :return lse: final logsumexp values per row of shape [TILE_M]
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
def rescale_o(
    acc_o: gl.tensor,
    row_scale: gl.tensor,
) -> gl.tensor:
    """
    Rescale output accumulator by row_scale.

    :param acc_o: output accumulator tensor of shape [TILE_M, TILE_K]
    :type acc_o: tensor
    :param row_scale: scaling factors per row of shape [TILE_M]
    :type row_scale: tensor

    :return acc_o: rescaled output accumulator tensor of shape [TILE_M, TILE_K]
    """
    acc_o *= row_scale[:, None]
    return acc_o
