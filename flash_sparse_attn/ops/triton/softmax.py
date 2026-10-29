# Copyright (c) 2026, Jingze Shi.
import triton
import triton.language as tl


@triton.jit
def log2(
    x: tl.tensor,
) -> tl.tensor:
    """
    Compute logarithm base 2 of the input tensor.

    :param x: input tensor
    :type x: tensor

    :return: tensor containing log2 values of the input
    """
    return log2(x)


@triton.jit
def exp2(
    x: tl.tensor,
) -> tl.tensor:
    """
    Compute exponent base 2 of the input tensor.

    :param x: input tensor
    :type x: tensor
    :type FAST_MATH: bool

    :return: tensor containing 2^x values
    """
    return tl.exp2(x)


@triton.jit
def exp(
    x: tl.tensor,
) -> tl.tensor:
    """
    Compute exponential of the input tensor.

    :param x: input tensor
    :type x: tensor

    :return: tensor containing e^x values
    """
    log2_e: tl.constexpr = 1.4426950408889634
    return exp2(x * log2_e)


@triton.jit
def check_neginf(
    x: tl.tensor,
) -> tl.tensor:
    """
    Clamp -inf values in the input tensor to -1e6 to avoid NaN issues in subsequent computations.

    :param x: input tensor
    :type x: tensor

    :return: tensor with -inf values clamp values below -1e6
    """
    return tl.maximum(x, -1e6)


@triton.jit
def online_softmax(
    acc_s: tl.tensor,
    row_max: tl.tensor,
    row_sum: tl.tensor,
    scale_log2: tl.tensor,
    row_max_curr: tl.tensor = None,
    CHECK_INF: tl.constexpr = False,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
    :param row_max_curr: precomputed current row maximum values
    :type row_max_curr: tensor
    :param CHECK_INF: boolean flag indicating if -inf row_max should be clamp values below -1e6
    :type CHECK_INF: bool

    :return p: online softmax probabilities of shape [TILE_M, TILE_N]
    :return row_max_new: updated running maximum values per row of shape [TILE_M]
    :return row_sum_new: updated running sum values per row of shape [TILE_M]
    :return row_scale: scaling factors per row of shape [TILE_M]
    """
    if row_max_curr is None:
        # Compute current row max
        row_max_curr = tl.max(acc_s, axis=1)

    # Update row max
    row_max_new = tl.maximum(row_max_curr, row_max)

    # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
    if CHECK_INF:
        row_max_new = check_neginf(row_max_new)

    # Compute scaled differences to new row max
    acc_scale_log2 = (row_max - row_max_new) * scale_log2

    # Compute row scale
    row_scale = exp2(acc_scale_log2)

    # Compute attention weights
    p = exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

    # Update row sum
    row_sum_new = row_sum * row_scale + tl.sum(p, axis=1)

    return p, row_max_new, row_sum_new, row_scale


@triton.jit
def score_skip(
    acc_s: tl.tensor,
    row_max: tl.tensor,
    row_sum: tl.tensor,
    scale_log2: tl.tensor,
    softmax_threshold_log2: tl.tensor,
) -> tuple[tl.tensor, tl.tensor]:
    """
    Determine whether the current score can be skipped.

    :param acc_s: attention scores of shape [TILE_M, TILE_N]
    :type acc_s: tensor
    :param row_max: running maximum values per row of shape [TILE_M]
    :type row_max: tensor
    :param row_sum: running sum values per row of shape [TILE_M]
    :type row_sum: tensor
    :param scale_log2: log2 of the scaling factor to be applied to acc_s
    :type scale_log2: tensor
    :param softmax_threshold_log2: threshold in log2-domain for block-level skip
    :type softmax_threshold_log2: tensor

    :return is_skip: boolean indicating whether this block can be skipped
    :return row_max_curr: current row maximum values
    """
    # Compute current row max
    row_max_curr = tl.max(acc_s, axis=1)

    # Compute scaled differences to new row max
    row_max_diff_log2 = (row_max_curr - row_max) * scale_log2

    # Compute approximate final probability with current row max diff and row sum
    row_max_diff_log2 -= log2(row_sum)

    # Update skip condition based on threshold
    is_skip = tl.max(row_max_diff_log2 - softmax_threshold_log2) < 0.0

    return is_skip, row_max_curr


@triton.jit
def prob_skip(
    p_log2: tl.tensor,
    softmax_threshold_log2: tl.tensor,
) -> tl.tensor:
    """
    Determine whether the current probability can be skipped.

    :param p_log2: attention probabilities in log2-domain of shape [TILE_M, TILE_N]
    :type p_log2: tensor
    :param softmax_threshold_log2: threshold in log2-domain for block-level skip
    :type softmax_threshold_log2: tensor

    :return: boolean indicating whether this block can be skipped
    """
    # Update skip condition based on threshold
    is_skip = tl.max(p_log2 - softmax_threshold_log2[None, :]) < 0.0
    return is_skip


@triton.jit
def finalize(
    row_max: tl.tensor,
    row_sum: tl.tensor,
    scale_log2: tl.tensor,
    final_scale: tl.tensor,
    IS_LOG2: tl.constexpr,
    CHECK_NAN: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """
    Finalize online softmax by computing output scale and logsumexp.

    :param row_max: final maximum values per row of shape [TILE_M]
    :type row_max: tensor
    :param row_sum: final sum values per row of shape [TILE_M]
    :type row_sum: tensor
    :param scale_log2: log2 of the scaling factor applied to acc_S
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
        row_scale = tl.where(invalid, 1.0, final_scale / row_sum)
        lse = tl.where(
            invalid,
            float("-inf"),
            row_max * scale_log2 + log2(row_sum),
        )
    else:
        row_scale = final_scale / row_sum
        lse = row_max * scale_log2 + log2(row_sum)
    if not IS_LOG2:
        # ln2 = math.log(2.0)
        ln2: tl.constexpr = 0.6931471805599453
        lse *= ln2
    return row_scale, lse


@triton.jit
def rescale_o(
    acc_o: tl.tensor,
    row_scale: tl.tensor,
) -> tl.tensor:
    """
    Rescale output accumulator by row_scale.

    :param acc_o: output accumulator tensor of shape [TILE_M, TILE_K]
    :type acc_o: tensor
    :param row_scale: scaling factors per row of shape [TILE_M]
    :type row_scale: tensor

    :return acc_o: rescaled output accumulator tensor of shape [TILE_M, TILE_K]
    """
    acc_o = acc_o * row_scale[:, None]
    return acc_o
