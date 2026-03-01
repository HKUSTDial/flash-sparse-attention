import math
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
    # RESCALE_THRESHOLD: tl.constexpr = 0.0,
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
    # Update row max
    row_max_new = tl.maximum(tl.max(acc_s, axis=1), row_max)

    # Avoid exp(-inf - (-inf)) = nan by clamping -inf to 0
    if CHECK_INF:
        row_max_new = check_inf(row_max_new)

    # Compute row scale
    acc_scale_log2 = (row_max - row_max_new) * scale_log2
    row_scale = tl.exp2(acc_scale_log2)

    # TODO: Triton 3.6 currently does not support enabling LAZY_RESCALE
    # # If max update is tiny, keep the old max
    # if RESCALE_THRESHOLD > 0.0:
    #     if tl.min(acc_scale_log2) >= -RESCALE_THRESHOLD:
    #         row_max_new = row_max
    #         row_scale = row_scale * 0.0 + 1.0

    # Compute attention weights
    p = tl.exp2(acc_s * scale_log2 - row_max_new[:, None] * scale_log2)

    # Update row sum
    row_sum_cur = tl.sum(p, axis=1)
    row_sum_new = row_sum * row_scale + row_sum_cur

    return p, row_max_new, row_sum_new, row_scale


@triton.jit
def finalize(
    row_max,
    row_sum,
    scale_log2,
    final_scale,
):
    """
    Finalize online softmax by computing output scale and logsumexp.

    :param row_max: Final maximum values per row of shape [BLOCK_M].
    :param row_sum: Final sum values per row of shape [BLOCK_M].
    :param final_scale: Scaling factor to be applied to the output.

    :return row_scale: Final scaling factors per row of shape [BLOCK_M].
    :return lse: Logsumexp values per row of shape [BLOCK_M].
    """
    # if row_sum is zero or nan, set it to 1 to avoid division by zero
    acc_o_is_zero_or_nan = (row_sum == 0.0) | (row_sum != row_sum)
    row_scale = tl.where(acc_o_is_zero_or_nan, 1.0, 1.0 / row_sum) * final_scale
    ln2 = math.log(2.0)
    lse = tl.where(
        acc_o_is_zero_or_nan,
        float("-inf"),
        (row_max * scale_log2 + tl.log2(row_sum)) * ln2,
    )
    return row_scale, lse


@triton.jit
def rescale_o(
    acc_o,
    row_scale,
    # LAZY_RESCALE: tl.constexpr,
):
    """
    Rescale output accumulator by row_scale.

    :param acc_o: Output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    :param row_scale: Scaling factors per row of shape [BLOCK_M].

    :return: Rescaled output accumulator tensor of shape [BLOCK_M, BLOCK_N].
    """
    # TODO: In Triton 3.6, combining tensor condition with early return
    # in a jitted helper can trigger compile errors
    # if LAZY_RESCALE:
    #     if tl.min(row_scale) == 1.0:
    #         return acc_o
    return acc_o * row_scale[:, None]


@triton.jit
def log_sigmoid(x, mask):
    x = x.to(tl.float32)
    neg_abs_x = -tl.abs(x)
    # TODO: In Triton 3.6, tl.where cannot reduce the actual computation
    # correction = tl.where(neg_abs_x < -8.0, 0.0, tl.log(1.0 + tl.exp(neg_abs_x)))
    # return tl.where(mask, tl.minimum(x, 0.0) - correction, float("-inf"))
    return tl.where(
        mask, tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(neg_abs_x)), float("-inf")
    )


@triton.jit
def gate_skip(a_max, a_min, d_max, d_min, g_thr_min):
    g_upper = tl.maximum(
        tl.maximum(a_max * d_max, a_max * d_min),
        tl.maximum(a_min * d_max, a_min * d_min),
    )
    return g_upper >= g_thr_min
