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
    IS_FIRST: tl.constexpr,
    CHECK_INF: tl.constexpr,
):
    """
    Apply online softmax to acc_s, and update row_max and row_sum.

    Args:
        acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
        row_max: Current maximum values per row of shape [BLOCK_M].
        row_sum: Current sum values per row of shape [BLOCK_M].
        IS_FIRST: Boolean flag indicating if this is the first block.
        CHECK_INF: Boolean flag indicating if inf values should be checked.

    Returns:
        p: Softmax probabilities tensor of shape [BLOCK_M, BLOCK_N].
        row_max_cur: Updated maximum values per row of shape [BLOCK_M].
        row_sum_new: Updated sum values per row of shape [BLOCK_M].
        row_scale: Scaling factors per row of shape [BLOCK_M].
    """
    # compute the current row max
    row_max_cur = tl.max(acc_s, axis=1)

    # if not the first block, combine with previous row max
    if not IS_FIRST:
        row_max_cur = tl.maximum(row_max_cur, row_max)

    # avoid exp(0) by checking for -inf
    if CHECK_INF:
        row_max_cur = check_inf(row_max_cur)

    # compute the exponentials and current row sum
    p = tl.exp(acc_s - row_max_cur[:, None])
    row_sum_cur = tl.sum(p, axis=1)

    if IS_FIRST:
        # no rescaling needed
        row_scale = tl.full(row_sum_cur.shape, 1.0)
        row_sum_new = row_sum_cur
    else:
        # compute rescaling factor and update row sum
        row_scale = tl.exp(row_max - row_max_cur)
        row_sum_new = row_sum * row_scale + row_sum_cur

    return p, row_max_cur, row_sum_new, row_scale


@triton.jit
def finalize(
    row_max,
    row_sum,
    final_scale,
):
    """
    Finalize online softmax by computing output scale and logsumexp.

    Args:
        row_max: Final maximum values per row of shape [BLOCK_M].
        row_sum: Final sum values per row of shape [BLOCK_M].
        final_scale: Scaling factor to be applied to the output.

    Returns:
        o_scale: Output scaling factors per row of shape [BLOCK_M].
        lse: Logsumexp values per row of shape [BLOCK_M].
    """
    # if row_sum is zero or nan, set it to 1 to avoid division by zero
    acc_o_is_zero_or_nan = (row_sum == 0.0) | (row_sum != row_sum)

    row_sum = tl.where(acc_o_is_zero_or_nan, 1.0, row_sum)
    o_scale = (1.0 / row_sum) * final_scale
    lse = tl.where(acc_o_is_zero_or_nan, float("-inf"), row_max + tl.log(row_sum))
    return o_scale, lse


@triton.jit
def rescale_o(
    acc_o,
    row_scale,
):
    return acc_o * row_scale[:, None]
