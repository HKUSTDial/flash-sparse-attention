import triton
import triton.language as tl


@triton.jit
def apply_mask(
    acc_s,
    m_idx,
    n_idx,
    seqlen_k,
    causal_offset,
    IS_CAUSAL: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Apply causal and padding mask to the attention scores.

    :param acc_s: Attention scores tensor of shape [BLOCK_M, BLOCK_N].
    :param m_idx: Row indices corresponding to BLOCK_M.
    :param n_idx: Column indices corresponding to BLOCK_N.
    :param seqlen_k: The sequence length of the key.
    :param causal_offset: Offset used for causal masking.
    :param IS_CAUSAL: Boolean flag indicating if causal masking should be applied.
    :param EVEN_N: Boolean flag indicating if BLOCK_N is even.

    :return acc_s: Masked attention scores tensor of shape [BLOCK_M, BLOCK_N].
    """
    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        acc_s = tl.where(
            n_idx[None, :] < seqlen_k,
            acc_s,
            float("-inf"),
        )
    if IS_CAUSAL:
        acc_s = tl.where(
            m_idx[:, None] + causal_offset >= n_idx[None, :],
            acc_s,
            float("-inf"),
        )
    return acc_s
