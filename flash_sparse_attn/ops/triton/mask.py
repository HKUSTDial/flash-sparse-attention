import triton
import triton.language as tl


@triton.jit
def apply_mask(
    acc_s,  # [BLOCK_M, BLOCK_N]
    m_idx,  # [BLOCK_M]
    n_idx,  # [BLOCK_N]
    seqlen_k,  # int
    causal_offset,  # int
    IS_CAUSAL: tl.constexpr,
    EVEN_N: tl.constexpr,
):
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
