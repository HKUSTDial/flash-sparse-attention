def _get(c, key, default=None):
    try:
        return c[key]
    except (KeyError, TypeError):
        return default


def fwd_dense_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitkv" if _get(c, "IS_SPLIT_KV") else ""
    return f"flash_dense_fwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{split}{varlen}"


def fwd_sparse_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitkv" if _get(c, "IS_SPLIT_KV") else ""
    return f"flash_sparse_fwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{split}{varlen}"


def fwd_gated_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitkv" if _get(c, "IS_SPLIT_KV") else ""
    return f"flash_gated_fwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{split}{varlen}"


def bwd_dense_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    return f"flash_dense_bwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{varlen}"


def bwd_sparse_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    return f"flash_sparse_bwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{varlen}"


def bwd_gated_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    return f"flash_gated_bwd_{mask}_{c['TILE_M']}x{c['TILE_N']}{varlen}"


def dec_dense_repr(specialization):
    c = specialization.constants
    mask = "local" if _get(c, "IS_LOCAL") else "full"
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    gather = "_topk" if _get(c, "HAS_GATHER_KV") else ""
    return f"flash_dense_dec_{mask}_{c['TILE_M']}x{c['TILE_N']}{gather}{varlen}"


def dec_sparse_repr(specialization):
    c = specialization.constants
    mask = "local" if _get(c, "IS_LOCAL") else "full"
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    gather = "_topk" if _get(c, "HAS_GATHER_KV") else ""
    return f"flash_sparse_dec_{mask}_{c['TILE_M']}x{c['TILE_N']}{gather}{varlen}"


def dec_gated_repr(specialization):
    c = specialization.constants
    mask = "local" if _get(c, "IS_LOCAL") else "full"
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    gather = "_topk" if _get(c, "HAS_GATHER_KV") else ""
    return f"flash_gated_dec_{mask}_{c['TILE_M']}x{c['TILE_N']}{gather}{varlen}"


def fwd_combine_repr(specialization):
    c = specialization.constants
    return f"flash_fwd_combine_{c['TILE_M']}"


def dec_combine_repr(specialization):
    c = specialization.constants
    return f"flash_dec_combine_{c['TILE_K']}"


def bwd_preprocess_repr(specialization):
    c = specialization.constants
    return f"flash_bwd_preprocess_{c['TILE_M']}"


def bwd_postprocess_repr(specialization):
    c = specialization.constants
    return f"flash_bwd_postprocess_{c['TILE_M']}"
