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
    quant = "_quant" if _get(c, "IS_QUANT") else ""
    return f"flash_dense_fwd_{mask}_M{c['TILE_M']}N{c['TILE_N']}K{c['TILE_K']}{split}{quant}{varlen}"


def fwd_sparse_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitkv" if _get(c, "IS_SPLIT_KV") else ""
    quant = "_quant" if _get(c, "IS_QUANT") else ""
    return f"flash_sparse_fwd_{mask}_M{c['TILE_M']}N{c['TILE_N']}K{c['TILE_K']}{split}{quant}{varlen}"


def bwd_dense_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitqo" if _get(c, "IS_SPLIT_QO") else ""
    quant = "_quant" if _get(c, "IS_QUANT") else ""
    return f"flash_dense_bwd_{mask}_M{c['TILE_M']}N{c['TILE_N']}K{c['TILE_K']}{split}{quant}{varlen}"


def bwd_sparse_repr(specialization):
    c = specialization.constants
    mask = (
        "causal"
        if _get(c, "IS_CAUSAL")
        else ("local" if _get(c, "IS_LOCAL") else "full")
    )
    varlen = "_varlen" if _get(c, "HAS_CU_SEQLENS_Q") else ""
    split = "_splitqo" if _get(c, "IS_SPLIT_QO") else ""
    quant = "_quant" if _get(c, "IS_QUANT") else ""
    return f"flash_sparse_bwd_{mask}_M{c['TILE_M']}N{c['TILE_N']}K{c['TILE_K']}{split}{quant}{varlen}"


def fwd_combine_repr(specialization):
    c = specialization.constants
    return f"flash_fwd_combine_M{c['TILE_M']}K{c['TILE_K']}"


def bwd_preprocess_repr(specialization):
    c = specialization.constants
    return f"flash_bwd_preprocess_M{c['TILE_M']}K{c['TILE_K']}"


def bwd_postprocess_repr(specialization):
    c = specialization.constants
    return f"flash_bwd_postprocess_M{c['TILE_M']}K{c['TILE_K']}"
