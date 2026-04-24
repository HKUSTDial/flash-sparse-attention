import triton

from flash_sparse_attn.ops.triton import cache_utils


def get_fwd_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    pack_gqa: bool,
    num_splits: int,
):
    """
    Get the grid function for the forward kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param pack_gqa: Whether GQA packing is used
    :param num_splits: Number of KV splits

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(
                seqlen_q * (num_heads_q // num_heads_kv) if pack_gqa else seqlen_q,
                META["TILE_M"],
            ),
            num_heads_kv if pack_gqa else num_heads_q,
            batch_size * num_splits,
        )

    return grid


get_fwd_grid = cache_utils.cache_launch_grid(get_fwd_grid)


def get_bwd_grid(
    batch_size: int,
    seqlen_k: int,
    num_heads_q: int,
):
    """
    Get the grid function for the backward kernel.

    :param batch_size: Batch size
    :param seqlen_k: Sequence length of keys
    :param num_heads_q: Number of query heads

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(seqlen_k, META["TILE_N"]),
            num_heads_q,
            batch_size,
        )

    return grid


get_bwd_grid = cache_utils.cache_launch_grid(get_bwd_grid)


def get_dec_combine_grid(
    batch_size: int,
    num_heads_q: int,
):
    """
    Get the grid function for the decode combine kernel.

    :param batch_size: Batch size
    :param num_heads_q: Number of query heads

    :return grid: Grid function
    """

    def grid(META):
        return (batch_size * num_heads_q,)

    return grid


get_dec_combine_grid = cache_utils.cache_launch_grid(get_dec_combine_grid)


def get_bwd_preprocess_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
):
    """
    Get the grid function for the backward preprocess kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(seqlen_q, META["TILE_M"]),
            num_heads_q,
            batch_size,
        )

    return grid


get_bwd_preprocess_grid = cache_utils.cache_launch_grid(get_bwd_preprocess_grid)


def get_bwd_postprocess_grid(
    batch_size: int,
    seqlen_q: int,
    num_heads_q: int,
):
    """
    Get the grid function for the backward postprocess kernel.

    :param batch_size: Batch size
    :param seqlen_q: Sequence length of queries
    :param num_heads_q: Number of query heads

    :return grid: Grid function
    """

    def grid(META):
        return (
            triton.cdiv(seqlen_q, META["TILE_M"]),
            num_heads_q,
            batch_size,
        )

    return grid


get_bwd_postprocess_grid = cache_utils.cache_launch_grid(get_bwd_postprocess_grid)


def get_dec_grid(
    batch_size: int,
    num_heads_kv: int,
    num_splits: int,
):
    """
    Get the grid function for the decode kernel.

    :param batch_size: Batch size
    :param num_heads_kv: Number of key/value heads
    :param num_splits: Number of KV splits

    :return grid: Grid function
    """

    def grid(META):
        return (
            num_heads_kv,
            batch_size * num_splits,
        )

    return grid


get_dec_grid = cache_utils.cache_launch_grid(get_dec_grid)
