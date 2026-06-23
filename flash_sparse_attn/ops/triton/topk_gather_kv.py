import triton
import triton.language as tl


@triton.jit
def load_topk_indices(
    gather_kv_base,
    stride_gn,
    n_block,
    TILE_N: tl.constexpr,
):
    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)
    return tl.load(gather_kv_base + offs_n * stride_gn)


@triton.jit
def make_valid_mask(indices):
    return indices >= 0


@triton.jit
def load_gathered_k(
    k_base,
    topk_indices,
    stride_kn,
    head_dim,
    TILE_K: tl.constexpr,
):
    valid_mask = make_valid_mask(topk_indices)
    offs_k = tl.arange(0, TILE_K)
    return tl.load(
        k_base + topk_indices[:, None] * stride_kn + offs_k[None, :],
        mask=valid_mask[:, None] & (offs_k[None, :] < head_dim),
        other=0.0,
        cache_modifier=".ca",
    )


@triton.jit
def load_gathered_v(
    v_base,
    topk_indices,
    stride_vn,
    head_dim,
    TILE_K: tl.constexpr,
):
    valid_mask = make_valid_mask(topk_indices)
    offs_k = tl.arange(0, TILE_K)
    return tl.load(
        v_base + topk_indices[:, None] * stride_vn + offs_k[None, :],
        mask=valid_mask[:, None] & (offs_k[None, :] < head_dim),
        other=0.0,
        cache_modifier=".ca",
    )


@triton.jit
def load_gathered_d(
    d_base,
    topk_indices,
):
    valid_mask = make_valid_mask(topk_indices)
    return tl.load(
        d_base + topk_indices,
        mask=valid_mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)


@triton.jit
def apply_gather_mask(
    acc_s,
    topk_indices,
    actual_seqlen_q,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
):
    valid_mask = make_valid_mask(topk_indices)
    offs_m = tl.arange(0, TILE_M)
    q_idx = offs_m // QHEAD_PER_KVHEAD_PACKGQA
    valid_m = q_idx < actual_seqlen_q
    return tl.where(valid_m[:, None] & valid_mask[None, :], acc_s, float("-inf"))
