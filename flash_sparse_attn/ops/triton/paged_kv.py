import triton
import triton.language as tl


@triton.jit
def load_page(page_table_base, n_block):
    return tl.load(page_table_base + n_block)


@triton.jit
def load_paged_k(
    k_base,
    page_table_base,
    stride_kb,
    stride_kn,
    n_block,
    actual_seqlen_k,
    head_dim,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    page = load_page(page_table_base, n_block)
    offs_n = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    return tl.load(
        k_base + page * stride_kb + offs_n[:, None] * stride_kn + offs_k[None, :],
        mask=(n_block * TILE_N + offs_n < actual_seqlen_k)[:, None]
        & (offs_k[None, :] < head_dim),
        other=0.0,
        cache_modifier=".ca",
    )


@triton.jit
def load_paged_v(
    v_base,
    page_table_base,
    stride_vb,
    stride_vn,
    n_block,
    actual_seqlen_k,
    head_dim,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    page = load_page(page_table_base, n_block)
    offs_n = tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    return tl.load(
        v_base + page * stride_vb + offs_n[:, None] * stride_vn + offs_k[None, :],
        mask=(n_block * TILE_N + offs_n < actual_seqlen_k)[:, None]
        & (offs_k[None, :] < head_dim),
        other=0.0,
        cache_modifier=".ca",
    )


@triton.jit
def load_paged_d(
    d_base,
    page_table_base,
    stride_db,
    stride_dn,
    n_block,
    actual_seqlen_k,
    TILE_N: tl.constexpr,
):
    page = load_page(page_table_base, n_block)
    offs_n = tl.arange(0, TILE_N)
    return tl.load(
        d_base + page * stride_db + offs_n * stride_dn,
        mask=n_block * TILE_N + offs_n < actual_seqlen_k,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
