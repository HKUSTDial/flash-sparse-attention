from typing import Tuple, Optional

import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    utils,
    cache_utils,
    launch_template,
    launch_grid,
    flash_dec_combine,
    kernel_repr,
    autotuner,
)
from flash_sparse_attn.ops.triton.scheduler import (
    AttnDecGridIndex,
    AttnDecConfig,
    AttnDecBlockScheduler,
    AttnDecPointerScheduler,
    AttnMaskScheduler,
    SoftmaxScheduler,
)


@triton.jit
def _dec_inner_sparse_kernel(
    config: AttnDecConfig,
    ptrs_sched: AttnDecPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    q_tile,
    k_tile,
    topk_indices,
    k_ptrs,
    v_ptrs,
    acc_o,
    row_max,
    row_sum,
    n_block,
    n_block_min,
    IS_MASK: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    MASK_SINK: tl.constexpr,
    CHECK_INF: tl.constexpr,
    HAS_GATHER_KV: tl.constexpr,
):
    # Compute attention scores
    acc_s = tl.dot(q_tile, k_tile.T)

    # Prefetch next key tile
    next_topk_indices = topk_indices
    if n_block > n_block_min:
        if HAS_GATHER_KV:
            # Load next topk indices
            next_topk_indices = ptrs_sched.load_topk_indices(config, n_block - 1)

        # Load next key tile
        k_tile = ptrs_sched.load_k(
            config,
            k_ptrs,
            n_block - 1,
            next_topk_indices,
            HAS_GATHER_KV=HAS_GATHER_KV,
        )

    if HAS_GATHER_KV:
        # Apply mask to attention scores
        acc_s = ptrs_sched.apply_gather_mask(
            config,
            acc_s,
            topk_indices,
        )

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask_sched.apply_mask(
            acc_s=acc_s,
            iter_block=n_block,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Apply online sparse softmax
    p, row_max, row_sum, row_scale, skip_softmax = softmax_sched.online_sparse_softmax(
        acc_s=acc_s,
        row_max=row_max,
        row_sum=row_sum,
        softmax_threshold_log2=config.softmax_threshold_log2,
        CHECK_INF=CHECK_INF,
    )

    if not skip_softmax:
        # Load value tile
        v_tile = ptrs_sched.load_v(
            config, v_ptrs, n_block, topk_indices, HAS_GATHER_KV=HAS_GATHER_KV
        )

        # Rescale output accumulator
        acc_o = softmax_sched.rescale_o(
            acc_o=acc_o,
            row_scale=row_scale,
        )

        # Update output accumulator
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

    return k_tile, next_topk_indices, acc_o, row_max, row_sum


@triton.jit(repr=kernel_repr.dec_sparse_repr)
def _dec_sparse_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    softmax_threshold,
    window_sizes,
    gather_kv_indices,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_os,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_ls,
    stride_wh,
    stride_gb,
    stride_gn,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_splits,
    seqlen_q,
    seqlen_k,
    head_dim,
    SEQLEN_Q_CACHE: tl.constexpr,
    SEQLEN_K_CACHE: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    topk_seqlen_k: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    HAS_GATHER_KV: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnDecGridIndex.create(
        num_splits=num_splits,
    )

    # Load window sizes
    (
        window_size_sink,
        window_size_left,
        window_size_right,
        window_size_dist,
    ) = grid_idx.load_window_sizes(
        window_sizes=window_sizes,
        stride_wh=stride_wh,
        IS_LOCAL=IS_LOCAL,
    )

    # Create config
    config = AttnDecConfig.create(
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        batch_idx=grid_idx.batch_idx,
        window_size_sink=window_size_sink,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        window_size_dist=window_size_dist,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create pointer scheduler
    ptrs_sched = AttnDecPointerScheduler.create(
        config=config,
        Q=Q,
        K=K,
        V=V,
        GatherKVIndices=gather_kv_indices,
        Out=Out,
        Lse=Lse,
        batch_idx=grid_idx.batch_idx,
        head_idx=grid_idx.head_idx,
        head_kv_idx=grid_idx.head_kv_idx,
        split_idx=grid_idx.split_idx,
        stride_qb=stride_qb,
        stride_qh=stride_qh,
        stride_qm=stride_qm,
        stride_kb=stride_kb,
        stride_kh=stride_kh,
        stride_kn=stride_kn,
        stride_vb=stride_vb,
        stride_vh=stride_vh,
        stride_vn=stride_vn,
        stride_ob=stride_ob,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_os=stride_os,
        stride_lb=stride_lb,
        stride_lh=stride_lh,
        stride_lm=stride_lm,
        stride_ls=stride_ls,
        stride_gb=stride_gb,
        stride_gn=stride_gn,
        HAS_GATHER_KV=HAS_GATHER_KV,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnDecBlockScheduler.create(
        config=config,
        split_idx=grid_idx.split_idx,
        num_splits=num_splits,
        topk_seqlen_k=topk_seqlen_k,
        IS_LOCAL=IS_LOCAL,
        HAS_GATHER_KV=HAS_GATHER_KV,
    )

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Create pointers
    out_ptrs = ptrs_sched.make_out_ptrs(config)
    lse_ptrs = ptrs_sched.make_lse_ptrs(config)

    # Early exit if no n_blocks to process
    if block_sched.is_empty():
        ptrs_sched.store_empty(config, out_ptrs, lse_ptrs, Out)
        return

    q_ptrs = ptrs_sched.make_q_ptrs(config)
    k_ptrs = ptrs_sched.make_k_ptrs(config)
    v_ptrs = ptrs_sched.make_v_ptrs(config)

    # Initialize accumulators
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Initialize topk indices
    topk_indices = tl.arange(0, TILE_N)

    # Load query tile
    q_tile = ptrs_sched.load_q(config, q_ptrs)

    if HAS_GATHER_KV:
        # Load topk indices
        topk_indices = ptrs_sched.load_topk_indices(config, block_sched.n_block_max - 1)

    # Load key tile
    k_tile = ptrs_sched.load_k(
        config,
        k_ptrs,
        block_sched.n_block_max - 1,
        topk_indices,
        HAS_GATHER_KV=HAS_GATHER_KV,
    )

    # Process n_blocks with seqlen masking
    for n_block in tl.range(
        block_sched.n_block_max - 1, block_sched.n_block_max_no_mask - 1, -1
    ):
        k_tile, topk_indices, acc_o, row_max, row_sum = _dec_inner_sparse_kernel(
            config=config,
            ptrs_sched=ptrs_sched,
            mask_sched=mask_sched,
            softmax_sched=softmax_sched,
            q_tile=q_tile,
            k_tile=k_tile,
            topk_indices=topk_indices,
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
            acc_o=acc_o,
            row_max=row_max,
            row_sum=row_sum,
            n_block=n_block,
            n_block_min=block_sched.n_block_max_no_mask,
            IS_MASK=True,
            MASK_LOCAL=True if IS_LOCAL else False,
            MASK_SINK=False,
            CHECK_INF=True,
            HAS_GATHER_KV=HAS_GATHER_KV,
        )

    # Process n_blocks without masking
    if (
        not IS_LOCAL or HAS_GATHER_KV
    ) and block_sched.n_block_max_no_mask > block_sched.n_block_min:
        if HAS_GATHER_KV:
            # Load topk indices
            topk_indices = ptrs_sched.load_topk_indices(
                config,
                block_sched.n_block_max_no_mask - 1,
            )

        # Load key tile
        k_tile = ptrs_sched.load_k(
            config,
            k_ptrs,
            block_sched.n_block_max_no_mask - 1,
            topk_indices,
            HAS_GATHER_KV=HAS_GATHER_KV,
        )

        for n_block in tl.range(
            block_sched.n_block_max_no_mask - 1, block_sched.n_block_min - 1, -1
        ):
            k_tile, topk_indices, acc_o, row_max, row_sum = _dec_inner_sparse_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                q_tile=q_tile,
                k_tile=k_tile,
                topk_indices=topk_indices,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                acc_o=acc_o,
                row_max=row_max,
                row_sum=row_sum,
                n_block=n_block,
                n_block_min=block_sched.n_block_min,
                IS_MASK=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
                CHECK_INF=True if HAS_GATHER_KV else False,
                HAS_GATHER_KV=HAS_GATHER_KV,
            )

    if IS_LOCAL and not HAS_GATHER_KV:
        # Process n_blocks with local right masking
        if block_sched.n_block_window_max > block_sched.n_block_window_max_no_mask:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, k_ptrs, block_sched.n_block_window_max - 1
            )

            for n_block in tl.range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                k_tile, topk_indices, acc_o, row_max, row_sum = (
                    _dec_inner_sparse_kernel(
                        config=config,
                        ptrs_sched=ptrs_sched,
                        mask_sched=mask_sched,
                        softmax_sched=softmax_sched,
                        q_tile=q_tile,
                        k_tile=k_tile,
                        topk_indices=topk_indices,
                        k_ptrs=k_ptrs,
                        v_ptrs=v_ptrs,
                        acc_o=acc_o,
                        row_max=row_max,
                        row_sum=row_sum,
                        n_block=n_block,
                        n_block_min=block_sched.n_block_window_max_no_mask,
                        IS_MASK=True,
                        MASK_LOCAL=True,
                        MASK_SINK=False,
                        CHECK_INF=True,
                        HAS_GATHER_KV=False,
                    )
                )

        # Process n_blocks without masking
        if (
            block_sched.n_block_window_max_no_mask
            > block_sched.n_block_window_min_no_mask
        ):
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, k_ptrs, block_sched.n_block_window_max_no_mask - 1
            )

            for n_block in tl.range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                k_tile, topk_indices, acc_o, row_max, row_sum = (
                    _dec_inner_sparse_kernel(
                        config=config,
                        ptrs_sched=ptrs_sched,
                        mask_sched=mask_sched,
                        softmax_sched=softmax_sched,
                        q_tile=q_tile,
                        k_tile=k_tile,
                        topk_indices=topk_indices,
                        k_ptrs=k_ptrs,
                        v_ptrs=v_ptrs,
                        acc_o=acc_o,
                        row_max=row_max,
                        row_sum=row_sum,
                        n_block=n_block,
                        n_block_min=block_sched.n_block_window_min_no_mask,
                        IS_MASK=False,
                        MASK_LOCAL=False,
                        MASK_SINK=False,
                        CHECK_INF=False,
                        HAS_GATHER_KV=False,
                    )
                )

        # Process n_blocks with local left masking
        if block_sched.n_block_window_min_no_mask > block_sched.n_block_window_min:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, k_ptrs, block_sched.n_block_window_min_no_mask - 1
            )

            for n_block in tl.range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                k_tile, topk_indices, acc_o, row_max, row_sum = (
                    _dec_inner_sparse_kernel(
                        config=config,
                        ptrs_sched=ptrs_sched,
                        mask_sched=mask_sched,
                        softmax_sched=softmax_sched,
                        q_tile=q_tile,
                        k_tile=k_tile,
                        topk_indices=topk_indices,
                        k_ptrs=k_ptrs,
                        v_ptrs=v_ptrs,
                        acc_o=acc_o,
                        row_max=row_max,
                        row_sum=row_sum,
                        n_block=n_block,
                        n_block_min=block_sched.n_block_window_min,
                        IS_MASK=True,
                        MASK_LOCAL=True,
                        MASK_SINK=False,
                        CHECK_INF=True,
                        HAS_GATHER_KV=False,
                    )
                )

        # Process n_blocks with local sink masking
        if block_sched.n_block_sink_max > block_sched.n_block_sink_min:
            # Load key tile
            k_tile = ptrs_sched.load_k(config, k_ptrs, block_sched.n_block_sink_max - 1)

            for n_block in tl.range(
                block_sched.n_block_sink_max - 1,
                block_sched.n_block_sink_min - 1,
                -1,
            ):
                k_tile, topk_indices, acc_o, row_max, row_sum = (
                    _dec_inner_sparse_kernel(
                        config=config,
                        ptrs_sched=ptrs_sched,
                        mask_sched=mask_sched,
                        softmax_sched=softmax_sched,
                        q_tile=q_tile,
                        k_tile=k_tile,
                        topk_indices=topk_indices,
                        k_ptrs=k_ptrs,
                        v_ptrs=v_ptrs,
                        acc_o=acc_o,
                        row_max=row_max,
                        row_sum=row_sum,
                        n_block=n_block,
                        n_block_min=block_sched.n_block_sink_min,
                        IS_MASK=True,
                        MASK_LOCAL=True,
                        MASK_SINK=True,
                        CHECK_INF=True,
                        HAS_GATHER_KV=False,
                    )
                )

    # Finalize softmax
    row_scale, lse_tile = softmax_sched.finalize(
        row_max=row_max,
        row_sum=row_sum,
        IS_LOG2=True,
    )

    # Store LSE
    ptrs_sched.store_lse(config, lse_ptrs, lse_tile)

    # Finalize rescale
    acc_o = softmax_sched.rescale_o(
        acc_o=acc_o,
        row_scale=row_scale,
    )

    # Store output
    ptrs_sched.store_out(config, out_ptrs, acc_o)


_dec_sparse_kernel = cache_utils.wrap_kernel(_dec_sparse_kernel)


_dec_sparse_kernel_autotuned = None


def _get_autotuned_kernel():
    global _dec_sparse_kernel_autotuned
    if _dec_sparse_kernel_autotuned is None:
        jit_kernel = _dec_sparse_kernel._kernel
        autotuned = autotuner.make_dec_sparse_autotuned_kernel(jit_kernel)
        _dec_sparse_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _dec_sparse_kernel_autotuned


def _flash_sparse_attn_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    gather_kv_indices: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    topk_seqlen_k = (
        gather_kv_indices.shape[-1] if gather_kv_indices is not None else seqlen_k
    )
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else 1 / seqlen_k
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 4), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_dec_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_k=None,
            seqused_k=None,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="dec_sparse",
        seqlen_q=1,
        seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _dec_sparse_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        TILE_M = max(triton.next_power_of_2(qhead_per_kvhead), 16)
        TILE_N = 128
        num_warps = num_stages = num_ctas = None

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        max_split_blocks=utils.max_split_blocks_from_window_sizes(window_sizes, TILE_N)
        if is_local
        else None,
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = (
        out
        if out is not None
        else torch.empty(query.shape, dtype=out_dtype, device=device)
    )
    lse = (
        lse
        if lse is not None
        else torch.empty((batch_size, num_heads_q), dtype=torch.float32, device=device)
    )

    out_partial = torch.empty(
        (num_splits, batch_size, num_heads_q, head_dim),
        dtype=torch.float32,
        device=device,
    )
    lse_partial = torch.empty(
        (num_splits, batch_size, num_heads_q),
        dtype=torch.float32,
        device=device,
    )

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    grid = launch_grid.get_dec_grid(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        out_partial,
        lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        window_sizes,
        gather_kv_indices,
        query.stride(0),
        query.stride(-2),
        1,
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        window_sizes.stride(0),
        gather_kv_indices.stride(0) if gather_kv_indices is not None else 0,
        gather_kv_indices.stride(-1) if gather_kv_indices is not None else 0,
        None,
        None,
        None,
        None,
        num_splits,
        seqlen_q=1,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=0,
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        topk_seqlen_k=topk_seqlen_k,
        IS_LOCAL=is_local,
        HAS_GATHER_KV=gather_kv_indices is not None,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="dec_sparse",
                seqlen_q=1,
                seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_quant=is_quant,
            )

    flash_dec_combine._flash_attn_dec_combine(
        out_partial,
        lse_partial,
        out,
        lse,
    )

    return out, lse


def _flash_sparse_attn_varlen_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    seqlen_k = max_seqlen_k
    topk_seqlen_k = (
        gather_kv_indices.shape[-1] if gather_kv_indices is not None else seqlen_k
    )
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else 1 / seqlen_k
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 4), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_dec_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="dec_sparse",
        seqlen_q=1,
        seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _dec_sparse_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        TILE_M = max(triton.next_power_of_2(qhead_per_kvhead), 16)
        TILE_N = 128
        num_warps = num_stages = num_ctas = None

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        max_split_blocks=utils.max_split_blocks_from_window_sizes(window_sizes, TILE_N)
        if is_local
        else None,
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = (
        out
        if out is not None
        else torch.empty(query.shape, dtype=out_dtype, device=device)
    )
    lse = (
        lse
        if lse is not None
        else torch.empty((batch_size, num_heads_q), dtype=torch.float32, device=device)
    )

    out_partial = torch.empty(
        (num_splits, batch_size, num_heads_q, head_dim),
        dtype=torch.float32,
        device=device,
    )
    lse_partial = torch.empty(
        (num_splits, batch_size, num_heads_q),
        dtype=torch.float32,
        device=device,
    )

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    grid = launch_grid.get_dec_grid(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        out_partial,
        lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        window_sizes,
        gather_kv_indices,
        query.stride(0),
        query.stride(-2),
        1,
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        window_sizes.stride(0),
        gather_kv_indices.stride(0) if gather_kv_indices is not None else 0,
        gather_kv_indices.stride(-1) if gather_kv_indices is not None else 0,
        None,
        cu_seqlens_k,
        None,
        seqused_k,
        num_splits,
        seqlen_q=1,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=0,
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        topk_seqlen_k=topk_seqlen_k,
        IS_LOCAL=is_local,
        HAS_GATHER_KV=gather_kv_indices is not None,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="dec_sparse",
                seqlen_q=1,
                seqlen_k=seqlen_k if gather_kv_indices is None else topk_seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_quant=is_quant,
            )

    flash_dec_combine._flash_attn_dec_combine(
        out_partial,
        lse_partial,
        out,
        lse,
    )

    return out, lse
