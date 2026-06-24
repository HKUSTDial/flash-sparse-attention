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
    flash_fwd_combine,
    kernel_repr,
    autotuner,
)
from flash_sparse_attn.ops.triton.scheduler import (
    AttnFwdGridIndex,
    AttnFwdConfig,
    AttnFwdBlockScheduler,
    AttnFwdPointerScheduler,
    AttnMaskScheduler,
    SoftmaxScheduler,
)


@triton.jit
def _fwd_inner_dense_kernel(
    config: AttnFwdConfig,
    ptrs_sched: AttnFwdPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    q_tile,
    k_tile,
    k_ptrs,
    v_ptrs,
    acc_o,
    row_max,
    row_sum,
    n_block,
    n_block_min,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    MASK_SINK: tl.constexpr,
    CHECK_INF: tl.constexpr,
):
    # Compute attention scores
    acc_s = tl.dot(q_tile, k_tile.T)

    # Prefetch next key tile
    if n_block > n_block_min:
        # Load next key tile
        k_tile = ptrs_sched.load_k(config, k_ptrs, n_block - 1)

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask_sched.apply_mask(
            acc_s=acc_s,
            iter_block=n_block,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Apply online softmax
    p, row_max, row_sum, row_scale = softmax_sched.online_softmax(
        acc_s=acc_s,
        row_max=row_max,
        row_sum=row_sum,
        CHECK_INF=CHECK_INF,
    )

    # Load value tile
    v_tile = ptrs_sched.load_v(config, v_ptrs, n_block)

    # Rescale output accumulator
    acc_o = softmax_sched.rescale_o(
        acc_o=acc_o,
        row_scale=row_scale,
    )

    # Update output accumulator
    acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

    return k_tile, acc_o, row_max, row_sum


@triton.jit(repr=kernel_repr.fwd_dense_repr)
def _fwd_dense_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    window_sizes,
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
    stride_ls,
    stride_wh,
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
    QHEAD_PER_KVHEAD: tl.constexpr,
    PACK_GQA: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_QUANT: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnFwdGridIndex.create(
        num_splits=num_splits,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        IS_SPLIT_KV=IS_SPLIT_KV,
        PACK_GQA=PACK_GQA,
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
    config = AttnFwdConfig.create(
        softmax_scale=softmax_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        m_block=grid_idx.m_block,
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
        PACK_GQA=PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
        IS_QUANT=IS_QUANT,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create pointer scheduler
    ptrs_sched = AttnFwdPointerScheduler.create(
        config=config,
        Q=Q,
        K=K,
        V=V,
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
        stride_ls=stride_ls,
        IS_SPLIT_KV=IS_SPLIT_KV,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnFwdBlockScheduler.create(
        config=config,
        split_idx=grid_idx.split_idx,
        num_splits=num_splits,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=IS_SPLIT_KV,
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
        ptrs_sched.store_empty(config, out_ptrs, lse_ptrs, IS_SPLIT_KV=IS_SPLIT_KV)
        return

    q_ptrs = ptrs_sched.make_q_ptrs(config)
    k_ptrs = ptrs_sched.make_k_ptrs(config)
    v_ptrs = ptrs_sched.make_v_ptrs(config)

    # Initialize accumulators
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load query tile
    q_tile = ptrs_sched.load_q(config, q_ptrs)

    # Load key tile
    k_tile = ptrs_sched.load_k(config, k_ptrs, block_sched.n_block_max - 1)

    # Process n_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        n_block_max_no_mask = block_sched.n_block_max_no_mask
        for n_block in tl.range(
            block_sched.n_block_max - 1, block_sched.n_block_max_no_mask - 1, -1
        ):
            k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                q_tile=q_tile,
                k_tile=k_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                acc_o=acc_o,
                row_max=row_max,
                row_sum=row_sum,
                n_block=n_block,
                n_block_min=block_sched.n_block_max_no_mask,
                IS_MASK=True,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=True if IS_LOCAL else False,
                MASK_SINK=False,
                CHECK_INF=True,
            )
    else:
        # First iteration with seqlen masking
        n_block = block_sched.n_block_max - 1
        # Triton aggregate does not support attribute reassignment yet
        # block_sched.n_block_max_no_mask = n_block
        n_block_max_no_mask = n_block

        k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
            config=config,
            ptrs_sched=ptrs_sched,
            mask_sched=mask_sched,
            softmax_sched=softmax_sched,
            q_tile=q_tile,
            k_tile=k_tile,
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
            acc_o=acc_o,
            row_max=row_max,
            row_sum=row_sum,
            n_block=n_block,
            n_block_min=n_block,
            IS_MASK=True,
            MASK_CAUSAL=False,
            MASK_LOCAL=False,
            MASK_SINK=False,
            CHECK_INF=True,
        )

    # Process n_blocks without masking
    if not IS_LOCAL and n_block_max_no_mask > block_sched.n_block_min:
        # Load key tile
        k_tile = ptrs_sched.load_k(config, k_ptrs, n_block_max_no_mask - 1)

        for n_block in tl.range(
            n_block_max_no_mask - 1, block_sched.n_block_min - 1, -1
        ):
            k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                q_tile=q_tile,
                k_tile=k_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                acc_o=acc_o,
                row_max=row_max,
                row_sum=row_sum,
                n_block=n_block,
                n_block_min=block_sched.n_block_min,
                IS_MASK=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
                CHECK_INF=False,
            )

    if IS_LOCAL:
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
                k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_max_no_mask,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                    CHECK_INF=True,
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
                k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_min_no_mask,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                    CHECK_INF=False,
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
                k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_min,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                    CHECK_INF=True,
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
                k_tile, acc_o, row_max, row_sum = _fwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    q_tile=q_tile,
                    k_tile=k_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    acc_o=acc_o,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_sink_min,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                    CHECK_INF=True,
                )

    # Finalize softmax
    row_scale, lse_tile = softmax_sched.finalize(
        row_max=row_max,
        row_sum=row_sum,
        IS_LOG2=IS_SPLIT_KV,
    )

    # Store LSE
    ptrs_sched.store_lse(config, lse_ptrs, lse_tile)

    # Finalize rescale
    acc_o = softmax_sched.rescale_o(
        acc_o=acc_o,
        row_scale=row_scale,
    )

    # Store output
    ptrs_sched.store_out(config, out_ptrs, acc_o, IS_SPLIT_KV=IS_SPLIT_KV)


_fwd_dense_kernel = cache_utils.wrap_kernel(_fwd_dense_kernel)


_fwd_dense_kernel_autotuned = None


def _get_autotuned_kernel():
    global _fwd_dense_kernel_autotuned
    if _fwd_dense_kernel_autotuned is None:
        jit_kernel = _fwd_dense_kernel._kernel
        autotuned = autotuner.make_fwd_dense_autotuned_kernel(jit_kernel)
        _fwd_dense_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _fwd_dense_kernel_autotuned


def _flash_dense_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    _kernel_name = f"fwd_dense{'_split' if is_split_kv else ''}"
    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name=_kernel_name,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        pack_gqa=pack_gqa,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _fwd_dense_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            max_split_blocks=utils.max_split_blocks_from_window_sizes(
                window_sizes, TILE_N
            )
            if is_local
            else None,
        )
        if is_split_kv
        else 1
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = out if out is not None else torch.empty_like(query, dtype=out_dtype)
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )
    )

    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, batch_size, seqlen_q, num_heads_q, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        lse_partial = torch.empty(
            (num_splits, batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        query.stride(0),
        query.stride(-2),
        query.stride(-3),
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        out.stride(0) if not is_split_kv else out_partial.stride(1),
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(-3) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        lse.stride(0) if not is_split_kv else lse_partial.stride(1),
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
        window_sizes.stride(0) if window_sizes is not None else 0,
        None,
        None,
        None,
        None,
        num_splits,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=max(triton.next_power_of_2(seqlen_q), 256),
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_QUANT=is_quant,
        IS_SPLIT_KV=is_split_kv,
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
                kernel_name=_kernel_name,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                pack_gqa=pack_gqa,
                is_quant=is_quant,
            )

    if is_split_kv:
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
        )

    return out, lse, softmax_scale


def _flash_dense_attn_varlen_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    total_seqlen_q, num_heads_q, head_dim = query.shape
    _, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            query,
            key,
            value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            is_quant=is_quant,
            device=device,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    _kernel_name = f"fwd_dense{'_split' if is_split_kv else ''}"
    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name=_kernel_name,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        pack_gqa=pack_gqa,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _fwd_dense_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
            seqlen_q=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            max_split_blocks=utils.max_split_blocks_from_window_sizes(
                window_sizes, TILE_N
            )
            if is_local
            else None,
        )
        if is_split_kv
        else 1
    )

    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = torch.empty_like(query, dtype=out_dtype)
    lse = torch.empty(
        (num_heads_q, total_seqlen_q),
        dtype=torch.float32,
        device=query.device,
    )

    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, total_seqlen_q, num_heads_q, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        lse_partial = torch.empty(
            (num_splits, num_heads_q, total_seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )

    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        0,
        query.stride(-2),
        query.stride(0),
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        0,
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(0) if not is_split_kv else out_partial.stride(-3),
        0 if not is_split_kv else out_partial.stride(0),
        0,
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(0),
        window_sizes.stride(0) if window_sizes is not None else 0,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        num_splits,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=max(triton.next_power_of_2(seqlen_q), 256),
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_QUANT=is_quant,
        IS_SPLIT_KV=is_split_kv,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
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
                kernel_name=_kernel_name,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                pack_gqa=pack_gqa,
                is_quant=is_quant,
            )

    if is_split_kv:
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
        )

    return out, lse, softmax_scale
