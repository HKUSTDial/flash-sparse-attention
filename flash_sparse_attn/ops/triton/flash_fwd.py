from typing import Tuple, Optional

import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    kernel_cache,
    utils,
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
def _fwd_inner_kernel(
    config: AttnFwdConfig,
    ptrs_sched: AttnFwdPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    q_tile,
    k_tile,
    acc_o,
    row_max,
    row_sum,
    n_block,
    n_block_min,
    IS_MASK: tl.constexpr = False,
    MASK_SEQLEN: tl.constexpr = False,
    MASK_CAUSAL: tl.constexpr = False,
    MASK_LOCAL: tl.constexpr = False,
    MASK_SINK: tl.constexpr = False,
):
    # Compute attention scores
    acc_s = tl.dot(q_tile, k_tile.T)

    if n_block > n_block_min:
        # Load next key tile
        k_tile = ptrs_sched.load_k(config, ptrs_sched.make_k_ptrs(config), n_block - 1)

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask_sched.apply_mask(
            acc_s,
            n_block,
            offs_m=config.offs_m(),
            offs_n=config.offs_n(),
            MASK_SEQLEN=MASK_SEQLEN,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    if not config.IS_SPARSE:
        is_skip: tl.constexpr = False
        row_max_curr: tl.constexpr = None
    else:
        # Determine skip condition
        is_skip, row_max_curr = softmax_sched.score_skip(
            acc_s, row_max, row_sum, config.get_softmax_threshold_log2()
        )

    if not is_skip:
        # Load value tile
        v_tile = ptrs_sched.load_v(config, ptrs_sched.make_v_ptrs(config), n_block)

        # Apply online softmax
        p, row_max, row_sum, row_scale = softmax_sched.online_softmax(
            acc_s, row_max, row_sum, row_max_curr=row_max_curr, CHECK_INF=IS_MASK
        )

        # Rescale output accumulator
        acc_o = softmax_sched.rescale_o(acc_o, row_scale)

        # Update output accumulator
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

    return k_tile, acc_o, row_max, row_sum


@triton.jit(repr=kernel_repr.fwd_repr)
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    WindowSizes,
    CuSeqlensQ,
    CuSeqlensK,
    SeqUsedQ,
    SeqUsedK,
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
    seqlen_q,
    seqlen_k,
    softmax_scale,
    softmax_threshold,
    q_scale,
    k_scale,
    v_scale,
    head_dim,
    QHEAD_PER_KVHEAD: tl.constexpr,
    PACK_GQA: tl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    IS_SPLIT_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPARSE: tl.constexpr,
    IS_QUANT: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    SEQLEN_Q_CACHE: tl.constexpr,
    SEQLEN_K_CACHE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnFwdGridIndex.create(
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        NUM_SPLITS=NUM_SPLITS,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        PACK_GQA=PACK_GQA,
        IS_SPLIT_KV=IS_SPLIT_KV,
        IS_LPT=not PACK_GQA and not IS_SPLIT_KV,
    )

    # Load window sizes
    window_size_sink, window_size_left, window_size_right, window_size_near = (
        grid_idx.load_window_sizes(WindowSizes, stride_wh, IS_LOCAL=IS_LOCAL)
    )

    # Create config
    config = AttnFwdConfig.create(
        batch_idx=grid_idx.batch_idx,
        head_idx=grid_idx.head_idx,
        head_kv_idx=grid_idx.head_kv_idx,
        split_idx=grid_idx.split_idx,
        m_block=grid_idx.m_block,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        query_scale=q_scale,
        key_scale=k_scale,
        value_scale=v_scale,
        window_size_sink=window_size_sink,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        window_size_near=window_size_near,
        head_dim=head_dim,
        cu_seqlens_q=CuSeqlensQ,
        cu_seqlens_k=CuSeqlensK,
        seqused_q=SeqUsedQ,
        seqused_k=SeqUsedK,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        PACK_GQA=PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        NUM_SPLITS=NUM_SPLITS,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPARSE=IS_SPARSE,
        IS_SPLIT_KV=IS_SPLIT_KV,
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
    )

    # Create block scheduler
    block_sched = AttnFwdBlockScheduler.create(config)

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Early exit if no n_blocks to process
    if block_sched.is_empty():
        ptrs_sched.store_empty(
            config,
            ptrs_sched.make_out_ptrs(config),
            ptrs_sched.make_lse_ptrs(config),
        )
        return

    # Initialize accumulators
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load query tile
    q_tile = ptrs_sched.load_q(config, ptrs_sched.make_q_ptrs(config))

    # First iteration with seqlen masking
    n_block_max = block_sched.n_block_max - 1

    # Load key tile
    k_tile = ptrs_sched.load_k(config, ptrs_sched.make_k_ptrs(config), block_sched.n_block_max - 1)

    # Process n_block with seqlen masking
    k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
        config,
        ptrs_sched,
        mask_sched,
        softmax_sched,
        q_tile,
        k_tile,
        acc_o,
        row_max,
        row_sum,
        n_block_max,
        block_sched.n_block_min,
        IS_MASK=True,
        MASK_SEQLEN=True,
        MASK_CAUSAL=IS_CAUSAL,
        MASK_LOCAL=IS_LOCAL,
        MASK_SINK=IS_LOCAL,
    )

    if IS_CAUSAL or IS_LOCAL:
        # Process n_blocks with causal masking
        for n_block in tl.range(
            n_block_max - 1,
            block_sched.n_block_max_no_mask - 1,
            -1,
        ):
            k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                q_tile,
                k_tile,
                acc_o,
                row_max,
                row_sum,
                n_block,
                block_sched.n_block_min,
                IS_MASK=True,
                MASK_SEQLEN=False,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=IS_LOCAL,
                MASK_SINK=False,
            )

    if not IS_LOCAL and block_sched.n_block_max_no_mask > block_sched.n_block_min:
        # Process n_blocks without masking
        for n_block in tl.range(
            block_sched.n_block_max_no_mask - 1,
            block_sched.n_block_min - 1,
            -1,
        ):
            k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                q_tile,
                k_tile,
                acc_o,
                row_max,
                row_sum,
                n_block,
                block_sched.n_block_min,
                IS_MASK=False,
                MASK_SEQLEN=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

    if IS_LOCAL:
        if block_sched.n_block_window_max > block_sched.n_block_window_min:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, ptrs_sched.make_k_ptrs(config), block_sched.n_block_window_max - 1
            )

            # Process n_blocks with local right masking
            for n_block in tl.range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    q_tile,
                    k_tile,
                    acc_o,
                    row_max,
                    row_sum,
                    n_block,
                    block_sched.n_block_window_min,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

            # Process n_blocks without masking
            for n_block in tl.range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    q_tile,
                    k_tile,
                    acc_o,
                    row_max,
                    row_sum,
                    n_block,
                    block_sched.n_block_window_min,
                    IS_MASK=False,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                )

            # Process n_blocks with local left masking
            for n_block in tl.range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    q_tile,
                    k_tile,
                    acc_o,
                    row_max,
                    row_sum,
                    n_block,
                    block_sched.n_block_window_min,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

        if block_sched.n_block_sink_max > block_sched.n_block_sink_min:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, ptrs_sched.make_k_ptrs(config), block_sched.n_block_sink_max - 1
            )

            # Process n_blocks with prefix-sink masking
            for n_block in tl.range(
                block_sched.n_block_sink_max - 1,
                block_sched.n_block_sink_min - 1,
                -1,
            ):
                k_tile, acc_o, row_max, row_sum = _fwd_inner_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    q_tile,
                    k_tile,
                    acc_o,
                    row_max,
                    row_sum,
                    n_block,
                    block_sched.n_block_sink_min,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

    # Finalize softmax
    row_scale, lse_tile = softmax_sched.finalize(row_max, row_sum, IS_LOG2=IS_SPLIT_KV)

    # Store LSE
    ptrs_sched.store_lse(config, ptrs_sched.make_lse_ptrs(config), lse_tile)

    # Finalize rescale
    acc_o = softmax_sched.rescale_o(acc_o, row_scale)

    # Store O
    ptrs_sched.store_out(config, ptrs_sched.make_out_ptrs(config), acc_o)


_fwd_kernel_autotuned = None


def _get_autotuned_kernel():
    global _fwd_kernel_autotuned
    if _fwd_kernel_autotuned is None:
        autotuned = autotuner.make_fwd_autotuned_kernel(_fwd_kernel)
        _fwd_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _fwd_kernel_autotuned


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    is_local: bool = False,
    is_sparse: bool = False,
    is_split_kv: bool = False,
    is_quant: bool = False,
    pack_gqa: bool = False,
    softmax_scale: Optional[float] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    num_splits: Optional[int] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    is_autotune: bool = True,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    num_SMs = utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

    # Assert the validity of inputs
    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            q=q,
            k=k,
            v=v,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            window_sizes=window_sizes,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    seqlen_q_cache = autotuner.get_seqlen_bucket(seqlen_q)
    seqlen_k_cache = autotuner.get_seqlen_bucket(seqlen_k)
    kernel_name = f"fwd{'_split' if is_split_kv else ''}"
    launch_config = None
    if not is_autotune:
        kernel = _fwd_kernel
        TILE_M = tile_m if tile_m is not None else 128
        TILE_N = tile_n if tile_n is not None else 64
        num_warps, num_stages, num_ctas = 4, 1, 1
    else:
        launch_config = kernel_cache._launch_config_cache.load(
            device=device,
            kernel_name=kernel_name,
            seqlen_q=seqlen_q_cache,
            seqlen_k=seqlen_k_cache,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            is_causal=is_causal,
            is_local=is_local,
            is_quant=is_quant,
            tile_k=TILE_K,
        )
        if launch_config is not None:
            kernel = _fwd_kernel
            TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
        else:
            kernel = _get_autotuned_kernel()
            # Placeholder for pre-launch computations
            TILE_M = 128
            TILE_N = 64
            num_warps = num_stages = num_ctas = None
    if is_split_kv and num_splits is None:
        num_splits = utils.num_splits_heuristic(
            batch_size=batch_size,
            seqlen_parallel=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_loop=seqlen_k,
            num_heads_parallel=num_heads_q // qhead_per_kvhead_packgqa,
            num_heads_kv=num_heads_kv,
            num_SMs=num_SMs,
            TILE_PARALLEL=TILE_M,
            TILE_LOOP=TILE_N,
            is_local=is_local,
        )
        is_split_kv = True if num_splits > 1 else False
    elif not is_split_kv:
        num_splits = 1

    # Allocate outputs
    out = (
        out
        if out is not None
        else torch.empty_like(q, dtype=q_scale.dtype if is_quant else q.dtype)
    )
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=q.device,
        )
    )
    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, batch_size, num_heads_q, seqlen_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        lse_partial = torch.empty(
            (num_splits, batch_size, num_heads_q, seqlen_q),
            dtype=torch.float32,
            device=q.device,
        )

    # Get the grid for the kernel launch
    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    # Launch the kernel
    kernel[grid](
        q,
        k,
        v,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        window_sizes,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        seqused_k,
        q.stride(-4),
        q.stride(-2),
        q.stride(-3),
        k.stride(-4),
        k.stride(-2),
        k.stride(-3),
        v.stride(-4),
        v.stride(-2),
        v.stride(-3),
        out.stride(-4) if not is_split_kv else out_partial.stride(-4),
        out.stride(-2) if not is_split_kv else out_partial.stride(-3),
        out.stride(-3) if not is_split_kv else out_partial.stride(-2),
        0 if not is_split_kv else out_partial.stride(-5),
        lse.stride(-3) if not is_split_kv else lse_partial.stride(-3),
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(-4),
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        softmax_threshold,
        q_scale,
        k_scale,
        v_scale,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPARSE=is_sparse,
        IS_SPLIT_KV=is_split_kv,
        IS_QUANT=is_quant,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        SEQLEN_Q_CACHE=seqlen_q_cache,
        SEQLEN_K_CACHE=seqlen_k_cache,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_autotune and launch_config is None:
        best_config = _get_autotuned_kernel().best_config
        if best_config is not None:
            kernel_cache._launch_config_cache.store(
                config=(
                    best_config.kwargs["TILE_M"],
                    best_config.kwargs["TILE_N"],
                    best_config.num_warps,
                    best_config.num_stages,
                    best_config.num_ctas,
                ),
                device=device,
                kernel_name=kernel_name,
                seqlen_q=seqlen_q_cache,
                seqlen_k=seqlen_k_cache,
                qhead_per_kvhead=qhead_per_kvhead,
                pack_gqa=pack_gqa,
                is_causal=is_causal,
                is_local=is_local,
                is_quant=is_quant,
                tile_k=TILE_K,
            )

    if is_split_kv:
        # Combine partial O and LSE
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
        )

    return out, lse


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    is_local: bool = False,
    is_sparse: bool = False,
    is_split_kv: bool = False,
    is_quant: bool = False,
    pack_gqa: bool = False,
    softmax_scale: Optional[float] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    num_splits: Optional[int] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    is_autotune: bool = True,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = q.device
    num_SMs = utils.get_device_num_sms(device)
    total_q, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

    # Assert the validity of inputs
    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            q=q,
            k=k,
            v=v,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            window_sizes=window_sizes,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    # Setup launch configuration
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    seqlen_q_cache = autotuner.get_seqlen_bucket(seqlen_q)
    seqlen_k_cache = autotuner.get_seqlen_bucket(seqlen_k)
    kernel_name = f"fwd{'_split' if is_split_kv else ''}"
    launch_config = None
    if not is_autotune:
        kernel = _fwd_kernel
        TILE_M = tile_m if tile_m is not None else 128
        TILE_N = tile_n if tile_n is not None else 64
        num_warps, num_stages, num_ctas = 4, 1, 1
    else:
        launch_config = kernel_cache._launch_config_cache.load(
            device=device,
            kernel_name=kernel_name,
            seqlen_q=seqlen_q_cache,
            seqlen_k=seqlen_k_cache,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            is_causal=is_causal,
            is_local=is_local,
            is_quant=is_quant,
            tile_k=TILE_K,
        )
        if launch_config is not None:
            kernel = _fwd_kernel
            TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
        else:
            kernel = _get_autotuned_kernel()
            # Placeholder for pre-launch computations
            TILE_M = 128
            TILE_N = 64
            num_warps = num_stages = num_ctas = None
    if is_split_kv and num_splits is None:
        num_splits = utils.num_splits_heuristic(
            batch_size=batch_size,
            seqlen_parallel=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_loop=seqlen_k,
            num_heads_parallel=num_heads_q // qhead_per_kvhead_packgqa,
            num_heads_kv=num_heads_kv,
            num_SMs=num_SMs,
            TILE_PARALLEL=TILE_M,
            TILE_LOOP=TILE_N,
            is_local=is_local,
        )
        is_split_kv = True if num_splits > 1 else False
    elif not is_split_kv:
        num_splits = 1

    # Allocate outputs
    out = torch.empty_like(q, dtype=q_scale.dtype if is_quant else q.dtype)
    lse = torch.empty(
        (num_heads_q, total_q),
        dtype=torch.float32,
        device=q.device,
    )
    if is_split_kv:
        out_partial = torch.empty(
            (num_splits, num_heads_q, total_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        lse_partial = torch.empty(
            (num_splits, num_heads_q, total_q),
            dtype=torch.float32,
            device=q.device,
        )

    # Get the grid for the kernel launch
    grid = launch_grid.get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    # Launch the kernel
    kernel[grid](
        q,
        k,
        v,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        window_sizes,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        0,
        q.stride(-2),
        q.stride(-3),
        0,
        k.stride(-2),
        k.stride(-3),
        0,
        v.stride(-2),
        v.stride(-3),
        0,
        out.stride(-2) if not is_split_kv else out_partial.stride(-3),
        out.stride(-3) if not is_split_kv else out_partial.stride(-2),
        0 if not is_split_kv else out_partial.stride(-4),
        0,
        lse.stride(-2) if not is_split_kv else lse_partial.stride(-2),
        0 if not is_split_kv else lse_partial.stride(-3),
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        softmax_threshold,
        q_scale,
        k_scale,
        v_scale,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPARSE=is_sparse,
        IS_SPLIT_KV=is_split_kv,
        IS_QUANT=is_quant,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        SEQLEN_Q_CACHE=seqlen_q_cache,
        SEQLEN_K_CACHE=seqlen_k_cache,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_autotune and launch_config is None:
        best_config = _get_autotuned_kernel().best_config
        if best_config is not None:
            kernel_cache._launch_config_cache.store(
                config=(
                    best_config.kwargs["TILE_M"],
                    best_config.kwargs["TILE_N"],
                    best_config.num_warps,
                    best_config.num_stages,
                    best_config.num_ctas,
                ),
                device=device,
                kernel_name=kernel_name,
                seqlen_q=seqlen_q_cache,
                seqlen_k=seqlen_k_cache,
                qhead_per_kvhead=qhead_per_kvhead,
                pack_gqa=pack_gqa,
                is_causal=is_causal,
                is_local=is_local,
                is_quant=is_quant,
                tile_k=TILE_K,
            )

    if is_split_kv:
        # Combine partial O and LSE
        flash_fwd_combine._flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
        )

    return out, lse, softmax_scale
