from typing import Optional, Tuple

import math
import torch
import triton
import triton.language as tl

from flash_sparse_attn.ops.triton import (
    assert_inputs,
    utils,
    cache_utils,
    launch_template,
    launch_grid,
    activations,
    flash_bwd_preprocess,
    flash_bwd_postprocess,
    kernel_repr,
    autotuner,
)
from flash_sparse_attn.ops.triton.scheduler import (
    AttnBwdGridIndex,
    AttnBwdConfig,
    AttnBwdBlockScheduler,
    AttnBwdPointerScheduler,
    AttnMaskScheduler,
)


@triton.jit
def _bwd_inner_dense_kernel(
    config,
    ptrs_sched,
    mask_sched,
    acc_dk,
    acc_dv,
    k_tile,
    v_tile,
    q_ptrs,
    do_ptrs,
    dq_accum_ptrs,
    lse_ptrs,
    dpsum_ptrs,
    m_block,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    MASK_SINK: tl.constexpr,
):
    # Load query tile
    q_tile = ptrs_sched.load_q(config, q_ptrs, m_block)

    # Compute attention scores
    acc_s = tl.dot(k_tile, q_tile.T)

    # Load LSE
    lse_log2 = ptrs_sched.load_lse(config, lse_ptrs, m_block)

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask_sched.apply_mask(
            acc_s=acc_s,
            iter_block=m_block,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Compute attention weights
    p = activations.exp2(acc_s * config.softmax_scale_log2 - lse_log2[None, :]).to(
        q_tile.dtype
    )

    # Load output gradients tile
    do_tile = ptrs_sched.load_do(config, do_ptrs, m_block)

    # Compute value gradients
    acc_dv += tl.dot(p, do_tile)

    # Compute attention weight gradients
    acc_dp = tl.dot(v_tile, tl.trans(do_tile))

    # Load dpsum
    dpsum = ptrs_sched.load_dpsum(config, dpsum_ptrs, m_block)

    # Compute attention score gradients
    ds = p * (acc_dp - dpsum[None, :]).to(q_tile.dtype)

    # Compute query gradients
    dq = tl.dot(tl.trans(ds), k_tile)

    # Store query gradients
    ptrs_sched.store_dq(config, dq_accum_ptrs, m_block, dq)

    # Compute key gradients
    acc_dk += tl.dot(ds, q_tile)

    return acc_dk, acc_dv


@triton.jit(repr=kernel_repr.bwd_dense_repr)
def _bwd_dense_kernel(
    Q,
    K,
    V,
    dO,
    LSELog2,
    dPsum,
    dQaccum,
    dK,
    dV,
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
    stride_dob,
    stride_doh,
    stride_dom,
    stride_lb,
    stride_lh,
    stride_pb,
    stride_ph,
    stride_dqab,
    stride_dqah,
    stride_dqam,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dks,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvs,
    stride_wh,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    seqlen_q,
    seqlen_k,
    head_dim,
    SEQLEN_Q_CACHE: tl.constexpr,
    SEQLEN_K_CACHE: tl.constexpr,
    QHEAD_PER_KVHEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_QUANT: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnBwdGridIndex.create(
        NUM_SPLITS=NUM_SPLITS,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        IS_SPLIT_QO=IS_SPLIT_QO,
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
    config = AttnBwdConfig.create(
        softmax_scale=softmax_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        n_block=grid_idx.n_block,
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
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
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
    ptrs_sched = AttnBwdPointerScheduler.create(
        config=config,
        Q=Q,
        K=K,
        V=V,
        dO=dO,
        LSELog2=LSELog2,
        dPsum=dPsum,
        dQaccum=dQaccum,
        dK=dK,
        dV=dV,
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
        stride_dob=stride_dob,
        stride_doh=stride_doh,
        stride_dom=stride_dom,
        stride_lb=stride_lb,
        stride_lh=stride_lh,
        stride_pb=stride_pb,
        stride_ph=stride_ph,
        stride_dqab=stride_dqab,
        stride_dqah=stride_dqah,
        stride_dqam=stride_dqam,
        stride_dkb=stride_dkb,
        stride_dkh=stride_dkh,
        stride_dkn=stride_dkn,
        stride_dks=stride_dks,
        stride_dvb=stride_dvb,
        stride_dvh=stride_dvh,
        stride_dvn=stride_dvn,
        stride_dvs=stride_dvs,
        IS_SPLIT_QO=IS_SPLIT_QO,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnBwdBlockScheduler.create(
        config=config,
        split_idx=grid_idx.split_idx,
        NUM_SPLITS=NUM_SPLITS,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_QO=IS_SPLIT_QO,
    )

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config, SWAP_AB=True)

    # Early exit if no n_blocks to process
    if grid_idx.n_block * TILE_N >= config.actual_seqlen_k:
        return

    # Create pointers
    q_ptrs = ptrs_sched.make_q_ptrs(config)
    k_ptrs = ptrs_sched.make_k_ptrs(config)
    v_ptrs = ptrs_sched.make_v_ptrs(config)
    do_ptrs = ptrs_sched.make_do_ptrs(config)
    lse_ptrs = ptrs_sched.make_lse_ptrs(config)
    dpsum_ptrs = ptrs_sched.make_dpsum_ptrs(config)
    dq_accum_ptrs = ptrs_sched.make_dq_accum_ptrs(config)
    dk_ptrs = ptrs_sched.make_dk_ptrs(config)
    dv_ptrs = ptrs_sched.make_dv_ptrs(config)

    # Initialize accumulators
    acc_dk = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dv = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)

    # Load key tile
    k_tile = ptrs_sched.load_k(config, k_ptrs)

    # Load value tile
    v_tile = ptrs_sched.load_v(config, v_ptrs)

    # Process m_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for m_block in tl.range(
            block_sched.m_block_min, block_sched.m_block_min_no_mask
        ):
            acc_dk, acc_dv = _bwd_inner_dense_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                k_tile=k_tile,
                v_tile=v_tile,
                q_ptrs=q_ptrs,
                do_ptrs=do_ptrs,
                dq_accum_ptrs=dq_accum_ptrs,
                lse_ptrs=lse_ptrs,
                dpsum_ptrs=dpsum_ptrs,
                m_block=m_block,
                IS_MASK=True,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=True if IS_LOCAL else False,
                MASK_SINK=False,
            )

    # Process m_blocks without masking
    if not IS_LOCAL and block_sched.m_block_min_no_mask < block_sched.m_block_max:
        for m_block in tl.range(
            block_sched.m_block_min_no_mask, block_sched.m_block_max
        ):
            acc_dk, acc_dv = _bwd_inner_dense_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                k_tile=k_tile,
                v_tile=v_tile,
                q_ptrs=q_ptrs,
                do_ptrs=do_ptrs,
                dq_accum_ptrs=dq_accum_ptrs,
                lse_ptrs=lse_ptrs,
                dpsum_ptrs=dpsum_ptrs,
                m_block=m_block,
                IS_MASK=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

    if IS_LOCAL:
        # Process m_blocks with local right masking
        if block_sched.m_block_window_min < block_sched.m_block_window_min_no_mask:
            for m_block in tl.range(
                block_sched.m_block_window_min,
                block_sched.m_block_window_min_no_mask,
            ):
                acc_dk, acc_dv = _bwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    m_block=m_block,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

        # Process m_blocks without masking
        if (
            block_sched.m_block_window_min_no_mask
            < block_sched.m_block_window_max_no_mask
        ):
            for m_block in tl.range(
                block_sched.m_block_window_min_no_mask,
                block_sched.m_block_window_max_no_mask,
            ):
                acc_dk, acc_dv = _bwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    m_block=m_block,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                )

        # Process m_blocks with local left masking
        if block_sched.m_block_window_max_no_mask < block_sched.m_block_window_max:
            for m_block in tl.range(
                block_sched.m_block_window_max_no_mask,
                block_sched.m_block_window_max,
            ):
                acc_dk, acc_dv = _bwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    m_block=m_block,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

        # Process m_blocks with local sink masking
        if block_sched.m_block_sink_max > block_sched.m_block_sink_min:
            for m_block in tl.range(
                block_sched.m_block_sink_min,
                block_sched.m_block_sink_max,
            ):
                acc_dk, acc_dv = _bwd_inner_dense_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    q_ptrs=q_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    m_block=m_block,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

    # Store value gradients
    ptrs_sched.store_dv(config, dv_ptrs, acc_dv)

    # Scale key gradients
    acc_dk = acc_dk * softmax_scale

    # Store key gradients
    ptrs_sched.store_dk(config, dk_ptrs, acc_dk)


_bwd_dense_kernel = cache_utils.wrap_kernel(_bwd_dense_kernel)


_bwd_dense_kernel_autotuned = None


def _get_autotuned_kernel():
    global _bwd_dense_kernel_autotuned
    if _bwd_dense_kernel_autotuned is None:
        jit_kernel = _bwd_dense_kernel._kernel
        autotuned = autotuner.make_bwd_dense_autotuned_kernel(jit_kernel)
        _bwd_dense_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _bwd_dense_kernel_autotuned


def _flash_dense_attn_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_qo: bool = False,
    num_splits: Optional[int] = None,
    is_autotune: bool = True,
    tile_mn: Optional[Tuple[int, int]] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
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

    launch_config = None
    if not is_autotune:
        kernel = _bwd_dense_kernel
        TILE_M, TILE_N = tile_mn if tile_mn is not None else (64, 64)
        num_warps, num_stages, num_ctas = 8, 1, 1
    else:
        launch_config = launch_template.load_launch_config(
            device=device,
            kernel_name="bwd_dense",
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            tile_k=TILE_K,
            is_local=is_local,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=is_causal,
            is_quant=is_quant,
        )
        if launch_config is not None:
            kernel = _bwd_dense_kernel
            TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
        else:
            kernel = _get_autotuned_kernel()
            # Placeholder for pre-launch computations
            TILE_M = TILE_N = 64
            num_warps = num_stages = num_ctas = None

    if is_split_qo and num_splits is None:
        num_splits = utils.num_splits_heuristic(
            seqlen_q=seqlen_k,
            seqlen_k=seqlen_q,
            num_SMs=num_SMs,
            TILE_M=TILE_N,
            TILE_N=TILE_M,
            max_split_blocks=utils.max_split_blocks_from_window_sizes(
                window_sizes, TILE_M
            )
            if is_local
            else None,
        )
    elif not is_split_qo:
        num_splits = 1

    seqlen_q_rounded = int(math.ceil(seqlen_q / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    dq = torch.empty_like(query, dtype=dout.dtype)
    dk = torch.empty_like(key, dtype=dout.dtype)
    dv = torch.empty_like(value, dtype=dout.dtype)
    lse_log2 = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded * head_dim_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dk_accum = torch.zeros(
        (num_splits, batch_size, seqlen_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (batch_size, seqlen_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        (num_splits, batch_size, seqlen_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (batch_size, seqlen_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )

    flash_bwd_preprocess._flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    grid = launch_grid.get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
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
        dout.stride(0),
        dout.stride(-2),
        dout.stride(-3),
        lse_log2.stride(0),
        lse_log2.stride(-2),
        dpsum.stride(0),
        dpsum.stride(-2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        head_dim_rounded,
        dk_accum.stride(-4) if is_split_qo and num_splits > 1 else dk_accum.stride(0),
        dk_accum.stride(-2),
        dk_accum.stride(-3),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dv_accum.stride(-4) if is_split_qo and num_splits > 1 else dv_accum.stride(0),
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0) if window_sizes is not None else 0,
        None,
        None,
        None,
        None,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=max(triton.next_power_of_2(seqlen_q), 256),
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_QUANT=is_quant,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_autotune and launch_config is None:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_dense",
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_quant=is_quant,
            )

    flash_bwd_postprocess._flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv


def _flash_dense_attn_varlen_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_qo: bool = False,
    num_splits: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    is_autotune: bool = True,
    tile_mn: Optional[Tuple[int, int]] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    total_q, num_heads_q, head_dim = query.shape
    total_k, num_heads_kv, _ = key.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
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

    launch_config = None
    if not is_autotune:
        kernel = _bwd_dense_kernel
        TILE_M, TILE_N = tile_mn if tile_mn is not None else (64, 64)
        num_warps, num_stages, num_ctas = 8, 1, 1
    else:
        launch_config = launch_template.load_launch_config(
            device=device,
            kernel_name="bwd_dense",
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            tile_k=TILE_K,
            is_local=is_local,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=is_causal,
            is_quant=is_quant,
        )
        if launch_config is not None:
            kernel = _bwd_dense_kernel
            TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
        else:
            kernel = _get_autotuned_kernel()
            # Placeholder for pre-launch computations
            TILE_M = TILE_N = 64
            num_warps = num_stages = num_ctas = None

    if is_split_qo and num_splits is None:
        num_splits = utils.num_splits_heuristic(
            seqlen_q=seqlen_k,
            seqlen_k=seqlen_q,
            num_SMs=num_SMs,
            TILE_M=TILE_N,
            TILE_N=TILE_M,
            max_split_blocks=utils.max_split_blocks_from_window_sizes(
                window_sizes, TILE_M
            )
            if is_local
            else None,
        )
    elif not is_split_qo:
        num_splits = 1

    total_q_rounded_padded = int(math.ceil((total_q + batch_size * 128) / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    dq = torch.empty_like(query, dtype=dout.dtype)
    dk = torch.empty_like(key, dtype=dout.dtype)
    dv = torch.empty_like(value, dtype=dout.dtype)
    lse_log2 = torch.empty(
        num_heads_q,
        total_q_rounded_padded,
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        num_heads_q,
        total_q_rounded_padded,
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        num_heads_q,
        total_q_rounded_padded * head_dim_rounded,
        dtype=torch.float32,
        device=query.device,
    )
    dk_accum = torch.zeros(
        (num_splits, total_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (total_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        (num_splits, total_k, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (total_k, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )

    flash_bwd_preprocess._flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim_rounded=head_dim_rounded,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    grid = launch_grid.get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
        num_splits=num_splits,
    )

    triton.set_allocator(utils.alloc_fn)

    kernel[grid](
        query,
        key,
        value,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
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
        dout.stride(-2),
        dout.stride(0),
        0,
        lse_log2.stride(0),
        0,
        dpsum.stride(0),
        0,
        dq_accum.stride(0),
        head_dim_rounded,
        0,
        dk_accum.stride(-2),
        dk_accum.stride(-3),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        0,
        dv_accum.stride(-2),
        dv_accum.stride(-3),
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0) if window_sizes is not None else 0,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=max(triton.next_power_of_2(seqlen_q), 256),
        SEQLEN_K_CACHE=max(triton.next_power_of_2(seqlen_k), 256),
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_QUANT=is_quant,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if is_autotune and launch_config is None:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="bwd_dense",
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_k=TILE_K,
                config=best,
                is_local=is_local,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_quant=is_quant,
            )

    flash_bwd_postprocess._flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)

    return dq, dk, dv
