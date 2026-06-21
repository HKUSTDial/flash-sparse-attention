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
    SoftmaxScheduler,
)


@triton.jit
def _bwd_inner_gated_kernel(
    config,
    ptrs_sched,
    mask_sched,
    softmax_sched,
    acc_dk,
    acc_dv,
    acc_dd,
    k_tile,
    v_tile,
    d_tile,
    q_ptrs,
    a_ptrs,
    do_ptrs,
    dq_accum_ptrs,
    da_ptrs,
    lse_ptrs,
    dpsum_ptrs,
    d_max,
    d_min,
    gate_max,
    m_block,
    IS_MASK: tl.constexpr,
    MASK_CAUSAL: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    MASK_SINK: tl.constexpr,
):
    skip_gate = False
    skip_softmax = False

    # Load alpha tile
    a_tile = ptrs_sched.load_a(config, a_ptrs, m_block)
    a_max = tl.max(a_tile)
    a_min = tl.min(a_tile)

    # Compute gate threshold for this m_block
    gate_threshold_log2 = config.get_gate_threshold_log2(
        m_block=m_block,
    )

    # Check if any gates are active for current tile
    gate_max, skip_gate = softmax_sched.online_gate(
        a_max=a_max,
        a_min=a_min,
        d_max=d_max,
        d_min=d_min,
        gate_max=gate_max,
        gate_threshold_log2=gate_threshold_log2,
    )

    if not skip_gate:
        # Compute attention gates
        acc_s = d_tile[:, None] * a_tile[None, :]

        # Compute scaling factor for gated attention score gradients
        if config.IS_LOGSIGMOID_GATE:
            ds_scale = tl.sigmoid(-acc_s)
        else:
            ds_scale = 1.0

        # Load query tile
        q_tile = ptrs_sched.load_q(config, q_ptrs, m_block)

        # Compute attention scores
        if config.IS_LOGSIGMOID_GATE:
            acc_s = softmax_sched.log_sigmoid(
                acc_s=acc_s,
            )
        acc_s += tl.dot(k_tile, q_tile.T)

        if IS_MASK:
            # Apply mask to attention scores
            acc_s = mask_sched.apply_mask(
                acc_s=acc_s,
                iter_block=m_block,
                MASK_CAUSAL=MASK_CAUSAL,
                MASK_LOCAL=MASK_LOCAL,
                MASK_SINK=MASK_SINK,
            )

        # Load LSE
        lse_log2 = ptrs_sched.load_lse(config, lse_ptrs, m_block)

        # Compute attention weights in log2-domain
        p_log2 = acc_s * config.softmax_scale_log2 - lse_log2[None, :]

        # Compute softmax threshold for this m_block
        softmax_threshold_log2 = config.get_softmax_threshold_log2(
            m_block=m_block,
        )

        # Update skip condition based on threshold
        skip_softmax = tl.max(p_log2 - softmax_threshold_log2[None, :]) < 0.0

        if not skip_softmax:
            # Compute attention weights
            p = activations.exp2(p_log2).to(q_tile.dtype)

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

            # Compute alpha gradients
            da = tl.sum(ds * ds_scale * d_tile[:, None], axis=0)

            # Store alpha gradients
            ptrs_sched.store_da(config, da_ptrs, m_block, da)

            # Compute delta gradients
            acc_dd += tl.sum(ds * ds_scale * a_tile[None, :], axis=1)

    return (
        acc_dk,
        acc_dv,
        acc_dd,
        gate_max,
    )


@triton.jit(repr=kernel_repr.bwd_gated_repr)
def _bwd_gated_kernel(
    Q,
    K,
    V,
    A,
    D,
    dO,
    LSELog2,
    dPsum,
    dQaccum,
    dK,
    dV,
    dA,
    dD,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    window_sizes,
    softmax_threshold,
    gate_threshold,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ab,
    stride_ah,
    stride_db,
    stride_dh,
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
    stride_dab,
    stride_dah,
    stride_ddb,
    stride_ddh,
    stride_dds,
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
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    IS_SPLIT_QO: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
    IS_ADAPT_GATE: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnBwdGridIndex.create(
        num_splits=num_splits,
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
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
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
        IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
        IS_ADAPT_GATE=IS_ADAPT_GATE,
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
        A=A,
        D=D,
        dO=dO,
        LSELog2=LSELog2,
        dPsum=dPsum,
        dQaccum=dQaccum,
        dK=dK,
        dV=dV,
        dA=dA,
        dD=dD,
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
        stride_ab=stride_ab,
        stride_ah=stride_ah,
        stride_db=stride_db,
        stride_dh=stride_dh,
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
        stride_dab=stride_dab,
        stride_dah=stride_dah,
        stride_ddb=stride_ddb,
        stride_ddh=stride_ddh,
        stride_dds=stride_dds,
        IS_SPLIT_QO=IS_SPLIT_QO,
        IS_GATED=True,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnBwdBlockScheduler.create(
        config=config,
        split_idx=grid_idx.split_idx,
        num_splits=num_splits,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_QO=IS_SPLIT_QO,
    )

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config, SWAP_AB=True)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Early exit if no n_blocks to process
    if grid_idx.n_block * TILE_N >= config.actual_seqlen_k:
        return

    # Create pointers
    q_ptrs = ptrs_sched.make_q_ptrs(config)
    k_ptrs = ptrs_sched.make_k_ptrs(config)
    v_ptrs = ptrs_sched.make_v_ptrs(config)
    a_ptrs = ptrs_sched.make_a_ptrs(config)
    d_ptrs = ptrs_sched.make_d_ptrs(config)
    do_ptrs = ptrs_sched.make_do_ptrs(config)
    lse_ptrs = ptrs_sched.make_lse_ptrs(config)
    dpsum_ptrs = ptrs_sched.make_dpsum_ptrs(config)
    dq_accum_ptrs = ptrs_sched.make_dq_accum_ptrs(config)
    dk_ptrs = ptrs_sched.make_dk_ptrs(config)
    dv_ptrs = ptrs_sched.make_dv_ptrs(config)
    da_ptrs = ptrs_sched.make_da_ptrs(config)
    dd_ptrs = ptrs_sched.make_dd_ptrs(config)

    # Initialize accumulators
    gate_max = tl.full((), float("-inf"), dtype=tl.float32)
    acc_dk = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dv = tl.zeros((TILE_N, TILE_K), dtype=tl.float32)
    acc_dd = tl.zeros((TILE_N,), dtype=tl.float32)

    # Load key tile
    k_tile = ptrs_sched.load_k(config, k_ptrs)

    # Load value tile
    v_tile = ptrs_sched.load_v(config, v_ptrs)

    # Load delta tile
    d_tile = ptrs_sched.load_d(config, d_ptrs)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    # Process m_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for m_block in tl.range(
            block_sched.m_block_min, block_sched.m_block_min_no_mask
        ):
            (
                acc_dk,
                acc_dv,
                acc_dd,
                gate_max,
            ) = _bwd_inner_gated_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                acc_dd=acc_dd,
                k_tile=k_tile,
                v_tile=v_tile,
                d_tile=d_tile,
                q_ptrs=q_ptrs,
                a_ptrs=a_ptrs,
                do_ptrs=do_ptrs,
                dq_accum_ptrs=dq_accum_ptrs,
                da_ptrs=da_ptrs,
                lse_ptrs=lse_ptrs,
                dpsum_ptrs=dpsum_ptrs,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
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
            (
                acc_dk,
                acc_dv,
                acc_dd,
                gate_max,
            ) = _bwd_inner_gated_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                acc_dk=acc_dk,
                acc_dv=acc_dv,
                acc_dd=acc_dd,
                k_tile=k_tile,
                v_tile=v_tile,
                d_tile=d_tile,
                q_ptrs=q_ptrs,
                a_ptrs=a_ptrs,
                do_ptrs=do_ptrs,
                dq_accum_ptrs=dq_accum_ptrs,
                da_ptrs=da_ptrs,
                lse_ptrs=lse_ptrs,
                dpsum_ptrs=dpsum_ptrs,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
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
                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_ptrs=q_ptrs,
                    a_ptrs=a_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    da_ptrs=da_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
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
                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_ptrs=q_ptrs,
                    a_ptrs=a_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    da_ptrs=da_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
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
                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_ptrs=q_ptrs,
                    a_ptrs=a_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    da_ptrs=da_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
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
                (
                    acc_dk,
                    acc_dv,
                    acc_dd,
                    gate_max,
                ) = _bwd_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    acc_dk=acc_dk,
                    acc_dv=acc_dv,
                    acc_dd=acc_dd,
                    k_tile=k_tile,
                    v_tile=v_tile,
                    d_tile=d_tile,
                    q_ptrs=q_ptrs,
                    a_ptrs=a_ptrs,
                    do_ptrs=do_ptrs,
                    dq_accum_ptrs=dq_accum_ptrs,
                    da_ptrs=da_ptrs,
                    lse_ptrs=lse_ptrs,
                    dpsum_ptrs=dpsum_ptrs,
                    d_max=d_max,
                    d_min=d_min,
                    gate_max=gate_max,
                    m_block=m_block,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

    # Store value gradients
    ptrs_sched.store_dv(config, dv_ptrs, acc_dv)

    # Scale delta gradients
    acc_dd = acc_dd * softmax_scale

    # Store delta gradients
    ptrs_sched.store_dd(config, dd_ptrs, acc_dd)

    # Scale key gradients
    acc_dk = acc_dk * softmax_scale

    # Store key gradients
    ptrs_sched.store_dk(config, dk_ptrs, acc_dk)


_bwd_gated_kernel = cache_utils.wrap_kernel(_bwd_gated_kernel)


_bwd_gated_kernel_autotuned = None


def _get_autotuned_kernel():
    global _bwd_gated_kernel_autotuned
    if _bwd_gated_kernel_autotuned is None:
        jit_kernel = _bwd_gated_kernel._kernel
        autotuned = autotuner.make_bwd_gated_autotuned_kernel(jit_kernel)
        _bwd_gated_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _bwd_gated_kernel_autotuned


def _flash_gated_attn_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    seqlen_q, batch_size, num_heads_q, head_dim = query.shape
    seqlen_k, _, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else 1 / seqlen_k
    )
    gate_threshold = (
        gate_threshold if gate_threshold is not None else head_dim / seqlen_k
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 4), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
            alpha=alpha,
            delta=delta,
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

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="bwd_gated",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
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
        if is_split_qo
        else 1
    )

    seqlen_q_rounded = int(math.ceil(seqlen_q / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    dq = torch.empty_like(query, dtype=query_scale.dtype)
    dk = torch.empty_like(key, dtype=key_scale.dtype)
    dv = torch.empty_like(value, dtype=value_scale.dtype)
    da = torch.empty_like(alpha)
    dd = torch.empty_like(delta)
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
        (num_splits, seqlen_k, batch_size, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (seqlen_k, batch_size, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    dv_accum = torch.zeros(
        (num_splits, seqlen_k, batch_size, num_heads_kv, head_dim)
        if is_split_qo and num_splits > 1
        else (seqlen_k, batch_size, num_heads_kv, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    da_accum = torch.zeros(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=query.device,
    )
    dd_accum = torch.zeros(
        (num_splits, batch_size, num_heads_kv, seqlen_k)
        if is_split_qo and num_splits > 1
        else (batch_size, num_heads_kv, seqlen_k),
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
        alpha,
        delta,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        da_accum,
        dd_accum,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
        query.stride(1),
        query.stride(-2),
        query.stride(0),
        key.stride(1),
        key.stride(-2),
        key.stride(0),
        value.stride(1),
        value.stride(-2),
        value.stride(0),
        alpha.stride(1),
        alpha.stride(0),
        delta.stride(1),
        delta.stride(0),
        dout.stride(1),
        dout.stride(-2),
        dout.stride(0),
        lse_log2.stride(0),
        lse_log2.stride(-2),
        dpsum.stride(0),
        dpsum.stride(-2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        head_dim_rounded,
        dk_accum.stride(-3) if is_split_qo and num_splits > 1 else dk_accum.stride(1),
        dk_accum.stride(-2),
        dk_accum.stride(-4) if is_split_qo and num_splits > 1 else dk_accum.stride(0),
        dk_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        dv_accum.stride(-3) if is_split_qo and num_splits > 1 else dv_accum.stride(1),
        dv_accum.stride(-2),
        dv_accum.stride(-4) if is_split_qo and num_splits > 1 else dv_accum.stride(0),
        dv_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        da_accum.stride(0),
        da_accum.stride(-2),
        dd_accum.stride(-3) if is_split_qo and num_splits > 1 else dd_accum.stride(0),
        dd_accum.stride(-2),
        dd_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0),
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
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
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
                kernel_name="bwd_gated",
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
        da_accum=da_accum,
        da=da,
        scale=softmax_scale,
        head_dim_rounded=head_dim_rounded,
        tile_m=TILE_M,
        tile_k=TILE_K,
    )

    if is_split_qo and num_splits > 1:
        dk_accum = dk_accum.sum(dim=0)
        dv_accum = dv_accum.sum(dim=0)
        dd_accum = dd_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)
    dd.copy_(dd_accum)

    return dq, dk, dv, da, dd


def _flash_gated_attn_varlen_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
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
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    is_split_qo: bool = False,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    softmax_threshold = (
        softmax_threshold if softmax_threshold is not None else 1 / seqlen_k
    )
    gate_threshold = (
        gate_threshold if gate_threshold is not None else head_dim / seqlen_k
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = utils.window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    elif not is_local:
        window_sizes = torch.zeros((num_heads_kv, 4), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_bwd_inputs(
            query,
            key,
            value,
            out,
            dout,
            lse,
            alpha=alpha,
            delta=delta,
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

    launch_config = launch_template.load_launch_config(
        device=device,
        kernel_name="bwd_gated",
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_causal=is_causal,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _bwd_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        # Placeholder for pre-launch computations
        TILE_M = TILE_N = 64
        num_warps = num_stages = num_ctas = None

    num_splits = (
        utils.num_splits_heuristic(
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
        if is_split_qo
        else 1
    )

    total_q_rounded_padded = int(math.ceil((total_q + batch_size * 128) / 128) * 128)
    head_dim_rounded = int(math.ceil(head_dim / 32) * 32)

    if not is_quant:
        query_scale = torch.ones(1, device=device, dtype=query.dtype)
        key_scale = torch.ones(1, device=device, dtype=query.dtype)
        value_scale = torch.ones(1, device=device, dtype=query.dtype)

    dq = torch.empty_like(query, dtype=query_scale.dtype)
    dk = torch.empty_like(key, dtype=key_scale.dtype)
    dv = torch.empty_like(value, dtype=value_scale.dtype)
    da = torch.empty_like(alpha)
    dd = torch.empty_like(delta)
    lse_log2 = torch.empty(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dpsum = torch.empty(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dq_accum = torch.empty(
        (num_heads_q, total_q_rounded_padded * head_dim_rounded),
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
    da_accum = torch.zeros(
        (num_heads_q, total_q_rounded_padded),
        dtype=torch.float32,
        device=query.device,
    )
    dd_accum = torch.zeros(
        (num_splits, num_heads_kv, total_k)
        if is_split_qo and num_splits > 1
        else (num_heads_kv, total_k),
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
        alpha,
        delta,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
        da_accum,
        dd_accum,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
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
        alpha.stride(-2),
        0,
        delta.stride(-2),
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
        0,
        da_accum.stride(0),
        0,
        dd_accum.stride(-2),
        dd_accum.stride(0) if is_split_qo and num_splits > 1 else 0,
        window_sizes.stride(0),
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
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_QO=is_split_qo and num_splits > 1,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        IS_ADAPT_GATE=is_adapt_gate,
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
                kernel_name="bwd_gated",
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
        da_accum=da_accum,
        da=da,
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
        dd_accum = dd_accum.sum(dim=0)

    dk.copy_(dk_accum)
    dv.copy_(dv_accum)
    dd.copy_(dd_accum)

    return dq, dk, dv, da, dd
