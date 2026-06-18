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
def _dec_inner_gated_kernel(
    config: AttnDecConfig,
    ptrs_sched: AttnDecPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    skip_gate_curr,
    acc_s,
    acc_o,
    q_tile,
    a_tile,
    k_ptrs,
    v_ptrs,
    d_ptrs,
    a_max,
    a_min,
    gate_max,
    row_max,
    row_sum,
    n_block,
    n_block_min,
    IS_MASK: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    CHECK_INF: tl.constexpr,
):
    # Load next delta tile
    d_tile = ptrs_sched.load_d(config, d_ptrs, n_block - 1)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    skip_gate_next = True
    if not skip_gate_curr:
        if n_block > n_block_min:
            # Check if any gates are active for next tile
            gate_max, skip_gate_next = softmax_sched.online_gate(
                a_max=a_max,
                a_min=a_min,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                gate_threshold_log2=config.gate_threshold_log2,
            )

        if IS_MASK:
            # Apply mask to attention scores
            acc_s = mask_sched.apply_mask(
                acc_s=acc_s,
                iter_block=n_block,
                MASK_LOCAL=MASK_LOCAL,
            )

        # Apply online softmax
        p, row_max, row_sum, row_scale, skip_softmax = (
            softmax_sched.online_sparse_softmax(
                acc_s=acc_s,
                row_max=row_max,
                row_sum=row_sum,
                softmax_threshold_log2=config.softmax_threshold_log2,
                CHECK_INF=CHECK_INF,
            )
        )

        if not skip_softmax:
            # Load value tile
            v_tile = ptrs_sched.load_v(config, v_ptrs, n_block)

            # Rescale output accumulator
            acc_o = softmax_sched.rescale_o(
                acc_o=acc_o,
                row_scale=row_scale,
            )

            # Update output accumulator
            acc_o += tl.dot(p.to(v_tile.dtype), v_tile)

    else:
        if n_block > n_block_min:
            # Check if any gates are active for next tile
            gate_max, skip_gate_next = softmax_sched.online_gate(
                a_max=a_max,
                a_min=a_min,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                gate_threshold_log2=config.gate_threshold_log2,
            )

    if not skip_gate_next:
        # Compute attention gates for next tile
        acc_s = a_tile[:, None] * d_tile[None, :]
        if config.IS_LOGSIGMOID_GATE:
            acc_s = softmax_sched.log_sigmoid(
                acc_s=acc_s,
            )

        # Load next key tile
        k_tile = ptrs_sched.load_k(config, k_ptrs, n_block - 1)

        # Compute attention scores for next tile
        acc_s += tl.dot(q_tile, k_tile.T)

    return (
        skip_gate_next,
        acc_s,
        acc_o,
        gate_max,
        row_max,
        row_sum,
    )


@triton.jit(repr=kernel_repr.dec_gated_repr)
def _dec_gated_kernel(
    Q,
    K,
    V,
    A,
    D,
    Out,
    Lse,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    softmax_threshold,
    gate_threshold,
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
    stride_ab,
    stride_ah,
    stride_db,
    stride_dh,
    stride_ob,
    stride_oh,
    stride_om,
    stride_os,
    stride_lb,
    stride_lh,
    stride_lm,
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
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    HAS_CU_SEQLENS_Q: tl.constexpr,
    HAS_CU_SEQLENS_K: tl.constexpr,
    HAS_SEQUSED_Q: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    IS_LOGSIGMOID_GATE: tl.constexpr,
):
    # Create grid index
    grid_idx = AttnDecGridIndex.create(
        num_splits=num_splits,
    )

    # Load window sizes
    window_size_left, window_size_right, window_size_dist = grid_idx.load_window_sizes(
        window_sizes=window_sizes,
        stride_wh=stride_wh,
        IS_LOCAL=IS_LOCAL,
    )

    # Create config
    config = AttnDecConfig.create(
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        batch_idx=grid_idx.batch_idx,
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
        IS_LOGSIGMOID_GATE=IS_LOGSIGMOID_GATE,
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
        A=A,
        D=D,
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
        stride_ab=stride_ab,
        stride_ah=stride_ah,
        stride_db=stride_db,
        stride_dh=stride_dh,
        stride_ob=stride_ob,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_os=stride_os,
        stride_lb=stride_lb,
        stride_lh=stride_lh,
        stride_lm=stride_lm,
        stride_ls=stride_ls,
        IS_GATED=True,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnDecBlockScheduler.create(
        config=config,
        split_idx=grid_idx.split_idx,
        num_splits=num_splits,
        IS_LOCAL=IS_LOCAL,
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
    a_ptrs = ptrs_sched.make_a_ptrs(config)
    d_ptrs = ptrs_sched.make_d_ptrs(config)

    # Initialize accumulators
    gate_max = tl.full((), float("-inf"), dtype=tl.float32)
    row_max = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((TILE_M,), dtype=tl.float32)
    acc_o = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Load query tile
    q_tile = ptrs_sched.load_q(config, q_ptrs)

    # Load alpha tile
    a_tile = ptrs_sched.load_a(config, a_ptrs)
    a_max = tl.max(a_tile)
    a_min = tl.min(a_tile)

    # Load delta tile
    d_tile = ptrs_sched.load_d(config, d_ptrs, block_sched.n_block_max - 1)
    d_max = tl.max(d_tile)
    d_min = tl.min(d_tile)

    # Check if any gates are active for current tile
    gate_max, skip_gate_curr = softmax_sched.online_gate(
        a_max=a_max,
        a_min=a_min,
        d_max=d_max,
        d_min=d_min,
        gate_max=gate_max,
        gate_threshold_log2=config.gate_threshold_log2,
    )

    # Initialize skip_gate_curr to False for the first iteration
    skip_gate_curr = False

    # Compute attention gates for first tile
    acc_s = a_tile[:, None] * d_tile[None, :]
    if IS_LOGSIGMOID_GATE:
        acc_s = softmax_sched.log_sigmoid(
            acc_s=acc_s,
        )

    # Load key tile
    k_tile = ptrs_sched.load_k(config, k_ptrs, block_sched.n_block_max - 1)

    # Compute attention scores for first tile
    acc_s += tl.dot(q_tile, k_tile.T)

    # Process n_blocks with seqlen masking
    for n_block in tl.range(
        block_sched.n_block_max - 1, block_sched.n_block_max_no_mask - 1, -1
    ):
        (
            skip_gate_curr,
            acc_s,
            acc_o,
            gate_max,
            row_max,
            row_sum,
        ) = _dec_inner_gated_kernel(
            config=config,
            ptrs_sched=ptrs_sched,
            mask_sched=mask_sched,
            softmax_sched=softmax_sched,
            skip_gate_curr=skip_gate_curr,
            acc_s=acc_s,
            acc_o=acc_o,
            q_tile=q_tile,
            a_tile=a_tile,
            k_ptrs=k_ptrs,
            v_ptrs=v_ptrs,
            d_ptrs=d_ptrs,
            a_max=a_max,
            a_min=a_min,
            gate_max=gate_max,
            row_max=row_max,
            row_sum=row_sum,
            n_block=n_block,
            n_block_min=block_sched.n_block_max_no_mask,
            IS_MASK=True,
            MASK_LOCAL=False,
            CHECK_INF=True,
        )

    # Process n_blocks without masking
    if not IS_LOCAL and block_sched.n_block_max_no_mask > block_sched.n_block_min:
        # Load key tile
        k_tile = ptrs_sched.load_k(config, k_ptrs, block_sched.n_block_max_no_mask - 1)

        # Load delta tile
        d_tile = ptrs_sched.load_d(config, d_ptrs, block_sched.n_block_max_no_mask - 1)
        d_max = tl.max(d_tile)
        d_min = tl.min(d_tile)

        # Check if any gates are active for current tile
        gate_max, skip_gate_curr = softmax_sched.online_gate(
            a_max=a_max,
            a_min=a_min,
            d_max=d_max,
            d_min=d_min,
            gate_max=gate_max,
            gate_threshold_log2=config.gate_threshold_log2,
        )

        # Compute attention gates
        acc_s = a_tile[:, None] * d_tile[None, :]
        if IS_LOGSIGMOID_GATE:
            acc_s = softmax_sched.log_sigmoid(
                acc_s=acc_s,
            )

        # Compute attention scores
        acc_s += tl.dot(q_tile, k_tile.T)

        for n_block in tl.range(
            block_sched.n_block_max_no_mask - 1, block_sched.n_block_min - 1, -1
        ):
            (
                skip_gate_curr,
                acc_s,
                acc_o,
                gate_max,
                row_max,
                row_sum,
            ) = _dec_inner_gated_kernel(
                config=config,
                ptrs_sched=ptrs_sched,
                mask_sched=mask_sched,
                softmax_sched=softmax_sched,
                skip_gate_curr=skip_gate_curr,
                acc_s=acc_s,
                acc_o=acc_o,
                q_tile=q_tile,
                a_tile=a_tile,
                k_ptrs=k_ptrs,
                v_ptrs=v_ptrs,
                d_ptrs=d_ptrs,
                a_max=a_max,
                a_min=a_min,
                gate_max=gate_max,
                row_max=row_max,
                row_sum=row_sum,
                n_block=n_block,
                n_block_min=block_sched.n_block_min,
                IS_MASK=False,
                MASK_LOCAL=False,
                CHECK_INF=False,
            )

    if IS_LOCAL:
        # Process n_blocks with local right masking
        if block_sched.n_block_window_max > block_sched.n_block_window_max_no_mask:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, k_ptrs, block_sched.n_block_window_max - 1
            )

            # Load delta tile
            d_tile = ptrs_sched.load_d(
                config, d_ptrs, block_sched.n_block_window_max - 1
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = softmax_sched.online_gate(
                a_max=a_max,
                a_min=a_min,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                gate_threshold_log2=config.gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = softmax_sched.log_sigmoid(
                    acc_s=acc_s,
                )

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile.T)

            for n_block in tl.range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    gate_max,
                    row_max,
                    row_sum,
                ) = _dec_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_max_no_mask,
                    IS_MASK=True,
                    MASK_LOCAL=True,
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

            # Load delta tile
            d_tile = ptrs_sched.load_d(
                config, d_ptrs, block_sched.n_block_window_max_no_mask - 1
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = softmax_sched.online_gate(
                a_max=a_max,
                a_min=a_min,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                gate_threshold_log2=config.gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = softmax_sched.log_sigmoid(
                    acc_s=acc_s,
                )

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile.T)

            for n_block in tl.range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    gate_max,
                    row_max,
                    row_sum,
                ) = _dec_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_min_no_mask,
                    IS_MASK=False,
                    MASK_LOCAL=False,
                    CHECK_INF=False,
                )

        # Process n_blocks with local left masking
        if block_sched.n_block_window_min_no_mask > block_sched.n_block_window_min:
            # Load key tile
            k_tile = ptrs_sched.load_k(
                config, k_ptrs, block_sched.n_block_window_min_no_mask - 1
            )

            # Load delta tile
            d_tile = ptrs_sched.load_d(
                config, d_ptrs, block_sched.n_block_window_min_no_mask - 1
            )
            d_max = tl.max(d_tile)
            d_min = tl.min(d_tile)

            # Check if any gates are active for current tile
            gate_max, skip_gate_curr = softmax_sched.online_gate(
                a_max=a_max,
                a_min=a_min,
                d_max=d_max,
                d_min=d_min,
                gate_max=gate_max,
                gate_threshold_log2=config.gate_threshold_log2,
            )

            # Compute attention gates
            acc_s = a_tile[:, None] * d_tile[None, :]
            if IS_LOGSIGMOID_GATE:
                acc_s = softmax_sched.log_sigmoid(
                    acc_s=acc_s,
                )

            # Compute attention scores
            acc_s += tl.dot(q_tile, k_tile.T)

            for n_block in tl.range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                (
                    skip_gate_curr,
                    acc_s,
                    acc_o,
                    gate_max,
                    row_max,
                    row_sum,
                ) = _dec_inner_gated_kernel(
                    config=config,
                    ptrs_sched=ptrs_sched,
                    mask_sched=mask_sched,
                    softmax_sched=softmax_sched,
                    skip_gate_curr=skip_gate_curr,
                    acc_s=acc_s,
                    acc_o=acc_o,
                    q_tile=q_tile,
                    a_tile=a_tile,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    d_ptrs=d_ptrs,
                    a_max=a_max,
                    a_min=a_min,
                    gate_max=gate_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    n_block=n_block,
                    n_block_min=block_sched.n_block_window_min,
                    IS_MASK=True,
                    MASK_LOCAL=True,
                    CHECK_INF=True,
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


_dec_gated_kernel = cache_utils.wrap_kernel(_dec_gated_kernel)


_dec_gated_kernel_autotuned = None


def _get_autotuned_kernel():
    global _dec_gated_kernel_autotuned
    if _dec_gated_kernel_autotuned is None:
        jit_kernel = _dec_gated_kernel._kernel
        autotuned = autotuner.make_dec_gated_autotuned_kernel(jit_kernel)
        _dec_gated_kernel_autotuned = autotuner.AutotunedKernel(autotuned)
    return _dec_gated_kernel_autotuned


def _flash_gated_attn_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = query.device
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
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
        window_sizes = torch.zeros((num_heads_kv, 3), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_dec_inputs(
            query,
            key,
            value,
            alpha=alpha,
            delta=delta,
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
        kernel_name="dec_gated",
        seqlen_q=1,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _dec_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        TILE_M = max(triton.next_power_of_2(qhead_per_kvhead), 16)
        TILE_N = 128
        num_warps = num_stages = num_ctas = None

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
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
        alpha,
        delta,
        out_partial,
        lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        gate_threshold,
        window_sizes,
        query.stride(0),
        query.stride(-2),
        1,
        key.stride(0),
        key.stride(-2),
        key.stride(-3),
        value.stride(0),
        value.stride(-2),
        value.stride(-3),
        alpha.stride(0),
        1,
        delta.stride(0),
        delta.stride(-2),
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        window_sizes.stride(0),
        None,
        None,
        None,
        None,
        num_splits,
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=0,
        SEQLEN_K_CACHE=seqlen_k // 1024,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_LOCAL=is_local,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=False,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="dec_gated",
                seqlen_q=1,
                seqlen_k=seqlen_k,
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


def _flash_gated_attn_varlen_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float = None,
    softmax_threshold: float = None,
    gate_threshold: float = None,
    is_logsigmoid_gate: bool = True,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_k: Optional[torch.Tensor] = None,
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
        window_sizes = torch.zeros((num_heads_kv, 3), dtype=torch.int32, device=device)

    if not skip_checks:
        assert_inputs.assert_dec_inputs(
            query,
            key,
            value,
            alpha=alpha,
            delta=delta,
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
        kernel_name="dec_gated",
        seqlen_q=1,
        seqlen_k=seqlen_k,
        tile_k=TILE_K,
        is_local=is_local,
        qhead_per_kvhead=qhead_per_kvhead,
        is_quant=is_quant,
    )
    if launch_config is not None and not is_autotune:
        kernel = _dec_gated_kernel
        TILE_M, TILE_N, num_warps, num_stages, num_ctas = launch_config
    else:
        kernel = _get_autotuned_kernel()
        TILE_M = max(triton.next_power_of_2(qhead_per_kvhead), 16)
        TILE_N = 128
        num_warps = num_stages = num_ctas = None

    num_splits = utils.num_splits_heuristic(
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k,
        num_SMs=num_SMs,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
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
        alpha,
        delta,
        out_partial,
        lse_partial,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        softmax_threshold,
        gate_threshold,
        window_sizes,
        query.stride(0),
        query.stride(-2),
        1,
        0,
        key.stride(-2),
        key.stride(0),
        0,
        value.stride(-2),
        value.stride(0),
        alpha.stride(0),
        1,
        0,
        delta.stride(-2),
        out_partial.stride(1),
        out_partial.stride(-2),
        1,
        out_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(-1),
        1,
        lse_partial.stride(0),
        window_sizes.stride(0),
        None,
        cu_seqlens_k,
        None,
        seqused_k,
        num_splits,
        seqlen_q=qhead_per_kvhead,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        SEQLEN_Q_CACHE=0,
        SEQLEN_K_CACHE=seqlen_k // 1024,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_LOCAL=is_local,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        IS_LOGSIGMOID_GATE=is_logsigmoid_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    if launch_config is None or is_autotune:
        best = launch_template.extract_best_config(_get_autotuned_kernel())
        if best is not None:
            launch_template.store_launch_config(
                device=device,
                kernel_name="dec_gated",
                seqlen_q=1,
                seqlen_k=seqlen_k,
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
