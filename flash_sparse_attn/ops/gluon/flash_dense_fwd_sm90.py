# Copyright (c) 2026, Jingze Shi.
from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
)

from flash_sparse_attn.ops.gluon.scheduler import (
    AttnFwdGridIndex,
    AttnFwdConfig,
    AttnFwdBlockScheduler,
    AttnFwdPointerScheduler,
    AttnMaskScheduler,
    SoftmaxScheduler,
)

from flash_sparse_attn.ops.gluon.assert_inputs import assert_fwd_inputs
from flash_sparse_attn.ops.gluon.utils import (
    get_device_num_sms,
    window_sizes_heuristic,
    num_splits_heuristic,
    alloc_fn,
)
from flash_sparse_attn.ops.gluon.launch_grid import get_fwd_grid
from flash_sparse_attn.ops.gluon.kernel_repr import fwd_dense_repr
from flash_sparse_attn.ops.gluon.hopper_helpers import (
    warpgroup_mma_wait,
    pack_mma_operand,
    unpack_mma_operand,
)
from flash_sparse_attn.ops.gluon.flash_fwd_combine import _flash_attn_fwd_combine


@gluon.jit
def _fwd_load_dense_kernel(
    config: AttnFwdConfig,
    desc,
    smem,
    ready,
    n_block,
    HAS_CU_SEQLENS_K: gl.constexpr,
):
    if HAS_CU_SEQLENS_K:
        coords = [config.offset_k + n_block * config.TILE_N, config.head_kv_idx, 0]
        shape: gl.constexpr = [config.TILE_N, 1, config.TILE_K]
    else:
        coords = [config.batch_idx, n_block * config.TILE_N, config.head_kv_idx, 0]
        shape: gl.constexpr = [1, config.TILE_N, 1, config.TILE_K]
    mbarrier.expect(ready, desc.block_type.nbytes)
    tma.async_load(desc, coords, ready, smem.reshape(shape))


@gluon.jit
def _fwd_producer_dense_kernel(
    config: AttnFwdConfig,
    block_sched: AttnFwdBlockScheduler,
    desc_K,
    desc_V,
    sK,
    sV,
    barriers,
    HAS_CU_SEQLENS_K: gl.constexpr,
):
    k_ready, v_ready, k_release, v_release = barriers

    # This counter is continuous across diagonal, window, and sink ranges
    kv_iter = gl.to_tensor(0)
    prev_n_block = gl.to_tensor(0)
    for range_idx in gl.static_range(3 if config.IS_LOCAL else 1):
        if range_idx == 0:
            n_block_max = block_sched.n_block_max
            n_block_min = (
                block_sched.n_block_max_no_mask if config.IS_LOCAL else block_sched.n_block_min
            )
        elif range_idx == 1:
            n_block_max = block_sched.n_block_window_max
            n_block_min = block_sched.n_block_window_min
        else:
            n_block_max = block_sched.n_block_sink_max
            n_block_min = block_sched.n_block_sink_min

        for n_block in range(n_block_max - 1, n_block_min - 1, -1):
            k_stage = kv_iter % 2

            # First fill needs no release
            if kv_iter >= 2:
                mbarrier.wait(k_release.index(k_stage), ((kv_iter // 2) - 1) & 1)
            _fwd_load_dense_kernel(
                config,
                desc_K,
                sK.index(k_stage),
                k_ready.index(k_stage),
                n_block,
                HAS_CU_SEQLENS_K,
            )

            if kv_iter > 0:
                # Issue the previous V only after the next K has been issued
                if kv_iter >= 2:
                    mbarrier.wait(v_release, (kv_iter - 2) & 1)
                _fwd_load_dense_kernel(config, desc_V, sV, v_ready, prev_n_block, HAS_CU_SEQLENS_K)
            prev_n_block = n_block
            kv_iter += 1

    # Drain the final V
    if kv_iter > 0:
        if kv_iter >= 2:
            mbarrier.wait(v_release, (kv_iter - 2) & 1)
        _fwd_load_dense_kernel(config, desc_V, sV, v_ready, prev_n_block, HAS_CU_SEQLENS_K)


@gluon.jit
def _fwd_compute_dense_kernel(
    config: AttnFwdConfig,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    sQ,
    sK,
    sV,
    barriers,
    row_max,
    row_sum,
    acc_O,
    rP,
    n_block,
    kv_iter,
    mma_layout: gl.constexpr,
    IS_FIRST: gl.constexpr,
    IS_MASK: gl.constexpr,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr = False,
    MASK_SINK: gl.constexpr = False,
):
    # MMA operand layouts for P
    mma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mma_layout,
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )

    k_ready, v_ready, k_release, v_release = barriers
    k_stage = kv_iter % 2
    k_phase = (kv_iter // 2) & 1
    v_phase = (kv_iter - 1) & 1

    # Wait for K
    mbarrier.wait(k_ready.index(k_stage), k_phase)

    # Compute S
    acc_S = gl.zeros([config.TILE_M, config.TILE_N], gl.float32, mma_layout)
    acc_S = warpgroup_mma_init(acc_S)
    acc_S = warpgroup_mma(
        sQ,
        sK.index(k_stage).permute((1, 0)),
        acc_S,
        use_acc=False,
        is_async=True,
    )
    if IS_FIRST:
        acc_S = warpgroup_mma_wait(num_outstanding=0, deps=(acc_S,))

    if not IS_FIRST:
        # Complete QK while leaving the previous PV in flight
        mbarrier.wait(v_ready, v_phase)
        rP = unpack_mma_operand(rP, sV.dtype, mma_lhs_layout)
        O_pending = warpgroup_mma_init(acc_O)
        O_pending = warpgroup_mma(rP, sV, O_pending, is_async=True)
        acc_S = warpgroup_mma_wait(num_outstanding=1, deps=(acc_S,))

    # Each warpgroup has finished QK
    mbarrier.arrive(k_release.index(k_stage))

    if IS_MASK:
        # Apply mask to S
        acc_S = mask_sched.apply_mask(
            acc_S,
            n_block,
            config.mma_row_m_offs(),
            config.mma_col_n_offs(),
            MASK_SEQLEN=MASK_SEQLEN,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Apply online softmax
    next_rP, row_max, row_sum, row_scale = softmax_sched.online_softmax(
        acc_S, row_max, row_sum, CHECK_INF=IS_MASK, PARTIAL_SUM=True
    )

    if not IS_FIRST:
        # Keep P alive until its asynchronous MMA completes
        # TODO: Improve steady-state softmax/PV overlap in generated code
        # Gluon emits no FFMA/EX2 before wait0 yet
        acc_O, rP = warpgroup_mma_wait(num_outstanding=0, deps=(O_pending, rP))
        mbarrier.arrive(v_release)

    # Rescale O
    acc_O = softmax_sched.rescale_o(acc_O, row_scale)

    # Carry P to the next iteration
    rP_next = pack_mma_operand(next_rP, sV.dtype, mma_lhs_layout)

    return row_max, row_sum, acc_O, rP_next


@gluon.jit
def _fwd_consumer_dense_kernel(
    config: AttnFwdConfig,
    block_sched: AttnFwdBlockScheduler,
    ptrs_sched: AttnFwdPointerScheduler,
    sQ,
    sK,
    sV,
    barriers,
    copy_layout: gl.constexpr,
    mma_layout: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    # MMA operand layouts for P
    mma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mma_layout,
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Initialize accumulators
    row_max = gl.full([config.TILE_M], float("-inf"), gl.float32, config.mma_row_layout)
    partial_sum = gl.zeros([config.TILE_M, config.TILE_N], gl.float32, mma_layout)
    row_sum = gl.sum(partial_sum.reshape([config.TILE_M, config.TILE_N // 8, 4, 2]).sum(3), 1)
    acc_O = gl.zeros([config.TILE_M, config.TILE_K], gl.float32, mma_layout)

    # This counter is continuous across diagonal, window, and sink ranges
    kv_iter = gl.to_tensor(0)

    # Load Q
    if not EVEN_M:
        pQ = ptrs_sched.make_m_predicate(config, config.copy_row_m_offs()[:, None])
    gQ = ptrs_sched.make_q_ptrs(config, config.copy_row_m_offs(), config.copy_col_k_offs())
    async_copy.async_load(
        sQ,
        gQ,
        mask=pQ if not EVEN_M else None,
        cache_modifier=".ca",
        eviction_policy="evict_last",
    )

    # Commit Q group
    async_copy.commit_group()

    # Wait for Q
    async_copy.wait_group(0)
    gl.barrier()

    # First iteration with seqlen masking
    n_block_max = block_sched.n_block_max - 1
    row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
        config,
        mask_sched,
        softmax_sched,
        sQ,
        sK,
        sV,
        barriers,
        row_max,
        row_sum,
        acc_O,
        None,
        n_block_max,
        kv_iter,
        mma_layout,
        IS_FIRST=True,
        IS_MASK=config.IS_CAUSAL or config.IS_LOCAL or not EVEN_N,
        MASK_SEQLEN=not EVEN_N,
        MASK_CAUSAL=config.IS_CAUSAL,
        MASK_LOCAL=config.IS_LOCAL,
        MASK_SINK=config.IS_LOCAL,
    )

    # Advance the K/V pipeline
    kv_iter += 1

    if config.IS_CAUSAL or config.IS_LOCAL:
        # Process n_blocks with causal masking
        for n_block in range(
            n_block_max - 1,
            block_sched.n_block_max_no_mask - 1,
            -1,
        ):
            row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                config,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                barriers,
                row_max,
                row_sum,
                acc_O,
                rP,
                n_block,
                kv_iter,
                mma_layout,
                IS_FIRST=False,
                IS_MASK=True,
                MASK_SEQLEN=False,
                MASK_CAUSAL=config.IS_CAUSAL,
                MASK_LOCAL=config.IS_LOCAL,
                MASK_SINK=False,
            )

            # Advance the K/V pipeline
            kv_iter += 1

    if not config.IS_LOCAL and block_sched.n_block_max_no_mask > block_sched.n_block_min:
        # Process n_blocks without masking
        for n_block in range(
            block_sched.n_block_max_no_mask - 1,
            block_sched.n_block_min - 1,
            -1,
        ):
            row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                config,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                barriers,
                row_max,
                row_sum,
                acc_O,
                rP,
                n_block,
                kv_iter,
                mma_layout,
                IS_FIRST=False,
                IS_MASK=False,
                MASK_SEQLEN=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

            # Advance the K/V pipeline
            kv_iter += 1

    if config.IS_LOCAL:
        if block_sched.n_block_window_max > block_sched.n_block_window_min:
            # Process n_blocks with local right masking
            for n_block in range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                    config,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    barriers,
                    row_max,
                    row_sum,
                    acc_O,
                    rP,
                    n_block,
                    kv_iter,
                    mma_layout,
                    IS_FIRST=False,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Advance the K/V pipeline
                kv_iter += 1

            # Process n_blocks without masking
            for n_block in range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                    config,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    barriers,
                    row_max,
                    row_sum,
                    acc_O,
                    rP,
                    n_block,
                    kv_iter,
                    mma_layout,
                    IS_FIRST=False,
                    IS_MASK=False,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                )

                # Advance the K/V pipeline
                kv_iter += 1

            # Process n_blocks with local left masking
            for n_block in range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                    config,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    barriers,
                    row_max,
                    row_sum,
                    acc_O,
                    rP,
                    n_block,
                    kv_iter,
                    mma_layout,
                    IS_FIRST=False,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Advance the K/V pipeline
                kv_iter += 1

        if block_sched.n_block_sink_max > block_sched.n_block_sink_min:
            # Process n_blocks with prefix-sink masking
            for n_block in range(
                block_sched.n_block_sink_max - 1,
                block_sched.n_block_sink_min - 1,
                -1,
            ):
                row_max, row_sum, acc_O, rP = _fwd_compute_dense_kernel(
                    config,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    barriers,
                    row_max,
                    row_sum,
                    acc_O,
                    rP,
                    n_block,
                    kv_iter,
                    mma_layout,
                    IS_FIRST=False,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

                # Advance the K/V pipeline
                kv_iter += 1

    # Drain the last PV once
    k_ready, v_ready, k_release, v_release = barriers
    mbarrier.wait(v_ready, (kv_iter - 1) & 1)
    rP = unpack_mma_operand(rP, sV.dtype, mma_lhs_layout)
    acc_O = warpgroup_mma_init(acc_O)
    acc_O = warpgroup_mma(rP, sV, acc_O, is_async=True)
    acc_O = warpgroup_mma_wait(num_outstanding=0, deps=(acc_O,))

    # Finalize softmax
    row_sum = gl.convert_layout(gl.sum(row_sum, 1), config.mma_row_layout)
    row_scale, lse = softmax_sched.finalize(
        row_max,
        row_sum,
        IS_LOG2=config.IS_SPLIT_KV,
    )

    # Store LSE
    if not EVEN_M:
        pLSE = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs())
    gLSE = ptrs_sched.make_lse_ptrs(config, config.mma_row_m_offs())
    gl.store(gLSE, lse, mask=pLSE if not EVEN_M else None)

    # Finalize rescale
    acc_O = softmax_sched.rescale_o(acc_O, row_scale)

    # Store O
    if config.IS_SPLIT_KV:
        if not EVEN_M:
            pO = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs()[:, None])
        gO = ptrs_sched.make_out_ptrs(config, config.mma_row_m_offs(), config.mma_col_k_offs())
        gl.store(gO, acc_O, mask=pO if not EVEN_M else None)
    else:
        sQ.store(acc_O.to(sQ.dtype))
        gl.barrier()
        rO = sQ.load(copy_layout)
        if not EVEN_M:
            pO = ptrs_sched.make_m_predicate(config, config.copy_row_m_offs()[:, None])
        gO = ptrs_sched.make_out_ptrs(config, config.copy_row_m_offs(), config.copy_col_k_offs())
        gl.store(gO, rO, mask=pO if not EVEN_M else None)


@triton.heuristics(
    {
        "EVEN_M": lambda args: (
            not args["HAS_CU_SEQLENS_Q"]
            and not args["HAS_SEQUSED_Q"]
            and args["seqlen_q"] % args["TILE_M"] == 0
        ),
        "EVEN_N": lambda args: (
            not args["HAS_CU_SEQLENS_K"]
            and not args["HAS_SEQUSED_K"]
            and args["seqlen_k"] % args["TILE_N"] == 0
        ),
    }
)
@gluon.jit(repr=fwd_dense_repr)
def _fwd_dense_kernel(
    mQ,
    desc_K,
    desc_V,
    mOut,
    mLse,
    mWindowSizes,
    mCuSeqlensQ,
    mCuSeqlensK,
    mSeqUsedQ,
    mSeqUsedK,
    stride_qb,
    stride_qh,
    stride_qm,
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
    head_dim,
    QHEAD_PER_KVHEAD: gl.constexpr,
    PACK_GQA: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    IS_SPLIT_KV: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_CU_SEQLENS_K: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    HAS_SEQUSED_K: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    warp_size: gl.constexpr = 32
    element_bits: gl.constexpr = mQ.dtype.element_ty.primitive_bitwidth
    elements_per_thread: gl.constexpr = 128 // element_bits
    copy_atom_k: gl.constexpr = min(TILE_K, 1024 // element_bits)
    threads_per_row: gl.constexpr = copy_atom_k // elements_per_thread
    rows_per_warp: gl.constexpr = warp_size // threads_per_row

    copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, min(128, TILE_N, TILE_K), 16],
    )

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
        grid_idx.load_window_sizes(mWindowSizes, stride_wh, IS_LOCAL=IS_LOCAL)
    )

    # Create config
    config = AttnFwdConfig.create(
        batch_idx=grid_idx.batch_idx,
        head_idx=grid_idx.head_idx,
        head_kv_idx=grid_idx.head_kv_idx,
        split_idx=grid_idx.split_idx,
        m_block=grid_idx.m_block,
        softmax_scale=softmax_scale,
        window_size_sink=window_size_sink,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        window_size_near=window_size_near,
        head_dim=head_dim,
        cu_seqlens_q=mCuSeqlensQ,
        cu_seqlens_k=mCuSeqlensK,
        seqused_q=mSeqUsedQ,
        seqused_k=mSeqUsedK,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        copy_layout=copy_layout,
        mma_layout=mma_layout,
        PACK_GQA=PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        NUM_SPLITS=NUM_SPLITS,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=IS_SPLIT_KV,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create block scheduler
    block_sched = AttnFwdBlockScheduler.create(config=config)

    # Create pointer scheduler
    ptrs_sched = AttnFwdPointerScheduler.create(
        config=config,
        Q=mQ,
        Out=mOut,
        Lse=mLse,
        stride_qb=stride_qb,
        stride_qh=stride_qh,
        stride_qm=stride_qm,
        stride_ob=stride_ob,
        stride_oh=stride_oh,
        stride_om=stride_om,
        stride_os=stride_os,
        stride_lb=stride_lb,
        stride_lh=stride_lh,
        stride_ls=stride_ls,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Early exit if no n_blocks to process
    if block_sched.is_empty():
        empty_o = gl.zeros([TILE_M, TILE_K], mOut.dtype.element_ty, mma_layout)
        if not EVEN_M:
            pO = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs()[:, None])
        gO = ptrs_sched.make_out_ptrs(config, config.mma_row_m_offs(), config.mma_col_k_offs())
        gl.store(gO, empty_o, mask=pO if not EVEN_M else None)
        empty_lse = gl.full([TILE_M], float("-inf"), mLse.dtype.element_ty, config.mma_row_layout)
        if not EVEN_M:
            pLSE = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs())
        gLSE = ptrs_sched.make_lse_ptrs(config, config.mma_row_m_offs())
        gl.store(gLSE, empty_lse, mask=pLSE if not EVEN_M else None)
        return

    # Allocate shared memory
    sQ_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_M, TILE_K], mQ.dtype.element_ty
    )
    sK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], desc_K.block_type.element_ty
    )
    sV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], desc_V.block_type.element_ty
    )
    sQ = gl.allocate_shared_memory(mQ.dtype.element_ty, [TILE_M, TILE_K], sQ_layout)
    sK = gl.allocate_shared_memory(desc_K.block_type.element_ty, [2, TILE_N, TILE_K], sK_layout)
    sV = gl.allocate_shared_memory(desc_V.block_type.element_ty, [TILE_N, TILE_K], sV_layout)

    # Independent ready barriers
    k_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    v_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(k_ready.index(0), count=1)
    mbarrier.init(k_ready.index(1), count=1)
    mbarrier.init(v_ready, count=1)

    # Independent release barriers
    k_release = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    v_release = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(k_release.index(0), count=1)
    mbarrier.init(k_release.index(1), count=1)
    mbarrier.init(v_release, count=1)

    # Bundle all barriers into a single tuple for convenience
    barriers = (k_ready, v_ready, k_release, v_release)

    # Launch warp specialization pipeline
    gl.warp_specialize(
        [
            (
                _fwd_consumer_dense_kernel,
                (
                    config,
                    block_sched,
                    ptrs_sched,
                    sQ,
                    sK,
                    sV,
                    barriers,
                    copy_layout,
                    mma_layout,
                    EVEN_M,
                    EVEN_N,
                ),
            ),
            (
                _fwd_producer_dense_kernel,
                (
                    config,
                    block_sched,
                    desc_K,
                    desc_V,
                    sK,
                    sV,
                    barriers,
                    HAS_CU_SEQLENS_K,
                ),
            ),
        ],
        [1],
        [24],
    )

    # Teardown barriers after warp specialization has completed
    mbarrier.invalidate(k_ready.index(0))
    mbarrier.invalidate(k_release.index(0))
    mbarrier.invalidate(k_ready.index(1))
    mbarrier.invalidate(k_release.index(1))
    mbarrier.invalidate(v_ready)
    mbarrier.invalidate(v_release)


def _flash_dense_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    num_splits: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = q.device
    num_SMs = get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    # Assert the validity of inputs
    if not skip_checks:
        assert_fwd_inputs(
            q=q,
            k=k,
            v=v,
            window_sizes=window_sizes,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    # Setup launch configuration
    TILE_M = tile_m if tile_m is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_N = tile_n if tile_n is not None else min(128, 16384 // TILE_K)
    num_warps = num_threads // 32 if num_threads is not None else 8
    if is_split_kv and num_splits is None:
        num_splits = num_splits_heuristic(
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
    out = out if out is not None else torch.empty_like(q)
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
    grid = get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    # Setup block shape and layout for TMA
    block_shape = [1] * (k.ndim - 3) + [TILE_N, 1, TILE_K]
    dtype = gl.bfloat16 if k.dtype == torch.bfloat16 else gl.float16
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, dtype)
    desc_K = TensorDescriptor.from_tensor(k, block_shape, layout)
    desc_V = TensorDescriptor.from_tensor(v, block_shape, layout)
    triton.set_allocator(alloc_fn)

    # Launch the kernel
    _fwd_dense_kernel[grid](
        q,
        desc_K,
        desc_V,
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
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_SPLIT_KV=is_split_kv,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=1,  # only change compiler metadata
        maxnreg=168,  # redistribute registers from the producer to compute warps
    )

    if is_split_kv:
        # Combine partial O and LSE
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            skip_checks=True,
        )

    return out, lse


def _flash_dense_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    num_splits: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = q.device
    num_SMs = get_device_num_sms(device)
    total_q, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    # Assert the validity of inputs
    if not skip_checks:
        assert_fwd_inputs(
            q=q,
            k=k,
            v=v,
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
    TILE_M = tile_m if tile_m is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_N = tile_n if tile_n is not None else min(128, 16384 // TILE_K)
    num_warps = num_threads // 32 if num_threads is not None else 8
    if is_split_kv and num_splits is None:
        num_splits = num_splits_heuristic(
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
    out = out if out is not None else torch.empty_like(q)
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (num_heads_q, total_q),
            dtype=torch.float32,
            device=q.device,
        )
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
    grid = get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    # Setup block shape and layout for TMA
    block_shape = [1] * (k.ndim - 3) + [TILE_N, 1, TILE_K]
    dtype = gl.bfloat16 if k.dtype == torch.bfloat16 else gl.float16
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, dtype)
    desc_K = TensorDescriptor.from_tensor(k, block_shape, layout)
    desc_V = TensorDescriptor.from_tensor(v, block_shape, layout)
    triton.set_allocator(alloc_fn)

    # Launch the kernel
    _fwd_dense_kernel[grid](
        q,
        desc_K,
        desc_V,
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
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_SPLIT_KV=is_split_kv,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=1,  # only change compiler metadata
        maxnreg=168,  # redistribute registers from the producer to compute warps
    )

    if is_split_kv:
        # Combine partial O and LSE
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            max_seqlen_q=max_seqlen_q,
            skip_checks=True,
        )

    return out, lse
