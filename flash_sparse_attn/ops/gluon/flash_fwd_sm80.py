# Copyright (c) 2026, Jingze Shi.
from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy

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
)
from flash_sparse_attn.ops.gluon.launch_grid import get_fwd_grid
from flash_sparse_attn.ops.gluon.kernel_repr import fwd_repr
from flash_sparse_attn.ops.gluon.ampere_helpers import gemm, gemm_rs
from flash_sparse_attn.ops.gluon.flash_fwd_combine import _flash_attn_fwd_combine


@gluon.jit
def _fwd_inner_sm80_kernel(
    config: AttnFwdConfig,
    ptrs_sched: AttnFwdPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    sQ,
    sK,
    sV,
    row_max,
    row_sum,
    acc_O,
    n_block,
    n_block_min,
    k_stage,
    mma_layout: gl.constexpr,
    IS_FIRST: gl.constexpr = False,
    IS_MASK: gl.constexpr = False,
    MASK_SEQLEN: gl.constexpr = False,
    MASK_CAUSAL: gl.constexpr = False,
    MASK_LOCAL: gl.constexpr = False,
    MASK_SINK: gl.constexpr = False,
):
    # MMA operand layouts
    mma_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mma_layout,
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )
    mma_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mma_layout,
        k_width=max(32 // sK.dtype.primitive_bitwidth, 1),
    )

    # TODO: Support a CTA-uniform issue predicate for both copy and commit_group to reduce empty copy in last iteration.
    # prefetch_mask = n_block > n_block_min

    # Wait for Q and K
    async_copy.wait_group(1)
    gl.barrier()

    # Load next K
    gK_next = ptrs_sched.make_k_ptrs(
        config, n_block - 1, config.copy_row_n_offs(), config.copy_col_k_offs()
    )
    async_copy.async_load(
        sK.index(1 - k_stage),
        gK_next,
        mask=n_block > n_block_min,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit next K group
    async_copy.commit_group()

    # Compute S
    acc_S = gl.zeros([config.TILE_M, config.TILE_N], gl.float32, mma_layout)
    acc_S = gemm(
        acc_S,
        sQ,
        sK.index(k_stage).permute((1, 0)),
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

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

    if not config.IS_SPARSE or IS_FIRST:
        is_skip: gl.constexpr = False
        row_max_curr: gl.constexpr = None
    else:
        # Determine skip condition
        is_skip, row_max_curr = softmax_sched.score_skip(
            acc_S,
            row_max,
            row_sum,
            config.get_softmax_threshold_log2(),
        )

    # Wait for V
    async_copy.wait_group(1)
    gl.barrier()

    if not is_skip:
        # Apply online softmax
        rP, row_max, row_sum, row_scale = softmax_sched.online_softmax(
            acc_S, row_max, row_sum, row_max_curr=row_max_curr, CHECK_INF=IS_MASK
        )

        # Rescale O accumulator
        acc_O = softmax_sched.rescale_o(acc_O, row_scale)

        # Compute O
        acc_O = gemm_rs(
            acc_O,
            rP,
            sV,
            lhs_layout=mma_lhs_layout,
            rhs_layout=mma_rhs_layout,
        )
    gl.barrier()

    # Load next V
    gV_next = ptrs_sched.make_v_ptrs(
        config, n_block - 1, config.copy_row_n_offs(), config.copy_col_k_offs()
    )
    async_copy.async_load(
        sV,
        gV_next,
        mask=n_block > n_block_min,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit next V group
    async_copy.commit_group()

    return row_max, row_sum, acc_O


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
@gluon.jit(repr=fwd_repr)
def _fwd_sm80_kernel(
    mQ,
    mK,
    mV,
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
    head_dim,
    QHEAD_PER_KVHEAD: gl.constexpr,
    PACK_GQA: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    IS_SPLIT_KV: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    IS_SPARSE: gl.constexpr,
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
        version=[2, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, 8],
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
        softmax_threshold=softmax_threshold,
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
        IS_SPARSE=IS_SPARSE,
        IS_SPLIT_KV=IS_SPLIT_KV,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create pointer scheduler
    ptrs_sched = AttnFwdPointerScheduler.create(
        config=config,
        Q=mQ,
        K=mK,
        V=mV,
        Out=mOut,
        Lse=mLse,
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
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnFwdBlockScheduler.create(config)

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

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
        [TILE_M, TILE_K],
        mQ.dtype.element_ty,
    )
    sK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K],
        mK.dtype.element_ty,
    )
    sV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K],
        mV.dtype.element_ty,
    )
    sQ = gl.allocate_shared_memory(
        mQ.dtype.element_ty,
        [TILE_M, TILE_K],
        sQ_layout,
    )
    sK = gl.allocate_shared_memory(
        mK.dtype.element_ty,
        [2, TILE_N, TILE_K],
        sK_layout,
    )
    sV = gl.allocate_shared_memory(
        mV.dtype.element_ty,
        [TILE_N, TILE_K],
        sV_layout,
    )

    # Initialize accumulators
    row_max = gl.full([TILE_M], float("-inf"), gl.float32, config.mma_row_layout)
    row_sum = gl.zeros([TILE_M], gl.float32, config.mma_row_layout)
    acc_O = gl.zeros([TILE_M, TILE_K], gl.float32, mma_layout)

    # Initialize the K stage for the asynchronous copy pipeline
    k_stage = gl.to_tensor(0)

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

    # First iteration with seqlen masking
    n_block_max = block_sched.n_block_max - 1

    # Load K
    if not EVEN_N:
        pK = ptrs_sched.make_n_predicate(
            config, block_sched.n_block_max - 1, config.copy_row_n_offs()[:, None]
        )
    gK = ptrs_sched.make_k_ptrs(
        config,
        block_sched.n_block_max - 1,
        config.copy_row_n_offs(),
        config.copy_col_k_offs(),
    )
    async_copy.async_load(
        sK.index(k_stage),
        gK,
        mask=pK if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit K group
    async_copy.commit_group()

    # Load V
    if not EVEN_N:
        pV = ptrs_sched.make_n_predicate(
            config, block_sched.n_block_max - 1, config.copy_row_n_offs()[:, None]
        )
    gV = ptrs_sched.make_v_ptrs(
        config,
        block_sched.n_block_max - 1,
        config.copy_row_n_offs(),
        config.copy_col_k_offs(),
    )
    async_copy.async_load(
        sV,
        gV,
        mask=pV if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit V group
    async_copy.commit_group()

    # Process n_block with seqlen masking
    row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
        config,
        ptrs_sched,
        mask_sched,
        softmax_sched,
        sQ,
        sK,
        sV,
        row_max,
        row_sum,
        acc_O,
        n_block_max,
        block_sched.n_block_min,
        k_stage,
        mma_layout,
        IS_FIRST=True,
        IS_MASK=IS_CAUSAL or IS_LOCAL or not EVEN_N,
        MASK_SEQLEN=not EVEN_N,
        MASK_CAUSAL=IS_CAUSAL,
        MASK_LOCAL=IS_LOCAL,
        MASK_SINK=IS_LOCAL,
    )

    # Toggle the k_stage for the remaining range
    k_stage = 1 - k_stage

    if IS_CAUSAL or IS_LOCAL:
        # Process n_blocks with causal masking
        for n_block in range(
            n_block_max - 1,
            block_sched.n_block_max_no_mask - 1,
            -1,
        ):
            row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                row_max,
                row_sum,
                acc_O,
                n_block,
                block_sched.n_block_min,
                k_stage,
                mma_layout,
                IS_MASK=True,
                MASK_SEQLEN=False,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=IS_LOCAL,
                MASK_SINK=False,
            )

            # Toggle the k_stage for the remaining range
            k_stage = 1 - k_stage

    if not IS_LOCAL and block_sched.n_block_max_no_mask > block_sched.n_block_min:
        # Process n_blocks without masking
        for n_block in range(
            block_sched.n_block_max_no_mask - 1,
            block_sched.n_block_min - 1,
            -1,
        ):
            row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                row_max,
                row_sum,
                acc_O,
                n_block,
                block_sched.n_block_min,
                k_stage,
                mma_layout,
                IS_MASK=False,
                MASK_SEQLEN=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

            # Toggle the k_stage for the remaining range
            k_stage = 1 - k_stage

    if IS_LOCAL:
        if block_sched.n_block_window_max > block_sched.n_block_window_min:
            # Synchronize to ensure all previous shared-memory reads are complete
            gl.barrier()

            # Load K
            gK = ptrs_sched.make_k_ptrs(
                config,
                block_sched.n_block_window_max - 1,
                config.copy_row_n_offs(),
                config.copy_col_k_offs(),
            )
            async_copy.async_load(
                sK.index(k_stage),
                gK,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit K group
            async_copy.commit_group()

            # Load V
            gV = ptrs_sched.make_v_ptrs(
                config,
                block_sched.n_block_window_max - 1,
                config.copy_row_n_offs(),
                config.copy_col_k_offs(),
            )
            async_copy.async_load(sV, gV, cache_modifier=".cg", eviction_policy="evict_first")

            # Commit V group
            async_copy.commit_group()

            # Process n_blocks with local right masking
            for n_block in range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_O,
                    n_block,
                    block_sched.n_block_window_min,
                    k_stage,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Toggle the k_stage for the remaining range
                k_stage = 1 - k_stage

            # Process n_blocks without masking
            for n_block in range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_O,
                    n_block,
                    block_sched.n_block_window_min,
                    k_stage,
                    mma_layout,
                    IS_MASK=False,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                )

                # Toggle the k_stage for the remaining range
                k_stage = 1 - k_stage

            # Process n_blocks with local left masking
            for n_block in range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_O,
                    n_block,
                    block_sched.n_block_window_min,
                    k_stage,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Toggle the k_stage for the remaining range
                k_stage = 1 - k_stage

        if block_sched.n_block_sink_max > block_sched.n_block_sink_min:
            # Synchronize to ensure all previous shared-memory reads are complete
            gl.barrier()

            # Load K
            gK = ptrs_sched.make_k_ptrs(
                config,
                block_sched.n_block_sink_max - 1,
                config.copy_row_n_offs(),
                config.copy_col_k_offs(),
            )
            async_copy.async_load(
                sK.index(k_stage),
                gK,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit K group
            async_copy.commit_group()

            # Load V
            gV = ptrs_sched.make_v_ptrs(
                config,
                block_sched.n_block_sink_max - 1,
                config.copy_row_n_offs(),
                config.copy_col_k_offs(),
            )
            async_copy.async_load(sV, gV, cache_modifier=".cg", eviction_policy="evict_first")

            # Commit V group
            async_copy.commit_group()

            # Process n_blocks with prefix-sink masking
            for n_block in range(
                block_sched.n_block_sink_max - 1,
                block_sched.n_block_sink_min - 1,
                -1,
            ):
                row_max, row_sum, acc_O = _fwd_inner_sm80_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_O,
                    n_block,
                    block_sched.n_block_sink_min,
                    k_stage,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

                # Toggle the k_stage for the remaining range
                k_stage = 1 - k_stage

    # Finalize softmax
    row_scale, lse = softmax_sched.finalize(row_max, row_sum, IS_LOG2=IS_SPLIT_KV)

    # Store LSE
    if not EVEN_M:
        pLSE = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs())
    gLSE = ptrs_sched.make_lse_ptrs(config, config.mma_row_m_offs())
    gl.store(gLSE, lse, mask=pLSE if not EVEN_M else None)

    # Finalize rescale
    acc_O = softmax_sched.rescale_o(acc_O, row_scale)

    # Store O
    if IS_SPLIT_KV:
        if not EVEN_M:
            pO = ptrs_sched.make_m_predicate(config, config.mma_row_m_offs()[:, None])
        gO = ptrs_sched.make_out_ptrs(config, config.mma_row_m_offs(), config.mma_col_k_offs())
        gl.store(gO, acc_O, mask=pO if not EVEN_M else None)
    else:
        sQ.store(acc_O.to(mOut.dtype.element_ty))
        gl.barrier()
        rO = sQ.load(copy_layout)
        if not EVEN_M:
            pO = ptrs_sched.make_m_predicate(config, config.copy_row_m_offs()[:, None])
        gO = ptrs_sched.make_out_ptrs(config, config.copy_row_m_offs(), config.copy_col_k_offs())
        gl.store(gO, rO, mask=pO if not EVEN_M else None)


def _flash_attn_forward_sm80(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_sparse: bool = False,
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
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

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
    TILE_N = tile_n if tile_n is not None else 64
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    num_warps = num_threads // 32 if num_threads is not None else 4
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

    # Launch the kernel
    _fwd_sm80_kernel[grid](
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
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_SPLIT_KV=is_split_kv,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPARSE=is_sparse,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=1,  # only change compiler metadata
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


def _flash_attn_varlen_forward_sm80(
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
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_sparse: bool = False,
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
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

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
    TILE_N = tile_n if tile_n is not None else 64
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    num_warps = num_threads // 32 if num_threads is not None else 4
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

    # Launch the kernel
    _fwd_sm80_kernel[grid](
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
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        PACK_GQA=pack_gqa,
        QHEAD_PER_KVHEAD_PACKGQA=qhead_per_kvhead_packgqa,
        NUM_SPLITS=num_splits,
        IS_SPLIT_KV=is_split_kv,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPARSE=is_sparse,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        num_warps=num_warps,
        num_stages=1,  # only change compiler metadata
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
