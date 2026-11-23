# Copyright (c) 2026, Jingze Shi.
from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
)

from flash_sparse_attn.ops.gluon.scheduler import (
    AttnBwdGridIndex,
    AttnBwdConfig,
    AttnBwdBlockScheduler,
    AttnMaskScheduler,
    SoftmaxScheduler,
)
from flash_sparse_attn.ops.gluon.assert_inputs import assert_bwd_inputs
from flash_sparse_attn.ops.gluon.utils import (
    get_device_num_sms,
    window_sizes_heuristic,
    num_splits_heuristic,
    alloc_fn,
)
from flash_sparse_attn.ops.gluon.launch_grid import get_bwd_grid
from flash_sparse_attn.ops.gluon.kernel_repr import bwd_repr
from flash_sparse_attn.ops.gluon.hopper_helpers import (
    pack_mma_operand,
    unpack_mma_operand,
    warpgroup_mma_wait,
)
from flash_sparse_attn.ops.gluon.softmax import exp2
from flash_sparse_attn.ops.gluon.flash_bwd_preprocess import _flash_attn_bwd_preprocess
from flash_sparse_attn.ops.gluon.flash_bwd_postprocess import _flash_attn_bwd_postprocess
from flash_sparse_attn.ops.gluon.flash_bwd_combine import _flash_attn_bwd_combine


@gluon.jit
def _bwd_load_sm90_kernel(
    config: AttnBwdConfig,
    desc,
    smem,
    ready,
    release,
    m_block,
    iter,
):
    if config.HAS_CU_SEQLENS_Q:
        coords = (
            [config.offset_q + m_block * config.TILE_M, config.head_idx, 0]
            if gl.constexpr(len(desc.block_shape) == 3)
            else [config.head_idx, config.padded_offset_q + m_block * config.TILE_M]
        )
        shape: gl.constexpr = (
            [config.TILE_M, 1, config.TILE_K]
            if gl.constexpr(len(desc.block_shape) == 3)
            else [1, config.TILE_M]
        )
    else:
        coords = (
            [config.batch_idx, m_block * config.TILE_M, config.head_idx, 0]
            if gl.constexpr(len(desc.block_shape) == 4)
            else [config.batch_idx, config.head_idx, m_block * config.TILE_M]
        )
        shape: gl.constexpr = (
            [1, config.TILE_M, 1, config.TILE_K]
            if gl.constexpr(len(desc.block_shape) == 4)
            else [1, 1, config.TILE_M]
        )

    stage = iter % 2
    # Wait for the reusable stage
    if iter >= 2:
        mbarrier.wait(release.index(stage), ((iter // 2) - 1) & 1)
    mbarrier.expect(ready.index(stage), desc.block_type.nbytes)
    tma.async_load(desc, coords, ready.index(stage), smem.index(stage).reshape(shape))


@gluon.jit
def _bwd_producer_sm90_kernel(
    config: AttnBwdConfig,
    block_sched: AttnBwdBlockScheduler,
    desc_Q,
    desc_K,
    desc_V,
    desc_LSE,
    desc_dO,
    desc_dPsum,
    sQ,
    sK,
    sV,
    sdO,
    sLSE,
    sdPsum,
    k_ready,
    v_ready,
    q_ready,
    lse_ready,
    do_ready,
    dpsum_ready,
    q_release,
    lse_release,
    do_release,
    dpsum_release,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_CU_SEQLENS_K: gl.constexpr,
):
    # Initialize pipeline iteration
    iter = gl.to_tensor(0)

    # Load K and V
    if HAS_CU_SEQLENS_K:
        k_coords = [config.offset_k + config.n_block * config.TILE_N, config.head_kv_idx, 0]
    else:
        k_coords = [config.batch_idx, config.n_block * config.TILE_N, config.head_kv_idx, 0]
    mbarrier.expect(k_ready, desc_K.block_type.nbytes)
    tma.async_load(desc_K, k_coords, k_ready, sK.reshape(desc_K.block_shape))
    mbarrier.expect(v_ready, desc_V.block_type.nbytes)
    tma.async_load(desc_V, k_coords, v_ready, sV.reshape(desc_V.block_shape))

    # First iteration
    m_block_max = block_sched.m_block_max - 1
    for m_block in range(m_block_max, block_sched.m_block_max):
        # Load LSE
        _bwd_load_sm90_kernel(
            config,
            desc_LSE,
            sLSE,
            lse_ready,
            lse_release,
            m_block,
            iter,
        )

        # Load dPsum
        _bwd_load_sm90_kernel(
            config,
            desc_dPsum,
            sdPsum,
            dpsum_ready,
            dpsum_release,
            m_block,
            iter,
        )

        # Load Q
        _bwd_load_sm90_kernel(
            config,
            desc_Q,
            sQ,
            q_ready,
            q_release,
            m_block,
            iter,
        )

        # Load dO
        _bwd_load_sm90_kernel(
            config,
            desc_dO,
            sdO,
            do_ready,
            do_release,
            m_block,
            iter,
        )

        # Advance the Q/dO pipeline
        iter += 1

    if config.IS_CAUSAL or config.IS_LOCAL:
        for m_block in range(block_sched.m_block_min, block_sched.m_block_min_no_mask):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1

    if not config.IS_LOCAL:
        for m_block in range(block_sched.m_block_min_no_mask, m_block_max):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1

    if config.IS_LOCAL:
        for m_block in range(
            block_sched.m_block_window_min,
            block_sched.m_block_window_min_no_mask,
        ):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1

        for m_block in range(
            block_sched.m_block_window_min_no_mask,
            block_sched.m_block_window_max_no_mask,
        ):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1

        for m_block in range(
            block_sched.m_block_window_max_no_mask,
            block_sched.m_block_window_max,
        ):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1

        for m_block in range(
            block_sched.m_block_sink_min,
            block_sched.m_block_sink_max,
        ):
            # Load LSE
            _bwd_load_sm90_kernel(
                config,
                desc_LSE,
                sLSE,
                lse_ready,
                lse_release,
                m_block,
                iter,
            )

            # Load dPsum
            _bwd_load_sm90_kernel(
                config,
                desc_dPsum,
                sdPsum,
                dpsum_ready,
                dpsum_release,
                m_block,
                iter,
            )

            # Load Q
            _bwd_load_sm90_kernel(
                config,
                desc_Q,
                sQ,
                q_ready,
                q_release,
                m_block,
                iter,
            )

            # Load dO
            _bwd_load_sm90_kernel(
                config,
                desc_dO,
                sdO,
                do_ready,
                do_release,
                m_block,
                iter,
            )

            # Advance the Q/dO pipeline
            iter += 1


@gluon.jit
def _bwd_inner_sm90_kernel(
    config: AttnBwdConfig,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    desc_dQ,
    sQ,
    sK,
    sV,
    sLSE,
    sdO,
    sdPsum,
    sdS,
    sdQ,
    acc_dK,
    acc_dV,
    q_ready,
    lse_ready,
    do_ready,
    dpsum_ready,
    q_release,
    lse_release,
    do_release,
    dpsum_release,
    m_block,
    iter,
    MASK_SEQLEN: gl.constexpr = False,
    MASK_CAUSAL: gl.constexpr = False,
    MASK_LOCAL: gl.constexpr = False,
    MASK_SINK: gl.constexpr = False,
):
    mma_dk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=config.mma_dk_layout,
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )
    mma_dv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=config.mma_dv_layout,
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )

    stage = iter % 2
    phase = (iter // 2) & 1

    # Wait for Q
    mbarrier.wait(q_ready.index(stage), phase)

    # Compute S
    S_pending = warpgroup_mma(
        sK,
        sQ.index(stage).permute((1, 0)),
        gl.zeros([config.TILE_N, config.TILE_M], gl.float32, config.mma_s_layout),
        is_async=True,
        descriptor_mode="inline",
    )
    if not config.IS_SPARSE:
        # Wait for dO
        mbarrier.wait(do_ready.index(stage), phase)

        # Compute dP
        dP_pending = warpgroup_mma(
            sV,
            sdO.index(stage).permute((1, 0)),
            gl.zeros([config.TILE_N, config.TILE_M], gl.float32, config.mma_dp_layout),
            is_async=True,
            descriptor_mode="inline",
        )

        # Wait for S
        acc_S = warpgroup_mma_wait(1, (S_pending,))
    else:
        # Wait for S
        acc_S = warpgroup_mma_wait(0, (S_pending,))

    # Apply mask to S
    acc_S = mask_sched.apply_mask(
        acc_S,
        m_block,
        config.mma_s_col_m_offs(),
        config.mma_s_row_n_offs(),
        MASK_SEQLEN=MASK_SEQLEN,
        MASK_CAUSAL=MASK_CAUSAL,
        MASK_LOCAL=MASK_LOCAL,
        MASK_SINK=MASK_SINK,
    )

    # Wait for LSE
    mbarrier.wait(lse_ready.index(stage), phase)

    # Load LSE
    rLSE = sLSE.index(stage).load(config.mma_s_col_layout)
    fence_async_shared()

    # Release LSE
    mbarrier.arrive(lse_release.index(stage))

    # Compute Plog2
    rPlog2 = acc_S * config.softmax_scale_log2 - rLSE[None, :]

    if not config.IS_SPARSE:
        is_skip: gl.constexpr = False
    else:
        # Determine skip condition
        is_skip = softmax_sched.prob_skip(
            rPlog2, config.get_softmax_threshold_log2(m_block), PRED_SCOPE="partition"
        )

        # Wait for dO
        mbarrier.wait(do_ready.index(stage), phase)

    if not is_skip:
        if config.IS_SPARSE:
            # Compute dP
            dP_pending = warpgroup_mma(
                sV,
                sdO.index(stage).permute((1, 0)),
                gl.zeros([config.TILE_N, config.TILE_M], gl.float32, config.mma_dp_layout),
                is_async=True,
                descriptor_mode="inline",
            )

        # Compute P
        rP = exp2(rPlog2)

        # Pack P for dV
        rP_packed = pack_mma_operand(rP, sQ.dtype, mma_dv_lhs_layout)

        # Wait for dP
        acc_dP = warpgroup_mma_wait(0, (dP_pending,))

        # Wait for dPsum
        mbarrier.wait(dpsum_ready.index(stage), phase)

        # Load dPsum
        rdPsum = sdPsum.index(stage).load(config.mma_dp_col_layout)
        fence_async_shared()

        # Release dPsum
        mbarrier.arrive(dpsum_release.index(stage))

        # Compute dS
        rP_dp = gl.convert_layout(rP, config.mma_dp_layout)
        rdS = rP_dp * (acc_dP - rdPsum[None, :])

        # Pack dS
        rdS_packed = pack_mma_operand(rdS, sQ.dtype, mma_dk_lhs_layout)

        # NOTE: dQ changes warpgroup ownership, so dS crosses shared memory.
        # Store dS
        sdS.index(stage).store(rdS.to(sQ.dtype))

        # Restore P
        rP = unpack_mma_operand(rP_packed, sQ.dtype, mma_dv_lhs_layout)

        # Compute dV
        dV_pending = warpgroup_mma(
            rP, sdO.index(stage), acc_dV, is_async=True, descriptor_mode="inline"
        )

        # Synchronize dS
        fence_async_shared()
        gl.barrier()

        # Compute dQ
        dQ_pending = warpgroup_mma(
            sK.permute((1, 0)),
            sdS.index(stage),
            gl.zeros([config.TILE_K, config.TILE_M], gl.float32, config.mma_dq_layout),
            is_async=True,
            descriptor_mode="inline",
        )

        # Wait for dV
        acc_dV, rP = warpgroup_mma_wait(1, (dV_pending, rP))

        # Release dO
        mbarrier.arrive(do_release.index(stage))

        # Restore dS
        rdS = unpack_mma_operand(rdS_packed, sQ.dtype, mma_dk_lhs_layout)

        # Compute dK
        dK_pending = warpgroup_mma(
            rdS, sQ.index(stage), acc_dK, is_async=True, descriptor_mode="inline"
        )

        # Wait for dQ
        acc_dQ = warpgroup_mma_wait(1, (dQ_pending,))

        # Store dQ
        if config.HAS_CU_SEQLENS_Q:
            dq_coords = [config.head_idx, config.padded_offset_q + m_block * config.TILE_M, 0]
        else:
            dq_coords = [config.batch_idx, config.head_idx, m_block * config.TILE_M, 0]
        # NOTE: Wait for the previous dQ store
        if iter >= 2:
            tma.store_wait(1)
            gl.barrier()
        sdQ.index(stage).store(acc_dQ.permute((1, 0)))
        fence_async_shared()
        gl.barrier()
        tma.async_atomic_add(desc_dQ, dq_coords, sdQ.index(stage).reshape(desc_dQ.block_shape))

        # Wait for dK
        acc_dK, rdS = warpgroup_mma_wait(0, (dK_pending, rdS))
    else:
        # Wait for dPsum
        mbarrier.wait(dpsum_ready.index(stage), phase)

        # Release dPsum
        mbarrier.arrive(dpsum_release.index(stage))

        # Release dO
        mbarrier.arrive(do_release.index(stage))

        # Wait for dQ
        tma.store_wait(0)

    # Release Q
    mbarrier.arrive(q_release.index(stage))
    return acc_dK, acc_dV


@gluon.jit
def _bwd_consumer_sm90_kernel(
    config: AttnBwdConfig,
    block_sched: AttnBwdBlockScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    desc_dQ,
    desc_dK,
    desc_dV,
    sQ,
    sK,
    sV,
    sLSE,
    sdO,
    sdPsum,
    sdS,
    sdQ,
    sdK,
    sdV,
    k_ready,
    v_ready,
    q_ready,
    lse_ready,
    do_ready,
    dpsum_ready,
    q_release,
    lse_release,
    do_release,
    dpsum_release,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    # Initialize accumulators
    acc_dK = gl.zeros([config.TILE_N, config.TILE_K], gl.float32, config.mma_dk_layout)
    acc_dV = gl.zeros([config.TILE_N, config.TILE_K], gl.float32, config.mma_dv_layout)

    # Initialize pipeline iteration
    iter = gl.to_tensor(0)

    # Wait for K
    mbarrier.wait(k_ready, 0)

    # Wait for V
    mbarrier.wait(v_ready, 0)

    # First iteration with seqlen masking
    # NOTE: Keep this single-iteration range
    # a direct call extended WGMMA descriptor lifetimes and caused 128B spills on D128
    # near-use descriptor views recovered performance, but split cases still spilled
    # revisit after compiler updates
    m_block_max = block_sched.m_block_max - 1
    for m_block in range(m_block_max, block_sched.m_block_max):
        acc_dK, acc_dV = _bwd_inner_sm90_kernel(
            config,
            mask_sched,
            softmax_sched,
            desc_dQ,
            sQ,
            sK,
            sV,
            sLSE,
            sdO,
            sdPsum,
            sdS,
            sdQ,
            acc_dK,
            acc_dV,
            q_ready,
            lse_ready,
            do_ready,
            dpsum_ready,
            q_release,
            lse_release,
            do_release,
            dpsum_release,
            m_block,
            iter,
            MASK_SEQLEN=not EVEN_M or not EVEN_N,
            MASK_CAUSAL=config.IS_CAUSAL,
            MASK_LOCAL=config.IS_LOCAL,
            MASK_SINK=config.IS_LOCAL,
        )

        # Advance the Q/dO pipeline
        iter += 1

    if config.IS_CAUSAL or config.IS_LOCAL:
        # Process m_blocks with causal masking
        for m_block in range(block_sched.m_block_min, block_sched.m_block_min_no_mask):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=config.IS_CAUSAL,
                MASK_LOCAL=config.IS_LOCAL,
                MASK_SINK=False,
            )

            # Advance the Q/dO pipeline
            iter += 1

    if not config.IS_LOCAL:
        # Process m_blocks without masking
        for m_block in range(block_sched.m_block_min_no_mask, m_block_max):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

            # Advance the Q/dO pipeline
            iter += 1

    if config.IS_LOCAL:
        # Process m_blocks with local right masking
        for m_block in range(
            block_sched.m_block_window_min,
            block_sched.m_block_window_min_no_mask,
        ):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=False,
                MASK_LOCAL=True,
                MASK_SINK=False,
            )

            # Advance the Q/dO pipeline
            iter += 1

        # Process m_blocks without masking
        for m_block in range(
            block_sched.m_block_window_min_no_mask,
            block_sched.m_block_window_max_no_mask,
        ):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

            # Advance the Q/dO pipeline
            iter += 1

        # Process m_blocks with local left masking
        for m_block in range(
            block_sched.m_block_window_max_no_mask,
            block_sched.m_block_window_max,
        ):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=False,
                MASK_LOCAL=True,
                MASK_SINK=False,
            )

            # Advance the Q/dO pipeline
            iter += 1

        # Process m_blocks with local sink masking
        for m_block in range(
            block_sched.m_block_sink_min,
            block_sched.m_block_sink_max,
        ):
            acc_dK, acc_dV = _bwd_inner_sm90_kernel(
                config,
                mask_sched,
                softmax_sched,
                desc_dQ,
                sQ,
                sK,
                sV,
                sLSE,
                sdO,
                sdPsum,
                sdS,
                sdQ,
                acc_dK,
                acc_dV,
                q_ready,
                lse_ready,
                do_ready,
                dpsum_ready,
                q_release,
                lse_release,
                do_release,
                dpsum_release,
                m_block,
                iter,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=False,
                MASK_LOCAL=True,
                MASK_SINK=True,
            )

            # Advance the Q/dO pipeline
            iter += 1

    # Store dK and dV
    if config.HAS_CU_SEQLENS_K:
        dk_coords = [
            config.split_idx,
            config.head_kv_idx,
            config.padded_offset_k + config.n_block * config.TILE_N,
            0,
        ]
    else:
        dk_coords = [
            config.split_idx,
            config.batch_idx,
            config.head_kv_idx,
            config.n_block * config.TILE_N,
            0,
        ]
    tma.store_wait(0, read_only=False)
    gl.barrier()
    sdV.store(acc_dV)
    fence_async_shared()
    gl.barrier()
    if config.QHEAD_PER_KVHEAD > 1:
        tma.async_atomic_add(desc_dV, dk_coords, sdV.reshape(desc_dV.block_shape))
    else:
        tma.async_store(desc_dV, dk_coords, sdV.reshape(desc_dV.block_shape))
    tma.store_wait(0)
    gl.barrier()
    sdK.store(acc_dK * config.softmax_scale)
    fence_async_shared()
    gl.barrier()
    if config.QHEAD_PER_KVHEAD > 1:
        tma.async_atomic_add(desc_dK, dk_coords, sdK.reshape(desc_dK.block_shape))
    else:
        tma.async_store(desc_dK, dk_coords, sdK.reshape(desc_dK.block_shape))
    tma.store_wait(0, read_only=False)


@triton.heuristics(
    {
        "EVEN_N": lambda args: (
            not args["HAS_CU_SEQLENS_K"]
            and not args["HAS_SEQUSED_K"]
            and args["seqlen_k"] % args["TILE_N"] == 0
        ),
        "EVEN_M": lambda args: (
            not args["HAS_CU_SEQLENS_Q"]
            and not args["HAS_SEQUSED_Q"]
            and args["seqlen_q"] % args["TILE_M"] == 0
        ),
    }
)
@gluon.jit(repr=bwd_repr)
def _bwd_sm90_kernel(
    desc_Q,
    desc_K,
    desc_V,
    desc_LSE,
    desc_dO,
    desc_dPsum,
    desc_dQ,
    desc_dK,
    desc_dV,
    mWindowSizes,
    mCuSeqlensQ,
    mCuSeqlensK,
    mSeqUsedQ,
    mSeqUsedK,
    stride_wh,
    seqlen_q,
    seqlen_k,
    softmax_scale,
    softmax_threshold,
    head_dim,
    QHEAD_PER_KVHEAD: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    IS_SPLIT_QO: gl.constexpr,
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
    element_bits: gl.constexpr = desc_Q.dtype.primitive_bitwidth
    elements_per_thread: gl.constexpr = 128 // element_bits
    copy_atom_k: gl.constexpr = min(TILE_K, 1024 // element_bits)
    threads_per_row: gl.constexpr = copy_atom_k // elements_per_thread
    rows_per_warp: gl.constexpr = warp_size // threads_per_row

    copy_q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    copy_k_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    copy_v_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    copy_do_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, elements_per_thread],
        threads_per_warp=[rows_per_warp, threads_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    mma_s_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, 64, 16],
    )
    mma_dp_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, 64, 16],
    )
    mma_dk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, TILE_K, 16],
    )
    mma_dv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[8, 1],
        instr_shape=[16, TILE_K, 16],
    )
    mma_dq_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[min(num_warps, TILE_K // 16), num_warps // min(num_warps, TILE_K // 16)],
        instr_shape=[16, min(TILE_M, TILE_K // 2), 16],
    )

    # Create grid index
    grid_idx = AttnBwdGridIndex.create(
        seqlen_q=seqlen_q,
        head_dim=head_dim,
        NUM_SPLITS=NUM_SPLITS,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        IS_SPLIT_QO=IS_SPLIT_QO,
        IS_SPT=False,
    )

    # Load window sizes
    window_size_sink, window_size_left, window_size_right, window_size_near = (
        grid_idx.load_window_sizes(mWindowSizes, stride_wh, IS_LOCAL=IS_LOCAL)
    )

    # Create config
    config = AttnBwdConfig.create(
        batch_idx=grid_idx.batch_idx,
        head_idx=grid_idx.head_idx,
        head_kv_idx=grid_idx.head_kv_idx,
        split_idx=grid_idx.split_idx,
        n_block=grid_idx.n_block,
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
        copy_q_layout=copy_q_layout,
        copy_k_layout=copy_k_layout,
        copy_v_layout=copy_v_layout,
        copy_do_layout=copy_do_layout,
        mma_s_layout=mma_s_layout,
        mma_dp_layout=mma_dp_layout,
        mma_dq_layout=mma_dq_layout,
        mma_dk_layout=mma_dk_layout,
        mma_dv_layout=mma_dv_layout,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        NUM_SPLITS=NUM_SPLITS,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPARSE=IS_SPARSE,
        IS_SPLIT_QO=IS_SPLIT_QO,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create block scheduler
    block_sched = AttnBwdBlockScheduler.create(config)

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config, SWAP_AB=True)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Early exit if no m_blocks to process
    if block_sched.is_empty():
        return

    # Allocate shared memory
    sQ_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_M, TILE_K],
        dtype=desc_Q.block_type.element_ty,
    )
    sK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_N, TILE_K],
        dtype=desc_K.block_type.element_ty,
    )
    sV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_N, TILE_K],
        dtype=desc_V.block_type.element_ty,
    )
    sLSE_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=0,
        element_bitwidth=32,
        rank=1,
    )
    sdO_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_M, TILE_K],
        dtype=desc_Q.block_type.element_ty,
    )
    sdPsum_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=0,
        element_bitwidth=32,
        rank=1,
    )
    sdS_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_N, TILE_M],
        dtype=desc_Q.block_type.element_ty,
    )
    sdQ_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[TILE_M, TILE_K],
        dtype=gl.float32,
    )
    sdK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[config.TILE_N, config.TILE_K],
        dtype=gl.float32,
    )
    sdV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        block_shape=[config.TILE_N, config.TILE_K],
        dtype=gl.float32,
    )
    sQ = gl.allocate_shared_memory(
        element_ty=desc_Q.block_type.element_ty,
        shape=[2, TILE_M, TILE_K],
        layout=sQ_layout,
    )
    sK = gl.allocate_shared_memory(
        element_ty=desc_Q.block_type.element_ty,
        shape=[TILE_N, TILE_K],
        layout=sK_layout,
    )
    sV = gl.allocate_shared_memory(
        element_ty=desc_Q.block_type.element_ty,
        shape=[TILE_N, TILE_K],
        layout=sV_layout,
    )
    sLSE = gl.allocate_shared_memory(
        element_ty=gl.float32,
        shape=[2, TILE_M],
        layout=sLSE_layout,
    )
    sdO = gl.allocate_shared_memory(
        element_ty=desc_Q.block_type.element_ty,
        shape=[2, TILE_M, TILE_K],
        layout=sdO_layout,
    )
    sdPsum = gl.allocate_shared_memory(
        element_ty=gl.float32,
        shape=[2, TILE_M],
        layout=sdPsum_layout,
    )
    sdS = gl.allocate_shared_memory(
        element_ty=desc_Q.block_type.element_ty,
        shape=[2, TILE_N, TILE_M],
        layout=sdS_layout,
    )
    sdQ = gl.allocate_shared_memory(
        element_ty=gl.float32,
        shape=[2, TILE_M, TILE_K],
        layout=sdQ_layout,
    )
    sdK = sdQ.reinterpret(
        dtype=gl.float32,
        shape=[TILE_N, TILE_K],
        layout=sdK_layout,
    )
    sdV = sdQ.reinterpret(
        dtype=gl.float32,
        shape=[TILE_N, TILE_K],
        layout=sdV_layout,
    )

    # Create K ready barriers
    k_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(k_ready, count=1)

    # Create V ready barriers
    v_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(v_ready, count=1)

    # Create Q ready barrier
    q_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(q_ready.index(0), count=1)
    mbarrier.init(q_ready.index(1), count=1)

    # Create LSE ready barrier
    lse_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(lse_ready.index(0), count=1)
    mbarrier.init(lse_ready.index(1), count=1)

    # Create dO ready barrier
    do_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(do_ready.index(0), count=1)
    mbarrier.init(do_ready.index(1), count=1)

    # Create dPsum ready barrier
    dpsum_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(dpsum_ready.index(0), count=1)
    mbarrier.init(dpsum_ready.index(1), count=1)

    # Create Q release barrier
    q_release = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(q_release.index(0), count=1)
    mbarrier.init(q_release.index(1), count=1)

    # Create LSE release barrier
    lse_release = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(lse_release.index(0), count=1)
    mbarrier.init(lse_release.index(1), count=1)

    # Create dO release barrier
    do_release = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(do_release.index(0), count=1)
    mbarrier.init(do_release.index(1), count=1)

    # Create dPsum release barrier
    dpsum_release = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(dpsum_release.index(0), count=1)
    mbarrier.init(dpsum_release.index(1), count=1)

    # Launch warp-specialized pipeline
    gl.warp_specialize(
        [
            (
                _bwd_consumer_sm90_kernel,
                (
                    config,
                    block_sched,
                    mask_sched,
                    softmax_sched,
                    desc_dQ,
                    desc_dK,
                    desc_dV,
                    sQ,
                    sK,
                    sV,
                    sLSE,
                    sdO,
                    sdPsum,
                    sdS,
                    sdQ,
                    sdK,
                    sdV,
                    k_ready,
                    v_ready,
                    q_ready,
                    lse_ready,
                    do_ready,
                    dpsum_ready,
                    q_release,
                    lse_release,
                    do_release,
                    dpsum_release,
                    EVEN_M,
                    EVEN_N,
                ),
            ),
            (
                _bwd_producer_sm90_kernel,
                (
                    config,
                    block_sched,
                    desc_Q,
                    desc_K,
                    desc_V,
                    desc_LSE,
                    desc_dO,
                    desc_dPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    k_ready,
                    v_ready,
                    q_ready,
                    lse_ready,
                    do_ready,
                    dpsum_ready,
                    q_release,
                    lse_release,
                    do_release,
                    dpsum_release,
                    HAS_CU_SEQLENS_Q,
                    HAS_CU_SEQLENS_K,
                ),
            ),
        ],
        [1],
        [24],
    )

    # Invalidate barriers
    mbarrier.invalidate(k_ready)
    mbarrier.invalidate(v_ready)
    mbarrier.invalidate(q_ready.index(0))
    mbarrier.invalidate(q_ready.index(1))
    mbarrier.invalidate(lse_ready.index(0))
    mbarrier.invalidate(lse_ready.index(1))
    mbarrier.invalidate(do_ready.index(0))
    mbarrier.invalidate(do_ready.index(1))
    mbarrier.invalidate(dpsum_ready.index(0))
    mbarrier.invalidate(dpsum_ready.index(1))
    mbarrier.invalidate(q_release.index(0))
    mbarrier.invalidate(q_release.index(1))
    mbarrier.invalidate(lse_release.index(0))
    mbarrier.invalidate(lse_release.index(1))
    mbarrier.invalidate(do_release.index(0))
    mbarrier.invalidate(do_release.index(1))
    mbarrier.invalidate(dpsum_release.index(0))
    mbarrier.invalidate(dpsum_release.index(1))


def _flash_attn_backward_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_sparse: bool = False,
    is_split_qo: bool = False,
    num_splits: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q.device
    num_SMs = get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    softmax_threshold = softmax_threshold if softmax_threshold is not None else 1.0
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

    # Assert the validity of inputs
    if not skip_checks:
        assert_bwd_inputs(
            q=q,
            k=k,
            v=v,
            lse=lse,
            out=out,
            dout=dout,
            window_sizes=window_sizes,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    # Setup launch configuration
    TILE_N = tile_n if tile_n is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else min(64, 8192 // TILE_K)
    num_warps = num_threads // 32 if num_threads is not None else 8
    maxnreg = 168 if head_dim >= 128 else 152
    if is_split_qo and num_splits is None:
        num_splits = num_splits_heuristic(
            batch_size=batch_size,
            seqlen_parallel=seqlen_k,
            seqlen_loop=seqlen_q,
            num_heads_parallel=num_heads_q,
            num_heads_kv=num_heads_kv,
            num_SMs=num_SMs,
            TILE_PARALLEL=TILE_N,
            TILE_LOOP=TILE_M,
            is_local=is_local,
        )
        is_split_qo = True if num_splits > 1 else False
    elif not is_split_qo:
        num_splits = 1

    # Allocate outputs
    seqlen_q_rounded = (seqlen_q + TILE_M - 1) // TILE_M * TILE_M
    seqlen_k_rounded = (seqlen_k + TILE_N - 1) // TILE_N * TILE_N
    dq = dq if dq is not None else torch.empty_like(q, dtype=dout.dtype)
    dk = dk if dk is not None else torch.empty_like(k, dtype=dout.dtype)
    dv = dv if dv is not None else torch.empty_like(v, dtype=dout.dtype)
    lse_log2 = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=q.device,
    )
    dpsum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded),
        dtype=torch.float32,
        device=q.device,
    )
    dq_accum = torch.empty(
        (batch_size, num_heads_q, seqlen_q_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dk_accum = torch.zeros(
        (num_splits, batch_size, num_heads_kv, seqlen_k_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dv_accum = torch.zeros(
        (num_splits, batch_size, num_heads_kv, seqlen_k_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    # Preprocess
    _flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim=head_dim,
        padded_tile_m=TILE_M,
    )

    # Setup TMA descriptors
    q_dtype = gl.bfloat16 if q.dtype == torch.bfloat16 else gl.float16
    k_dtype = gl.bfloat16 if k.dtype == torch.bfloat16 else gl.float16
    v_dtype = gl.bfloat16 if v.dtype == torch.bfloat16 else gl.float16
    do_dtype = gl.bfloat16 if v.dtype == torch.bfloat16 else gl.float16
    q_block_shape = [1, TILE_M, 1, TILE_K]
    k_block_shape = [1, TILE_N, 1, TILE_K]
    v_block_shape = [1, TILE_N, 1, TILE_K]
    lse_block_shape = [1, 1, TILE_M]
    do_block_shape = [1, TILE_M, 1, TILE_K]
    dpsum_block_shape = [1, 1, TILE_M]
    dq_block_shape = [1, 1, TILE_M, TILE_K]
    dk_block_shape = [1, 1, 1, TILE_N, TILE_K]
    dv_block_shape = [1, 1, 1, TILE_N, TILE_K]
    q_layout = gl.NVMMASharedLayout.get_default_for(q_block_shape, q_dtype)
    k_layout = gl.NVMMASharedLayout.get_default_for(k_block_shape, k_dtype)
    v_layout = gl.NVMMASharedLayout.get_default_for(v_block_shape, v_dtype)
    lse_layout = gl.NVMMASharedLayout(0, 32, rank=3)
    do_layout = gl.NVMMASharedLayout.get_default_for(do_block_shape, do_dtype)
    dpsum_layout = gl.NVMMASharedLayout(0, 32, rank=3)
    dq_layout = gl.NVMMASharedLayout.get_default_for(dq_block_shape, gl.float32)
    dk_layout = gl.NVMMASharedLayout.get_default_for(dk_block_shape, gl.float32)
    dv_layout = gl.NVMMASharedLayout.get_default_for(dv_block_shape, gl.float32)
    desc_q = TensorDescriptor.from_tensor(q, q_block_shape, q_layout)
    desc_k = TensorDescriptor.from_tensor(k, k_block_shape, k_layout)
    desc_v = TensorDescriptor.from_tensor(v, v_block_shape, v_layout)
    desc_lse = TensorDescriptor.from_tensor(lse_log2, lse_block_shape, lse_layout)
    desc_do = TensorDescriptor.from_tensor(dout, do_block_shape, do_layout)
    desc_dpsum = TensorDescriptor.from_tensor(dpsum, dpsum_block_shape, dpsum_layout)
    desc_dq = TensorDescriptor.from_tensor(dq_accum, dq_block_shape, dq_layout)
    desc_dk = TensorDescriptor.from_tensor(dk_accum, dk_block_shape, dk_layout)
    desc_dv = TensorDescriptor.from_tensor(dv_accum, dv_block_shape, dv_layout)
    triton.set_allocator(alloc_fn)

    # Get the grid for the kernel launch
    grid = get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
        num_splits=num_splits,
    )

    # Launch the kernel
    _bwd_sm90_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_lse,
        desc_do,
        desc_dpsum,
        desc_dq,
        desc_dk,
        desc_dv,
        window_sizes,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        seqused_k,
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        softmax_threshold,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        IS_SPLIT_QO=is_split_qo,
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
        maxnreg=maxnreg,  # redistribute registers from the producer to compute warps
    )

    # Postprocess dQ
    _flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim=head_dim,
        padded_tile_m=TILE_M,
    )

    # Combine partial dK and dV
    _flash_attn_bwd_combine(
        dk_partial=dk_accum,
        dv_partial=dv_accum,
        dk=dk,
        dv=dv,
        seqused_k=seqused_k,
        padded_tile_n=TILE_N,
        skip_checks=True,
    )

    return dq, dk, dv


def _flash_attn_varlen_backward_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_sparse: bool = False,
    is_split_qo: bool = False,
    num_splits: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    num_threads: Optional[int] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q.device
    num_SMs = get_device_num_sms(device)
    total_q, num_heads_q, head_dim = q.shape
    total_k, num_heads_kv, _ = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    softmax_threshold = softmax_threshold if softmax_threshold is not None else 1.0
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)
    if softmax_threshold is None:
        softmax_threshold = 1.0 if is_sparse else 0.0

    # Assert the validity of inputs
    if not skip_checks:
        assert_bwd_inputs(
            q=q,
            k=k,
            v=v,
            lse=lse,
            out=out,
            dout=dout,
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
    TILE_N = tile_n if tile_n is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    TILE_M = tile_m if tile_m is not None else min(64, 8192 // TILE_K)
    num_warps = num_threads // 32 if num_threads is not None else 8
    maxnreg = 168 if head_dim >= 128 else 152
    if is_split_qo and num_splits is None:
        num_splits = num_splits_heuristic(
            batch_size=batch_size,
            seqlen_parallel=seqlen_k,
            seqlen_loop=seqlen_q,
            num_heads_parallel=num_heads_q,
            num_heads_kv=num_heads_kv,
            num_SMs=num_SMs,
            TILE_PARALLEL=TILE_N,
            TILE_LOOP=TILE_M,
            is_local=is_local,
        )
        is_split_qo = True if num_splits > 1 else False
    elif not is_split_qo:
        num_splits = 1

    # Allocate outputs
    total_q_rounded = (total_q + batch_size * TILE_M - 1) // TILE_M * TILE_M
    total_k_rounded = (total_k + batch_size * TILE_N - 1) // TILE_N * TILE_N
    dq = dq if dq is not None else torch.empty_like(q, dtype=dout.dtype)
    dk = dk if dk is not None else torch.empty_like(k, dtype=dout.dtype)
    dv = dv if dv is not None else torch.empty_like(v, dtype=dout.dtype)
    lse_log2 = torch.empty(
        (num_heads_q, total_q_rounded),
        dtype=torch.float32,
        device=q.device,
    )
    dpsum = torch.empty(
        (num_heads_q, total_q_rounded),
        dtype=torch.float32,
        device=q.device,
    )
    dq_accum = torch.empty(
        (num_heads_q, total_q_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dk_accum = torch.zeros(
        (num_splits, num_heads_kv, total_k_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dv_accum = torch.zeros(
        (num_splits, num_heads_kv, total_k_rounded, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    # Preprocess
    _flash_attn_bwd_preprocess(
        out=out,
        dout=dout,
        dpsum=dpsum,
        lse=lse,
        lse_log2=lse_log2,
        dq_accum=dq_accum,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        padded_tile_m=TILE_M,
    )

    # Get the grid for the kernel launch
    grid = get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
        num_splits=num_splits,
    )

    # Setup TMA descriptors
    q_dtype = gl.bfloat16 if q.dtype == torch.bfloat16 else gl.float16
    k_dtype = gl.bfloat16 if k.dtype == torch.bfloat16 else gl.float16
    v_dtype = gl.bfloat16 if v.dtype == torch.bfloat16 else gl.float16
    do_dtype = gl.bfloat16 if v.dtype == torch.bfloat16 else gl.float16
    q_block_shape = [TILE_M, 1, TILE_K]
    k_block_shape = [TILE_N, 1, TILE_K]
    v_block_shape = [TILE_N, 1, TILE_K]
    lse_block_shape = [1, TILE_M]
    do_block_shape = [TILE_M, 1, TILE_K]
    dpsum_block_shape = [1, TILE_M]
    dq_block_shape = [1, TILE_M, TILE_K]
    dk_block_shape = [1, 1, TILE_N, TILE_K]
    dv_block_shape = [1, 1, TILE_N, TILE_K]
    q_layout = gl.NVMMASharedLayout.get_default_for(q_block_shape, q_dtype)
    k_layout = gl.NVMMASharedLayout.get_default_for(k_block_shape, k_dtype)
    v_layout = gl.NVMMASharedLayout.get_default_for(v_block_shape, v_dtype)
    lse_layout = gl.NVMMASharedLayout(0, 32, rank=2)
    do_layout = gl.NVMMASharedLayout.get_default_for(do_block_shape, do_dtype)
    dpsum_layout = gl.NVMMASharedLayout(0, 32, rank=2)
    dq_layout = gl.NVMMASharedLayout.get_default_for(dq_block_shape, gl.float32)
    dk_layout = gl.NVMMASharedLayout.get_default_for(dk_block_shape, gl.float32)
    dv_layout = gl.NVMMASharedLayout.get_default_for(dv_block_shape, gl.float32)
    desc_q = TensorDescriptor.from_tensor(q, q_block_shape, q_layout)
    desc_k = TensorDescriptor.from_tensor(k, k_block_shape, k_layout)
    desc_v = TensorDescriptor.from_tensor(v, v_block_shape, v_layout)
    desc_lse = TensorDescriptor.from_tensor(lse_log2, lse_block_shape, lse_layout)
    desc_do = TensorDescriptor.from_tensor(dout, do_block_shape, do_layout)
    desc_dpsum = TensorDescriptor.from_tensor(dpsum, dpsum_block_shape, dpsum_layout)
    desc_dq = TensorDescriptor.from_tensor(dq_accum, dq_block_shape, dq_layout)
    desc_dk = TensorDescriptor.from_tensor(dk_accum, dk_block_shape, dk_layout)
    desc_dv = TensorDescriptor.from_tensor(dv_accum, dv_block_shape, dv_layout)
    triton.set_allocator(alloc_fn)

    # Launch the kernel
    _bwd_sm90_kernel[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_lse,
        desc_do,
        desc_dpsum,
        desc_dq,
        desc_dk,
        desc_dv,
        window_sizes,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        softmax_threshold,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        IS_SPLIT_QO=is_split_qo,
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
        maxnreg=maxnreg,  # redistribute registers from the producer to compute warps
    )

    # Postprocess dQ
    _flash_attn_bwd_postprocess(
        dq_accum=dq_accum,
        dq=dq,
        scale=softmax_scale,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        padded_tile_m=TILE_M,
    )

    # Combine partial dK and dV
    _flash_attn_bwd_combine(
        dk_partial=dk_accum,
        dv_partial=dv_accum,
        dk=dk,
        dv=dv,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        padded_tile_n=TILE_N,
        skip_checks=True,
    )

    return dq, dk, dv
