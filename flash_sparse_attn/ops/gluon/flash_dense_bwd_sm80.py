from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy

from flash_sparse_attn.ops.gluon.scheduler import (
    AttnBwdGridIndex,
    AttnBwdConfig,
    AttnBwdBlockScheduler,
    AttnBwdPointerScheduler,
    AttnMaskScheduler,
)

from flash_sparse_attn.ops.gluon.assert_inputs import assert_bwd_inputs
from flash_sparse_attn.ops.gluon.utils import (
    get_device_num_sms,
    window_sizes_heuristic,
    num_splits_heuristic,
)
from flash_sparse_attn.ops.gluon.launch_grid import get_bwd_grid
from flash_sparse_attn.ops.gluon.kernel_repr import bwd_dense_repr
from flash_sparse_attn.ops.gluon.ampere_helpers import gemm, atomic_add
from flash_sparse_attn.ops.gluon.softmax import exp2
from flash_sparse_attn.ops.gluon.flash_bwd_preprocess import (
    _flash_attn_bwd_preprocess,
)
from flash_sparse_attn.ops.gluon.flash_bwd_postprocess import (
    _flash_attn_bwd_postprocess,
)
from flash_sparse_attn.ops.gluon.flash_bwd_combine import _flash_attn_bwd_combine


@gluon.jit
def _bwd_inner_dense_kernel(
    config: AttnBwdConfig,
    ptrs_sched: AttnBwdPointerScheduler,
    mask_sched: AttnMaskScheduler,
    sQ,
    sK,
    sV,
    sdO,
    sP,
    sdS,
    sLSE,
    sdPsum,
    acc_dK,
    acc_dV,
    m_block,
    q_stage,
    m_block_max,
    copy_row_m_offs,
    copy_col_k_offs,
    mma_col_m_offs,
    mma_row_n_offs,
    mma_layout: gl.constexpr,
    IS_MASK: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr,
    MASK_SINK: gl.constexpr,
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
        k_width=max(32 // sQ.dtype.primitive_bitwidth, 1),
    )
    mma_col_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)

    # TODO: Support a CTA-uniform issue predicate for both copy and commit_group to reduce empty copy in last iteration.
    # prefetch_mask = m_block < m_block_max - 1

    # Wait Q and LSE
    async_copy.wait_group(1)
    gl.barrier()

    # Compute S
    acc_S = gl.zeros([config.TILE_N, config.TILE_M], gl.float32, mma_layout)
    acc_S = gemm(
        acc=acc_S,
        sA=sK,
        sB=sQ.index(q_stage).permute((1, 0)),
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

    if IS_MASK:
        # Apply mask to S
        acc_S = mask_sched.apply_mask(
            acc_S,
            m_block,
            mma_col_m_offs,
            mma_row_n_offs,
            MASK_SEQLEN=True,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Compute P
    rP = exp2(
        acc_S * config.softmax_scale_log2
        - sLSE.index(q_stage).load(mma_col_layout)[None, :]
    )

    # Wait for dO and dPsum
    async_copy.wait_group(0)
    gl.barrier()

    # Load next Q
    gQ_next = ptrs_sched.make_q_ptrs(
        config, m_block + 1, copy_row_m_offs, copy_col_k_offs
    )
    async_copy.async_copy_global_to_shared(
        sQ.index(1 - q_stage),
        gQ_next,
        mask=m_block < m_block_max - 1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Load next LSE
    gLSE_next = ptrs_sched.make_lse_ptrs(config, m_block + 1, copy_row_m_offs)
    async_copy.async_copy_global_to_shared(
        sLSE.index(1 - q_stage),
        gLSE_next,
        mask=m_block < m_block_max - 1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit next Q and LSE
    async_copy.commit_group()

    # Compute dP
    acc_dP = gl.zeros([config.TILE_N, config.TILE_M], gl.float32, mma_layout)
    acc_dP = gemm(
        acc=acc_dP,
        sA=sV,
        sB=sdO.permute((1, 0)),
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

    # Compute dS
    dS = (rP * (acc_dP - sdPsum.index(0).load(mma_col_layout)[None, :])).to(sQ.dtype)

    # Materialize MMA LHS operands for dV, dQ, and dK
    sP.store(rP.to(sQ.dtype))
    sdS.store(dS)
    gl.barrier()

    # Compute dV
    acc_dV = gemm(
        acc=acc_dV,
        sA=sP,
        sB=sdO,
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

    # Load next dO
    gdO_next = ptrs_sched.make_do_ptrs(
        config, m_block + 1, copy_row_m_offs, copy_col_k_offs
    )
    async_copy.async_copy_global_to_shared(
        sdO,
        gdO_next,
        mask=m_block < m_block_max - 1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Load next dPsum
    gdPsum_next = ptrs_sched.make_dpsum_ptrs(config, m_block + 1, copy_row_m_offs)
    async_copy.async_copy_global_to_shared(
        sdPsum.index(0),
        gdPsum_next,
        mask=m_block < m_block_max - 1,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    # Commit next dO and dPsum
    async_copy.commit_group()

    # Compute dQ
    acc_dQ = gl.zeros([config.TILE_M, config.TILE_K], gl.float32, mma_layout)
    acc_dQ = gemm(
        acc=acc_dQ,
        sA=sdS.permute((1, 0)),
        sB=sK,
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

    # Compute dK
    acc_dK = gemm(
        acc=acc_dK,
        sA=sdS,
        sB=sQ.index(q_stage),
        lhs_layout=mma_lhs_layout,
        rhs_layout=mma_rhs_layout,
    )

    return acc_dK, acc_dV, acc_dQ


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
@gluon.jit(repr=bwd_dense_repr)
def _bwd_dense_kernel(
    mQ,
    mK,
    mV,
    mdO,
    mLSELog2,
    mdPsum,
    mdQaccum,
    mdK,
    mdV,
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
    seqlen_q,
    seqlen_k,
    softmax_scale,
    head_dim,
    QHEAD_PER_KVHEAD: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    IS_SPLIT_QO: gl.constexpr,
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
    copy_row_layout: gl.constexpr = gl.SliceLayout(1, copy_layout)
    copy_col_layout: gl.constexpr = gl.SliceLayout(0, copy_layout)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[num_warps // 2, 2],
        instr_shape=[16, 8],
    )
    mma_row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    mma_col_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)

    # Compute offsets for global memory copy and MMA operations
    copy_row_m_offs = gl.arange(0, TILE_M, copy_row_layout)
    copy_row_n_offs = gl.arange(0, TILE_N, copy_row_layout)
    copy_col_k_offs = gl.arange(0, TILE_K, copy_col_layout)
    mma_col_m_offs = gl.arange(0, TILE_M, mma_col_layout)
    mma_row_m_offs = gl.arange(0, TILE_M, mma_row_layout)
    mma_row_n_offs = gl.arange(0, TILE_N, mma_row_layout)
    mma_col_k_offs = gl.arange(0, TILE_K, mma_col_layout)

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
        window_size_near,
    ) = grid_idx.load_window_sizes(
        window_sizes=mWindowSizes,
        stride_wh=stride_wh,
        IS_LOCAL=IS_LOCAL,
    )

    # Create config
    config = AttnBwdConfig.create(
        batch_idx=grid_idx.batch_idx,
        head_idx=grid_idx.head_idx,
        head_kv_idx=grid_idx.head_kv_idx,
        split_idx=grid_idx.split_idx,
        n_block=grid_idx.n_block,
        row_offsets=mma_col_m_offs,
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
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        NUM_SPLITS=NUM_SPLITS,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_QO=IS_SPLIT_QO,
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q=HAS_SEQUSED_Q,
        HAS_SEQUSED_K=HAS_SEQUSED_K,
    )

    # Create pointer scheduler
    ptrs_sched = AttnBwdPointerScheduler.create(
        config=config,
        Q=mQ,
        K=mK,
        V=mV,
        dO=mdO,
        LSELog2=mLSELog2,
        dPsum=mdPsum,
        dQaccum=mdQaccum,
        dK=mdK,
        dV=mdV,
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
        HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
    )

    # Create block scheduler
    block_sched = AttnBwdBlockScheduler.create(config=config)

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config, SWAP_AB=True)

    # Early exit if no m_blocks to process
    if block_sched.is_empty():
        return

    # Allocate shared memory
    sQ_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_M, TILE_K], mQ.dtype.element_ty
    )
    sK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], mK.dtype.element_ty
    )
    sV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], mV.dtype.element_ty
    )
    sdO_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_M, TILE_K], mdO.dtype.element_ty
    )
    sP_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_M], mQ.dtype.element_ty
    )
    sdS_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_M], mQ.dtype.element_ty
    )
    sLSE_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    sdPsum_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    sQ = gl.allocate_shared_memory(mQ.dtype.element_ty, [2, TILE_M, TILE_K], sQ_layout)
    sK = gl.allocate_shared_memory(mK.dtype.element_ty, [TILE_N, TILE_K], sK_layout)
    sV = gl.allocate_shared_memory(mV.dtype.element_ty, [TILE_N, TILE_K], sV_layout)
    sdO = gl.allocate_shared_memory(mdO.dtype.element_ty, [TILE_M, TILE_K], sdO_layout)
    sP = gl.allocate_shared_memory(mQ.dtype.element_ty, [TILE_N, TILE_M], sP_layout)
    sdS = gl.allocate_shared_memory(mQ.dtype.element_ty, [TILE_N, TILE_M], sdS_layout)
    sLSE = gl.allocate_shared_memory(gl.float32, [2, TILE_M], sLSE_layout)
    sdPsum = gl.allocate_shared_memory(gl.float32, [1, TILE_M], sdPsum_layout)

    # Initialize the Q stage for the asynchronous copy pipeline
    q_stage = gl.to_tensor(0)

    # Initialize accumulators
    acc_dK = gl.zeros([TILE_N, TILE_K], gl.float32, mma_layout)
    acc_dV = gl.zeros([TILE_N, TILE_K], gl.float32, mma_layout)

    # Load K
    if not EVEN_N:
        pK = ptrs_sched.make_n_predicate(config, copy_row_n_offs[:, None])
    gK = ptrs_sched.make_k_ptrs(config, copy_row_n_offs, copy_col_k_offs)
    async_copy.async_copy_global_to_shared(
        sK,
        gK,
        mask=pK if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_last",
    )

    # Load V
    if not EVEN_N:
        pV = ptrs_sched.make_n_predicate(config, copy_row_n_offs[:, None])
    gV = ptrs_sched.make_v_ptrs(config, copy_row_n_offs, copy_col_k_offs)
    async_copy.async_copy_global_to_shared(
        sV,
        gV,
        mask=pV if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_last",
    )

    # Commit K and V group
    async_copy.commit_group()

    if block_sched.m_block_min < block_sched.m_block_max:
        # Load Q
        gQ = ptrs_sched.make_q_ptrs(
            config, block_sched.m_block_min, copy_row_m_offs, copy_col_k_offs
        )
        async_copy.async_copy_global_to_shared(
            sQ.index(q_stage),
            gQ,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Load LSE
        gLSE = ptrs_sched.make_lse_ptrs(
            config, block_sched.m_block_min, copy_row_m_offs
        )
        async_copy.async_copy_global_to_shared(
            sLSE.index(q_stage),
            gLSE,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Commit Q and LSE group
        async_copy.commit_group()

        # Load dO
        gdO = ptrs_sched.make_do_ptrs(
            config, block_sched.m_block_min, copy_row_m_offs, copy_col_k_offs
        )
        async_copy.async_copy_global_to_shared(
            sdO,
            gdO,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Load dPsum
        gdPsum = ptrs_sched.make_dpsum_ptrs(
            config, block_sched.m_block_min, copy_row_m_offs
        )
        async_copy.async_copy_global_to_shared(
            sdPsum.index(0),
            gdPsum,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Commit dO and dPsum group
        async_copy.commit_group()

    # Process m_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        for m_block in range(
            block_sched.m_block_min,
            block_sched.m_block_min_no_mask,
        ):
            # NOTE: Create gdQaccum here to avoid spilling
            gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                config, m_block, mma_row_m_offs, mma_col_k_offs
            )

            acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                config,
                ptrs_sched,
                mask_sched,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                acc_dK,
                acc_dV,
                m_block,
                q_stage,
                block_sched.m_block_max,
                copy_row_m_offs,
                copy_col_k_offs,
                mma_col_m_offs,
                mma_row_n_offs,
                mma_layout,
                IS_MASK=True,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=IS_LOCAL,
                MASK_SINK=False,
            )

            # Store dQ
            gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

            # Toggle the q_stage for the next iteration
            q_stage = 1 - q_stage

    # Process m_blocks without masking
    if not IS_LOCAL and block_sched.m_block_min_no_mask < block_sched.m_block_max:
        for m_block in range(
            block_sched.m_block_min_no_mask,
            block_sched.m_block_max,
        ):
            # NOTE: Create gdQaccum here to avoid spilling
            gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                config, m_block, mma_row_m_offs, mma_col_k_offs
            )

            acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                config,
                ptrs_sched,
                mask_sched,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                acc_dK,
                acc_dV,
                m_block,
                q_stage,
                block_sched.m_block_max,
                copy_row_m_offs,
                copy_col_k_offs,
                mma_col_m_offs,
                mma_row_n_offs,
                mma_layout,
                IS_MASK=False,
                MASK_CAUSAL=False,
                MASK_LOCAL=False,
                MASK_SINK=False,
            )

            # Store dQ
            gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

            # Toggle the q_stage for the next iteration
            q_stage = 1 - q_stage

    if IS_LOCAL:
        if block_sched.m_block_window_min < block_sched.m_block_window_max:
            # Synchronize to ensure all previous shared-memory reads are complete
            gl.barrier()

            # Load Q
            gQ = ptrs_sched.make_q_ptrs(
                config, block_sched.m_block_window_min, copy_row_m_offs, copy_col_k_offs
            )
            async_copy.async_copy_global_to_shared(
                sQ.index(q_stage),
                gQ,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Load LSE
            gLSE = ptrs_sched.make_lse_ptrs(
                config, block_sched.m_block_window_min, copy_row_m_offs
            )
            async_copy.async_copy_global_to_shared(
                sLSE.index(q_stage),
                gLSE,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit Q and LSE group
            async_copy.commit_group()

            # Load dO
            gdO = ptrs_sched.make_do_ptrs(
                config, block_sched.m_block_window_min, copy_row_m_offs, copy_col_k_offs
            )
            async_copy.async_copy_global_to_shared(
                sdO,
                gdO,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Load dPsum
            gdPsum = ptrs_sched.make_dpsum_ptrs(
                config, block_sched.m_block_window_min, copy_row_m_offs
            )
            async_copy.async_copy_global_to_shared(
                sdPsum.index(0),
                gdPsum,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit dO and dPsum group
            async_copy.commit_group()

        # Process m_blocks with local right masking
        if block_sched.m_block_window_min < block_sched.m_block_window_min_no_mask:
            for m_block in range(
                block_sched.m_block_window_min,
                block_sched.m_block_window_min_no_mask,
            ):
                # NOTE: Create gdQaccum here to avoid spilling
                gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                    config, m_block, mma_row_m_offs, mma_col_k_offs
                )

                acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sP,
                    sdS,
                    sLSE,
                    sdPsum,
                    acc_dK,
                    acc_dV,
                    m_block,
                    q_stage,
                    block_sched.m_block_window_max,
                    copy_row_m_offs,
                    copy_col_k_offs,
                    mma_col_m_offs,
                    mma_row_n_offs,
                    mma_layout,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Store dQ
                gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

                # Toggle the q_stage for the next iteration
                q_stage = 1 - q_stage

        # Process m_blocks without masking
        if (
            block_sched.m_block_window_min_no_mask
            < block_sched.m_block_window_max_no_mask
        ):
            for m_block in range(
                block_sched.m_block_window_min_no_mask,
                block_sched.m_block_window_max_no_mask,
            ):
                # NOTE: Create gdQaccum here to avoid spilling
                gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                    config, m_block, mma_row_m_offs, mma_col_k_offs
                )

                acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sP,
                    sdS,
                    sLSE,
                    sdPsum,
                    acc_dK,
                    acc_dV,
                    m_block,
                    q_stage,
                    block_sched.m_block_window_max,
                    copy_row_m_offs,
                    copy_col_k_offs,
                    mma_col_m_offs,
                    mma_row_n_offs,
                    mma_layout,
                    IS_MASK=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=False,
                    MASK_SINK=False,
                )

                # Store dQ
                gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

                # Toggle the q_stage for the next iteration
                q_stage = 1 - q_stage

        # Process m_blocks with local left masking
        if block_sched.m_block_window_max_no_mask < block_sched.m_block_window_max:
            for m_block in range(
                block_sched.m_block_window_max_no_mask,
                block_sched.m_block_window_max,
            ):
                # NOTE: Create gdQaccum here to avoid spilling
                gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                    config, m_block, mma_row_m_offs, mma_col_k_offs
                )

                acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sP,
                    sdS,
                    sLSE,
                    sdPsum,
                    acc_dK,
                    acc_dV,
                    m_block,
                    q_stage,
                    block_sched.m_block_window_max,
                    copy_row_m_offs,
                    copy_col_k_offs,
                    mma_col_m_offs,
                    mma_row_n_offs,
                    mma_layout,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

                # Store dQ
                gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

                # Toggle the q_stage for the next iteration
                q_stage = 1 - q_stage

        # Process m_blocks with local sink masking
        if block_sched.m_block_sink_min < block_sched.m_block_sink_max:
            # Synchronize to ensure all previous shared-memory reads are complete
            gl.barrier()

            # Load Q
            gQ = ptrs_sched.make_q_ptrs(
                config, block_sched.m_block_sink_min, copy_row_m_offs, copy_col_k_offs
            )
            async_copy.async_copy_global_to_shared(
                sQ.index(q_stage),
                gQ,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Load LSE
            gLSE = ptrs_sched.make_lse_ptrs(
                config, block_sched.m_block_sink_min, copy_row_m_offs
            )
            async_copy.async_copy_global_to_shared(
                sLSE.index(q_stage),
                gLSE,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit Q and LSE group
            async_copy.commit_group()

            # Load dO
            gdO = ptrs_sched.make_do_ptrs(
                config, block_sched.m_block_sink_min, copy_row_m_offs, copy_col_k_offs
            )
            async_copy.async_copy_global_to_shared(
                sdO,
                gdO,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Load dPsum
            gdPsum = ptrs_sched.make_dpsum_ptrs(
                config, block_sched.m_block_sink_min, copy_row_m_offs
            )
            async_copy.async_copy_global_to_shared(
                sdPsum.index(0),
                gdPsum,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # Commit dO and dPsum group
            async_copy.commit_group()

            for m_block in range(
                block_sched.m_block_sink_min,
                block_sched.m_block_sink_max,
            ):
                # NOTE: Create gdQaccum here to avoid spilling
                gdQaccum = ptrs_sched.make_dq_accum_ptrs(
                    config, m_block, mma_row_m_offs, mma_col_k_offs
                )

                acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sP,
                    sdS,
                    sLSE,
                    sdPsum,
                    acc_dK,
                    acc_dV,
                    m_block,
                    q_stage,
                    block_sched.m_block_sink_max,
                    copy_row_m_offs,
                    copy_col_k_offs,
                    mma_col_m_offs,
                    mma_row_n_offs,
                    mma_layout,
                    IS_MASK=True,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

                # Store dQ
                gl.atomic_add(gdQaccum, acc_dQ, sem="relaxed")

                # Toggle the q_stage for the next iteration
                q_stage = 1 - q_stage

    # Last iteration with seqlen masking
    if not EVEN_M and block_sched.m_block_last >= 0:
        # TODO: wait for the previous empty copy to complete
        async_copy.wait_group(0)

        # Synchronize to ensure all previous shared-memory reads are complete
        gl.barrier()

        # Load Q
        pQ = ptrs_sched.make_m_predicate(
            config, block_sched.m_block_last, copy_row_m_offs[:, None]
        )
        gQ = ptrs_sched.make_q_ptrs(
            config, block_sched.m_block_last, copy_row_m_offs, copy_col_k_offs
        )
        async_copy.async_copy_global_to_shared(
            sQ.index(q_stage),
            gQ,
            mask=pQ,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Load LSE
        gLSE = ptrs_sched.make_lse_ptrs(
            config, block_sched.m_block_last, copy_row_m_offs
        )
        async_copy.async_copy_global_to_shared(
            sLSE.index(q_stage),
            gLSE,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Commit Q and LSE group
        async_copy.commit_group()

        # Load dO
        pdO = ptrs_sched.make_m_predicate(
            config, block_sched.m_block_last, copy_row_m_offs[:, None]
        )
        gdO = ptrs_sched.make_do_ptrs(
            config, block_sched.m_block_last, copy_row_m_offs, copy_col_k_offs
        )
        async_copy.async_copy_global_to_shared(
            sdO,
            gdO,
            mask=pdO,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Load dPsum
        gdPsum = ptrs_sched.make_dpsum_ptrs(
            config, block_sched.m_block_last, copy_row_m_offs
        )
        async_copy.async_copy_global_to_shared(
            sdPsum.index(0),
            gdPsum,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )

        # Commit dO and dPsum group
        async_copy.commit_group()

        acc_dK, acc_dV, acc_dQ = _bwd_inner_dense_kernel(
            config,
            ptrs_sched,
            mask_sched,
            sQ,
            sK,
            sV,
            sdO,
            sP,
            sdS,
            sLSE,
            sdPsum,
            acc_dK,
            acc_dV,
            block_sched.m_block_last,
            q_stage,
            block_sched.m_block_last + 1,
            copy_row_m_offs,
            copy_col_k_offs,
            mma_col_m_offs,
            mma_row_n_offs,
            mma_layout,
            IS_MASK=True,
            MASK_CAUSAL=IS_CAUSAL,
            MASK_LOCAL=IS_LOCAL,
            MASK_SINK=IS_LOCAL,
        )

        # Store dQ
        # NOTE: per-address atomic add to avoid spilling
        atomic_add(
            ptrs_sched.dq_accum_base,
            acc_dQ,
            block_sched.m_block_last,
            ptrs_sched.stride_dqam,
            mma_row_m_offs,
            mma_col_k_offs,
            TILE_M,
        )

    # Store dV
    gdV = ptrs_sched.make_dv_ptrs(config, mma_row_n_offs, mma_col_k_offs)
    if QHEAD_PER_KVHEAD > 1:
        gl.atomic_add(gdV, acc_dV, sem="relaxed")
    else:
        gl.store(gdV, acc_dV)

    # Scale dK
    acc_dK = acc_dK * softmax_scale

    # Store dK
    gdK = ptrs_sched.make_dk_ptrs(config, mma_row_n_offs, mma_col_k_offs)
    if QHEAD_PER_KVHEAD > 1:
        gl.atomic_add(gdK, acc_dK, sem="relaxed")
    else:
        gl.store(gdK, acc_dK)


def _flash_dense_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
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
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    # Assert the validity of inputs
    if not skip_checks:
        assert_bwd_inputs(
            q=q,
            k=k,
            v=v,
            out=out,
            dout=dout,
            lse=lse,
            window_sizes=window_sizes,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    # Setup launch configuration
    TILE_M = tile_m if tile_m is not None else 64
    TILE_N = tile_n if tile_n is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    num_warps = num_threads // 32 if num_threads is not None else 8
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

    # Get the grid for the kernel launch
    grid = get_bwd_grid(
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        batch_size=batch_size,
        num_splits=num_splits,
    )

    # Launch the kernel
    _bwd_dense_kernel[grid](
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
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
        dout.stride(-4),
        dout.stride(-2),
        dout.stride(-3),
        lse_log2.stride(-3),
        lse_log2.stride(-2),
        dpsum.stride(-3),
        dpsum.stride(-2),
        dq_accum.stride(-4),
        dq_accum.stride(-3),
        dq_accum.stride(-2),
        dk_accum.stride(-4) if not is_split_qo else dk_accum.stride(-4),
        dk_accum.stride(-3),
        dk_accum.stride(-2),
        0 if not is_split_qo else dk_accum.stride(0),
        dv_accum.stride(-4) if not is_split_qo else dv_accum.stride(-4),
        dv_accum.stride(-3),
        dv_accum.stride(-2),
        0 if not is_split_qo else dv_accum.stride(0),
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        IS_SPLIT_QO=is_split_qo,
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


def _flash_dense_attn_varlen_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
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
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    # Assert the validity of inputs
    if not skip_checks:
        assert_bwd_inputs(
            q=q,
            k=k,
            v=v,
            out=out,
            dout=dout,
            lse=lse,
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
    TILE_M = tile_m if tile_m is not None else 64
    TILE_N = tile_n if tile_n is not None else 128
    TILE_K = max(triton.next_power_of_2(head_dim), 32)
    num_warps = num_threads // 32 if num_threads is not None else 8
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

    # Launch the kernel
    _bwd_dense_kernel[grid](
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum,
        dv_accum,
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
        dout.stride(-2),
        dout.stride(-3),
        0,
        lse_log2.stride(-2),
        0,
        dpsum.stride(-2),
        0,
        dq_accum.stride(-3),
        dq_accum.stride(-2),
        0,
        dk_accum.stride(-3),
        dk_accum.stride(-2),
        0 if not is_split_qo else dk_accum.stride(-4),
        0,
        dv_accum.stride(-3),
        dv_accum.stride(-2),
        0 if not is_split_qo else dv_accum.stride(-4),
        window_sizes.stride(0) if window_sizes is not None else 0,
        seqlen_q,
        seqlen_k,
        softmax_scale,
        head_dim,
        QHEAD_PER_KVHEAD=qhead_per_kvhead,
        NUM_SPLITS=num_splits,
        IS_SPLIT_QO=is_split_qo,
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
