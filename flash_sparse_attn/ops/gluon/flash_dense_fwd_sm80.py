from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy

from flash_sparse_attn.ops.gluon.assert_inputs import assert_fwd_sm80_inputs
from flash_sparse_attn.ops.gluon.utils import (
    window_sizes_heuristic,
    num_splits_heuristic,
)
from flash_sparse_attn.ops.gluon.cache_utils import get_device_num_sms
from flash_sparse_attn.ops.gluon.launch_grid import get_fwd_grid

from flash_sparse_attn.ops.gluon.ampere_helpers import gemm, gemm_rs
from flash_sparse_attn.ops.gluon.scheduler import (
    AttnFwdBlockScheduler,
    AttnFwdConfig,
    AttnFwdGridIndex,
    AttnFwdPointerScheduler,
    AttnMaskScheduler,
    SoftmaxScheduler,
)

from flash_sparse_attn.ops.gluon.flash_fwd_combine import _flash_attn_fwd_combine


@gluon.jit
def _fwd_inner_dense_kernel(
    config: AttnFwdConfig,
    ptrs_sched: AttnFwdPointerScheduler,
    mask_sched: AttnMaskScheduler,
    softmax_sched: SoftmaxScheduler,
    sQ,
    sK,
    sV,
    row_max,
    row_sum,
    acc_o,
    n_block,
    n_block_min,
    mma_offs_m,
    mma_offs_n,
    copy_offs_n,
    copy_offs_k,
    mma_layout: gl.constexpr,
    IS_MASK: gl.constexpr,
    MASK_SEQLEN: gl.constexpr,
    MASK_CAUSAL: gl.constexpr,
    MASK_LOCAL: gl.constexpr = False,
    MASK_SINK: gl.constexpr = False,
):
    # MMA operand layouts for QK/PV
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

    # Wait for K
    async_copy.wait_group(1)

    # Compute attention scores
    acc_s = gl.zeros([config.TILE_M, config.TILE_N], gl.float32, layout=mma_layout)
    acc_s = gemm(
        acc_s,
        sQ,
        sK,
        mma_lhs_layout,
        mma_rhs_layout,
    )

    if IS_MASK:
        # Apply mask to attention scores
        acc_s = mask_sched.apply_mask(
            acc_s=acc_s,
            iter_block=n_block,
            offs_m=mma_offs_m,
            offs_n=mma_offs_n,
            MASK_SEQLEN=MASK_SEQLEN,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
        )

    # Apply online softmax
    p, row_max, row_sum, row_scale = softmax_sched.online_softmax(
        acc_s=acc_s,
        row_max=row_max,
        row_sum=row_sum,
        CHECK_INF=IS_MASK,
    )

    # Rescale output accumulator
    acc_o = softmax_sched.rescale_o(acc_o, row_scale)

    # Wait for V
    async_copy.wait_group(0)
    gl.barrier()

    # Make pointers for next K and V
    gK_next = ptrs_sched.make_k_ptrs(config, n_block - 1, copy_offs_n, copy_offs_k)
    gV_next = ptrs_sched.make_v_ptrs(config, n_block - 1, copy_offs_n, copy_offs_k)

    # TODO: Support a CTA-uniform issue predicate for both copy and commit_group to reduce empty copy in last iteration.
    prefetch_mask = n_block > n_block_min

    # Load next K
    async_copy.async_copy_global_to_shared(
        sK,
        gK_next,
        mask=prefetch_mask,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    async_copy.commit_group()

    # Update output accumulator
    acc_o = gemm_rs(
        acc_o,
        p,
        sV,
        mma_lhs_layout,
        mma_rhs_layout,
    )
    gl.barrier()

    # Load next V
    async_copy.async_copy_global_to_shared(
        sV,
        gV_next,
        mask=prefetch_mask,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    async_copy.commit_group()

    return row_max, row_sum, acc_o


@triton.heuristics(
    {
        "EVEN_M": lambda args: not args["HAS_CU_SEQLENS_Q"]
        and args["seqlen_q"] % args["TILE_M"] == 0,
        "EVEN_N": lambda args: not args["HAS_CU_SEQLENS_K"]
        and args["seqlen_k"] % args["TILE_N"] == 0,
    }
)
@gluon.jit
def _fwd_dense_kernel(
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
    head_dim,
    QHEAD_PER_KVHEAD: gl.constexpr,
    PACK_GQA: gl.constexpr,
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_K: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    IS_SPLIT_KV: gl.constexpr,
    HAS_CU_SEQLENS_Q: gl.constexpr,
    HAS_CU_SEQLENS_K: gl.constexpr,
    HAS_SEQUSED_Q: gl.constexpr,
    HAS_SEQUSED_K: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
):
    num_warps: gl.constexpr = gl.num_warps()

    copy_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, 8],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)

    # Create grid index
    grid_idx = AttnFwdGridIndex.create(
        NUM_SPLITS=NUM_SPLITS,
        QHEAD_PER_KVHEAD=QHEAD_PER_KVHEAD,
        IS_SPLIT_KV=IS_SPLIT_KV,
        PACK_GQA=PACK_GQA,
    )

    # Load window sizes
    window_size_sink, window_size_left, window_size_right, window_size_near = (
        grid_idx.load_window_sizes(
            window_sizes=mWindowSizes, stride_wh=stride_wh, IS_LOCAL=IS_LOCAL
        )
    )

    # Create config
    config = AttnFwdConfig.create(
        softmax_scale=softmax_scale,
        m_block=grid_idx.m_block,
        batch_idx=grid_idx.batch_idx,
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
        ROW_LAYOUT=row_layout,
        PACK_GQA=PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=IS_CAUSAL,
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
        NUM_SPLITS=NUM_SPLITS,
        IS_CAUSAL=IS_CAUSAL,
        IS_LOCAL=IS_LOCAL,
        IS_SPLIT_KV=IS_SPLIT_KV,
    )

    # Create mask scheduler
    mask_sched = AttnMaskScheduler.create(config)

    # Create softmax scheduler
    softmax_sched = SoftmaxScheduler.create(config)

    # Compute offsets for global memory copy and MMA operations
    copy_offs_m = gl.arange(0, TILE_M, gl.SliceLayout(1, copy_layout))
    copy_offs_n = gl.arange(0, TILE_N, gl.SliceLayout(1, copy_layout))
    copy_offs_k = gl.arange(0, TILE_K, gl.SliceLayout(0, copy_layout))
    mma_offs_m = gl.arange(0, TILE_M, gl.SliceLayout(1, mma_layout))
    mma_offs_n = gl.arange(0, TILE_N, gl.SliceLayout(0, mma_layout))
    mma_offs_k = gl.arange(0, TILE_K, gl.SliceLayout(0, mma_layout))

    # Compute predicates for global memory copy and MMA operations
    if not EVEN_M:
        copy_m_rows = (
            (config.m_block * TILE_M + copy_offs_m) // QHEAD_PER_KVHEAD_PACKGQA
            if PACK_GQA
            else config.m_block * TILE_M + copy_offs_m
        )
        mma_m_rows = (
            (config.m_block * TILE_M + mma_offs_m) // QHEAD_PER_KVHEAD_PACKGQA
            if PACK_GQA
            else config.m_block * TILE_M + mma_offs_m
        )
        predicate_load_m = copy_m_rows < config.actual_seqlen_q
        predicate_store_m = mma_m_rows < config.actual_seqlen_q
    if not EVEN_N:
        copy_n_cols = (block_sched.n_block_max - 1) * TILE_N + copy_offs_n
        predicate_load_n = copy_n_cols < config.actual_seqlen_k

    # Early exit if no n_blocks to process
    if block_sched.is_empty():
        empty_o = gl.zeros([TILE_M, TILE_K], mOut.dtype.element_ty, layout=mma_layout)
        gO = ptrs_sched.make_out_ptrs(config, mma_offs_m, mma_offs_k)
        empty_o = empty_o.to(gO.dtype.element_ty)
        gl.store(
            gO,
            empty_o,
            mask=predicate_store_m[:, None] if not EVEN_M else None,
        )
        empty_lse = gl.full(
            [TILE_M], float("-inf"), mLse.dtype.element_ty, layout=row_layout
        )
        gLSE = ptrs_sched.make_lse_ptrs(config, mma_offs_m)
        gl.store(gLSE, empty_lse, mask=predicate_store_m if not EVEN_M else None)
        return

    # Allocate shared memory for Q/K/V tiles
    sQ_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_M, TILE_K], mQ.dtype.element_ty
    )
    sK_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], mK.dtype.element_ty
    )
    sV_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_N, TILE_K], mV.dtype.element_ty
    )
    sQ = gl.allocate_shared_memory(mQ.dtype.element_ty, [TILE_M, TILE_K], sQ_layout)
    sK = gl.allocate_shared_memory(mK.dtype.element_ty, [TILE_N, TILE_K], sK_layout)
    sV = gl.allocate_shared_memory(mV.dtype.element_ty, [TILE_N, TILE_K], sV_layout)

    # Initialize accumulators
    row_max = gl.full([TILE_M], float("-inf"), gl.float32, layout=row_layout)
    row_sum = gl.zeros([TILE_M], gl.float32, layout=row_layout)
    acc_o = gl.zeros([TILE_M, TILE_K], gl.float32, layout=mma_layout)

    # Load Q
    gQ = ptrs_sched.make_q_ptrs(config, copy_offs_m, copy_offs_k)
    async_copy.async_copy_global_to_shared(
        sQ,
        gQ,
        mask=predicate_load_m[:, None] if not EVEN_M else None,
        cache_modifier=".ca",
        eviction_policy="evict_last",
    )
    async_copy.commit_group()

    # Load K for near diagonal
    gK = ptrs_sched.make_k_ptrs(
        config, block_sched.n_block_max - 1, copy_offs_n, copy_offs_k
    )
    async_copy.async_copy_global_to_shared(
        sK,
        gK,
        mask=predicate_load_n[:, None] if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    async_copy.commit_group()

    # Load V for near diagonal
    gV = ptrs_sched.make_v_ptrs(
        config, block_sched.n_block_max - 1, copy_offs_n, copy_offs_k
    )
    async_copy.async_copy_global_to_shared(
        sV,
        gV,
        mask=predicate_load_n[:, None] if not EVEN_N else None,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    async_copy.commit_group()

    # Wait for Q
    async_copy.wait_group(2)
    gl.barrier()

    # Process n_blocks with causal masking
    if IS_CAUSAL or IS_LOCAL:
        n_block_max_no_mask = block_sched.n_block_max_no_mask
        for n_block in range(
            block_sched.n_block_max - 1,
            block_sched.n_block_max_no_mask - 1,
            -1,
        ):
            row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                row_max,
                row_sum,
                acc_o,
                n_block,
                block_sched.n_block_min,
                mma_offs_m,
                mma_offs_n,
                copy_offs_n,
                copy_offs_k,
                mma_layout,
                IS_MASK=True,
                MASK_SEQLEN=not EVEN_N,
                MASK_CAUSAL=IS_CAUSAL,
                MASK_LOCAL=IS_LOCAL,
                MASK_SINK=False,
            )
    else:
        # First iteration with seqlen masking
        n_block = block_sched.n_block_max - 1
        n_block_max_no_mask = n_block
        row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
            config,
            ptrs_sched,
            mask_sched,
            softmax_sched,
            sQ,
            sK,
            sV,
            row_max,
            row_sum,
            acc_o,
            n_block,
            block_sched.n_block_min,
            mma_offs_m,
            mma_offs_n,
            copy_offs_n,
            copy_offs_k,
            mma_layout,
            IS_MASK=not EVEN_N,
            MASK_SEQLEN=not EVEN_N,
            MASK_CAUSAL=False,
        )

    if not IS_LOCAL and n_block_max_no_mask > block_sched.n_block_min:
        for n_block in range(
            n_block_max_no_mask - 1,
            block_sched.n_block_min - 1,
            -1,
        ):
            row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                config,
                ptrs_sched,
                mask_sched,
                softmax_sched,
                sQ,
                sK,
                sV,
                row_max,
                row_sum,
                acc_o,
                n_block,
                block_sched.n_block_min,
                mma_offs_m,
                mma_offs_n,
                copy_offs_n,
                copy_offs_k,
                mma_layout,
                IS_MASK=False,
                MASK_SEQLEN=False,
                MASK_CAUSAL=False,
            )

    if IS_LOCAL:
        if block_sched.n_block_window_max > block_sched.n_block_window_min:
            # Load K for long-range window
            gK = ptrs_sched.make_k_ptrs(
                config, block_sched.n_block_window_max - 1, copy_offs_n, copy_offs_k
            )
            async_copy.async_copy_global_to_shared(
                sK, gK, cache_modifier=".cg", eviction_policy="evict_first"
            )
            async_copy.commit_group()

            # Load V for long-range window
            gV = ptrs_sched.make_v_ptrs(
                config, block_sched.n_block_window_max - 1, copy_offs_n, copy_offs_k
            )
            async_copy.async_copy_global_to_shared(
                sV, gV, cache_modifier=".cg", eviction_policy="evict_first"
            )
            async_copy.commit_group()

            # Process n_blocks with local right masking
            for n_block in range(
                block_sched.n_block_window_max - 1,
                block_sched.n_block_window_max_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_o,
                    n_block,
                    block_sched.n_block_window_min,
                    mma_offs_m,
                    mma_offs_n,
                    copy_offs_n,
                    copy_offs_k,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

            # Process n_blocks without masking
            for n_block in range(
                block_sched.n_block_window_max_no_mask - 1,
                block_sched.n_block_window_min_no_mask - 1,
                -1,
            ):
                row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_o,
                    n_block,
                    block_sched.n_block_window_min,
                    mma_offs_m,
                    mma_offs_n,
                    copy_offs_n,
                    copy_offs_k,
                    mma_layout,
                    IS_MASK=False,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                )

            # Process n_blocks with local left masking
            for n_block in range(
                block_sched.n_block_window_min_no_mask - 1,
                block_sched.n_block_window_min - 1,
                -1,
            ):
                row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_o,
                    n_block,
                    block_sched.n_block_window_min,
                    mma_offs_m,
                    mma_offs_n,
                    copy_offs_n,
                    copy_offs_k,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=False,
                )

        if block_sched.n_block_sink_max > block_sched.n_block_sink_min:
            # Load K for prefix sink
            gK = ptrs_sched.make_k_ptrs(
                config, block_sched.n_block_sink_max - 1, copy_offs_n, copy_offs_k
            )
            async_copy.async_copy_global_to_shared(
                sK, gK, cache_modifier=".cg", eviction_policy="evict_first"
            )
            async_copy.commit_group()

            # Load V for prefix sink
            gV = ptrs_sched.make_v_ptrs(
                config, block_sched.n_block_sink_max - 1, copy_offs_n, copy_offs_k
            )
            async_copy.async_copy_global_to_shared(
                sV, gV, cache_modifier=".cg", eviction_policy="evict_first"
            )
            async_copy.commit_group()

            # Process n_blocks with prefix sink masking
            for n_block in range(
                block_sched.n_block_sink_max - 1,
                block_sched.n_block_sink_min - 1,
                -1,
            ):
                row_max, row_sum, acc_o = _fwd_inner_dense_kernel(
                    config,
                    ptrs_sched,
                    mask_sched,
                    softmax_sched,
                    sQ,
                    sK,
                    sV,
                    row_max,
                    row_sum,
                    acc_o,
                    n_block,
                    block_sched.n_block_sink_min,
                    mma_offs_m,
                    mma_offs_n,
                    copy_offs_n,
                    copy_offs_k,
                    mma_layout,
                    IS_MASK=True,
                    MASK_SEQLEN=False,
                    MASK_CAUSAL=False,
                    MASK_LOCAL=True,
                    MASK_SINK=True,
                )

    # Finalize softmax
    row_scale, lse = softmax_sched.finalize(
        row_max=row_max,
        row_sum=row_sum,
        IS_LOG2=IS_SPLIT_KV,
    )

    # Store LSE
    gLSE = ptrs_sched.make_lse_ptrs(config, mma_offs_m)
    gl.store(gLSE, lse, mask=predicate_store_m if not EVEN_M else None)

    # Finalize rescale
    acc_o = softmax_sched.rescale_o(
        acc_o=acc_o,
        row_scale=row_scale,
    )

    # Store output
    if IS_SPLIT_KV:
        gO = ptrs_sched.make_out_ptrs(config, mma_offs_m, mma_offs_k)
        gl.store(gO, acc_o, mask=predicate_store_m[:, None] if not EVEN_M else None)
    else:
        sQ.store(acc_o.to(mOut.dtype.element_ty))
        gl.barrier()
        rO = sQ.load(copy_layout)
        gO = ptrs_sched.make_out_ptrs(config, copy_offs_m, copy_offs_k)
        gl.store(gO, rO, mask=predicate_load_m[:, None] if not EVEN_M else None)


def _flash_dense_attn_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    num_SMs = get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = (
        softmax_scale if softmax_scale is not None else 1.0 / (head_dim**0.5)
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    qhead_per_kvhead_packgqa = num_heads_q // num_heads_kv if pack_gqa else 1
    if is_local and window_sizes is None:
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_fwd_sm80_inputs(
            query,
            key,
            value,
            window_sizes=window_sizes,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=seqused_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )

    TILE_M = 128
    TILE_N = 64
    TILE_K = max(triton.next_power_of_2(head_dim), 16)
    num_warps = 4
    num_stages = 1
    if is_split_kv and num_splits is None:
        num_splits = num_splits_heuristic(
            seqlen_q=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            is_local=is_local,
            num_heads_kv=num_heads_kv,
        )
    elif not is_split_kv:
        num_splits = 1

    out = out if out is not None else torch.empty_like(query)
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

    grid = get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    _fwd_dense_kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        window_sizes,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        seqused_k,
        query.stride(-4),
        query.stride(-2),
        query.stride(-3),
        key.stride(-4),
        key.stride(-2),
        key.stride(-3),
        value.stride(-4),
        value.stride(-2),
        value.stride(-3),
        out.stride(-4) if not is_split_kv else out_partial.stride(-4),
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(-3) if not is_split_kv else out_partial.stride(-3),
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
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        HAS_CU_SEQLENS_Q=False,
        HAS_CU_SEQLENS_K=False,
        HAS_SEQUSED_Q=False,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if is_split_kv:
        _flash_attn_fwd_combine(
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
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    num_splits: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    num_SMs = get_device_num_sms(device)
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
        window_sizes = window_sizes_heuristic(seqlen_k, num_heads_kv, device)

    if not skip_checks:
        assert_fwd_sm80_inputs(
            query,
            key,
            value,
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

    TILE_M = 128
    TILE_N = 64
    TILE_K = max(triton.next_power_of_2(head_dim), 16)
    num_warps = 4
    num_stages = 1
    if is_split_kv and num_splits is None:
        num_splits = num_splits_heuristic(
            seqlen_q=seqlen_q * qhead_per_kvhead_packgqa,
            seqlen_k=seqlen_k,
            num_SMs=num_SMs,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            is_local=is_local,
            num_heads_kv=num_heads_kv,
        )
    elif not is_split_kv:
        num_splits = 1

    out = out if out is not None else torch.empty_like(query)
    lse = (
        lse
        if lse is not None
        else torch.empty(
            (num_heads_q, total_seqlen_q),
            dtype=torch.float32,
            device=query.device,
        )
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

    grid = get_fwd_grid(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )

    _fwd_dense_kernel[grid](
        query,
        key,
        value,
        out if not is_split_kv else out_partial,
        lse if not is_split_kv else lse_partial,
        window_sizes,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        0,
        query.stride(-2),
        query.stride(-3),
        0,
        key.stride(-2),
        key.stride(-3),
        0,
        value.stride(-2),
        value.stride(-3),
        0,
        out.stride(-2) if not is_split_kv else out_partial.stride(-2),
        out.stride(-3) if not is_split_kv else out_partial.stride(-3),
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
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        IS_SPLIT_KV=is_split_kv,
        HAS_CU_SEQLENS_Q=True,
        HAS_CU_SEQLENS_K=True,
        HAS_SEQUSED_Q=seqused_q is not None,
        HAS_SEQUSED_K=seqused_k is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
        )

    return out, lse, softmax_scale
