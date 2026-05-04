"""
SM100 (Blackwell) warp-specialized dense forward attention kernel.

Architecture: 6-partition persistent kernel using Gluon SM100 primitives.
  - Partition 0 (rescale):   rescale O accumulator + finalize (scale + LSE writeback)
  - Partition 1 (softmax):   online softmax on upper SPLIT_M rows (tile_id=0)
  - Partition 2 (softmax):   online softmax on lower SPLIT_M rows (tile_id=1)
  - Partition 3 (mma):       tcgen05_mma Q*K^T -> S, P*V -> O
  - Partition 4 (load):      TMA load Q/K/V -> SMEM
  - Partition 5 (store):     TMA store O -> HBM
"""

from typing import Tuple, Optional
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
)

from flash_sparse_attn.ops.gluon import (
    assert_inputs,
    utils,
    cache_utils,
)
from flash_sparse_attn.ops.gluon.utils import (
    borrow_s_as_p,
)
from flash_sparse_attn.ops.gluon.activations import rescale_o, finalize, online_softmax
from flash_sparse_attn.ops.gluon.scheduling import (
    SharedMemoryChannel,
    TensorMemoryChannel,
    issue_async_tma_load,
    convert_smem_for_mma,
    AttentionConfig,
    ProgramScheduler,
)
from flash_sparse_attn.ops.gluon.launch_template import (
    KernelConfig,
    get_fwd_launch_config,
)


@gluon.jit
def _attn_fwd_softmax(tile_id: gl.constexpr, config, chnls, use_tmem_red: gl.constexpr):
    (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    if tile_id == 0:
        s_chnl = s0_tmem_chnl
        scale_chnl = scale0_mbarrier_chnl
        exp_chnl = exp_turnstile_mbarrier.create_producer()
    else:
        s_chnl = s1_tmem_chnl
        scale_chnl = scale1_mbarrier_chnl
        exp_chnl = exp_turnstile_mbarrier.create_consumer()
    online_softmax(
        tile_id,
        config,
        s_chnl,
        scale_chnl,
        exp_chnl,
        use_tmem_red,
    )


# ===-----------------------------------------------------------------------===#
# Partition: Load (TMA Q/K/V -> SMEM)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_load(config, chnls, descs):
    (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_producer = q_smem_chnl.create_producer()
    kv_producer = kv_smem_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        n_start, n_end = prog.get_loop_bounds()
        num_kv_tiles = (n_end - n_start) // config.TILE_N

        q0_smem, q0_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(
            q0_smem, q0_bar, desc_q, prog.batch_idx, prog.head_q_idx, prog.seq_offset
        )

        offs_kv_seq = n_end - config.TILE_N
        k_smem, k_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(
            k_smem, k_bar, desc_k, prog.batch_idx, prog.head_kv_idx, offs_kv_seq
        )

        q1_smem, q1_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(
            q1_smem,
            q1_bar,
            desc_q,
            prog.batch_idx,
            prog.head_q_idx,
            prog.seq_offset + config.SPLIT_M,
        )

        v_smem, v_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(
            v_smem, v_bar, desc_v, prog.batch_idx, prog.head_kv_idx, offs_kv_seq
        )

        for i in range(1, num_kv_tiles):
            offs_kv_seq = n_end - (1 + i) * config.TILE_N
            k_smem, k_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(
                k_smem, k_bar, desc_k, prog.batch_idx, prog.head_kv_idx, offs_kv_seq
            )
            v_smem, v_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(
                v_smem, v_bar, desc_v, prog.batch_idx, prog.head_kv_idx, offs_kv_seq
            )


# ===-----------------------------------------------------------------------===#
# Partition: MMA (tcgen05_mma Q*K^T and P*V)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_mma(config, chnls, descs):
    (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_consumer = q_smem_chnl.create_consumer()
    kv_consumer = kv_smem_chnl.create_consumer()
    o_producer = o_tmem_chnl.create_producer()

    s0_producer = s0_tmem_chnl.create_producer()
    s1_producer = s1_tmem_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        n_start, n_end = prog.get_loop_bounds()
        num_mmas = (n_end - n_start) // config.TILE_N

        q0_smem, q0_bar, q_consumer = q_consumer.acquire()
        k_smem, k_bar, kv_consumer = kv_consumer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        tcgen05_mma(
            convert_smem_for_mma(q0_smem),
            convert_smem_for_mma(k_smem).permute((1, 0)),
            s0_tmem,
            use_acc=False,
            mbarriers=[s0_bar],
        )

        q1_smem, q1_bar, q_consumer = q_consumer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        tcgen05_mma(
            convert_smem_for_mma(q1_smem),
            convert_smem_for_mma(k_smem).permute((1, 0)),
            s1_tmem,
            use_acc=False,
            mbarriers=[s1_bar, k_bar],
        )

        v_smem, v_bar, kv_consumer = kv_consumer.acquire()
        o0_tmem, o0_bar, o_producer = o_producer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        p0_tmem = borrow_s_as_p(config, s0_tmem)
        tcgen05_mma(
            p0_tmem,
            convert_smem_for_mma(v_smem),
            o0_tmem,
            use_acc=False,
            mbarriers=[o0_bar],
        )
        o1_init = False

        for _ in range(num_mmas - 1):
            k_smem, k_bar, kv_consumer = kv_consumer.acquire()
            tcgen05_mma(
                convert_smem_for_mma(q0_smem),
                convert_smem_for_mma(k_smem).permute((1, 0)),
                s0_tmem,
                use_acc=False,
                mbarriers=[s0_bar],
            )

            o1_tmem, o1_bar, o_producer = o_producer.acquire()
            s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
            p1_tmem = borrow_s_as_p(config, s1_tmem)
            tcgen05_mma(
                p1_tmem,
                convert_smem_for_mma(v_smem),
                o1_tmem,
                use_acc=o1_init,
                mbarriers=[o1_bar, v_bar],
            )
            o1_init = True

            tcgen05_mma(
                convert_smem_for_mma(q1_smem),
                convert_smem_for_mma(k_smem).permute((1, 0)),
                s1_tmem,
                use_acc=False,
                mbarriers=[s1_bar, k_bar],
            )

            v_smem, v_bar, kv_consumer = kv_consumer.acquire()
            o0_tmem, o0_bar, o_producer = o_producer.acquire()
            s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
            p0_tmem = borrow_s_as_p(config, s0_tmem)
            tcgen05_mma(
                p0_tmem, convert_smem_for_mma(v_smem), o0_tmem, mbarriers=[o0_bar]
            )

        tcgen05_commit(q0_bar)
        tcgen05_commit(q1_bar)

        o1_tmem, o1_bar, o_producer = o_producer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        p1_tmem = borrow_s_as_p(config, s1_tmem)
        tcgen05_mma(
            p1_tmem,
            convert_smem_for_mma(v_smem),
            o1_tmem,
            use_acc=o1_init,
            mbarriers=[o1_bar, v_bar, s0_bar, s1_bar],
        )


# ===-----------------------------------------------------------------------===#
# Partition: Rescale (rescale O + finalize writeback)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_rescale(config, chnls, Lse):
    (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls

    s0_tmem = s0_tmem_chnl.mem.index(0)
    s1_tmem = s1_tmem_chnl.mem.index(0)
    scale0_consumer = scale0_mbarrier_chnl.create_consumer()
    scale1_consumer = scale1_mbarrier_chnl.create_consumer()
    o_tmem_consumer = o_tmem_chnl.create_consumer()

    o_smem_producer = o_smem_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        n_start, n_end = prog.get_loop_bounds()
        num_corrections = (n_end - n_start) // config.TILE_N

        _, scale0_bar, scale0_consumer = scale0_consumer.acquire()
        mbarrier.arrive(scale0_bar, count=1)
        _, scale1_bar, scale1_consumer = scale1_consumer.acquire()
        mbarrier.arrive(scale1_bar, count=1)

        for i in range(num_corrections - 1):
            scale0_consumer, o_tmem_consumer = rescale_o(
                config, s0_tmem, scale0_consumer, o_tmem_consumer
            )
            scale1_consumer, o_tmem_consumer = rescale_o(
                config, s1_tmem, scale1_consumer, o_tmem_consumer
            )

        scale0_consumer, o_smem_producer, o_tmem_consumer = finalize(
            config,
            prog,
            s0_tmem,
            Lse,
            scale0_consumer,
            o_smem_producer,
            o_tmem_consumer,
            final_scale=1.0,
            CHECK_NAN=True,
            tile_id=0,
        )
        scale1_consumer, o_smem_producer, o_tmem_consumer = finalize(
            config,
            prog,
            s1_tmem,
            Lse,
            scale1_consumer,
            o_smem_producer,
            o_tmem_consumer,
            final_scale=1.0,
            CHECK_NAN=True,
            tile_id=1,
        )


# ===-----------------------------------------------------------------------===#
# Partition: Store (TMA store O -> HBM)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_store(config, chnls, descs):
    (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    o_smem_consumer = o_smem_chnl.create_consumer()
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        o0_smem, o0_bar, o_smem_consumer = o_smem_consumer.acquire()
        tma.async_copy_shared_to_global(
            desc_o, [prog.batch_idx, prog.head_q_idx, prog.seq_offset, 0], o0_smem
        )

        o1_smem, o1_bar, o_smem_consumer = o_smem_consumer.acquire()
        tma.async_copy_shared_to_global(
            desc_o,
            [prog.batch_idx, prog.head_q_idx, prog.seq_offset + config.SPLIT_M, 0],
            o1_smem,
        )

        tma.store_wait(1)
        mbarrier.arrive(o0_bar, count=1)
        tma.store_wait(0)
        mbarrier.arrive(o1_bar, count=1)


# ===-----------------------------------------------------------------------===#
# Kernel Entry Point
# ===-----------------------------------------------------------------------===#


def attention_repr(specialization):
    name = "gluon_attention"
    if specialization.constants["dtype"] == gl.float8e5:
        name = "cutlass_" + name
    return name


@gluon.jit(
    do_not_specialize=[
        "batch_size",
        "num_heads_q",
        "num_heads_kv",
        "seqlen_q",
        "seqlen_k",
    ],
    repr=attention_repr,
)
def attention_kernel(
    softmax_scale_log2,
    Lse,
    batch_size,
    num_heads_q,
    num_heads_kv,
    seqlen_q,
    seqlen_k,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    TILE_M: gl.constexpr,
    TILE_N: gl.constexpr,
    TILE_K: gl.constexpr,
    GROUP_SIZE_N: gl.constexpr,
    NUM_SMS: gl.constexpr,
    SPLIT_EXP_FACTOR: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    IS_LOCAL: gl.constexpr,
    WINDOW_SIZE_LEFT: gl.constexpr,
    WINDOW_SIZE_RIGHT: gl.constexpr,
    dtype: gl.constexpr,
    num_warps: gl.constexpr,
    use_tmem_red: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
    USE_EXP2_TURNSTILE: gl.constexpr,
):
    config = AttentionConfig(
        softmax_scale_log2,
        batch_size,
        num_heads_q,
        num_heads_kv,
        seqlen_q,
        seqlen_k,
        TILE_M,
        TILE_N,
        TILE_K,
        GROUP_SIZE_N,
        NUM_SMS,
        IS_CAUSAL,
        IS_LOCAL,
        WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT,
        SPLIT_EXP_FACTOR,
        dtype,
        num_warps,
        NUM_KV_BUFFERS,
        USE_EXP2_TURNSTILE,
    )

    q_smem_chnl = SharedMemoryChannel.alloc(
        config.qo_smem_shape, config.dtype, gl.constexpr(desc_q.layout), num_buffers=2
    )
    kv_smem_chnl = SharedMemoryChannel.alloc(
        config.kv_smem_shape,
        config.dtype,
        gl.constexpr(desc_k.layout),
        num_buffers=config.num_kv_buffers,
    )
    o_tmem_chnl = TensorMemoryChannel.alloc(
        config.o_tmem_shape, gl.float32, config.o_tmem_layout, num_buffers=2
    )
    o_smem_chnl = SharedMemoryChannel.alloc(
        config.qo_smem_shape, config.dtype, gl.constexpr(desc_o.layout), num_buffers=2
    )
    s0_tmem_chnl = TensorMemoryChannel.alloc(
        config.s_tmem_shape, gl.float32, config.s_tmem_layout, num_buffers=1
    )
    s1_tmem_chnl = TensorMemoryChannel.alloc(
        config.s_tmem_shape, gl.float32, config.s_tmem_layout, num_buffers=1
    )
    scale0_mbarrier_chnl = SharedMemoryChannel.alloc(
        [1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1
    )
    scale1_mbarrier_chnl = SharedMemoryChannel.alloc(
        [1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1
    )
    exp_turnstile_mbarrier = SharedMemoryChannel.alloc(
        [1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1
    )

    chnls = (
        q_smem_chnl,
        kv_smem_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    )
    descs = (desc_q, desc_k, desc_v, desc_o)
    gl.warp_specialize(
        [
            (_attn_fwd_rescale, (config, chnls, Lse)),
            (_attn_fwd_softmax, (0, config, chnls, use_tmem_red)),
            (_attn_fwd_softmax, (1, config, chnls, use_tmem_red)),
            (_attn_fwd_mma, (config, chnls, descs)),
            (_attn_fwd_load, (config, chnls, descs)),
            (_attn_fwd_store, (config, chnls, descs)),
        ],
        [4, 4, 1, 1, 1],
        [192, 192, 24, 24, 24],
    )

    q_smem_chnl.release()
    kv_smem_chnl.release()
    o_tmem_chnl.release()
    o_smem_chnl.release()
    s0_tmem_chnl.release()
    s1_tmem_chnl.release()
    scale0_mbarrier_chnl.release()
    scale1_mbarrier_chnl.release()
    exp_turnstile_mbarrier.release()


# ===-----------------------------------------------------------------------===#
# Dispatch Functions
# ===-----------------------------------------------------------------------===#


def _flash_dense_attn_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_size: Tuple[int, int] = (None, None),
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    skip_checks: bool = False,
    use_tmem_red: bool = False,
    kernel_config_override: KernelConfig | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    softmax_scale = softmax_scale or 1.0 / (head_dim**0.5)

    if not skip_checks:
        arch = cache_utils.get_device_arch(device)
        assert_inputs.assert_fwd_inputs(
            query,
            key,
            value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
            arch=arch,
        )

    TILE_K = head_dim
    is_local = window_size[0] is not None or window_size[1] is not None
    window_size_left = window_size[0] if window_size[0] is not None else None
    window_size_right = window_size[1] if window_size[1] is not None else None
    launch_config = get_fwd_launch_config(
        TILE_K,
        seqlen_q,
        query.dtype,
        is_causal,
        use_tmem_red,
        override=kernel_config_override,
    )

    TILE_M = launch_config.TILE_M
    TILE_N = launch_config.TILE_N
    SPLIT_M = TILE_M // 2
    GROUP_SIZE_N = launch_config.GROUP_SIZE_N
    NUM_SMS = (
        torch.cuda.get_device_properties(device).multi_processor_count
        * launch_config.OCCUPANCY
    )

    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads_q, seqlen_q),
        dtype=torch.float32,
        device=device,
    )

    desc_q = utils.make_tensor_desc(
        query,
        shape=[batch_size, num_heads_q, seqlen_q, head_dim],
        strides=[
            seqlen_q * num_heads_q * head_dim,
            head_dim,
            num_heads_q * head_dim,
            1,
        ],
        block_shape=[1, 1, SPLIT_M, TILE_K],
    )
    desc_k = utils.make_tensor_desc(
        key,
        shape=[batch_size, num_heads_kv, seqlen_k, head_dim],
        strides=[
            seqlen_k * num_heads_kv * head_dim,
            head_dim,
            num_heads_kv * head_dim,
            1,
        ],
        block_shape=[1, 1, TILE_N, TILE_K],
    )
    desc_v = utils.make_tensor_desc(
        value,
        shape=[batch_size, num_heads_kv, seqlen_k, head_dim],
        strides=[
            seqlen_k * num_heads_kv * head_dim,
            head_dim,
            num_heads_kv * head_dim,
            1,
        ],
        block_shape=[1, 1, TILE_N, TILE_K],
    )
    desc_o = utils.make_tensor_desc(
        out,
        shape=[batch_size, num_heads_q, seqlen_q, head_dim],
        strides=[
            seqlen_q * num_heads_q * head_dim,
            head_dim,
            num_heads_q * head_dim,
            1,
        ],
        block_shape=[1, 1, SPLIT_M, TILE_K],
    )

    dtype_gl = utils.torch_dtype_to_gluon(query.dtype)

    softmax_scale_log2 = softmax_scale * 1.44269504
    num_tiles_m = triton.cdiv(seqlen_q, TILE_M)
    num_tiles_bh = batch_size * num_heads_q
    grid = min(NUM_SMS, num_tiles_m * num_tiles_bh)

    attention_kernel[(grid,)](
        softmax_scale_log2,
        lse,
        batch_size,
        num_heads_q,
        num_heads_kv,
        seqlen_q,
        seqlen_k,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        TILE_M,
        TILE_N,
        TILE_K,
        GROUP_SIZE_N,
        NUM_SMS,
        SPLIT_EXP_FACTOR=launch_config.SPLIT_EXP_FACTOR,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        dtype=dtype_gl,
        num_warps=launch_config.NUM_WARPS,
        maxnreg=launch_config.MAXNREG,
        use_tmem_red=launch_config.USE_TMEM_RED,
        NUM_KV_BUFFERS=launch_config.NUM_KV_BUFFERS,
        USE_EXP2_TURNSTILE=launch_config.USE_EXP2_TURNSTILE,
    )

    return out, lse, softmax_scale


def _flash_dense_attn_varlen_base_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: float = None,
    window_size: Tuple[int, int] = (None, None),
    is_split_kv: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    raise NotImplementedError(
        "SM100 Gluon varlen forward is not yet implemented. "
        "Use the Triton backend for varlen attention."
    )
