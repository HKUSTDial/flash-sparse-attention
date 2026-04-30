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
from dataclasses import dataclass, fields
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
    get_desc_channel,
    issue_async_tma_load,
    AttentionConfig,
    ProgramScheduler,
)



@gluon.jit
def _attn_fwd_softmax(
    tile_id: gl.constexpr, config, chnls, use_tmem_red: gl.constexpr
):
    (
        q_tma_chnl,
        kv_tma_chnl,
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
        q_tma_chnl,
        kv_tma_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_producer = q_tma_chnl.create_producer()
    kv_producer = kv_tma_chnl.create_producer()

    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        n_start, n_end = prog.get_loop_bounds()
        num_kv_tiles = (n_end - n_start) // config.TILE_N

        q0_offset = prog.qo_offset + config.SPLIT_M * 0
        q0_smem, q0_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q0_smem, q0_bar, desc_q, q0_offset)

        offs_kv_tma = prog.kv_offset + n_end - config.TILE_N
        k_smem, k_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(k_smem, k_bar, desc_k, offs_kv_tma)

        q1_offset = prog.qo_offset + config.SPLIT_M * 1
        q1_smem, q1_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q1_smem, q1_bar, desc_q, q1_offset)

        v_smem, v_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(v_smem, v_bar, desc_v, offs_kv_tma)

        for i in range(1, num_kv_tiles):
            offs_kv_tma = prog.kv_offset + n_end - (1 + i) * config.TILE_N
            k_smem, k_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(k_smem, k_bar, desc_k, offs_kv_tma)
            v_smem, v_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(v_smem, v_bar, desc_v, offs_kv_tma)


# ===-----------------------------------------------------------------------===#
# Partition: MMA (tcgen05_mma Q*K^T and P*V)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_mma(config, chnls, descs):
    (
        q_tma_chnl,
        kv_tma_chnl,
        o_tmem_chnl,
        o_smem_chnl,
        s0_tmem_chnl,
        s1_tmem_chnl,
        scale0_mbarrier_chnl,
        scale1_mbarrier_chnl,
        exp_turnstile_mbarrier,
    ) = chnls
    desc_q, desc_k, desc_v, desc_o = descs

    q_consumer = q_tma_chnl.create_consumer()
    kv_consumer = kv_tma_chnl.create_consumer()
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
            q0_smem, k_smem.permute((1, 0)), s0_tmem, use_acc=False, mbarriers=[s0_bar]
        )

        q1_smem, q1_bar, q_consumer = q_consumer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        tcgen05_mma(
            q1_smem,
            k_smem.permute((1, 0)),
            s1_tmem,
            use_acc=False,
            mbarriers=[s1_bar, k_bar],
        )

        v_smem, v_bar, kv_consumer = kv_consumer.acquire()
        o0_tmem, o0_bar, o_producer = o_producer.acquire()
        s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
        p0_tmem = borrow_s_as_p(config, s0_tmem)
        tcgen05_mma(p0_tmem, v_smem, o0_tmem, use_acc=False, mbarriers=[o0_bar])
        o1_init = False

        for _ in range(num_mmas - 1):
            k_smem, k_bar, kv_consumer = kv_consumer.acquire()
            tcgen05_mma(
                q0_smem,
                k_smem.permute((1, 0)),
                s0_tmem,
                use_acc=False,
                mbarriers=[s0_bar],
            )

            o1_tmem, o1_bar, o_producer = o_producer.acquire()
            s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
            p1_tmem = borrow_s_as_p(config, s1_tmem)
            tcgen05_mma(
                p1_tmem, v_smem, o1_tmem, use_acc=o1_init, mbarriers=[o1_bar, v_bar]
            )
            o1_init = True

            tcgen05_mma(
                q1_smem,
                k_smem.permute((1, 0)),
                s1_tmem,
                use_acc=False,
                mbarriers=[s1_bar, k_bar],
            )

            v_smem, v_bar, kv_consumer = kv_consumer.acquire()
            o0_tmem, o0_bar, o_producer = o_producer.acquire()
            s0_tmem, s0_bar, s0_producer = s0_producer.acquire()
            p0_tmem = borrow_s_as_p(config, s0_tmem)
            tcgen05_mma(p0_tmem, v_smem, o0_tmem, mbarriers=[o0_bar])

        tcgen05_commit(q0_bar)
        tcgen05_commit(q1_bar)

        o1_tmem, o1_bar, o_producer = o_producer.acquire()
        s1_tmem, s1_bar, s1_producer = s1_producer.acquire()
        p1_tmem = borrow_s_as_p(config, s1_tmem)
        tcgen05_mma(
            p1_tmem,
            v_smem,
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
        q_tma_chnl,
        kv_tma_chnl,
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
        q_tma_chnl,
        kv_tma_chnl,
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
            desc_o, [prog.qo_offset + config.SPLIT_M * 0, 0], o0_smem
        )

        o1_smem, o1_bar, o_smem_consumer = o_smem_consumer.acquire()
        tma.async_copy_shared_to_global(
            desc_o, [prog.qo_offset + config.SPLIT_M * 1, 0], o1_smem
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
    do_not_specialize=["batch_size", "num_heads", "seqlen_q", "seqlen_k"],
    repr=attention_repr,
)
def attention_kernel(
    softmax_scale_log2,
    Lse,
    batch_size,
    num_heads,
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
        num_heads,
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

    q_tma_chnl = get_desc_channel(desc_q, num_buffers=2)
    kv_tma_chnl = get_desc_channel(desc_k, num_buffers=config.num_kv_buffers)
    o_tmem_chnl = TensorMemoryChannel.alloc(
        config.o_shape, gl.float32, config.o_tmem_layout, num_buffers=2
    )
    o_smem_chnl = SharedMemoryChannel.alloc(
        config.o_shape, config.dtype, gl.constexpr(desc_o.layout), num_buffers=2
    )
    s0_tmem_chnl = TensorMemoryChannel.alloc(
        config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1
    )
    s1_tmem_chnl = TensorMemoryChannel.alloc(
        config.qk_shape, gl.float32, config.qk_tmem_layout, num_buffers=1
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
        q_tma_chnl,
        kv_tma_chnl,
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

    q_tma_chnl.release()
    kv_tma_chnl.release()
    o_tmem_chnl.release()
    o_smem_chnl.release()
    s0_tmem_chnl.release()
    s1_tmem_chnl.release()
    scale0_mbarrier_chnl.release()
    scale1_mbarrier_chnl.release()
    exp_turnstile_mbarrier.release()


# ===-----------------------------------------------------------------------===#
# KernelConfig + select_kernel_config
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_blackwell_ultra():
    return is_cuda() and torch.cuda.get_device_capability()[0:2] == (10, 3)


@dataclass(frozen=True, slots=True)
class KernelConfig:
    TILE_M: int = 256
    TILE_N: int = 128
    GROUP_SIZE_N: int | None = None
    SPLIT_EXP_FACTOR: int | None = None
    NUM_WARPS: int = 4
    MAXNREG: int = 128
    OCCUPANCY: int = 1
    USE_TMEM_RED: bool = False
    NUM_KV_BUFFERS: int | None = None
    USE_EXP2_TURNSTILE: bool | None = None


def _default_split_exp_factor(head_dim: int) -> int:
    return max(1, 256 // head_dim)


def _default_num_kv_buffers(head_dim: int, dtype: torch.dtype) -> int:
    is_fp16 = dtype in [torch.float16, torch.bfloat16]
    if is_fp16:
        return 3 if head_dim == 128 else 6
    return 4 if head_dim == 128 else 8


def select_kernel_config(
    head_dim: int,
    seqlen: int,
    dtype: torch.dtype,
    causal: bool,
    use_tmem_red: bool,
    override: KernelConfig | None = None,
) -> KernelConfig:
    is_fp8 = dtype == torch.float8_e5m2
    is_bf16 = dtype == torch.bfloat16
    is_bwu = is_blackwell_ultra()

    block_m = 256
    block_n = 128
    group_size_n = 1
    split_exp_factor = _default_split_exp_factor(head_dim)
    num_warps = 4
    maxnreg = 128
    occupancy = 1
    use_selected_tmem_red = (use_tmem_red or (is_bwu and not causal)) and not causal
    num_kv_buffers = _default_num_kv_buffers(head_dim, dtype)
    use_exp2_turnstile = head_dim == 64

    if causal:
        group_size_n = 8 if head_dim == 64 or seqlen <= 2048 else 4

    if head_dim == 128:
        split_exp_factor = 4
        if not causal and is_bf16 and seqlen <= 2048:
            group_size_n = 4
    elif not causal and head_dim == 64 and use_selected_tmem_red:
        split_exp_factor = 1
        if seqlen <= 1024:
            num_kv_buffers = 2
        elif seqlen >= 8192:
            maxnreg = 112
    elif causal and head_dim == 64:
        num_kv_buffers = 2
        if seqlen <= 1024:
            split_exp_factor = 2
        else:
            use_exp2_turnstile = False

    if is_fp8:
        if causal and head_dim == 64:
            group_size_n = 8 if seqlen <= 2048 else 4
            split_exp_factor = 4 if seqlen <= 2048 else 2
            maxnreg = 112 if seqlen >= 4096 else 128
            use_selected_tmem_red = False
            num_kv_buffers = 2
            use_exp2_turnstile = seqlen <= 1024
        elif causal and head_dim == 128:
            group_size_n = 8 if seqlen <= 2048 else 4
            split_exp_factor = 2 if seqlen <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = False
            num_kv_buffers = 4
            use_exp2_turnstile = False
        elif not causal and head_dim == 64:
            group_size_n = 1
            split_exp_factor = 2
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 2 if seqlen <= 1024 else 8
            use_exp2_turnstile = True
        elif not causal and head_dim == 128:
            group_size_n = 1
            split_exp_factor = 4 if seqlen <= 2048 else 8
            maxnreg = 128
            use_selected_tmem_red = is_bwu
            num_kv_buffers = 4
            use_exp2_turnstile = False
        else:
            group_size_n = 4 if causal else 1
            split_exp_factor = _default_split_exp_factor(head_dim)
            use_selected_tmem_red = use_tmem_red and not causal

    config = KernelConfig(
        TILE_M=block_m,
        TILE_N=block_n,
        GROUP_SIZE_N=group_size_n,
        SPLIT_EXP_FACTOR=split_exp_factor,
        NUM_WARPS=num_warps,
        MAXNREG=maxnreg,
        OCCUPANCY=occupancy,
        USE_TMEM_RED=use_selected_tmem_red,
        NUM_KV_BUFFERS=num_kv_buffers,
        USE_EXP2_TURNSTILE=use_exp2_turnstile,
    )
    if override is None:
        return config

    values = {
        field.name: getattr(override, field.name) for field in fields(KernelConfig)
    }
    values = {
        name: getattr(config, name) if value is None else value
        for name, value in values.items()
    }
    return KernelConfig(**values)


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
    p = select_kernel_config(
        TILE_K,
        seqlen_q,
        query.dtype,
        is_causal,
        use_tmem_red,
        override=kernel_config_override,
    )

    TILE_M = p.TILE_M
    TILE_N = p.TILE_N
    SPLIT_M = TILE_M // 2
    GROUP_SIZE_N = p.GROUP_SIZE_N
    NUM_SMS = (
        torch.cuda.get_device_properties(device).multi_processor_count * p.OCCUPANCY
    )

    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads_q, seqlen_q),
        dtype=torch.float32,
        device=device,
    )

    qo_m_dim = batch_size * num_heads_q * seqlen_q
    kv_n_dim = batch_size * num_heads_q * seqlen_k

    desc_q = utils.make_tensor_desc(
        query,
        shape=[qo_m_dim, TILE_K],
        strides=[TILE_K, 1],
        block_shape=[SPLIT_M, TILE_K],
    )
    desc_k = utils.make_tensor_desc(
        key, shape=[kv_n_dim, TILE_K], strides=[TILE_K, 1], block_shape=[TILE_N, TILE_K]
    )
    desc_v = utils.make_tensor_desc(
        value,
        shape=[kv_n_dim, TILE_K],
        strides=[TILE_K, 1],
        block_shape=[TILE_N, TILE_K],
    )
    desc_o = utils.make_tensor_desc(
        out,
        shape=[qo_m_dim, TILE_K],
        strides=[TILE_K, 1],
        block_shape=[SPLIT_M, TILE_K],
    )

    dtype_gl = utils.torch_dtype_to_gluon(query.dtype)

    softmax_scale_log2 = softmax_scale * 1.44269504
    num_pid_m = triton.cdiv(seqlen_q, TILE_M)
    num_pid_n = batch_size * num_heads_q
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    attention_kernel[(grid,)](
        softmax_scale_log2,
        lse,
        batch_size,
        num_heads_q,
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
        SPLIT_EXP_FACTOR=p.SPLIT_EXP_FACTOR,
        IS_CAUSAL=is_causal,
        IS_LOCAL=is_local,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        dtype=dtype_gl,
        num_warps=p.NUM_WARPS,
        maxnreg=p.MAXNREG,
        use_tmem_red=p.USE_TMEM_RED,
        NUM_KV_BUFFERS=p.NUM_KV_BUFFERS,
        USE_EXP2_TURNSTILE=p.USE_EXP2_TURNSTILE,
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
