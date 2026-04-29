"""
SM100 (Blackwell) warp-specialized dense forward attention kernel.

Architecture: 5-partition persistent kernel using Gluon SM100 primitives.
  - Partition 0 (load):     TMA load Q/K/V -> SMEM
  - Partition 1 (mma):      tcgen05_mma Q*K^T -> S, P*V -> O
  - Partition 2 (softmax0): online softmax on upper SPLIT_M rows
  - Partition 3 (softmax1): online softmax on lower SPLIT_M rows
  - Partition 4 (epilogue): TMA store O -> HBM
"""

from typing import Tuple, Optional
import copy
import math
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    float2,
)
from triton.experimental.gluon.language.nvidia.blackwell.float2 import Float2Tensor

from flash_sparse_attn.ops.gluon import (
    assert_inputs,
    utils,
    cache_utils,
    launch_template,
    launch_grid,
    flash_fwd_combine,
)


# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


# ===-----------------------------------------------------------------------===#
# Channel / Barrier Infrastructure
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


def Channel(T, alloc_fn):

    @gluon.aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr
        num_consumers: gl.constexpr

        @gluon.jit
        def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr,
                  num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(
                gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
            )
            empty_bars = gl.allocate_shared_memory(
                gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
            )
            for i in gl.static_range(num_buffers):
                mbarrier.init(ready_bars.index(i), count=1)
                mbarrier.init(empty_bars.index(i), count=num_consumers)
                mbarrier.arrive(empty_bars.index(i), count=num_consumers)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers, num_consumers)

        @gluon.jit
        def acquire_producer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)
            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)
            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_counter(self):
            return BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), self.num_buffers)

        @gluon.jit
        def create_producer(self):
            return Producer(self, self.create_counter())

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, self.create_counter())

        @gluon.jit
        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in gl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @gluon.aggregate
    class Producer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            mem, ready_bar = self.channel.acquire_producer(self.counter)
            next = Producer(self.channel, self.counter.increment())
            return mem, ready_bar, next

    @gluon.aggregate
    class Consumer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            mem, empty_bar = self.channel.acquire_consumer(self.counter)
            next = Consumer(self.channel, self.counter.increment())
            return mem, empty_bar, next

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(
    gl.shared_memory_descriptor, gl.allocate_shared_memory
)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(
    tensor_memory_descriptor, allocate_tensor_memory
)


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, num_consumers)


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset_y, offset_x=0):
    mbarrier.expect(bar, desc.get_tma_size())
    from triton.experimental.gluon.language.nvidia.blackwell import fence_async_shared
    fence_async_shared()
    tma.async_copy_global_to_shared(desc, [offset_y, offset_x], smem, bar)


# ===-----------------------------------------------------------------------===#
# Persistent Tile Scheduler
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ProgramInfo:
    start_m: gl.tensor
    offset_y: gl.tensor
    qo_offset_y: gl.tensor

    @gluon.jit
    def get_loop_bounds(self, BLOCK_N: gl.constexpr, N_CTX: gl.tensor, IS_CAUSAL: gl.constexpr):
        lo = gl.to_tensor(0)
        if IS_CAUSAL:
            hi = gl.minimum((self.start_m + 1) * BLOCK_N, N_CTX)
        else:
            hi = N_CTX
        return lo, hi


@gluon.aggregate
class ProgramScheduler:
    start_pid: gl.tensor
    num_tiles: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    GROUP_SIZE_N: gl.constexpr
    BLOCK_M: gl.constexpr
    SPLIT_M: gl.constexpr

    @gluon.jit
    def create(BLOCK_M: gl.constexpr, SPLIT_M: gl.constexpr, GROUP_SIZE_N: gl.constexpr,
               num_pid_m, num_pid_n, NUM_SMS: gl.constexpr):
        start_pid = gl.program_id(0)
        num_tiles = num_pid_m * num_pid_n
        return ProgramScheduler(start_pid, num_tiles, num_pid_m, num_pid_n,
                                GROUP_SIZE_N, BLOCK_M, SPLIT_M)

    @gluon.jit
    def get_program(self, pid):
        if self.GROUP_SIZE_N > 1:
            group_id = pid // (self.GROUP_SIZE_N * self.num_pid_m)
            first_pid_in_group = group_id * self.GROUP_SIZE_N
            group_size = gl.minimum(self.num_pid_n - first_pid_in_group, self.GROUP_SIZE_N)
            pid_m = (pid % (group_size * self.num_pid_m)) // group_size
            pid_n = first_pid_in_group + (pid % (group_size * self.num_pid_m)) % group_size
        else:
            pid_m = pid // self.num_pid_n
            pid_n = pid % self.num_pid_n

        start_m = pid_m
        offset_y = pid_n * self.BLOCK_M
        qo_offset_y = offset_y + pid_m * self.BLOCK_M
        return ProgramInfo(start_m, offset_y, qo_offset_y)


# ===-----------------------------------------------------------------------===#
# TMEM Helpers
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _borrow_s_as_p(config_block_n, s_tmem, dtype, qk_shape, p_tmem_layout):
    p_tmem = s_tmem.slice(0, config_block_n // 2)
    return p_tmem._reinterpret(dtype, qk_shape, p_tmem_layout)


@gluon.jit
def _borrow_s_as_alpha(split_m, s_tmem, block_n):
    alpha_layout: gl.constexpr = TensorMemoryLayout([split_m, 1], col_stride=1)
    alpha_tmem = s_tmem.slice(block_n // 2, 1)
    return alpha_tmem._reinterpret(gl.float32, [split_m, 1], alpha_layout)


@gluon.jit
def _borrow_s_for_epilogue(split_m, s_tmem, block_n):
    layout: gl.constexpr = TensorMemoryLayout([split_m, 1], col_stride=1)
    m_i_tmem = s_tmem.slice(block_n // 2 + 1, 1)
    l_i_tmem = s_tmem.slice(block_n // 2 + 2, 1)
    m_i_tmem = m_i_tmem._reinterpret(gl.float32, [split_m, 1], layout)
    l_i_tmem = l_i_tmem._reinterpret(gl.float32, [split_m, 1], layout)
    return m_i_tmem, l_i_tmem


# ===-----------------------------------------------------------------------===#
# Inner Softmax Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _fwd_inner_dense_softmax(
    s_tmem,
    s_bar,
    corr_producer,
    corr_bar,
    offs_m,
    m_i,
    l_i,
    n_block,
    N_CTX,
    sm_scale,
    SPLIT_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    qk_layout: gl.constexpr,
    qk_slice_dim1: gl.constexpr,
):
    """Online softmax on one SPLIT_M x BLOCK_N tile loaded from TMEM."""
    # Load S tile from TMEM
    qk = s_tmem.load(qk_layout)

    # Scale
    qk = qk * sm_scale

    # Causal mask
    if IS_CAUSAL:
        offs_n = n_block * BLOCK_N + gl.arange(0, BLOCK_N)
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = gl.where(mask, qk, float("-inf"))

    # Boundary mask
    offs_n = n_block * BLOCK_N + gl.arange(0, BLOCK_N)
    boundary_mask = offs_n[None, :] < N_CTX
    qk = gl.where(boundary_mask, qk, float("-inf"))

    # Online softmax: new_max, exp, new_sum
    row_max_new = gl.max(qk, axis=1)
    row_max_new = gl.maximum(m_i, row_max_new)

    # Correction factor for previous accumulator
    alpha = gl.exp2((m_i - row_max_new) * 1.44269504)

    # exp(qk - new_max)
    p = gl.exp2((qk - row_max_new[:, None]) * 1.44269504)

    # Update running sum
    l_i_new = l_i * alpha + gl.sum(p, axis=1)

    # Store alpha for MMA partition to rescale O
    alpha_tmem = _borrow_s_as_alpha(SPLIT_M, s_tmem, BLOCK_N)
    alpha_2d_layout: gl.constexpr = TensorMemoryLayout([SPLIT_M, 1], col_stride=1)
    alpha_tmem.store(gl.convert_layout(alpha.expand_dims(1), alpha_2d_layout))

    # Store P back to TMEM for P*V MMA
    p_half = p.to(s_tmem.type.element_ty)
    s_tmem.store(p_half)

    # Signal MMA partition that softmax is done
    mbarrier.arrive(s_bar, count=1)

    # Signal correction
    _, corr_bar_new, corr_producer = corr_producer.acquire()
    mbarrier.arrive(corr_bar, count=1)

    return row_max_new, l_i_new, corr_producer, corr_bar_new


# ===-----------------------------------------------------------------------===#
# Partition: Load (TMA Q/K/V -> SMEM)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_load(
    scheduler, q_chnl, kv_chnl,
    desc_q, desc_k, desc_v,
    N_CTX,
    BLOCK_N: gl.constexpr,
    SPLIT_M: gl.constexpr,
    NUM_SMS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
):
    q_producer = q_chnl.create_producer()
    kv_producer = kv_chnl.create_producer()

    for pid in range(scheduler.start_pid, scheduler.num_tiles, NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_loop_bounds(BLOCK_N, N_CTX, IS_CAUSAL)

        # Load Q tile 0 (upper SPLIT_M)
        q0_smem, q0_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q0_smem, q0_bar, desc_q, prog.qo_offset_y + SPLIT_M * 0)

        # Load first K tile
        k_smem, k_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(k_smem, k_bar, desc_k, prog.offset_y + lo)

        # Load Q tile 1 (lower SPLIT_M)
        q1_smem, q1_bar, q_producer = q_producer.acquire()
        issue_async_tma_load(q1_smem, q1_bar, desc_q, prog.qo_offset_y + SPLIT_M * 1)

        # Load first V tile
        v_smem, v_bar, kv_producer = kv_producer.acquire()
        issue_async_tma_load(v_smem, v_bar, desc_v, prog.offset_y + lo)

        # Pipeline remaining KV tiles
        for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
            k_smem, k_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(k_smem, k_bar, desc_k, prog.offset_y + start_n)
            v_smem, v_bar, kv_producer = kv_producer.acquire()
            issue_async_tma_load(v_smem, v_bar, desc_v, prog.offset_y + start_n)

    q_chnl.release()
    kv_chnl.release()


# ===-----------------------------------------------------------------------===#
# Partition: MMA (tcgen05_mma Q*K^T and P*V)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_mma(
    scheduler, q_chnl, kv_chnl, s0_chnl, s1_chnl, o_chnl, c0_chnl, c1_chnl,
    desc_o, N_CTX,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SPLIT_M: gl.constexpr,
    NUM_SMS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    dtype: gl.constexpr,
):
    q_consumer = q_chnl.create_consumer()
    kv_consumer = kv_chnl.create_consumer()
    s0_producer = s0_chnl.create_producer()
    s1_producer = s1_chnl.create_producer()
    o0_consumer = o_chnl.create_consumer() if o_chnl is not None else None
    c0_consumer = c0_chnl.create_consumer()
    c1_consumer = c1_chnl.create_consumer()

    qk_shape: gl.constexpr = (SPLIT_M, BLOCK_N)
    mma_instr: gl.constexpr = get_mma_instr_shape(qk_shape, dtype)
    s_tmem_layout: gl.constexpr = TensorMemoryLayout(
        [mma_instr[0], BLOCK_N // 2 + 3], col_stride=1
    )
    o_tmem_layout: gl.constexpr = TensorMemoryLayout(
        [mma_instr[0], HEAD_DIM // 2], col_stride=1
    )

    for pid in range(scheduler.start_pid, scheduler.num_tiles, NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_loop_bounds(BLOCK_N, N_CTX, IS_CAUSAL)
        num_steps = (hi - lo) // BLOCK_N

        # Allocate TMEM for S (scores) and O (output) for both subtiles
        s0_tmem = allocate_tensor_memory(dtype, [SPLIT_M, BLOCK_N // 2 + 3], s_tmem_layout)
        s1_tmem = allocate_tensor_memory(dtype, [SPLIT_M, BLOCK_N // 2 + 3], s_tmem_layout)
        o0_tmem = allocate_tensor_memory(dtype, [SPLIT_M, HEAD_DIM // 2], o_tmem_layout)
        o1_tmem = allocate_tensor_memory(dtype, [SPLIT_M, HEAD_DIM // 2], o_tmem_layout)

        # Load Q tiles
        q0_smem, q0_bar, q_consumer = q_consumer.acquire()
        mbarrier.arrive(q0_bar, count=1)
        q1_smem, q1_bar, q_consumer = q_consumer.acquire()
        mbarrier.arrive(q1_bar, count=1)

        for step in range(num_steps):
            n_block = lo // BLOCK_N + step

            # Get K tile
            k_smem, k_bar, kv_consumer = kv_consumer.acquire()

            # Q0 * K^T -> S0
            s0_smem, s0_bar, s0_producer = s0_producer.acquire()
            from triton.experimental.gluon.language.nvidia.blackwell import fence_async_shared
            fence_async_shared()
            tcgen05_mma(q0_smem, k_smem, s0_tmem, transpose_b=True)
            tcgen05_commit(s0_bar)

            # Q1 * K^T -> S1
            s1_smem, s1_bar, s1_producer = s1_producer.acquire()
            tcgen05_mma(q1_smem, k_smem, s1_tmem, transpose_b=True)
            tcgen05_commit(s1_bar)

            mbarrier.arrive(k_bar, count=1)

            # Wait for softmax to produce P, then P * V -> O
            v_smem, v_bar, kv_consumer = kv_consumer.acquire()

            # Wait for softmax0 correction
            _, c0_bar, c0_consumer = c0_consumer.acquire()
            mbarrier.arrive(c0_bar, count=1)

            # P0 * V -> O0
            fence_async_shared()
            tcgen05_mma(s0_tmem, v_smem, o0_tmem)
            tcgen05_commit(s0_bar)

            # Wait for softmax1 correction
            _, c1_bar, c1_consumer = c1_consumer.acquire()
            mbarrier.arrive(c1_bar, count=1)

            # P1 * V -> O1
            tcgen05_mma(s1_tmem, v_smem, o1_tmem)
            tcgen05_commit(s1_bar)

            mbarrier.arrive(v_bar, count=1)

    q_chnl.release()
    kv_chnl.release()
    s0_chnl.release()
    s1_chnl.release()
    c0_chnl.release()
    c1_chnl.release()


# ===-----------------------------------------------------------------------===#
# Partition: Softmax (online softmax on SPLIT_M rows)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_softmax_tile(
    tile_id, scheduler, s_chnl, corr_chnl,
    N_CTX, sm_scale,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SPLIT_M: gl.constexpr,
    NUM_SMS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    qk_layout: gl.constexpr,
    qk_slice_dim1: gl.constexpr,
):
    s_consumer = s_chnl.create_consumer()
    corr_producer = corr_chnl.create_producer()
    _, corr_bar, corr_producer = corr_producer.acquire()

    for pid in range(scheduler.start_pid, scheduler.num_tiles, NUM_SMS):
        prog = scheduler.get_program(pid)
        lo, hi = prog.get_loop_bounds(BLOCK_N, N_CTX, IS_CAUSAL)
        num_steps = (hi - lo) // BLOCK_N

        offs_m = prog.start_m * BLOCK_M + gl.arange(
            tile_id * SPLIT_M, (1 + tile_id) * SPLIT_M
        )

        m_i = gl.full([SPLIT_M], float("-inf"), gl.float32, qk_slice_dim1)
        l_i = gl.full([SPLIT_M], 0.0, gl.float32, qk_slice_dim1)

        for step in range(num_steps):
            n_block = lo // BLOCK_N + step

            s_tmem, s_bar, s_consumer = s_consumer.acquire()

            m_i, l_i, corr_producer, corr_bar = _fwd_inner_dense_softmax(
                s_tmem=s_tmem,
                s_bar=s_bar,
                corr_producer=corr_producer,
                corr_bar=corr_bar,
                offs_m=offs_m,
                m_i=m_i,
                l_i=l_i,
                n_block=n_block,
                N_CTX=N_CTX,
                sm_scale=sm_scale,
                SPLIT_M=SPLIT_M,
                BLOCK_N=BLOCK_N,
                IS_CAUSAL=IS_CAUSAL,
                qk_layout=qk_layout,
                qk_slice_dim1=qk_slice_dim1,
            )

        # Store final m_i and l_i to TMEM for epilogue rescale
        s_tmem, s_bar, s_consumer = s_consumer.acquire()
        m_i_tmem, l_i_tmem = _borrow_s_for_epilogue(SPLIT_M, s_tmem, BLOCK_N)
        alpha_2d_layout: gl.constexpr = TensorMemoryLayout([SPLIT_M, 1], col_stride=1)
        m_i_tmem.store(gl.convert_layout(m_i.expand_dims(1), alpha_2d_layout))
        l_i_tmem.store(gl.convert_layout(l_i.expand_dims(1), alpha_2d_layout))

        mbarrier.arrive(corr_bar, count=1)
        _, corr_bar, corr_producer = corr_producer.acquire()
        mbarrier.arrive(s_bar, count=1)

    s_chnl.release()
    corr_chnl.release()


@gluon.jit
def _attn_fwd_softmax0(scheduler, chnls, N_CTX, sm_scale,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                        SPLIT_M: gl.constexpr, NUM_SMS: gl.constexpr,
                        IS_CAUSAL: gl.constexpr,
                        qk_layout: gl.constexpr, qk_slice_dim1: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl = chnls
    _attn_fwd_softmax_tile(0, scheduler, s0_chnl, c0_chnl, N_CTX, sm_scale,
                           BLOCK_M, BLOCK_N, SPLIT_M, NUM_SMS, IS_CAUSAL,
                           qk_layout, qk_slice_dim1)


@gluon.jit
def _attn_fwd_softmax1(scheduler, chnls, N_CTX, sm_scale,
                        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                        SPLIT_M: gl.constexpr, NUM_SMS: gl.constexpr,
                        IS_CAUSAL: gl.constexpr,
                        qk_layout: gl.constexpr, qk_slice_dim1: gl.constexpr):
    q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl = chnls
    _attn_fwd_softmax_tile(1, scheduler, s1_chnl, c1_chnl, N_CTX, sm_scale,
                           BLOCK_M, BLOCK_N, SPLIT_M, NUM_SMS, IS_CAUSAL,
                           qk_layout, qk_slice_dim1)


# ===-----------------------------------------------------------------------===#
# Partition: Epilogue (rescale O, TMA store)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _attn_fwd_epilogue(
    scheduler, epi_chnl, desc_o,
    SPLIT_M: gl.constexpr,
    NUM_SMS: gl.constexpr,
):
    epi_consumer = epi_chnl.create_consumer()

    for pid in range(scheduler.start_pid, scheduler.num_tiles, NUM_SMS):
        prog = scheduler.get_program(pid)

        # Store O tile 0
        o0_smem, o0_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + SPLIT_M * 0, 0], o0_smem)

        # Store O tile 1
        o1_smem, o1_bar, epi_consumer = epi_consumer.acquire()
        tma.async_copy_shared_to_global(desc_o, [prog.qo_offset_y + SPLIT_M * 1, 0], o1_smem)

        tma.store_wait(1)
        mbarrier.arrive(o0_bar, count=1)
        tma.store_wait(0)
        mbarrier.arrive(o1_bar, count=1)

    epi_chnl.release()


# ===-----------------------------------------------------------------------===#
# Main Kernel (warp-specialized)
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=["Z", "H", "N_CTX"])
def _fwd_dense_base_kernel(
    sm_scale, Lse,
    Z, H, N_CTX,
    desc_q, desc_k, desc_v, desc_o,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    GROUP_SIZE_N: gl.constexpr,
    NUM_SMS: gl.constexpr,
    SPLIT_M: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    dtype: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
):
    num_pid_m = triton.cdiv(N_CTX, BLOCK_M)
    num_pid_n = Z * H

    scheduler = ProgramScheduler.create(
        BLOCK_M, SPLIT_M, GROUP_SIZE_N, num_pid_m, num_pid_n, NUM_SMS
    )

    # Allocate channels
    q_chnl = get_desc_channel(desc_q, num_buffers=2)
    kv_chnl = get_desc_channel(desc_k, num_buffers=NUM_KV_BUFFERS)
    epi_chnl = SharedMemoryChannel.alloc(
        desc_o.block_type.shape, dtype, gl.constexpr(desc_o.layout), num_buffers=2
    )

    qk_shape: gl.constexpr = (SPLIT_M, BLOCK_N)
    mma_instr: gl.constexpr = get_mma_instr_shape(qk_shape, dtype)
    s_tmem_layout: gl.constexpr = TensorMemoryLayout(
        [mma_instr[0], BLOCK_N // 2 + 3], col_stride=1
    )
    o_tmem_layout: gl.constexpr = TensorMemoryLayout(
        [mma_instr[0], HEAD_DIM // 2], col_stride=1
    )
    s0_chnl = TensorMemoryChannel.alloc(
        [SPLIT_M, BLOCK_N // 2 + 3], gl.float32, s_tmem_layout, num_buffers=1
    )
    s1_chnl = TensorMemoryChannel.alloc(
        [SPLIT_M, BLOCK_N // 2 + 3], gl.float32, s_tmem_layout, num_buffers=1
    )
    o_chnl = TensorMemoryChannel.alloc(
        [SPLIT_M, HEAD_DIM // 2], gl.float32, o_tmem_layout, num_buffers=2
    )
    c0_chnl = SharedMemoryChannel.alloc(
        [1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1
    )
    c1_chnl = SharedMemoryChannel.alloc(
        [1], gl.int8, gl.constexpr(mbarrier.MBarrierLayout()), num_buffers=1
    )

    chnls = (q_chnl, kv_chnl, o_chnl, epi_chnl, s0_chnl, s1_chnl, c0_chnl, c1_chnl)

    qk_layout: gl.constexpr = TensorMemoryLayout(
        [mma_instr[0], BLOCK_N // 2], col_stride=1
    )
    qk_slice_dim1: gl.constexpr = gl.SliceLayout(1, qk_layout)

    # Warp-specialized dispatch:
    #   partition 0: MMA (4 warps, high register count)
    #   partition 1: softmax0 (1 warp)
    #   partition 2: softmax1 (1 warp)
    #   partition 3: load (1 warp, low register count)
    #   partition 4: epilogue (1 warp, low register count)
    gl.warp_specialize([
        (_attn_fwd_mma, (
            scheduler, q_chnl, kv_chnl, s0_chnl, s1_chnl, o_chnl, c0_chnl, c1_chnl,
            desc_o, N_CTX,
            BLOCK_M, BLOCK_N, HEAD_DIM, SPLIT_M, NUM_SMS, IS_CAUSAL, dtype,
        )),
        (_attn_fwd_softmax0, (
            scheduler, chnls, N_CTX, sm_scale,
            BLOCK_M, BLOCK_N, SPLIT_M, NUM_SMS, IS_CAUSAL, qk_layout, qk_slice_dim1,
        )),
        (_attn_fwd_softmax1, (
            scheduler, chnls, N_CTX, sm_scale,
            BLOCK_M, BLOCK_N, SPLIT_M, NUM_SMS, IS_CAUSAL, qk_layout, qk_slice_dim1,
        )),
        (_attn_fwd_load, (
            scheduler, q_chnl, kv_chnl,
            desc_q, desc_k, desc_v,
            N_CTX, BLOCK_N, SPLIT_M, NUM_SMS, IS_CAUSAL, NUM_KV_BUFFERS,
        )),
        (_attn_fwd_epilogue, (
            scheduler, epi_chnl, desc_o, SPLIT_M, NUM_SMS,
        )),
    ], [4, 1, 1, 1], [128, 64, 64, 24, 24])

    q_chnl.release()
    kv_chnl.release()
    o_chnl.release()
    epi_chnl.release()
    s0_chnl.release()
    s1_chnl.release()
    c0_chnl.release()
    c1_chnl.release()


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
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    device = query.device
    arch = cache_utils.get_device_arch(device)
    num_SMs = cache_utils.get_device_num_sms(device)
    batch_size, seqlen_q, num_heads_q, head_dim = query.shape
    _, seqlen_k, num_heads_kv, _ = key.shape
    window_size_left, window_size_right = window_size
    is_local = window_size_left is not None or window_size_right is not None
    softmax_scale = softmax_scale or 1.0 / (head_dim ** 0.5)
    qheads_per_kvhead = num_heads_q // num_heads_kv

    if not skip_checks:
        assert_inputs.assert_fwd_inputs(
            query, key, value,
            cu_seqlens_q=None, cu_seqlens_k=None,
            seqused_q=None, seqused_k=None,
            num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
            head_dim=head_dim, device=device, arch=arch,
        )

    TILE_K = max(triton.next_power_of_2(head_dim), 16)

    TILE_M, TILE_N, num_warps, num_stages, num_ctas = (
        launch_template.get_fwd_dense_launch_config(
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            qheads_per_kvhead=qheads_per_kvhead,
            tile_k=TILE_K,
            device=device,
            arch=arch,
        )
    )

    BLOCK_M = TILE_M
    BLOCK_N = TILE_N
    SPLIT_M = BLOCK_M // 2
    HEAD_DIM = head_dim
    GROUP_SIZE_N = 4
    NUM_KV_BUFFERS = 4

    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, num_heads_q, seqlen_q),
        dtype=torch.float32,
        device=query.device,
    )

    # Flatten to 2D for TMA: [batch * heads * seqlen, head_dim]
    y_dim = batch_size * num_heads_q * seqlen_q

    desc_q = utils.make_tensor_desc(
        query, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[SPLIT_M, HEAD_DIM],
    )
    desc_k = utils.make_tensor_desc(
        key, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_v = utils.make_tensor_desc(
        value, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = utils.make_tensor_desc(
        out, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[SPLIT_M, HEAD_DIM],
    )

    num_pid_m = triton.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = batch_size * num_heads_q
    NUM_SMS = num_SMs
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    dtype_gl = utils.torch_dtype_to_gluon(query.dtype)

    _fwd_dense_base_kernel[(grid,)](
        softmax_scale, lse,
        batch_size, num_heads_q, seqlen_q,
        desc_q, desc_k, desc_v, desc_o,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        GROUP_SIZE_N=GROUP_SIZE_N,
        NUM_SMS=NUM_SMS,
        SPLIT_M=SPLIT_M,
        IS_CAUSAL=is_causal,
        dtype=dtype_gl,
        NUM_KV_BUFFERS=NUM_KV_BUFFERS,
        num_warps=num_warps,
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
    # TODO: SM100 varlen support — for now fall back to non-varlen path
    raise NotImplementedError(
        "SM100 Gluon varlen forward is not yet implemented. "
        "Use the Triton backend for varlen attention."
    )
