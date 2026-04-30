"""
SM100 (Blackwell) attention tile scheduling primitives.

Provides AttentionConfig (tile shapes, layouts, constexprs),
AttentionProgram (per-tile offset computation), and
ProgramScheduler (persistent tile scheduler with swizzle).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tma,
    mbarrier,
)

from flash_sparse_attn.ops.gluon.utils import get_mma_instr_shape
from flash_sparse_attn.ops.gluon import block_info


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
        def alloc(
            shape: gl.constexpr,
            dtype: gl.constexpr,
            layout: gl.constexpr,
            num_buffers: gl.constexpr,
            num_consumers: gl.constexpr = 1,
        ):
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
    return SharedMemoryChannel.alloc(
        shape, desc.dtype, layout, num_buffers, num_consumers
    )


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [offset, 0], bar, smem)


# ===-----------------------------------------------------------------------===#
# AttentionConfig
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    softmax_scale_log2: gl.tensor
    batch_size: gl.tensor
    num_heads: gl.tensor
    seqlen_q: gl.tensor
    seqlen_k: gl.tensor

    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    TILE_K: gl.constexpr
    GROUP_SIZE_N: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    IS_CAUSAL: gl.constexpr
    IS_LOCAL: gl.constexpr
    WINDOW_SIZE_LEFT: gl.constexpr
    WINDOW_SIZE_RIGHT: gl.constexpr

    SPLIT_D_FACTOR: gl.constexpr
    SPLIT_EXP_FACTOR: gl.constexpr
    SPLIT_QK_LOAD_FACTOR: gl.constexpr
    SPLIT_M: gl.constexpr
    SPLIT_D: gl.constexpr

    q_shape: gl.constexpr
    k_shape: gl.constexpr
    v_shape: gl.constexpr
    qk_shape: gl.constexpr
    o_shape: gl.constexpr

    qk_tmem_layout: gl.constexpr
    o_tmem_layout: gl.constexpr
    p_tmem_layout: gl.constexpr

    qk_layout: gl.constexpr
    o_splitn_layout: gl.constexpr
    row_scale_2d_layout: gl.constexpr

    num_kv_buffers: gl.constexpr
    use_exp2_turnstile: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
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
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k

        self.TILE_M = gl.constexpr(TILE_M)
        self.TILE_N = gl.constexpr(TILE_N)
        self.TILE_K = gl.constexpr(TILE_K)
        self.GROUP_SIZE_N = gl.constexpr(GROUP_SIZE_N)
        self.NUM_SMS = gl.constexpr(NUM_SMS)
        self.dtype = gl.constexpr(dtype)
        self.num_warps = gl.constexpr(num_warps)

        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.IS_LOCAL = gl.constexpr(IS_LOCAL)
        self.WINDOW_SIZE_LEFT = gl.constexpr(WINDOW_SIZE_LEFT)
        self.WINDOW_SIZE_RIGHT = gl.constexpr(WINDOW_SIZE_RIGHT)

        self.SPLIT_D_FACTOR = gl.constexpr(2)
        self.SPLIT_EXP_FACTOR = gl.constexpr(SPLIT_EXP_FACTOR)
        self.SPLIT_QK_LOAD_FACTOR = gl.constexpr(
            2 if (not IS_CAUSAL and not IS_LOCAL) else 1
        )
        self.SPLIT_M = gl.constexpr(self.TILE_M // 2)
        self.SPLIT_D = gl.constexpr(self.TILE_K // self.SPLIT_D_FACTOR)

        self.q_shape = gl.constexpr([self.SPLIT_M, self.TILE_K])
        self.k_shape = gl.constexpr([self.TILE_N, self.TILE_K])
        self.qk_shape = gl.constexpr([self.SPLIT_M, self.TILE_N])
        self.v_shape = gl.constexpr([self.TILE_N, self.TILE_K])
        self.o_shape = gl.constexpr([self.SPLIT_M, self.TILE_K])

        qk_instr_shape = get_mma_instr_shape(self.qk_shape, gl.float32)
        o_instr_shape = get_mma_instr_shape(self.o_shape, gl.float32)
        self.qk_tmem_layout = gl.constexpr(
            TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1)
        )
        self.o_tmem_layout = gl.constexpr(
            TensorMemoryLayout((o_instr_shape[0], o_instr_shape[1]), col_stride=1)
        )
        self.p_tmem_layout = gl.constexpr(
            TensorMemoryLayout((qk_instr_shape[0], qk_instr_shape[1]), col_stride=1)
        )
        o_splitn_tmem_layout: gl.constexpr = TensorMemoryLayout(
            (o_instr_shape[0], o_instr_shape[1] // self.SPLIT_D_FACTOR), col_stride=1
        )
        qk_tmem_ty: gl.constexpr = tensor_memory_descriptor_type(
            gl.float32, self.qk_shape, self.qk_tmem_layout, self.qk_shape
        )
        o_splitn_tmem_ty: gl.constexpr = tensor_memory_descriptor_type(
            gl.float32,
            [self.o_shape[0], self.o_shape[1] // self.SPLIT_D_FACTOR],
            o_splitn_tmem_layout,
            self.o_shape,
        )

        self.qk_layout = gl.constexpr(
            qk_tmem_ty.get_reg_layout(
                num_warps=self.num_warps, instr_variant="32x32b_splitn"
            )
        )
        self.o_splitn_layout = gl.constexpr(
            o_splitn_tmem_ty.get_reg_layout(num_warps=self.num_warps)
        )
        self.row_scale_2d_layout = gl.constexpr(
            gl.BlockedLayout([1, 1], [32, 1], [self.num_warps, 1], [0, 1])
        )

        self.num_kv_buffers = gl.constexpr(NUM_KV_BUFFERS)
        self.use_exp2_turnstile = gl.constexpr(USE_EXP2_TURNSTILE)

    @gluon.jit
    def get_program(self, pid_m, pid_n):
        m_block = pid_m
        head_idx = pid_n
        batch_idx = head_idx // self.num_heads
        head_in_batch = head_idx % self.num_heads
        kv_offset = (
            batch_idx * (self.seqlen_k * self.num_heads) + head_in_batch * self.seqlen_k
        )
        qo_offset = (
            batch_idx * (self.seqlen_q * self.num_heads)
            + head_in_batch * self.seqlen_q
            + m_block * self.TILE_M
        )
        return AttentionProgram(self, m_block, head_idx, kv_offset, qo_offset)


# ===-----------------------------------------------------------------------===#
# AttentionProgram
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    config: AttentionConfig
    m_block: gl.tensor
    head_idx: gl.tensor
    kv_offset: gl.tensor
    qo_offset: gl.tensor

    @gluon.jit
    def get_n_block_min_max(self):
        return block_info.get_n_block_min_max(
            seqlen_q=self.config.seqlen_q,
            seqlen_k=self.config.seqlen_k,
            m_block=self.m_block,
            TILE_N=self.config.TILE_N,
            TILE_M=self.config.TILE_M,
            IS_CAUSAL=self.config.IS_CAUSAL,
            IS_LOCAL=self.config.IS_LOCAL,
            WINDOW_SIZE_LEFT=self.config.WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=self.config.WINDOW_SIZE_RIGHT,
        )

    @gluon.jit
    def get_n_block_min_causal_local_mask(self, n_block_min):
        return block_info.get_n_block_min_causal_local_mask(
            seqlen_q=self.config.seqlen_q,
            seqlen_k=self.config.seqlen_k,
            m_block=self.m_block,
            n_block_min=n_block_min,
            TILE_N=self.config.TILE_N,
            TILE_M=self.config.TILE_M,
            IS_LOCAL=self.config.IS_LOCAL,
            WINDOW_SIZE_RIGHT=self.config.WINDOW_SIZE_RIGHT,
        )

    @gluon.jit
    def get_n_block_min_before_local_mask(self, n_block_min):
        return block_info.get_n_block_min_before_local_mask(
            seqlen_q=self.config.seqlen_q,
            seqlen_k=self.config.seqlen_k,
            m_block=self.m_block,
            n_block_min=n_block_min,
            TILE_N=self.config.TILE_N,
            TILE_M=self.config.TILE_M,
            IS_LOCAL=self.config.IS_LOCAL,
            WINDOW_SIZE_LEFT=self.config.WINDOW_SIZE_LEFT,
        )

    @gluon.jit
    def get_loop_bounds(self):
        n_block_min, n_block_max = self.get_n_block_min_max()
        return n_block_min * self.config.TILE_N, n_block_max * self.config.TILE_N

    @gluon.jit
    def get_seg_bounds(self):
        TILE_N: gl.constexpr = self.config.TILE_N
        n_block_min, n_block_max = self.get_n_block_min_max()
        n_block_max_no_mask = self.get_n_block_min_causal_local_mask(n_block_min)
        n_block_min_no_mask = self.get_n_block_min_before_local_mask(n_block_min)
        n_block_min_no_mask = gl.minimum(n_block_min_no_mask, n_block_max_no_mask)

        return (
            n_block_min * TILE_N,
            n_block_min_no_mask * TILE_N,
            n_block_max_no_mask * TILE_N,
            n_block_max * TILE_N,
        )


# ===-----------------------------------------------------------------------===#
# Persistent Tile Scheduler
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ProgramScheduler:
    config: AttentionConfig
    start_pid: gl.tensor
    num_pid_n: gl.tensor
    num_pid_in_group: gl.tensor
    num_tiles: gl.tensor

    @gluon.jit
    def create(config):
        start_pid = gl.program_id(0)
        num_pid_m = gl.cdiv(config.seqlen_q, config.TILE_M)
        num_pid_n = config.batch_size * config.num_heads
        num_pid_in_group = num_pid_m * config.GROUP_SIZE_N
        num_tiles = num_pid_m * num_pid_n
        return ProgramScheduler(
            config, start_pid, num_pid_n, num_pid_in_group, num_tiles
        )

    @gluon.jit
    def get_program(self, tile_id):
        group_id = tile_id // self.num_pid_in_group
        first_pid_n = group_id * self.config.GROUP_SIZE_N
        group_size_n = min(self.num_pid_n - first_pid_n, self.config.GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % self.num_pid_in_group) // group_size_n
        return self.config.get_program(pid_m, pid_n)
