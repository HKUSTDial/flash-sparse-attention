import copy
import functools
import math
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
)


def get_device():
    """
    Get the appropriate device for computation.

    :return device: torch.device object
    """
    # TODO: add NPU
    # Works for both NVIDIA and AMD
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Intel XPU if available
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ensure_contiguous(fn):
    """
    Decorator to ensure that all tensor inputs to the decorated function are contiguous.

    :param fn: Function to be decorated

    :return wrapper: Wrapped function
    """

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


@functools.lru_cache(maxsize=4096)
def num_splits_heuristic(
    seqlen_q: int,
    seqlen_k: int,
    num_SMs: int,
    TILE_M: int,
    TILE_N: int,
) -> int:
    """
    Determine the number of KV splits for FlashDecoding.

    Splits only when there are enough KV blocks to benefit from parallelism,
    and targets full SM occupancy by over-subscribing the M-block count.

    :param seqlen_q: Sequence length of queries.
    :param seqlen_k: Sequence length of keys.
    :param num_SMs: Number of streaming multiprocessors on the device.
    :param TILE_M: Tile size for M dimension.
    :param TILE_N: Tile size for N dimension.

    :return: Number of splits.
    """
    total_mblocks = triton.cdiv(seqlen_q, TILE_M)
    num_n_blocks = triton.cdiv(seqlen_k, TILE_N)
    max_splits = 1 << (max(num_SMs, 1).bit_length() - 1)
    if num_n_blocks <= 4:
        # 1 means no splitting
        return 1
    return min(num_SMs // max(total_mblocks, 1), max_splits, num_n_blocks)


def is_sm100():
    """
    Check if the current CUDA device supports SM100 (Blackwell) or later.

    :return: True if device compute capability >= 10.0, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


_TORCH_TO_GLUON_DTYPE = {
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
    torch.float8_e5m2: gl.float8e5,
    torch.float8_e4m3fn: gl.float8e4nv,
}


def torch_dtype_to_gluon(dtype):
    """
    Convert a PyTorch dtype to the corresponding Gluon dtype.

    :param dtype: PyTorch dtype (e.g. torch.float16, torch.bfloat16).

    :return: Corresponding Gluon dtype.
    """
    return _TORCH_TO_GLUON_DTYPE[dtype]


def make_tensor_desc(tensor, shape, strides, block_shape):
    """
    Create a TMA tensor descriptor with the default NVMMA shared layout.

    :param tensor: Source PyTorch tensor.
    :param shape: Logical shape of the descriptor.
    :param strides: Strides of the descriptor.
    :param block_shape: Block shape for the TMA transfer.

    :return: TensorDescriptor configured for TMA loads/stores.
    """
    layout = gl.NVMMASharedLayout.get_default_for(
        block_shape, torch_dtype_to_gluon(tensor.dtype)
    )
    return TensorDescriptor(
        tensor, shape=shape, strides=strides, block_shape=block_shape, layout=layout
    )


@gluon.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    """
    Compute the MMA instruction shape for a given tile shape and element type.

    :param shape: Tile shape as (M, N).
    :param element_ty: Gluon element type (e.g. gl.float32).

    :return: Tuple (m, n, k) of MMA instruction dimensions.
    """
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@gluon.jit
def borrow_s_as_p(config, s_tmem):
    """
    Reinterpret the lower half of the S accumulator TMEM as the P (exp2 result) buffer.

    :param config: AttentionConfig with dtype, qk_shape, p_tmem_layout.
    :param s_tmem: S accumulator tensor memory descriptor.

    :return: TMEM descriptor reinterpreted for P storage.
    """
    p_tmem = s_tmem.slice(0, config.TILE_N // 2)
    return p_tmem._reinterpret(config.dtype, config.qk_shape, config.p_tmem_layout)


@gluon.jit
def borrow_s_as_row_scale(config, s_tmem):
    """
    Reinterpret one column of the S accumulator TMEM as the row_scale (rescale factor) buffer.

    :param config: AttentionConfig with TILE_N, SPLIT_M.
    :param s_tmem: S accumulator tensor memory descriptor.

    :return: TMEM descriptor of shape [SPLIT_M, 1] for the row_scale column.
    """
    row_scale_tmem = s_tmem.slice(config.TILE_N // 2, 1)
    row_scale_layout: gl.constexpr = TensorMemoryLayout(
        [config.SPLIT_M, 1], col_stride=1
    )
    return row_scale_tmem._reinterpret(
        gl.float32, [config.SPLIT_M, 1], row_scale_layout
    )


@gluon.jit
def borrow_s_for_finalize(config, s_tmem):
    """
    Reinterpret two columns of the S accumulator TMEM as row_max and row_sum buffers for the finalize.

    :param config: AttentionConfig with TILE_N, SPLIT_M.
    :param s_tmem: S accumulator tensor memory descriptor.

    :return: Tuple (row_max_tmem, row_sum_tmem), each of shape [SPLIT_M, 1].
    """
    row_max_tmem = s_tmem.slice(config.TILE_N // 2 + 1, 1)
    row_sum_tmem = s_tmem.slice(config.TILE_N // 2 + 2, 1)
    layout: gl.constexpr = TensorMemoryLayout([config.SPLIT_M, 1], col_stride=1)
    row_max_tmem = row_max_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    row_sum_tmem = row_sum_tmem._reinterpret(gl.float32, [config.SPLIT_M, 1], layout)
    return row_max_tmem, row_sum_tmem


@gluon.constexpr_function
def get_split_n_layout(layout: gl.constexpr, SPLIT_FACTOR: gl.constexpr = 2):
    """
    Compute a DistributedLinearLayout suitable for splitting a tensor along the N dimension.

    Swaps the last register basis with the half-N basis so that a subsequent
    reshape+split cleanly partitions the N dimension.

    :param layout: Source DistributedLinearLayout.
    :param SPLIT_FACTOR: Number of splits (1 or 2).

    :return: Adjusted DistributedLinearLayout with the half-N basis in the last register position.
    """
    assert isinstance(layout, gl.DistributedLinearLayout), (
        "split_n requires a distributed layout"
    )
    assert SPLIT_FACTOR == 1 or SPLIT_FACTOR == 2, (
        "split_n requires a split factor of 1 or 2"
    )
    if SPLIT_FACTOR == 1:
        return layout
    else:
        target = [0, layout.shape[1] // 2]
        last_reg_idx = len(layout.reg_bases) - 1
        reg_last = layout.reg_bases[last_reg_idx]

        if reg_last == target:
            return layout

        ret = copy.deepcopy(layout)
        for L in (ret.reg_bases, ret.lane_bases, ret.warp_bases, ret.block_bases):
            for i, b in enumerate(L):
                if b == target:
                    L[i], ret.reg_bases[last_reg_idx] = reg_last, target
                    return ret
        assert False, f"split_n requires having a basis {target}. Got\n{layout}"


@gluon.jit
def split_n(x, SPLIT_FACTOR: gl.constexpr = 2):
    """
    Recursively split a 2D tensor along the N (column) dimension.

    :param x: Input tensor of shape [M, N].
    :param SPLIT_FACTOR: Number of equal splits (must be a power of 2).

    :return: Tuple of SPLIT_FACTOR tensors, each of shape [M, N // SPLIT_FACTOR].
    """
    if SPLIT_FACTOR == 1:
        return (x,)
    else:
        layout: gl.constexpr = get_split_n_layout(x.type.layout)
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        x0 = gl.convert_layout(x0, layout, assert_trivial=True)
        x1 = gl.convert_layout(x1, layout, assert_trivial=True)
        return split_n(x0, SPLIT_FACTOR // 2) + split_n(x1, SPLIT_FACTOR // 2)


@gluon.constexpr_function
def get_join_n_layout(layout, SPLIT_FACTOR: gl.constexpr = 2):
    """
    Compute a DistributedLinearLayout for joining split tensors back along the N dimension.

    Extends the register bases to cover the wider joined shape.

    :param layout: Source DistributedLinearLayout of one split piece.
    :param SPLIT_FACTOR: Number of pieces being joined (must be a power of 2).

    :return: DistributedLinearLayout with shape scaled by SPLIT_FACTOR along N.
    """
    assert isinstance(layout, gl.DistributedLinearLayout), (
        "join_n requires a Linear layout"
    )
    shape = list(layout.shape)
    regs = [[0, shape[1] * (1 << i)] for i in range(int(math.log2(SPLIT_FACTOR)))]
    shape[1] *= SPLIT_FACTOR
    return gl.DistributedLinearLayout(
        layout.reg_bases + regs,
        layout.lane_bases,
        layout.warp_bases,
        layout.block_bases,
        shape,
    )


@gluon.jit
def join_n(xs):
    """
    Recursively join a tuple of split tensors back along the N (column) dimension.

    Inverse of split_n: join_n(split_n(x, k)) == x.

    :param xs: Tuple of tensors with identical shapes [M, N_piece].

    :return: Single tensor of shape [M, N_piece * len(xs)].
    """
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = join_n(xs[: len(xs) // 2])
        x1 = join_n(xs[len(xs) // 2 :])
        layout: gl.constexpr = get_join_n_layout(x0.type.layout)
        x = gl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.jit
def compute_and_store_exp2(config, acc_s, p_tmem):
    SIZE: gl.constexpr = p_tmem.shape[1] // config.SPLIT_EXP_FACTOR
    acc_s_splits = split_n(acc_s, config.SPLIT_EXP_FACTOR)
    ps = ()
    for i in gl.static_range(config.SPLIT_EXP_FACTOR):
        p = gl.exp2(acc_s_splits[i])
        p_tmem.slice(i * SIZE, SIZE).store(p.to(config.dtype))
        ps = ps + (p,)
    return join_n(ps)


@gluon.jit
def subtiled_qk_load(config, s_tmem, use_tmem_red: gl.constexpr):
    SIZE: gl.constexpr = s_tmem.shape[1] // config.SPLIT_QK_LOAD_FACTOR
    acc_s_splits = ()
    if use_tmem_red:
        red_total = None
        for i in gl.static_range(config.SPLIT_QK_LOAD_FACTOR):
            vals, reds = s_tmem.slice(i * SIZE, SIZE).load_max()
            red_total = reds if red_total is None else gl.maximum(red_total, reds)
            acc_s_splits = acc_s_splits + (vals,)
        return join_n(acc_s_splits), red_total
    else:
        for i in gl.static_range(config.SPLIT_QK_LOAD_FACTOR):
            acc_s_splits = acc_s_splits + (s_tmem.slice(i * SIZE, SIZE).load(),)
        return join_n(acc_s_splits), None
