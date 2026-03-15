from typing import Optional

import torch

from flash_sparse_attn.ops.triton import utils


def assert_fwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
):
    """
    Assert the validity of inputs for the forward kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension

    :raises AssertionError: If any of the assertions fail
    """
    device = query.device
    arch = utils.get_arch(device)
    assert query.is_cuda and key.is_cuda and value.is_cuda, (
        "All inputs must be on CUDA device"
    )
    if arch // 10 >= 9:  # Hopper or newer
        assert query.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2], (
            "Input dtype must be float16, bfloat16, or float8_e5m2"
        )
    else:
        assert query.dtype in [torch.float16, torch.bfloat16], (
            "Input dtype must be float16 or bfloat16"
        )
    assert query.dtype == key.dtype == value.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )


def assert_bwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
):
    """
    Assert the validity of inputs for the backward base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param out: Output tensor
    :param dout: Gradient of the output tensor
    :param lse: Log-sum-exp tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension

    :raises AssertionError: If any of the assertions fail
    """
    device = query.device
    arch = utils.get_arch(device)
    assert (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and out.is_cuda
        and dout.is_cuda
        and lse.is_cuda
    ), "All inputs must be on CUDA device"
    if arch // 10 >= 9:  # Hopper or newer
        assert query.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2], (
            "Input dtype must be float16, bfloat16, or float8_e5m2"
        )
    else:
        assert query.dtype in [torch.float16, torch.bfloat16], (
            "Input dtype must be float16 or bfloat16"
        )
    assert query.dtype == key.dtype == value.dtype == out.dtype == dout.dtype, (
        "All inputs must have the same dtype"
    )
    assert lse.dtype == torch.float32, (
        "lse must be float32 for numerical stability in backward pass"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )
    if seqused_q is not None and seqused_k is not None:
        assert seqused_q.is_cuda and seqused_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert seqused_q.dtype == seqused_k.dtype == torch.int32, (
            "seqused_q and seqused_k must be int32"
        )


def assert_fwd_sparse_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    gate_threshold: float = None,
):
    """
    Assert the validity of inputs for the forward sparse base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param alpha: Alpha tensor
    :param delta: Delta tensor
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param gate_threshold: Gate threshold for sparse attention

    :raises AssertionError: If any of the assertions fail
    """
    device = query.device
    arch = utils.get_arch(device)
    assert (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and alpha.is_cuda
        and delta.is_cuda
    ), "All inputs must be on CUDA device"
    if arch // 10 >= 9:  # Hopper or newer
        assert query.dtype in [torch.float16, torch.bfloat16, torch.float8_e5m2], (
            "Input dtype must be float16, bfloat16, or float8_e5m2"
        )
    else:
        assert query.dtype in [torch.float16, torch.bfloat16], (
            "Input dtype must be float16 or bfloat16"
        )
    assert query.dtype == key.dtype == value.dtype == alpha.dtype == delta.dtype, (
        "All inputs must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim % 16 == 0, (
        "head_dim must be a multiple of 16 for efficient memory access"
    )
    assert head_dim <= 256, (
        "head_dim must be less than or equal to 256 for efficient memory access"
    )
    assert 0.0 <= gate_threshold <= 1.0, (
        "gate_threshold must be in the range [0.0, 1.0] for meaningful gating behavior"
    )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, (
            "All inputs must be on CUDA device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlen_q and cu_seqlen_k must be int32"
        )
