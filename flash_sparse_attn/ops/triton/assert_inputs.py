from typing import Optional

import torch


def assert_fwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    device: torch.device = None,
    arch: int = None,
):
    """
    Assert the validity of inputs for the forward kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param alpha: Alpha tensor for gated attention
    :param delta: Delta tensor for gated attention
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param device: Device of the tensors
    :param arch: Architecture model as a number

    :raises AssertionError: If any of the assertions fail
    """
    assert device == query.device == key.device == value.device, (
        "All inputs must be on the same device"
    )
    if device.type == "cuda" and arch // 10 >= 9:  # Hopper or newer
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
    assert head_dim <= 512, (
        "head_dim must be less than or equal to 512 for efficient memory access"
    )
    if alpha is not None and delta is not None:
        assert device == alpha.device == delta.device, (
            "All inputs must be on the same device"
        )
        assert alpha.dtype == delta.dtype == query.dtype, (
            "Alpha and Delta tensors must have the same dtype as query/key/value"
        )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert device == cu_seqlens_q.device == cu_seqlens_k.device, (
            "All inputs must be on the same device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlens_q and cu_seqlens_k must be int32"
        )
    if seqused_q is not None and seqused_k is not None:
        assert device == seqused_q.device == seqused_k.device, (
            "All inputs must be on the same device"
        )
        assert seqused_q.dtype == seqused_k.dtype == torch.int32, (
            "seqused_q and seqused_k must be int32"
        )


def assert_bwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    device: torch.device = None,
    arch: int = None,
):
    """
    Assert the validity of inputs for the backward base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param out: Output tensor
    :param dout: Gradient of the output tensor
    :param lse: Log-sum-exp tensor
    :param alpha: Alpha tensor for gated attention
    :param delta: Delta tensor for gated attention
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param device: Device of the tensors
    :param arch: Architecture model as a number

    :raises AssertionError: If any of the assertions fail
    """
    assert (
        device
        == query.device
        == key.device
        == value.device
        == out.device
        == dout.device
        == lse.device
    ), "All inputs must be on the same device"
    if device.type == "cuda" and arch // 10 >= 9:  # Hopper or newer
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
    assert head_dim <= 512, (
        "head_dim must be less than or equal to 512 for efficient memory access"
    )
    if alpha is not None and delta is not None:
        assert device == alpha.device == delta.device, (
            "All inputs must be on the same device"
        )
        assert alpha.dtype == delta.dtype == query.dtype, (
            "Alpha and Delta tensors must have the same dtype as query/key/value"
        )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert device == cu_seqlens_q.device == cu_seqlens_k.device, (
            "All inputs must be on the same device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlens_q and cu_seqlens_k must be int32"
        )
    if seqused_q is not None and seqused_k is not None:
        assert device == seqused_q.device == seqused_k.device, (
            "All inputs must be on the same device"
        )
        assert seqused_q.dtype == seqused_k.dtype == torch.int32, (
            "seqused_q and seqused_k must be int32"
        )


def assert_dec_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    device: torch.device = None,
    arch: int = None,
):
    """
    Assert the validity of inputs for the decode kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param alpha: Alpha tensor for gated attention
    :param delta: Delta tensor for gated attention
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param device: Device of the tensors
    :param arch: Architecture model as a number

    :raises AssertionError: If any of the assertions fail
    """
    assert device == query.device == key.device == value.device, (
        "All inputs must be on the same device"
    )
    if device.type == "cuda" and arch // 10 >= 9:  # Hopper or newer
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
    assert head_dim <= 512, (
        "head_dim must be less than or equal to 512 for efficient memory access"
    )
    if alpha is not None and delta is not None:
        assert device == alpha.device == delta.device, (
            "All inputs must be on the same device"
        )
        assert alpha.dtype == delta.dtype == query.dtype, (
            "Alpha and Delta tensors must have the same dtype as query/key/value"
        )
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        assert device == cu_seqlens_q.device == cu_seqlens_k.device, (
            "All inputs must be on the same device"
        )
        assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32, (
            "cu_seqlens_q and cu_seqlens_k must be int32"
        )
    if seqused_q is not None and seqused_k is not None:
        assert device == seqused_q.device == seqused_k.device, (
            "All inputs must be on the same device"
        )
        assert seqused_q.dtype == seqused_k.dtype == torch.int32, (
            "seqused_q and seqused_k must be int32"
        )
