# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
from flash_sparse_attn.ops.gluon.cache_utils import get_device_arch


def assert_fwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    device: torch.device = None,
):
    """
    Assert the validity of inputs for the forward kernel.

    :param query: query tensor
    :type query: torch.Tensor
    :param key: key tensor
    :type key: torch.Tensor
    :param value: value tensor
    :type value: torch.Tensor
    :param query_scale: Optional query scale tensor for quantized inputs
    :type query_scale: Optional[torch.Tensor]
    :param key_scale: Optional key scale tensor for quantized inputs
    :type key_scale: Optional[torch.Tensor]
    :param value_scale: Optional value scale tensor for quantized inputs
    :type value_scale: Optional[torch.Tensor]
    :param window_sizes: Optional window sizes tensor for local attention
    :type window_sizes: Optional[torch.Tensor]
    :param cu_seqlens_q: Optional cumulative sequence lengths for queries
    :type cu_seqlens_q: Optional[torch.Tensor]
    :param cu_seqlens_k: Optional cumulative sequence lengths for keys
    :type cu_seqlens_k: Optional[torch.Tensor]
    :param seqused_q: Optional sequence used for queries
    :type seqused_q: Optional[torch.Tensor]
    :param seqused_k: Optional sequence used for keys
    :type seqused_k: Optional[torch.Tensor]
    :param num_heads_q: number of query heads
    :type num_heads_q: int
    :param num_heads_kv: number of key/value heads
    :type num_heads_kv: int
    :param head_dim: head dimension
    :type head_dim: int
    :param device: device of the tensors
    :type device: torch.device

    :raises AssertionError: If any of the assertions fail
    """
    arch = get_device_arch(device)
    assert device == query.device == key.device == value.device, (
        "All inputs must be on the same device"
    )
    if arch >= 90:
        assert query.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], (
            "query tensor dtype must be float16 or bfloat16 or float8_e5m2 or float8_e4m3fn"
        )
    else:
        assert query.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "query tensor dtype must be float16 or bfloat16"
    assert query.dtype == key.dtype == value.dtype, (
        "query/key/value tensors must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim in [32, 64, 128, 256], (
        "head_dim must be one of [32, 64, 128, 256] for efficient memory access"
    )
    if query_scale is not None and key_scale is not None and value_scale is not None:
        assert device == query_scale.device == key_scale.device == value_scale.device, (
            "All inputs must be on the same device"
        )
        assert query_scale.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "scale tensors must be float16, bfloat16, or float32"
        assert query_scale.dtype == key_scale.dtype == value_scale.dtype, (
            "All scale tensors must have the same dtype"
        )
    if cu_seqlens_q is not None:
        assert device == cu_seqlens_q.device, "All inputs must be on the same device"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    if cu_seqlens_k is not None:
        assert device == cu_seqlens_k.device, "All inputs must be on the same device"
        assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
    if seqused_q is not None:
        assert device == seqused_q.device, "All inputs must be on the same device"
        assert seqused_q.dtype == torch.int32, "seqused_q must be int32"
    if seqused_k is not None:
        assert device == seqused_k.device, "All inputs must be on the same device"
        assert seqused_k.dtype == torch.int32, "seqused_k must be int32"
    if window_sizes is not None:
        assert window_sizes.dtype == torch.int32, "window_sizes must be int32"
        assert window_sizes.ndim == 2 and window_sizes.shape[1] == 4, (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )


def assert_bwd_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    device: torch.device = None,
):
    """
    Assert the validity of inputs for the backward base kernel.

    :param query: query tensor
    :type query: torch.Tensor
    :param key: key tensor
    :type key: torch.Tensor
    :param value: value tensor
    :type value: torch.Tensor
    :param out: output tensor
    :type out: torch.Tensor
    :param dout: gradient of the output tensor
    :type dout: torch.Tensor
    :param lse: log-sum-exp tensor
    :type lse: torch.Tensor
    :param query_scale: Optional query scale tensor for quantized inputs
    :type query_scale: torch.Tensor
    :param key_scale: Optional key scale tensor for quantized inputs
    :type key_scale: torch.Tensor
    :param value_scale: Optional value scale tensor for quantized inputs
    :type value_scale: torch.Tensor
    :param window_sizes: Optional window sizes tensor for local attention
    :type window_sizes: torch.Tensor
    :param cu_seqlens_q: Optional cumulative sequence lengths for queries
    :type cu_seqlens_q: torch.Tensor
    :param cu_seqlens_k: Optional cumulative sequence lengths for keys
    :type cu_seqlens_k: torch.Tensor
    :param seqused_q: Optional sequence used for queries
    :type seqused_q: torch.Tensor
    :param seqused_k: Optional sequence used for keys
    :type seqused_k: torch.Tensor
    :param num_heads_q: number of query heads
    :type num_heads_q: int
    :param num_heads_kv: number of key/value heads
    :type num_heads_kv: int
    :param head_dim: head dimension
    :type head_dim: int
    :param device: device of the tensors
    :type device: torch.device

    :raises AssertionError: If any of the assertions fail
    """
    arch = get_device_arch(device)
    assert (
        device
        == query.device
        == key.device
        == value.device
        == out.device
        == dout.device
        == lse.device
    ), "All inputs must be on the same device"
    if arch >= 90:
        assert query.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], (
            "query tensor dtype must be float16 or bfloat16 or float8_e5m2 or float8_e4m3fn"
        )
    else:
        assert query.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "query tensor dtype must be float16 or bfloat16"
    assert query.dtype == key.dtype == value.dtype, (
        "query/key/value tensors must have the same dtype"
    )
    assert out.dtype == dout.dtype, "out/dout tensors must have the same dtype"
    assert lse.dtype == torch.float32, (
        "lse must be float32 for numerical stability in backward pass"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim in [32, 64, 128, 256], (
        "head_dim must be one of [32, 64, 128, 256] for efficient memory access"
    )
    if query_scale is not None and key_scale is not None and value_scale is not None:
        assert device == query_scale.device == key_scale.device == value_scale.device, (
            "All inputs must be on the same device"
        )
        assert query_scale.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "scale tensors must be float16, bfloat16, or float32"
        assert query_scale.dtype == key_scale.dtype == value_scale.dtype, (
            "All scale tensors must have the same dtype"
        )
    if cu_seqlens_q is not None:
        assert device == cu_seqlens_q.device, "All inputs must be on the same device"
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    if cu_seqlens_k is not None:
        assert device == cu_seqlens_k.device, "All inputs must be on the same device"
        assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
    if seqused_q is not None:
        assert device == seqused_q.device, "All inputs must be on the same device"
        assert seqused_q.dtype == torch.int32, "seqused_q must be int32"
    if seqused_k is not None:
        assert device == seqused_k.device, "All inputs must be on the same device"
        assert seqused_k.dtype == torch.int32, "seqused_k must be int32"
    if window_sizes is not None:
        assert window_sizes.dtype == torch.int32, "window_sizes must be int32"
        assert window_sizes.ndim == 2 and window_sizes.shape[1] == 4, (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )
