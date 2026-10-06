# Copyright (c) 2026, Jingze Shi.
from typing import Optional

import torch
from flash_sparse_attn.ops.gluon.utils import get_device_arch


def assert_fwd_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
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

    :param q: query tensor
    :type q: torch.Tensor
    :param k: key tensor
    :type k: torch.Tensor
    :param v: value tensor
    :type v: torch.Tensor
    :param q_scale: query scale tensor for quantized inputs
    :type q_scale: Optional[torch.Tensor]
    :param k_scale: key scale tensor for quantized inputs
    :type k_scale: Optional[torch.Tensor]
    :param v_scale: value scale tensor for quantized inputs
    :type v_scale: Optional[torch.Tensor]
    :param window_sizes: window sizes tensor for local attention
    :type window_sizes: Optional[torch.Tensor]
    :param cu_seqlens_q: cumulative sequence lengths for queries
    :type cu_seqlens_q: Optional[torch.Tensor]
    :param cu_seqlens_k: cumulative sequence lengths for keys
    :type cu_seqlens_k: Optional[torch.Tensor]
    :param seqused_q: sequence used for queries
    :type seqused_q: Optional[torch.Tensor]
    :param seqused_k: sequence used for keys
    :type seqused_k: Optional[torch.Tensor]
    :param num_heads_q: number of query heads
    :type num_heads_q: int
    :param num_heads_kv: number of key/value heads
    :type num_heads_kv: int
    :param head_dim: dimension of each head
    :type head_dim: int
    :param device: device of the tensors
    :type device: torch.device

    :raises AssertionError: If any of the assertions fail
    """
    arch = get_device_arch(device)
    assert device == q.device == k.device == v.device, (
        "All inputs must be on the same device"
    )
    if arch >= 90:
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], (
            "query tensor dtype must be float16 or bfloat16 or float8_e5m2 or float8_e4m3fn"
        )
    else:
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "query tensor dtype must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, (
        "query/key/value tensors must have the same dtype"
    )
    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    assert head_dim in [32, 64, 128, 256], (
        "head_dim must be one of [32, 64, 128, 256] for efficient memory access"
    )
    if q_scale is not None and k_scale is not None and v_scale is not None:
        assert device == q_scale.device == k_scale.device == v_scale.device, (
            "All inputs must be on the same device"
        )
        assert q_scale.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "scale tensors must be float16, bfloat16, or float32"
        assert q_scale.dtype == k_scale.dtype == v_scale.dtype, (
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
        assert window_sizes.ndim == 2 and window_sizes.shape == (num_heads_kv, 4), (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )


def assert_bwd_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
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
    Assert the validity of inputs for the backward kernel.

    :param q: query tensor
    :type q: torch.Tensor
    :param k: key tensor
    :type k: torch.Tensor
    :param v: value tensor
    :type v: torch.Tensor
    :param out: output tensor
    :type out: torch.Tensor
    :param dout: gradient of the output tensor
    :type dout: torch.Tensor
    :param lse: log-sum-exp tensor
    :type lse: torch.Tensor
    :param q_scale: query scale tensor for quantized inputs
    :type q_scale: Optional[torch.Tensor]
    :param k_scale: key scale tensor for quantized inputs
    :type k_scale: Optional[torch.Tensor]
    :param v_scale: value scale tensor for quantized inputs
    :type v_scale: Optional[torch.Tensor]
    :param window_sizes: window sizes tensor for local attention
    :type window_sizes: Optional[torch.Tensor]
    :param cu_seqlens_q: cumulative sequence lengths for queries
    :type cu_seqlens_q: Optional[torch.Tensor]
    :param cu_seqlens_k: cumulative sequence lengths for keys
    :type cu_seqlens_k: Optional[torch.Tensor]
    :param seqused_q: sequence used for queries
    :type seqused_q: Optional[torch.Tensor]
    :param seqused_k: sequence used for keys
    :type seqused_k: Optional[torch.Tensor]
    :param num_heads_q: number of query heads
    :type num_heads_q: int
    :param num_heads_kv: number of key/value heads
    :type num_heads_kv: int
    :param head_dim: dimension of each head
    :type head_dim: int
    :param device: device of the tensors
    :type device: torch.device

    :raises AssertionError: If any of the assertions fail
    """
    arch = get_device_arch(device)
    assert (
        device
        == q.device
        == k.device
        == v.device
        == out.device
        == dout.device
        == lse.device
    ), "All inputs must be on the same device"
    if arch >= 90:
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], (
            "query tensor dtype must be float16 or bfloat16 or float8_e5m2 or float8_e4m3fn"
        )
    else:
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "query tensor dtype must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, (
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
    if q_scale is not None and k_scale is not None and v_scale is not None:
        assert device == q_scale.device == k_scale.device == v_scale.device, (
            "All inputs must be on the same device"
        )
        assert q_scale.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "scale tensors must be float16, bfloat16, or float32"
        assert q_scale.dtype == k_scale.dtype == v_scale.dtype, (
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
        assert window_sizes.ndim == 2 and window_sizes.shape == (num_heads_kv, 4), (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )


def assert_fwd_combine_inputs(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    num_splits: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
):
    """
    Assert the validity of inputs for the forward combine kernel.

    :param out_partial: partial output tensor
    :type out_partial: torch.Tensor
    :param lse_partial: partial logsumexp tensor
    :type lse_partial: torch.Tensor
    :param out: output tensor
    :type out: torch.Tensor
    :param lse: logsumexp tensor
    :type lse: torch.Tensor
    :param cu_seqlens_q: cumulative sequence lengths for queries
    :type cu_seqlens_q: Optional[torch.Tensor]
    :param seqused_q: sequence used for queries
    :type seqused_q: Optional[torch.Tensor]

    :raises AssertionError: If any of the assertions fail
    """
    assert out_partial.device == lse_partial.device == out.device == lse.device, (
        "All inputs must be on the same device"
    )
    assert out_partial.dtype == lse_partial.dtype == lse.dtype == torch.float32, (
        "out_partial/lse_partial/lse tensors must be float32 for numerical stability"
    )
    assert out.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "out tensor must be float16 or bfloat16"
    assert num_splits >= 1, "num_splits must be greater than or equal to 1"
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    if seqused_q is not None:
        assert seqused_q.dtype == torch.int32, "seqused_q must be int32"


def assert_bwd_combine_inputs(
    dk_partial: torch.Tensor,
    dv_partial: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    num_splits: int,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
):
    """
    Assert the validity of inputs for the backward combine kernel.

    :param dk_partial: partial key gradient tensor
    :type dk_partial: torch.Tensor
    :param dv_partial: partial value gradient tensor
    :type dv_partial: torch.Tensor
    :param dk: combined key gradient tensor
    :type dk: torch.Tensor
    :param dv: combined value gradient tensor
    :type dv: torch.Tensor
    :param num_splits: number of partial gradients to combine
    :type num_splits: int
    :param cu_seqlens_k: cumulative sequence lengths for keys
    :type cu_seqlens_k: Optional[torch.Tensor]
    :param seqused_k: actual sequence lengths for keys
    :type seqused_k: Optional[torch.Tensor]

    :raises AssertionError: If any of the assertions fail
    """
    device = dk_partial.device
    assert device == dv_partial.device == dk.device == dv.device, (
        "All inputs must be on the same device"
    )
    assert dk_partial.dtype == dv_partial.dtype == torch.float32, (
        "dk_partial/dv_partial tensors must be float32 for numerical stability"
    )
    assert dk.dtype == dv.dtype, "dk/dv tensors must have the same dtype"
    assert dk.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "dk/dv tensors must be float16 or bfloat16"
    assert num_splits >= 1, "num_splits must be greater than or equal to 1"
    if cu_seqlens_k is not None:
        assert device == cu_seqlens_k.device, "All inputs must be on the same device"
        assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
    if seqused_k is not None:
        assert device == seqused_k.device, "All inputs must be on the same device"
        assert seqused_k.dtype == torch.int32, "seqused_k must be int32"
