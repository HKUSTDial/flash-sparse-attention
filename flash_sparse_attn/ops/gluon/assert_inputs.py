from typing import Optional

import torch


def assert_fwd_sm80_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param window_sizes: Optional window sizes tensor for local attention
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param device: Device of the tensors

    :raises AssertionError: If any of the assertions fail
    """
    assert device == query.device == key.device == value.device, (
        "All inputs must be on the same device"
    )
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
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    is_quant: bool = False,
    device: torch.device = None,
):
    """
    Assert the validity of inputs for the backward base kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param out: Output tensor
    :param dout: Gradient of the output tensor
    :param lse: Log-sum-exp tensor
    :param query_scale: Optional query scale tensor for quantized inputs
    :param key_scale: Optional key scale tensor for quantized inputs
    :param value_scale: Optional value scale tensor for quantized inputs
    :param window_sizes: Optional window sizes tensor for local attention
    :param alpha: Alpha tensor for gated attention
    :param delta: Delta tensor for gated attention
    :param cu_seqlens_q: Cumulative sequence lengths for queries
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_q: Sequence used for queries
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param is_quant: Whether the inputs are quantized
    :param device: Device of the tensors

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
    if is_quant:
        assert query.dtype in [torch.float8_e5m2, torch.float8_e4m3fn], (
            "Input dtype must be float8_e5m2 or float8_e4m3fn"
        )
        assert (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        ), "Scale tensors must be provided for quantized inputs"
        assert query_scale.dtype in [torch.float16, torch.bfloat16, torch.float32], (
            "Scale tensors must be float16, bfloat16, or float32"
        )
        assert query_scale.dtype == key_scale.dtype == value_scale.dtype, (
            "All scale tensors must have the same dtype"
        )
        assert query_scale.device == key_scale.device == value_scale.device == device, (
            "All scale tensors must be on the same device as query/key/value"
        )
        assert query.dtype == key.dtype == value.dtype, (
            "All inputs must have the same dtype"
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
    if window_sizes is not None:
        assert window_sizes.dtype == torch.int32, "window_sizes must be int32"
        assert window_sizes.ndim == 2 and window_sizes.shape[1] == 4, (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )


def assert_dec_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_heads_q: int = None,
    num_heads_kv: int = None,
    head_dim: int = None,
    page_size: Optional[int] = None,
    tile_mn: Optional[tuple] = None,
    is_quant: bool = False,
    device: torch.device = None,
):
    """
    Assert the validity of inputs for the decode kernel.

    :param query: Query tensor
    :param key: Key tensor
    :param value: Value tensor
    :param query_scale: Optional query scale tensor for quantized inputs
    :param key_scale: Optional key scale tensor for quantized inputs
    :param value_scale: Optional value scale tensor for quantized inputs
    :param window_sizes: Optional window sizes tensor for local attention
    :param alpha: Alpha tensor for gated attention
    :param delta: Delta tensor for gated attention
    :param page_table: Optional paged KV table
    :param gather_kv_indices: Optional TopK gather indices
    :param cu_seqlens_k: Cumulative sequence lengths for keys
    :param seqused_k: Sequence used for keys
    :param num_heads_q: Number of query heads
    :param num_heads_kv: Number of key/value heads
    :param head_dim: Head dimension
    :param page_size: Optional paged KV page size
    :param tile_mn: Optional launch tile size
    :param is_quant: Whether the inputs are quantized
    :param device: Device of the tensors

    :raises AssertionError: If any of the assertions fail
    """
    assert device == query.device == key.device == value.device, (
        "All inputs must be on the same device"
    )
    if is_quant:
        assert query.dtype in [torch.float8_e5m2, torch.float8_e4m3fn], (
            "Input dtype must be float8_e5m2 or float8_e4m3fn"
        )
        assert (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        ), "Scale tensors must be provided for quantized inputs"
        assert query_scale.dtype in [torch.float16, torch.bfloat16, torch.float32], (
            "Scale tensors must be float16, bfloat16, or float32"
        )
        assert query_scale.dtype == key_scale.dtype == value_scale.dtype, (
            "All scale tensors must have the same dtype"
        )
        assert query_scale.device == key_scale.device == value_scale.device == device, (
            "All scale tensors must be on the same device as query/key/value"
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
    if page_size is not None:
        assert page_size in [32, 64, 128, 256], (
            "paged KV page_size must be one of 32, 64, 128, or 256"
        )
        if tile_mn is not None:
            assert tile_mn[1] == page_size, "paged KV requires tile_mn[1] == page_size"
    assert page_table is None or gather_kv_indices is None, (
        "decode does not support gather_kv_indices with paged KV"
    )
    if alpha is not None and delta is not None:
        assert device == alpha.device == delta.device, (
            "All inputs must be on the same device"
        )
    if cu_seqlens_k is not None:
        assert device == cu_seqlens_k.device, "All inputs must be on the same device"
        assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
    if seqused_k is not None:
        assert device == seqused_k.device, "All inputs must be on the same device"
        assert seqused_k.dtype == torch.int32, "seqused_k must be int32"
    if window_sizes is not None:
        assert window_sizes.dtype == torch.int32, "window_sizes must be int32"
        assert window_sizes.ndim == 2 and window_sizes.shape[1] == 4, (
            "window_sizes must have shape [num_kv_heads, 4] with columns [window_sink, window_left, window_right, window_near]"
        )
