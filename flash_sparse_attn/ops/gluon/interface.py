from typing import Optional, Tuple, Union

import torch

from flash_sparse_attn.ops.gluon.flash_fwd_sm80 import (
    _flash_attn_forward_sm80,
    _flash_attn_varlen_forward_sm80,
)
from flash_sparse_attn.ops.gluon.flash_bwd_sm80 import (
    _flash_attn_backward_sm80,
    _flash_attn_varlen_backward_sm80,
)
from flash_sparse_attn.ops.gluon.utils import ensure_contiguous


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        num_kv_splits: Optional[int] = None,
        num_qo_splits: Optional[int] = None,
        seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal
        is_local = True if window_sizes is not None else False
        is_sparse = True if softmax_threshold is not None else False
        if num_kv_splits is not None and num_kv_splits > 1:
            is_split_kv = True
        if num_qo_splits is not None and num_qo_splits > 1:
            is_split_qo = True

        out, lse = _flash_attn_forward_sm80(
            query,
            key,
            value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            window_sizes=window_sizes,
            softmax_threshold=softmax_threshold,
            is_local=is_local,
            is_sparse=is_sparse,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            num_splits=num_kv_splits,
            seqused_k=seqused_k,
            out=out,
            lse=lse,
            skip_checks=skip_checks,
        )

        ctx.save_for_backward(query, key, value, out, lse, seqused_k)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.is_local = is_local
        ctx.is_sparse = is_sparse
        ctx.is_split_qo = is_split_qo
        ctx.num_qo_splits = num_qo_splits
        ctx.skip_checks = skip_checks

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, out, lse, seqused_k = ctx.saved_tensors

        dq, dk, dv = _flash_attn_backward_sm80(
            query,
            key,
            value,
            out,
            lse,
            dout,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            is_local=ctx.is_local,
            is_sparse=ctx.is_sparse,
            is_split_qo=ctx.is_split_qo,
            num_splits=ctx.num_qo_splits,
            seqused_k=seqused_k,
            skip_checks=ctx.skip_checks,
        )

        return dq, dk, dv, *((None,) * 30)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        num_kv_splits: Optional[int] = None,
        num_qo_splits: Optional[int] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal
        is_local = True if window_sizes is not None else False
        is_sparse = True if softmax_threshold is not None else False
        if num_kv_splits is not None and num_kv_splits > 1:
            is_split_kv = True
        if num_qo_splits is not None and num_qo_splits > 1:
            is_split_qo = True

        out, lse = _flash_attn_varlen_forward_sm80(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            window_sizes=window_sizes,
            softmax_threshold=softmax_threshold,
            is_local=is_local,
            is_sparse=is_sparse,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            num_splits=num_kv_splits,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            out=out,
            lse=lse,
            skip_checks=skip_checks,
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.is_local = is_local
        ctx.is_sparse = is_sparse
        ctx.is_split_qo = is_split_qo
        ctx.num_qo_splits = num_qo_splits
        ctx.skip_checks = skip_checks

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        (
            query,
            key,
            value,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        ) = ctx.saved_tensors

        dq, dk, dv = _flash_attn_varlen_backward_sm80(
            query,
            key,
            value,
            out,
            lse,
            dout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            is_local=ctx.is_local,
            is_sparse=ctx.is_sparse,
            is_split_qo=ctx.is_split_qo,
            num_splits=ctx.num_qo_splits,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            skip_checks=ctx.skip_checks,
        )

        return dq, dk, dv, *((None,) * 30)


def flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    num_kv_splits: Optional[int] = None,
    num_qo_splits: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for query dequantization.
    :param key_scale: Optional per-tensor scale for key dequantization.
    :param value_scale: Optional per-tensor scale for value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [num_kv_heads, 4] with dtype int32, with columns [window_sink, window_left, window_right, window_near]. If provided, is_local is automatically set to True.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param num_kv_splits: Optional split count for split-KV. If provided, enables split-KV, if omitted with split-KV enabled, a heuristic is used.
    :param num_qo_splits: Optional split count for split-QO. If provided, enables split-QO, if omitted with split-QO enabled, a heuristic is used.
    :param seqused_k: Optional tensor of shape [batch_size] indicating the actual sequence lengths for keys/values.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashAttnFunc.apply(
        query,
        key,
        value,
        is_causal,
        softmax_scale,
        window_sizes,
        softmax_threshold,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        num_kv_splits,
        num_qo_splits,
        seqused_k,
        out,
        lse,
        skip_checks,
        return_lse,
    )


def flash_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    num_kv_splits: Optional[int] = None,
    num_qo_splits: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash attention function for variable-length sequences that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_q: Maximum sequence length for queries.
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for query dequantization.
    :param key_scale: Optional per-tensor scale for key dequantization.
    :param value_scale: Optional per-tensor scale for value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [num_kv_heads, 4] with dtype int32, with columns [window_sink, window_left, window_right, window_near]. If provided, is_local is automatically set to True.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param num_kv_splits: Optional split count for split-KV. If provided, enables split-KV, if omitted with split-KV enabled, a heuristic is used.
    :param num_qo_splits: Optional split count for split-QO. If provided, enables split-QO, if omitted with split-QO enabled, a heuristic is used.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    return FlashAttnVarlenFunc.apply(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        softmax_scale,
        window_sizes,
        softmax_threshold,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        num_kv_splits,
        num_qo_splits,
        seqused_q,
        seqused_k,
        out,
        lse,
        skip_checks,
        return_lse,
    )
