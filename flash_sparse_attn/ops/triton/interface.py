from typing import Optional, Tuple, Union

import torch

from flash_sparse_attn.ops.triton.flash_dense_fwd import (
    _flash_dense_attn_base_forward,
    _flash_dense_attn_varlen_base_forward,
)
from flash_sparse_attn.ops.triton.flash_dense_bwd import (
    _flash_dense_attn_base_backward,
    _flash_dense_attn_varlen_base_backward,
)
from flash_sparse_attn.ops.triton.flash_dense_dec import (
    _flash_dense_attn_base_decode,
    _flash_dense_attn_varlen_base_decode,
)
from flash_sparse_attn.ops.triton.flash_sparse_fwd import (
    _flash_sparse_attn_base_forward,
    _flash_sparse_attn_varlen_base_forward,
)
from flash_sparse_attn.ops.triton.flash_sparse_bwd import (
    _flash_sparse_attn_base_backward,
    _flash_sparse_attn_varlen_base_backward,
)
from flash_sparse_attn.ops.triton.flash_sparse_dec import (
    _flash_sparse_attn_base_decode,
    _flash_sparse_attn_varlen_base_decode,
)
from flash_sparse_attn.ops.triton.flash_gated_fwd import (
    _flash_gated_attn_base_forward,
    _flash_gated_attn_varlen_base_forward,
)
from flash_sparse_attn.ops.triton.flash_gated_bwd import (
    _flash_gated_attn_base_backward,
    _flash_gated_attn_varlen_base_backward,
)
from flash_sparse_attn.ops.triton.flash_gated_dec import (
    _flash_gated_attn_base_decode,
    _flash_gated_attn_varlen_base_decode,
)
from flash_sparse_attn.ops.triton.utils import ensure_contiguous


class FlashDenseAttnFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[2] != key.shape[2]
            and query.shape[1] == 1
            and query.shape[1] != key.shape[1]
        )
        out, lse, softmax_scale = _flash_dense_attn_base_forward(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            window_size=window_size,
            pack_gqa=pack_gqa,
        )

        ctx.save_for_backward(query, key, value, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, out, lse = ctx.saved_tensors

        dq, dk, dv = _flash_dense_attn_base_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            window_size=ctx.window_size,
        )

        return dq, dk, dv, *((None,) * 20)


class FlashDenseAttnVarlenFunc(torch.autograd.Function):
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
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[1] != key.shape[1]
            and max_seqlen_q == 1
            and max_seqlen_q != max_seqlen_k
        )
        out, lse, softmax_scale = _flash_dense_attn_varlen_base_forward(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            window_size=window_size,
            pack_gqa=pack_gqa,
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
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

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

        dq, dk, dv = _flash_dense_attn_varlen_base_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            window_size=ctx.window_size,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )

        return dq, dk, dv, *((None,) * 20)


class FlashSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_threshold: Optional[float] = None,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[2] != key.shape[2]
            and query.shape[1] == 1
            and query.shape[1] != key.shape[1]
        )
        out, lse, softmax_scale, softmax_threshold = _flash_sparse_attn_base_forward(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            window_size=window_size,
            pack_gqa=pack_gqa,
        )

        ctx.save_for_backward(query, key, value, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.softmax_threshold = softmax_threshold
        ctx.window_size = window_size

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, out, lse = ctx.saved_tensors

        dq, dk, dv = _flash_sparse_attn_base_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            softmax_threshold=ctx.softmax_threshold,
            window_size=ctx.window_size,
        )

        return dq, dk, dv, *((None,) * 20)


class FlashSparseAttnVarlenFunc(torch.autograd.Function):
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
        softmax_threshold: Optional[float] = None,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[1] != key.shape[1]
            and max_seqlen_q == 1
            and max_seqlen_q != max_seqlen_k
        )
        out, lse, softmax_scale, softmax_threshold = (
            _flash_sparse_attn_varlen_base_forward(
                query=query,
                key=key,
                value=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                is_causal=is_causal,
                softmax_scale=softmax_scale,
                softmax_threshold=softmax_threshold,
                window_size=window_size,
                pack_gqa=pack_gqa,
            )
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
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.softmax_threshold = softmax_threshold
        ctx.window_size = window_size
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

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

        dq, dk, dv = _flash_sparse_attn_varlen_base_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            softmax_threshold=ctx.softmax_threshold,
            window_size=ctx.window_size,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )

        return dq, dk, dv, *((None,) * 20)


class FlashGatedAttnFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        delta: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_threshold: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        is_logsigmoid_gate: bool = True,
        is_adapt_gate: bool = True,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[2] != key.shape[2]
            and query.shape[1] == 1
            and query.shape[1] != key.shape[1]
        )
        out, lse, softmax_scale, softmax_threshold, gate_threshold = (
            _flash_gated_attn_base_forward(
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                delta=delta,
                is_causal=is_causal,
                softmax_scale=softmax_scale,
                softmax_threshold=softmax_threshold,
                gate_threshold=gate_threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                is_adapt_gate=is_adapt_gate,
                window_size=window_size,
                pack_gqa=pack_gqa,
            )
        )

        ctx.save_for_backward(query, key, value, alpha, delta, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.softmax_threshold = softmax_threshold
        ctx.gate_threshold = gate_threshold
        ctx.is_logsigmoid_gate = is_logsigmoid_gate
        ctx.is_adapt_gate = is_adapt_gate
        ctx.window_size = window_size

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, alpha, delta, out, lse = ctx.saved_tensors

        dq, dk, dv, da, dd = _flash_gated_attn_base_backward(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            delta=delta,
            out=out,
            dout=dout,
            lse=lse,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            softmax_threshold=ctx.softmax_threshold,
            gate_threshold=ctx.gate_threshold,
            is_logsigmoid_gate=ctx.is_logsigmoid_gate,
            is_adapt_gate=ctx.is_adapt_gate,
            window_size=ctx.window_size,
        )

        return dq, dk, dv, da, dd, *((None,) * 20)


class FlashGatedAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        delta: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_threshold: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        is_logsigmoid_gate: bool = True,
        is_adapt_gate: bool = True,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal
        # Set pack_gqa to True if query and key have different number of heads and sequence length is 1 to enable GQA optimization
        pack_gqa = (
            query.shape[1] != key.shape[1]
            and max_seqlen_q == 1
            and max_seqlen_q != max_seqlen_k
        )
        out, lse, softmax_scale, softmax_threshold, gate_threshold = (
            _flash_gated_attn_varlen_base_forward(
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                delta=delta,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                is_causal=is_causal,
                softmax_scale=softmax_scale,
                softmax_threshold=softmax_threshold,
                gate_threshold=gate_threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                is_adapt_gate=is_adapt_gate,
                window_size=window_size,
                pack_gqa=pack_gqa,
            )
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            alpha,
            delta,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.softmax_threshold = softmax_threshold
        ctx.gate_threshold = gate_threshold
        ctx.is_logsigmoid_gate = is_logsigmoid_gate
        ctx.is_adapt_gate = is_adapt_gate
        ctx.window_size = window_size
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

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
            alpha,
            delta,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        ) = ctx.saved_tensors

        dq, dk, dv, da, dd = _flash_gated_attn_varlen_base_backward(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            delta=delta,
            out=out,
            dout=dout,
            lse=lse,
            softmax_scale=ctx.softmax_scale,
            softmax_threshold=ctx.softmax_threshold,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            is_causal=ctx.is_causal,
            gate_threshold=ctx.gate_threshold,
            is_logsigmoid_gate=ctx.is_logsigmoid_gate,
            is_adapt_gate=ctx.is_adapt_gate,
            window_size=ctx.window_size,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )

        return dq, dk, dv, da, dd, *((None,) * 20)


def flash_dense_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashDenseAttnFunc.apply(
        query,
        key,
        value,
        is_causal,
        softmax_scale,
        window_size,
        return_lse,
    )


def flash_dense_attn_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """
    out, lse = _flash_dense_attn_base_decode(
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        window_size=window_size,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out


def flash_dense_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function for variable-length sequences that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_q: Maximum sequence length for queries.
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    return FlashDenseAttnVarlenFunc.apply(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        softmax_scale,
        window_size,
        seqused_q,
        seqused_k,
        return_lse,
    )


def flash_dense_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor indicating the actual sequence lengths for queries.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [total_seqlen_q, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [total_seqlen_q, num_heads_q].

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    out, lse = _flash_dense_attn_varlen_base_decode(
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        window_size=window_size,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out


def flash_sparse_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / seqlen_k.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashSparseAttnFunc.apply(
        query,
        key,
        value,
        is_causal,
        softmax_scale,
        softmax_threshold,
        window_size,
        return_lse,
    )


def flash_sparse_attn_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / seqlen_k.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """
    out, lse = _flash_sparse_attn_base_decode(
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        window_size=window_size,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out


def flash_sparse_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function for variable-length sequences that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_q: Maximum sequence length for queries.
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / max_seqlen_k.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    return FlashSparseAttnVarlenFunc.apply(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        softmax_scale,
        softmax_threshold,
        window_size,
        seqused_q,
        seqused_k,
        return_lse,
    )


def flash_sparse_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / max_seqlen_k.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor indicating the actual sequence lengths for queries.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [total_seqlen_q, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [total_seqlen_q, num_heads_q].

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    out, lse = _flash_sparse_attn_varlen_base_decode(
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        window_size=window_size,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out


def flash_gated_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param alpha: Tensor of shape [batch_size, num_heads, seqlen_q] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [batch_size, num_kv_heads, seqlen_k] representing the sparsity pattern for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashGatedAttnFunc.apply(
        query,
        key,
        value,
        alpha,
        delta,
        is_causal,
        softmax_scale,
        softmax_threshold,
        gate_threshold,
        is_logsigmoid_gate,
        is_adapt_gate,
        window_size,
        return_lse,
    )


def flash_gated_attn_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param alpha: Tensor of shape [batch_size, num_heads] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [batch_size, num_kv_heads, seqlen_k] representing the sparsity pattern for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """
    out, lse = _flash_gated_attn_base_decode(
        query=query,
        key=key,
        value=value,
        alpha=alpha,
        delta=delta,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        is_logsigmoid_gate=is_logsigmoid_gate,
        is_adapt_gate=is_adapt_gate,
        window_size=window_size,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out


def flash_gated_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for variable-length sequences that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param alpha: Tensor of shape [num_heads_q, total_seqlen_q] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [num_heads_kv, total_seqlen_k] representing the sparsity pattern for keys/values.
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_q: Maximum sequence length for queries.
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    return FlashGatedAttnVarlenFunc.apply(
        query,
        key,
        value,
        alpha,
        delta,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        softmax_scale,
        softmax_threshold,
        gate_threshold,
        is_logsigmoid_gate,
        is_adapt_gate,
        window_size,
        seqused_q,
        seqused_k,
        return_lse,
    )


def flash_gated_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param alpha: Tensor of shape [num_heads_q, total_seqlen_q] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [num_heads_kv, total_seqlen_k] representing the sparsity pattern for keys/values.
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param window_size: Optional tuple (window_size_q, window_size_k) for local attention. If None, no local masking is applied.
    :param seqused_q: Optional tensor indicating the actual sequence lengths for queries.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.
    :param out: Optional preallocated output tensor with shape [total_seqlen_q, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [total_seqlen_q, num_heads_q].

    :returns: If return_lse is False, returns out with shape [total_seqlen_q, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [total_seqlen_q, num_heads_q].
    """
    out, lse = _flash_gated_attn_varlen_base_decode(
        query=query,
        key=key,
        value=value,
        alpha=alpha,
        delta=delta,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        is_logsigmoid_gate=is_logsigmoid_gate,
        is_adapt_gate=is_adapt_gate,
        window_size=window_size,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
    )

    if return_lse:
        return out, lse
    return out
