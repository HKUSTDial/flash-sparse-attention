from typing import Optional, Tuple, Union

import torch

from flash_sparse_attn.ops.triton.flash_dense_fwd import (
    _flash_dense_attn_forward,
    _flash_dense_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.flash_dense_bwd import (
    _flash_dense_attn_backward,
    _flash_dense_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_dense_dec import (
    _flash_dense_attn_decode,
    _flash_dense_attn_varlen_decode,
)
from flash_sparse_attn.ops.triton.flash_sparse_fwd import (
    _flash_sparse_attn_forward,
    _flash_sparse_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.flash_sparse_bwd import (
    _flash_sparse_attn_backward,
    _flash_sparse_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_sparse_dec import (
    _flash_sparse_attn_decode,
    _flash_sparse_attn_varlen_decode,
)
from flash_sparse_attn.ops.triton.flash_gated_fwd import (
    _flash_gated_attn_forward,
    _flash_gated_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.flash_gated_bwd import (
    _flash_gated_attn_backward,
    _flash_gated_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_gated_dec import (
    _flash_gated_attn_decode,
    _flash_gated_attn_varlen_decode,
)
from flash_sparse_attn.ops.triton.utils import ensure_contiguous
from flash_sparse_attn.ops.triton import quant


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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale = _flash_dense_attn_forward(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            is_local=is_local,
            is_quant=is_quant,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            out=out,
            lse=lse,
            is_autotune=is_autotune,
            skip_checks=skip_checks,
        )

        ctx.save_for_backward(query, key, value, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
        ctx.skip_checks = skip_checks

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, out, lse = ctx.saved_tensors

        dq, dk, dv = _flash_dense_attn_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            is_split_qo=ctx.is_split_qo,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale = _flash_dense_attn_varlen_forward(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            is_local=is_local,
            is_quant=is_quant,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            out=out,
            lse=lse,
            is_autotune=is_autotune,
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
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
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

        dq, dk, dv = _flash_dense_attn_varlen_backward(
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
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            is_split_qo=ctx.is_split_qo,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        query_orig, key_orig, value_orig = query, key, value
        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale, softmax_threshold = _flash_sparse_attn_forward(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            softmax_threshold=softmax_threshold,
            is_local=is_local,
            is_quant=is_quant,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            out=out,
            lse=lse,
            is_autotune=is_autotune,
            skip_checks=skip_checks,
        )

        ctx.save_for_backward(query_orig, key_orig, value_orig, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
        ctx.skip_checks = skip_checks

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, out, lse = ctx.saved_tensors

        dq, dk, dv = _flash_sparse_attn_backward(
            query=query,
            key=key,
            value=value,
            out=out,
            dout=dout,
            lse=lse,
            is_causal=ctx.is_causal,
            softmax_scale=ctx.softmax_scale,
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            is_split_qo=ctx.is_split_qo,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        query_orig, key_orig, value_orig = query, key, value
        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale, softmax_threshold = _flash_sparse_attn_varlen_forward(
            query=query,
            key=key,
            value=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            window_sizes=window_sizes,
            softmax_threshold=softmax_threshold,
            is_local=is_local,
            is_quant=is_quant,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            out=out,
            lse=lse,
            is_autotune=is_autotune,
            skip_checks=skip_checks,
        )

        ctx.save_for_backward(
            query_orig,
            key_orig,
            value_orig,
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
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
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

        dq, dk, dv = _flash_sparse_attn_varlen_backward(
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
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            is_split_qo=ctx.is_split_qo,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        is_logsigmoid_gate: bool = True,
        is_adapt_gate: bool = True,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if query.shape[1] == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        query_orig, key_orig, value_orig = query, key, value
        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale, softmax_threshold, gate_threshold = (
            _flash_gated_attn_forward(
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                delta=delta,
                is_causal=is_causal,
                softmax_scale=softmax_scale,
                query_scale=query_scale,
                key_scale=key_scale,
                value_scale=value_scale,
                window_sizes=window_sizes,
                softmax_threshold=softmax_threshold,
                gate_threshold=gate_threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                is_adapt_gate=is_adapt_gate,
                is_local=is_local,
                is_quant=is_quant,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                out=out,
                lse=lse,
                is_autotune=is_autotune,
                skip_checks=skip_checks,
            )
        )

        ctx.save_for_backward(query_orig, key_orig, value_orig, alpha, delta, out, lse)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.gate_threshold = gate_threshold
        ctx.is_logsigmoid_gate = is_logsigmoid_gate
        ctx.is_adapt_gate = is_adapt_gate
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
        ctx.skip_checks = skip_checks

        if return_lse:
            # LSE gradient is not supported yet
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout: torch.Tensor, *args):
        query, key, value, alpha, delta, out, lse = ctx.saved_tensors

        dq, dk, dv, da, dd = _flash_gated_attn_backward(
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
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            gate_threshold=ctx.gate_threshold,
            is_logsigmoid_gate=ctx.is_logsigmoid_gate,
            is_adapt_gate=ctx.is_adapt_gate,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            is_split_qo=ctx.is_split_qo,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
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
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        window_sizes: Optional[torch.Tensor] = None,
        softmax_threshold: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        is_logsigmoid_gate: bool = True,
        is_adapt_gate: bool = True,
        is_local: bool = False,
        is_quant: bool = False,
        is_split_kv: bool = False,
        is_split_qo: bool = False,
        pack_gqa: bool = False,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        is_autotune: bool = False,
        skip_checks: bool = False,
        return_lse: bool = False,
    ):
        # Set is_causal to False if sequence length is 1 to avoid unnecessary masking overhead
        is_causal = False if max_seqlen_q == 1 else is_causal

        if window_sizes is not None:
            is_local = True

        query_orig, key_orig, value_orig = query, key, value
        if is_quant and (
            query_scale is None or key_scale is None or value_scale is None
        ):
            query, query_scale = quant.quantize_fp8(query)
            key, key_scale = quant.quantize_fp8(key)
            value, value_scale = quant.quantize_fp8(value)

        out, lse, softmax_scale, softmax_threshold, gate_threshold = (
            _flash_gated_attn_varlen_forward(
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
                query_scale=query_scale,
                key_scale=key_scale,
                value_scale=value_scale,
                window_sizes=window_sizes,
                softmax_threshold=softmax_threshold,
                gate_threshold=gate_threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                is_adapt_gate=is_adapt_gate,
                is_local=is_local,
                is_quant=is_quant,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                out=out,
                lse=lse,
                is_autotune=is_autotune,
                skip_checks=skip_checks,
            )
        )

        ctx.save_for_backward(
            query_orig,
            key_orig,
            value_orig,
            alpha,
            delta,
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
        ctx.query_scale = query_scale
        ctx.key_scale = key_scale
        ctx.value_scale = value_scale
        ctx.window_sizes = window_sizes
        ctx.softmax_threshold = softmax_threshold
        ctx.gate_threshold = gate_threshold
        ctx.is_logsigmoid_gate = is_logsigmoid_gate
        ctx.is_adapt_gate = is_adapt_gate
        ctx.is_local = is_local
        ctx.is_quant = is_quant
        ctx.is_split_qo = is_split_qo
        ctx.is_autotune = is_autotune
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
            alpha,
            delta,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        ) = ctx.saved_tensors

        dq, dk, dv, da, dd = _flash_gated_attn_varlen_backward(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            delta=delta,
            out=out,
            dout=dout,
            lse=lse,
            softmax_scale=ctx.softmax_scale,
            query_scale=ctx.query_scale,
            key_scale=ctx.key_scale,
            value_scale=ctx.value_scale,
            window_sizes=ctx.window_sizes,
            softmax_threshold=ctx.softmax_threshold,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            is_causal=ctx.is_causal,
            gate_threshold=ctx.gate_threshold,
            is_logsigmoid_gate=ctx.is_logsigmoid_gate,
            is_adapt_gate=ctx.is_adapt_gate,
            is_local=ctx.is_local,
            is_quant=ctx.is_quant,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            is_split_qo=ctx.is_split_qo,
            is_autotune=ctx.is_autotune,
            skip_checks=ctx.skip_checks,
        )

        return dq, dk, dv, da, dd, *((None,) * 20)


def flash_dense_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashDenseAttnFunc.apply(
        query,
        key,
        value,
        is_causal,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        out,
        lse,
        is_autotune,
        skip_checks,
        return_lse,
    )


def flash_dense_attn_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_dense_attn_decode(
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
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
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [total_seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
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
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        seqused_q,
        seqused_k,
        out,
        lse,
        is_autotune,
        skip_checks,
        return_lse,
    )


def flash_dense_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash dense attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads_q].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_dense_attn_varlen_decode(
        query=query,
        key=key,
        value=value,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / seqlen_k.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward ccupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, seqlen_q, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads, seqlen_q].
    """
    return FlashSparseAttnFunc.apply(
        query,
        key,
        value,
        is_causal,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        out,
        lse,
        is_autotune,
        skip_checks,
        return_lse,
    )


def flash_sparse_attn_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / seqlen_k.
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_sparse_attn_decode(
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
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
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [total_seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / max_seqlen_k.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
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
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        seqused_q,
        seqused_k,
        out,
        lse,
        is_autotune,
        skip_checks,
        return_lse,
    )


def flash_sparse_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash sparse attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax. If None, defaults to head_dim / max_seqlen_k.
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads_q].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_sparse_attn_varlen_decode(
        query=query,
        key=key,
        value=value,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, seqlen_q, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param alpha: Tensor of shape [batch_size, seqlen_q, num_heads] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [batch_size, seqlen_k, num_kv_heads] representing the sparsity pattern for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
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
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
        is_logsigmoid_gate,
        is_adapt_gate,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        out,
        lse,
        is_autotune,
        skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads, head_dim].
    :param key: Key tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param value: Value tensor of shape [batch_size, seqlen_k, num_kv_heads, head_dim].
    :param alpha: Tensor of shape [batch_size, num_heads] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [batch_size, seqlen_k, num_kv_heads] representing the sparsity pattern for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_gated_attn_decode(
        query=query,
        key=key,
        value=value,
        alpha=alpha,
        delta=delta,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        is_logsigmoid_gate=is_logsigmoid_gate,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
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
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    is_adapt_gate: bool = True,
    is_local: bool = False,
    is_quant: bool = False,
    is_split_kv: bool = False,
    is_split_qo: bool = False,
    pack_gqa: bool = False,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for variable-length sequences that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [total_seqlen_q, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param alpha: Tensor of shape [total_seqlen_q, num_heads_q] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [total_seqlen_k, num_heads_kv] representing the sparsity pattern for keys/values.
    :param cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch_size + 1].
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_q: Maximum sequence length for queries.
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param is_causal: Whether to apply a causal mask.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [total_seqlen_q, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param is_adapt_gate: Whether to adapt the gate threshold based on sequence length.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided or will be computed from the input tensors.
    :param is_split_kv: Whether to enable split-KV for forward occupancy.
    :param is_split_qo: Whether to enable split-QO for backward occupancy.
    :param pack_gqa: Whether to pack grouped-query attention.
    :param seqused_q: Optional tensor of shape [total_seqlen_q] indicating the actual sequence lengths for queries. If provided, overrides cu_seqlens_q for masking.
    :param seqused_k: Optional tensor of shape [total_seqlen_k] indicating the actual sequence lengths for keys/values. If provided, overrides cu_seqlens_k for masking.
    :param out: Optional preallocated output tensor with shape [batch_size, seqlen_q, num_heads, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads, seqlen_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
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
        query_scale,
        key_scale,
        value_scale,
        window_sizes,
        softmax_threshold,
        gate_threshold,
        is_logsigmoid_gate,
        is_adapt_gate,
        is_local,
        is_quant,
        is_split_kv,
        is_split_qo,
        pack_gqa,
        seqused_q,
        seqused_k,
        out,
        lse,
        is_autotune,
        skip_checks,
        return_lse,
    )


def flash_gated_attn_varlen_with_kvcache_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    softmax_threshold: Optional[float] = None,
    gate_threshold: Optional[float] = None,
    is_logsigmoid_gate: bool = True,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    window_sizes: Optional[torch.Tensor] = None,
    is_local: bool = False,
    is_quant: bool = False,
    seqused_k: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_autotune: bool = False,
    skip_checks: bool = False,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Flash gated attention function for variable-length decoding with KV cache that computes the attention output and optionally the logsumexp.

    :param query: Query tensor of shape [batch_size, num_heads_q, head_dim].
    :param key: Key tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param value: Value tensor of shape [total_seqlen_k, num_heads_kv, head_dim].
    :param alpha: Tensor of shape [batch_size, num_heads_q] representing the sparsity pattern for queries.
    :param delta: Tensor of shape [total_seqlen_k, num_heads_kv] representing the sparsity pattern for keys/values.
    :param cu_seqlens_k: Cumulative sequence lengths for keys/values, shape [batch_size + 1].
    :param max_seqlen_k: Maximum sequence length for keys/values.
    :param softmax_scale: Optional scaling factor for the softmax. If None, defaults to 1/sqrt(head_dim).
    :param softmax_threshold: Optional threshold for the sparse softmax.
    :param gate_threshold: Optional threshold for the sparsity gate.
    :param is_logsigmoid_gate: Whether to use a log-sigmoid function for the sparsity gate. If False, uses a linear function.
    :param query_scale: Optional per-tensor scale for FP8 query dequantization.
    :param key_scale: Optional per-tensor scale for FP8 key dequantization.
    :param value_scale: Optional per-tensor scale for FP8 value dequantization.
    :param window_sizes: Optional window sizes tensor for flexible local attention. Must be shape [batch_size, 1, num_kv_heads, 2] with dtype int32. Use size=1 in any dimension to broadcast. If provided, is_local is automatically set to True.
    :param is_local: Whether to apply a local mask.
    :param is_quant: Whether the inputs are quantized. If True, query_scale, key_scale, and value_scale must be provided for dequantization.
    :param seqused_k: Optional tensor indicating the actual sequence lengths for keys/values.
    :param out: Optional preallocated output tensor with shape [batch_size, num_heads_q, head_dim].
    :param lse: Optional preallocated logsumexp tensor with shape [batch_size, num_heads_q].
    :param is_autotune: Force re-run Triton autotuner and overwrite cache. Default False uses cached config.
    :param skip_checks: Whether to skip input validation checks for faster performance.
    :param return_lse: Whether to return the logsumexp tensor for numerical stability analysis. If True, returns a tuple (out, lse). If False, returns only out.

    :returns: If return_lse is False, returns out with shape [batch_size, num_heads_q, head_dim]. If return_lse is True, returns a tuple (out, lse), where lse has shape [batch_size, num_heads_q].
    """

    if is_quant and (query_scale is None or key_scale is None or value_scale is None):
        query, query_scale = quant.quantize_fp8(query)
        key, key_scale = quant.quantize_fp8(key)
        value, value_scale = quant.quantize_fp8(value)

    if window_sizes is not None:
        is_local = True

    out, lse = _flash_gated_attn_varlen_decode(
        query=query,
        key=key,
        value=value,
        alpha=alpha,
        delta=delta,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        is_logsigmoid_gate=is_logsigmoid_gate,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        window_sizes=window_sizes,
        is_local=is_local,
        is_quant=is_quant,
        seqused_k=seqused_k,
        out=out,
        lse=lse,
        is_autotune=is_autotune,
        skip_checks=skip_checks,
    )

    if return_lse:
        return out, lse
    return out
