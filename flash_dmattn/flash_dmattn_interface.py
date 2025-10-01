# Copyright (c) 2025, Jingze Shi.

from typing import Optional, Tuple, Any
from packaging import version
import torch

import flash_dmattn_cuda as flash_dmattn_gpu # type: ignore


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _sanitize_tensors(*tensors: Optional[torch.Tensor], nan: float = 0.0, posinf: float = 1e6, neginf: float = -1e6) -> None:
    for t in tensors:
        if t is not None and isinstance(t, torch.Tensor):
            torch.nan_to_num_(t, nan=nan, posinf=posinf, neginf=neginf)


def _get_block_size_n(device, head_dim, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (is_causal) else 32
        else:
            return 64
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if version.parse(torch.__version__) >= version.parse("2.4.0"):
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("flash_dmattn::_flash_dmattn_forward", mutates_args=(), device_types="cuda")
def _flash_dmattn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, mask, bias = [maybe_contiguous(x) for x in (q, k, v, mask, bias)]
    out, softmax_lse, S_dmask = flash_dmattn_gpu.fwd(
        q,
        k,
        v,
        mask,
        bias,
        None,
        softmax_scale,
        is_causal,
        softcap,
        return_softmax,
    )
    # _sanitize_tensors(out, nan=0.0, posinf=torch.finfo(out.dtype).max, neginf=torch.finfo(out.dtype).min)
    return out, softmax_lse, S_dmask


@_torch_register_fake_wrapper("flash_dmattn::_flash_dmattn_forward")
def _flash_dmattn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, mask, bias = [maybe_contiguous(x) for x in (q, k, v, mask, bias)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)

    return out, softmax_lse, p


_wrapped_flash_dmattn_forward = _flash_dmattn_forward


@_torch_custom_op_wrapper("flash_dmattn::_flash_dmattn_backward", mutates_args=("dq", "dk", "dv", "dbias"), device_types="cuda")
def _flash_dmattn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
) -> torch.Tensor:
    dout, dbias, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, dbias, q, k, v, mask, bias, out)]
    (
        dq,
        dk,
        dv,
        dbias,
        softmax_d,
    ) = flash_dmattn_gpu.bwd(
        dout,
        q,
        k,
        v,
        mask,
        bias,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dbias,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
    )
    # _sanitize_tensors(dq, dk, dv, dbias, nan=0.0, posinf=torch.finfo(dq.dtype).max, neginf=torch.finfo(dq.dtype).min)
    return softmax_d


@_torch_register_fake_wrapper("flash_dmattn::_flash_dmattn_backward")
def _flash_dmattn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
) -> torch.Tensor:
    dout, dbias, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, dbias, q, k, v, mask, bias, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if dbias is None:
        dbias = torch.empty_like(bias)
    batch_size, seqlen_q, num_heads, _ = q.shape
    softmax_d = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32)
    
    return softmax_d


_wrapped_flash_dmattn_backward = _flash_dmattn_backward


class FlashDMAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if is_causal is None:
            is_causal = False
        if softcap is None:
            softcap = 0.0
        if deterministic is None:
            deterministic = True
        if return_softmax is None:
            return_softmax = False

        # Padding to multiple of 8 for 16-bit memory allocations
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        seqlen_k_og = k.shape[1]
        if seqlen_k_og % 8 != 0:
            k = torch.nn.functional.pad(k, [0, 0, 0, 0, 0, 8 - seqlen_k_og % 8])
            v = torch.nn.functional.pad(v, [0, 0, 0, 0, 0, 8 - seqlen_k_og % 8])
            if mask is not None:
                mask = torch.nn.functional.pad(mask, [0, 8 - seqlen_k_og % 8], value=False)
            if bias is not None:
                bias = torch.nn.functional.pad(bias, [0, 8 - seqlen_k_og % 8], value=0.0)

        out_padded, softmax_lse, S_dmask = _wrapped_flash_dmattn_forward(
            q,
            k,
            v,
            mask,
            bias,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, mask, bias, out_padded, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.seqlen_k_og = seqlen_k_og

        out = out_padded[..., :head_size_og]

        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        dbias = torch.zeros_like(bias) if bias is not None else None

        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        _wrapped_flash_dmattn_backward(
            dout_padded,
            q,
            k,
            v,
            mask,
            bias,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dbias,
            ctx.softmax_scale,
            ctx.is_causal,
            ctx.softcap,
            ctx.deterministic,
        )

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        if ctx.seqlen_k_og % 8 != 0:
            dk = dk[:, : ctx.seqlen_k_og, :, :]
            dv = dv[:, : ctx.seqlen_k_og, :, :]
            if dbias is not None:
                dbias = dbias[..., : ctx.seqlen_k_og]

        return dq, dk, dv, None, dbias, None, None, None, None, None, None


def flash_dmattn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """
    Supports multi-query attention and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    Similarity, also supports attn_mask and attn_bias with head dimension of 1, nheads_k or nheads for MQA/GQA.
    For example, if Q has 6 heads, K, V have 2 heads, then attn_mask and attn_bias can have head dimension
    of 1, 2 or 6. If it is 1, all heads use the same mask/bias; if it is 2, head 0, 1, 2 of Q use head 0
    of mask/bias, head 3, 4, 5 of Q use head 1 of mask/bias. If it is 6, each head uses its own mask/bias.

    If is_causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    Arguments:
        query: torch.Tensor. The query tensor of shape (batch_size, seqlen, nheads, headdim)
        key: torch.Tensor. The key tensor of shape (batch_size, seqlen, nheads_k, headdim)
        value: torch.Tensor. The value tensor of shape (batch_size, seqlen, nheads_k, headdim)
        attn_mask: torch.Tensor, optional. The attention mask boolean tensor of
            shape (batch_size, nheads, seqlen_q, seqlen_k) to apply to the attention scores.
            Also supports shape (batch_size, nheads_k, seqlen_q, seqlen_k) or
            (batch_size, 1, seqlen_q, seqlen_k) for MQA/GQA.
            If None, no mask is applied.
        attn_bias: torch.Tensor, optional. The attention bias float tensor of
            shape (batch_size, nheads, seqlen_q, seqlen_k) to add to the attention scores.
            Also supports shape (batch_size, nheads_k, seqlen_q, seqlen_k) or
            (batch_size, 1, seqlen_q, seqlen_k) for MQA/GQA.
            If None, no bias is applied.
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling).
    """
    return FlashDMAttnFunc.apply(
        query,
        key,
        value,
        attn_mask,
        attn_bias,
        scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
