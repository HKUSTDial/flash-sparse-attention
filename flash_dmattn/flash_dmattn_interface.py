# Copyright (c) 2025, Jingze Shi.

from typing import Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import os

import flash_dmattn_cuda as flash_dmattn_gpu # type: ignore


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
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
if torch.__version__ >= "2.4.0":
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
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_dmattn_gpu.fwd(
        q,
        k,
        v,
        mask,
        bias,
        None,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        return_softmax,
        None,
    )
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_dmattn::_flash_dmattn_forward")
def _flash_dmattn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


_wrapped_flash_dmattn_forward = _flash_dmattn_forward


@_torch_custom_op_wrapper("flash_dmattn::_flash_dmattn_varlen_forward", mutates_args=(), device_types="cuda")
def _flash_dmattn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float = 0.0,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_dmattn_gpu.varlen_fwd(
        q,
        k,
        v,
        mask,
        bias,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        is_causal,
        softcap,
        return_softmax,
        None,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_dmattn::_flash_dmattn_varlen_forward")
def _flash_dmattn_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float = 0.0,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    paged_kv = block_table is not None
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape
    
    out = torch.empty_like(q)
    softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


_wrapped_flash_dmattn_varlen_forward = _flash_dmattn_varlen_forward


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
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv, dbias are allocated by us so they should already be contiguous
    dout, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, q, k, v, mask, bias, out)]
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
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        None,
        rng_state,
    )
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
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dout, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, q, k, v, mask, bias, out)]
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


@_torch_custom_op_wrapper("flash_dmattn::_flash_dmattn_varlen_backward", mutates_args=("dq", "dk", "dv", "dbias"), device_types="cuda")
def _flash_dmattn_varlen_backward(
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
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, q, k, v, mask, bias, out)]
    (
        dq,
        dk,
        dv,
        dbias,
        softmax_d,
    ) = flash_dmattn_gpu.varlen_bwd(
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
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        is_causal,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return softmax_d


@_torch_register_fake_wrapper("flash_dmattn::_flash_dmattn_varlen_backward")
def _flash_dmattn_varlen_backward_fake(
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
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    dout, q, k, v, mask, bias, out = [maybe_contiguous(x) for x in (dout, q, k, v, mask, bias, out)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if dbias is None:
        dbias = torch.empty_like(bias)
    softmax_d = torch.empty((num_heads, total_q + 128 * batch_size), device=q.device, dtype=torch.float32)
    
    return softmax_d


_wrapped_flash_dmattn_varlen_backward = _flash_dmattn_varlen_backward


class FlashDMAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        # qkv is expected to be of shape (batch_size, seqlen, 3, num_heads, head_size)
        batch_size, seqlen, _, num_heads, head_size = qkv.shape
        is_grad = is_grad_enabled and qkv.requires_grad
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen, seqlen), dtype=qkv.dtype, device=qkv.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen, seqlen), dtype=qkv.dtype, device=qkv.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
        if is_causal is None:
            is_causal = False
        if softcap is None:
            softcap = 0.0
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if deterministic is None:
            deterministic = True
        if return_softmax is None:
            return_softmax = False

        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state =  _wrapped_flash_dmattn_forward(
            q,
            k,
            v,
            mask,
            bias,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, mask, bias, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])

        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dbias = torch.empty_like(bias, dtype=bias.dtype, device=bias.device)

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
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            dbias,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.is_causal,
            ctx.softcap,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


class FlashDMAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        qkv: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        # qkv is expected to be of shape (batch_size, seqlen, 3, num_heads, head_size)
        batch_size, seqlen, _, num_heads, head_size = qkv.shape
        is_grad = is_grad_enabled and qkv.requires_grad
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen, seqlen), dtype=qkv.dtype, device=qkv.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen, seqlen), dtype=qkv.dtype, device=qkv.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if is_causal is None:
            is_causal = False
        if softcap is None:
            softcap = 0.0
        if deterministic is None:
            deterministic = True
        if return_softmax is None:
            return_softmax = False
        
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_dmattn_varlen_forward(
            q,
            k,
            v,
            mask,
            bias,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, mask, bias, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])

        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dbias = torch.empty_like(bias, dtype=bias.dtype, device=bias.device)

        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        _wrapped_flash_dmattn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            mask,
            bias,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            dbias,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashDMAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        # q is expected to be of shape (batch_size, seqlen_q, num_heads, head_size)
        # kv is expected to be of shape (batch_size, seqlen_k, 2, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, kv]
        )
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen_q, seqlen_q), dtype=q.dtype, device=q.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen_q, seqlen_q), dtype=q.dtype, device=q.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
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

        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_dmattn_forward(
            q,
            k,
            v,
            mask,
            bias,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, mask, bias, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any
    ):
        q, k, v, mask, bias, out, softmax_lse, rng_state = ctx.saved_tensors
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])

        dq = torch.empty_like(q)
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dbias = torch.empty_like(bias, dtype=bias.dtype, device=bias.device)

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
            dkv[:, :, 0],
            dkv[:, :, 1],
            dbias,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashDMAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        # q is expected to be of shape (batch_size, seqlen_q, num_heads, head_size)
        # kv is expected to be of shape (batch_size, seqlen_k, 2, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, kv]
        )
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen_q, seqlen_q), dtype=q.dtype, device=q.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen_q, seqlen_q), dtype=q.dtype, device=q.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
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

        k, v = kv[:, 0].detach(), kv[:, 1].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_dmattn_varlen_forward(
            q,
            k,
            v,
            mask,
            bias,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )

        if is_grad:
            ctx.save_for_backward(
                q, k, v, mask, bias, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])

        dq = torch.empty_like(q)
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dbias = torch.empty_like(bias, dtype=bias.dtype, device=bias.device)

        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    
        _wrapped_flash_dmattn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            mask,
            bias,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            dbias,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.is_causal,
            ctx.softcap,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashDMAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        is_grad_enabled: bool = True,
    ):
        # q, k, v are expected to be of shape (batch_size, seqlen, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        seqlen_k = k.shape[1]
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
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

        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_dmattn_forward(
            q,
            k,
            v,
            mask,
            bias,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, mask, bias, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv, dbias = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bias)

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
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.is_causal,
            ctx.softcap,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FlashDMAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: Optional[float],
        softmax_scale: Optional[float],
        is_causal: Optional[bool],
        softcap: Optional[float],
        deterministic: Optional[bool],
        return_softmax: Optional[bool],
        block_table: Optional[torch.Tensor] = None,
        is_grad_enabled: bool = True,
    ):
        # q, k, v are expected to be of shape (batch_size, seqlen_q, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        seqlen_k = k.shape[1]
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if mask is None:
            mask = torch.ones((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device)
        if bias is None:
            bias = torch.zeros((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device)
        if dropout_p is None:
            dropout_p = 0.0
        if dropout_p < 0.0 or dropout_p > 1.0:
            raise ValueError(f"Invalid dropout_p: {dropout_p}. It should be in [0, 1].")
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

        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_dmattn_varlen_forward(
            q,
            k,
            v,
            mask,
            bias,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            is_causal=is_causal,
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )

        if is_grad:
            ctx.save_for_backward(
                q, k, v, mask, bias, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.is_causal = is_causal
            ctx.softcap = softcap
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        dout: torch.Tensor,
        *args: Any,
    ):
        q, k, v, mask, bias, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv, dbias = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bias)

        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])

        _wrapped_flash_dmattn_varlen_backward(
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
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.is_causal,
            ctx.softcap,
            ctx.deterministic,
            rng_state=rng_state,
        )

        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_dmattn_qkvpacked_func(
    qkv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_dmattn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_dmattn_kvpacked_func and flash_dmattn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        attn_mask: (batch_size, nheads, seqlen, seqlen). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen, seqlen). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
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
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnQKVPackedFunc.apply(
        qkv,
        attn_mask,
        attn_bias,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_dmattn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_dmattn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

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
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen_q, seqlen_k). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
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
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnKVPackedFunc.apply(
        q,
        kv,
        attn_mask,
        attn_bias,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_dmattn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

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
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen_q, seqlen_k). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
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
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnFunc.apply(
        q,
        k,
        v,
        attn_mask,
        attn_bias,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_dmattn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_dmattn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_dmattn_varlen_kvpacked_func and flash_dmattn_varlen_func.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen_q, seqlen_k). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnVarlenQKVPackedFunc.apply(
        qkv,
        attn_mask,
        attn_bias,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_dmattn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_dmattn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen_q, seqlen_k). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        attn_mask,
        attn_bias,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_dmattn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    dropout_p: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    return_attn_probs: Optional[bool] = None,
    block_table: Optional[torch.Tensor] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

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
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k). Attention mask to apply to the attention scores.
            If None, no mask is applied.
        attn_bias: (batch_size, nheads, seqlen_q, seqlen_k). Attention Bias to add to the attention scores.
            If None, no bias is applied.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        is_causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashDMAttnVarlenFunc.apply(
        q,
        k,
        v,
        attn_mask,
        attn_bias,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        is_causal,
        softcap,
        deterministic,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )