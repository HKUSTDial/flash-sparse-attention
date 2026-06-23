from typing import Literal, Optional, Sequence

import pytest
import torch

from flash_sparse_attn.ops.triton.interface import (
    flash_dense_attn_varlen_with_kvcache_func,
    flash_dense_attn_with_kvcache_func,
    flash_gated_attn_varlen_with_kvcache_func,
    flash_gated_attn_with_kvcache_func,
    flash_sparse_attn_varlen_with_kvcache_func,
    flash_sparse_attn_with_kvcache_func,
)
from test_utils import (
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _assert_close,
    make_cu_seqlens,
    set_seed,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

KernelType = Literal["dense", "sparse", "gated"]


def _repeat_kv_heads(x: torch.Tensor, num_heads_q: int) -> torch.Tensor:
    if x.shape[1] == num_heads_q:
        return x
    repeat = num_heads_q // x.shape[1]
    return torch.repeat_interleave(x, repeats=repeat, dim=1)


def _make_topk_indices(
    lengths: Sequence[int],
    topk_seqlen_k: int,
    device: torch.device,
    *,
    include_invalid: bool,
) -> torch.Tensor:
    indices = torch.empty(len(lengths), topk_seqlen_k, device=device, dtype=torch.int32)
    for batch_idx, seqlen_k in enumerate(lengths):
        valid_count = min(seqlen_k, topk_seqlen_k)
        perm = torch.randperm(seqlen_k, device=device, dtype=torch.int32)
        indices[batch_idx, :valid_count] = perm[:valid_count]
        if valid_count < topk_seqlen_k:
            indices[batch_idx, valid_count:] = -1
        if include_invalid:
            indices[batch_idx, batch_idx::23] = -1
    return indices


def _gather_topk_3d(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).to(torch.long)
    gathered = torch.stack(
        [
            x[batch_idx].index_select(0, safe_indices[batch_idx])
            for batch_idx in range(x.shape[0])
        ],
        dim=0,
    )
    return gathered.masked_fill(~valid[:, :, None, None], 0.0)


def _gather_topk_2d(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).to(torch.long)
    gathered = torch.stack(
        [
            x[batch_idx].index_select(0, safe_indices[batch_idx])
            for batch_idx in range(x.shape[0])
        ],
        dim=0,
    )
    return gathered.masked_fill(~valid[:, :, None], 0.0)


def _pack_varlen_to_batch(
    x: torch.Tensor,
    lengths: Sequence[int],
) -> torch.Tensor:
    padded = []
    offset = 0
    for seqlen_k in lengths:
        padded.append(x[offset : offset + seqlen_k])
        offset += seqlen_k
    return torch.nn.utils.rnn.pad_sequence(padded, batch_first=True)


def _reference_topk_decode(
    kind: KernelType,
    q: torch.Tensor,
    k_topk: torch.Tensor,
    v_topk: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    alpha: Optional[torch.Tensor] = None,
    delta_topk: Optional[torch.Tensor] = None,
    is_logsigmoid_gate: bool = True,
) -> torch.Tensor:
    valid = gather_kv_indices >= 0
    num_heads_q = q.shape[1]

    qh = q.float().unsqueeze(2)
    kh = _repeat_kv_heads(k_topk.transpose(1, 2).float(), num_heads_q)
    vh = _repeat_kv_heads(v_topk.transpose(1, 2).float(), num_heads_q)

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))

    if kind == "gated":
        delta_h = _repeat_kv_heads(delta_topk.transpose(1, 2).float(), num_heads_q)
        raw_gate = alpha.float().unsqueeze(-1).unsqueeze(-1) * delta_h.unsqueeze(2)
        gate = (
            torch.nn.functional.logsigmoid(raw_gate) if is_logsigmoid_gate else raw_gate
        )
        scores = scores + gate * softmax_scale

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, vh).squeeze(2).to(q.dtype)


def _run_base_topk_case(
    kind: KernelType,
    dtype: torch.dtype,
    topk_seqlen_k: int,
    is_local: bool,
) -> None:
    device = torch.device("cuda")
    batch_size = 2
    seqlen_k = 768
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    softmax_scale = head_dim**-0.5
    threshold = -128.0

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    alpha = torch.randn(batch_size, num_heads_q, device=device, dtype=dtype)
    delta = torch.randn(batch_size, seqlen_k, num_heads_kv, device=device, dtype=dtype)
    gather_kv_indices = _make_topk_indices(
        [seqlen_k] * batch_size,
        topk_seqlen_k,
        device,
        include_invalid=True,
    )

    k_topk = _gather_topk_3d(k, gather_kv_indices)
    v_topk = _gather_topk_3d(v, gather_kv_indices)
    delta_topk = _gather_topk_2d(delta, gather_kv_indices)

    if kind == "dense":
        out = flash_dense_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_local=is_local,
            gather_kv_indices=gather_kv_indices,
        )
    elif kind == "sparse":
        out = flash_sparse_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            gather_kv_indices=gather_kv_indices,
        )
    else:
        out = flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=True,
            is_local=is_local,
            gather_kv_indices=gather_kv_indices,
        )

    ref = _reference_topk_decode(
        kind,
        q,
        k_topk,
        v_topk,
        gather_kv_indices,
        softmax_scale,
        alpha=alpha,
        delta_topk=delta_topk,
    )
    _assert_close(
        name=f"{kind}-base-topk-decode",
        got=out,
        ref=ref,
        rtol=_DEFAULT_RTOL[kind],
        atol=_DEFAULT_ATOL[kind],
    )


def _run_varlen_topk_case(kind: KernelType, dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    lengths = [384, 640, 768]
    topk_seqlen_k = 512
    batch_size = len(lengths)
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    softmax_scale = head_dim**-0.5
    threshold = -128.0

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k = torch.cat(
        [
            torch.randn(seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seqlen_k in lengths
        ],
        dim=0,
    )
    v = torch.cat(
        [
            torch.randn(seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seqlen_k in lengths
        ],
        dim=0,
    )
    alpha = torch.randn(batch_size, num_heads_q, device=device, dtype=dtype)
    delta = torch.cat(
        [
            torch.randn(seqlen_k, num_heads_kv, device=device, dtype=dtype)
            for seqlen_k in lengths
        ],
        dim=0,
    )
    cu_seqlens_k = make_cu_seqlens(lengths, device)
    gather_kv_indices = _make_topk_indices(
        lengths,
        topk_seqlen_k,
        device,
        include_invalid=True,
    )

    k_batch = _pack_varlen_to_batch(k, lengths)
    v_batch = _pack_varlen_to_batch(v, lengths)
    delta_batch = _pack_varlen_to_batch(delta, lengths)
    k_topk = _gather_topk_3d(k_batch, gather_kv_indices)
    v_topk = _gather_topk_3d(v_batch, gather_kv_indices)
    delta_topk = _gather_topk_2d(delta_batch, gather_kv_indices)

    if kind == "dense":
        out = flash_dense_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lengths),
            softmax_scale=softmax_scale,
            gather_kv_indices=gather_kv_indices,
        )
    elif kind == "sparse":
        out = flash_sparse_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lengths),
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gather_kv_indices=gather_kv_indices,
        )
    else:
        out = flash_gated_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lengths),
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=True,
            gather_kv_indices=gather_kv_indices,
        )

    ref = _reference_topk_decode(
        kind,
        q,
        k_topk,
        v_topk,
        gather_kv_indices,
        softmax_scale,
        alpha=alpha,
        delta_topk=delta_topk,
    )
    _assert_close(
        name=f"{kind}-varlen-topk-decode",
        got=out,
        ref=ref,
        rtol=_DEFAULT_RTOL[kind],
        atol=_DEFAULT_ATOL[kind],
    )


def _run_contiguous_topk_matches_regular_case(kind: KernelType) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    seqlen_k = 512
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    softmax_scale = head_dim**-0.5
    threshold = -128.0

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    alpha = torch.randn(batch_size, num_heads_q, device=device, dtype=dtype)
    delta = torch.randn(batch_size, seqlen_k, num_heads_kv, device=device, dtype=dtype)
    gather_kv_indices = (
        torch.arange(seqlen_k, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    if kind == "dense":
        out = flash_dense_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            gather_kv_indices=gather_kv_indices,
        )
        ref = flash_dense_attn_with_kvcache_func(q, k, v, softmax_scale=softmax_scale)
    elif kind == "sparse":
        out = flash_sparse_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gather_kv_indices=gather_kv_indices,
        )
        ref = flash_sparse_attn_with_kvcache_func(
            q, k, v, softmax_scale=softmax_scale, softmax_threshold=threshold
        )
    else:
        out = flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            gather_kv_indices=gather_kv_indices,
        )
        ref = flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
        )

    _assert_close(
        name=f"{kind}-contiguous-topk-decode",
        got=out,
        ref=ref,
        rtol=_DEFAULT_RTOL[kind],
        atol=_DEFAULT_ATOL[kind],
    )


@pytest.mark.parametrize("kind", ["dense", "sparse", "gated"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk_seqlen_k", [256, 512])
@pytest.mark.parametrize("is_local", [False, True])
def test_topk_gather_base_decode_matches_reference(
    kind: KernelType,
    dtype: torch.dtype,
    topk_seqlen_k: int,
    is_local: bool,
) -> None:
    set_seed(0)
    _run_base_topk_case(kind, dtype, topk_seqlen_k, is_local)


@pytest.mark.parametrize("kind", ["dense", "sparse", "gated"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_topk_gather_varlen_decode_matches_reference(
    kind: KernelType,
    dtype: torch.dtype,
) -> None:
    set_seed(1)
    _run_varlen_topk_case(kind, dtype)


@pytest.mark.parametrize("kind", ["dense", "sparse", "gated"])
def test_contiguous_topk_matches_regular_decode(kind: KernelType) -> None:
    set_seed(2)
    _run_contiguous_topk_matches_regular_case(kind)
