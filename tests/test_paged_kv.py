import pytest
import torch

from flash_sparse_attn import (
    flash_dense_attn_func,
    flash_dense_attn_with_kvcache_func,
    flash_gated_attn_func,
    flash_gated_attn_with_kvcache_func,
    flash_sparse_attn_func,
    flash_sparse_attn_with_kvcache_func,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _repeat_kv_heads(x: torch.Tensor, num_heads_q: int) -> torch.Tensor:
    num_heads_kv = x.shape[0]
    repeat = num_heads_q // num_heads_kv
    return x.repeat_interleave(repeat, dim=0)


def _make_paged_kv(
    batch_size: int,
    max_pages_per_seq: int,
    page_size: int,
    num_heads_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    seqlen_k = max_pages_per_seq * page_size
    k_logical = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v_logical = torch.randn_like(k_logical)

    # Use a nontrivial physical page order so the test verifies page_table mapping.
    page_table = torch.empty(
        batch_size, max_pages_per_seq, device=device, dtype=torch.int32
    )
    physical_order = (
        [2, 0, 3, 1]
        if batch_size * max_pages_per_seq == 4
        else list(reversed(range(batch_size * max_pages_per_seq)))
    )

    k_paged = torch.empty(
        batch_size * max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_paged = torch.empty_like(k_paged)
    for b in range(batch_size):
        for page_idx in range(max_pages_per_seq):
            physical_page = physical_order[b * max_pages_per_seq + page_idx]
            page_table[b, page_idx] = physical_page
            start = page_idx * page_size
            stop = start + page_size
            k_paged[physical_page] = k_logical[b, start:stop]
            v_paged[physical_page] = v_logical[b, start:stop]

    seqused_k = torch.tensor(
        [seqlen_k - 3, page_size + 5],
        device=device,
        dtype=torch.int32,
    )
    return k_logical, v_logical, k_paged, v_paged, page_table, seqused_k


def _make_paged_delta(
    batch_size: int,
    max_pages_per_seq: int,
    page_size: int,
    num_heads_kv: int,
    page_table: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqlen_k = max_pages_per_seq * page_size
    delta_logical = torch.randn(
        batch_size, seqlen_k, num_heads_kv, device=device, dtype=dtype
    )
    delta_paged = torch.empty(
        batch_size * max_pages_per_seq,
        page_size,
        num_heads_kv,
        device=device,
        dtype=dtype,
    )
    for b in range(batch_size):
        for page_idx in range(max_pages_per_seq):
            physical_page = int(page_table[b, page_idx].item())
            start = page_idx * page_size
            stop = start + page_size
            delta_paged[physical_page] = delta_logical[b, start:stop]
    return delta_logical, delta_paged


def _reference_dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seqlen_q, num_heads_q, _ = q.shape
    out = torch.empty_like(q)
    for b in range(batch_size):
        actual_k = int(seqused_k[b].item())
        q_b = q[b].transpose(0, 1).float()
        k_b = _repeat_kv_heads(k[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        v_b = _repeat_kv_heads(v[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        scores = torch.matmul(q_b, k_b.transpose(-1, -2)) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.matmul(probs, v_b).transpose(0, 1).to(q.dtype)
    return out


def _reference_dense_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, num_heads_q, _ = q.shape
    out = torch.empty_like(q)
    for b in range(batch_size):
        actual_k = int(seqused_k[b].item())
        k_b = _repeat_kv_heads(k[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        v_b = _repeat_kv_heads(v[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        scores = torch.matmul(q[b].float().unsqueeze(-2), k_b.transpose(-1, -2))
        probs = torch.softmax(scores.squeeze(-2) * softmax_scale, dim=-1)
        out[b] = torch.matmul(probs.unsqueeze(-2), v_b).squeeze(-2).to(q.dtype)
    return out


def _reference_gated_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    is_logsigmoid_gate: bool = True,
) -> torch.Tensor:
    batch_size, seqlen_q, num_heads_q, _ = q.shape
    out = torch.empty_like(q)
    for b in range(batch_size):
        actual_k = int(seqused_k[b].item())
        q_b = q[b].transpose(0, 1).float()
        k_b = _repeat_kv_heads(k[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        v_b = _repeat_kv_heads(v[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        alpha_b = alpha[b].transpose(0, 1).float()
        delta_b = _repeat_kv_heads(
            delta[b, :actual_k].transpose(0, 1).float(), num_heads_q
        )
        gate = alpha_b.unsqueeze(-1) * delta_b.unsqueeze(-2)
        if is_logsigmoid_gate:
            gate = torch.nn.functional.logsigmoid(gate)
        scores = (torch.matmul(q_b, k_b.transpose(-1, -2)) + gate) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.matmul(probs, v_b).transpose(0, 1).to(q.dtype)
    return out


def _reference_gated_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    is_logsigmoid_gate: bool = True,
) -> torch.Tensor:
    batch_size, num_heads_q, _ = q.shape
    out = torch.empty_like(q)
    for b in range(batch_size):
        actual_k = int(seqused_k[b].item())
        k_b = _repeat_kv_heads(k[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        v_b = _repeat_kv_heads(v[b, :actual_k].transpose(0, 1).float(), num_heads_q)
        delta_b = _repeat_kv_heads(
            delta[b, :actual_k].transpose(0, 1).float(), num_heads_q
        )
        gate = alpha[b].float().unsqueeze(-1) * delta_b
        if is_logsigmoid_gate:
            gate = torch.nn.functional.logsigmoid(gate)
        scores = torch.matmul(q[b].float().unsqueeze(-2), k_b.transpose(-1, -2))
        scores = (scores.squeeze(-2) + gate) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.matmul(probs.unsqueeze(-2), v_b).squeeze(-2).to(q.dtype)
    return out


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_dense_paged_kv_forward_matches_reference(page_size: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    seqlen_q = 37
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )

    out = flash_dense_attn_func(
        q,
        k_paged,
        v_paged,
        softmax_scale=softmax_scale,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_dense_fwd(q, k_logical, v_logical, seqused_k, softmax_scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_dense_paged_kv_decode_matches_reference(page_size: int) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )

    out = flash_dense_attn_with_kvcache_func(
        q,
        k_paged,
        v_paged,
        softmax_scale=softmax_scale,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_dense_decode(q, k_logical, v_logical, seqused_k, softmax_scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_sparse_paged_kv_forward_matches_reference(page_size: int) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    seqlen_q = 37
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )

    out = flash_sparse_attn_func(
        q,
        k_paged,
        v_paged,
        softmax_scale=softmax_scale,
        softmax_threshold=-128.0,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_dense_fwd(q, k_logical, v_logical, seqused_k, softmax_scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_sparse_paged_kv_decode_matches_reference(page_size: int) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )

    out = flash_sparse_attn_with_kvcache_func(
        q,
        k_paged,
        v_paged,
        softmax_scale=softmax_scale,
        softmax_threshold=-128.0,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_dense_decode(q, k_logical, v_logical, seqused_k, softmax_scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_gated_paged_kv_forward_matches_reference(page_size: int) -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    seqlen_q = 37
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    alpha = torch.randn(batch_size, seqlen_q, num_heads_q, device=device, dtype=dtype)
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )
    delta_logical, delta_paged = _make_paged_delta(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        page_table,
        dtype,
        device,
    )

    out = flash_gated_attn_func(
        q,
        k_paged,
        v_paged,
        alpha,
        delta_paged,
        softmax_scale=softmax_scale,
        softmax_threshold=-128.0,
        gate_threshold=-128.0,
        is_logsigmoid_gate=True,
        is_adapt_gate=False,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_gated_fwd(
        q,
        k_logical,
        v_logical,
        alpha,
        delta_logical,
        seqused_k,
        softmax_scale,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_gated_paged_kv_decode_matches_reference(page_size: int) -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    num_heads_q = 8
    num_heads_kv = 2
    head_dim = 64
    max_pages_per_seq = 2
    softmax_scale = head_dim**-0.5

    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=dtype)
    alpha = torch.randn(batch_size, num_heads_q, device=device, dtype=dtype)
    k_logical, v_logical, k_paged, v_paged, page_table, seqused_k = _make_paged_kv(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        head_dim,
        dtype,
        device,
    )
    delta_logical, delta_paged = _make_paged_delta(
        batch_size,
        max_pages_per_seq,
        page_size,
        num_heads_kv,
        page_table,
        dtype,
        device,
    )

    out = flash_gated_attn_with_kvcache_func(
        q,
        k_paged,
        v_paged,
        alpha,
        delta_paged,
        softmax_scale=softmax_scale,
        softmax_threshold=-128.0,
        gate_threshold=-128.0,
        is_logsigmoid_gate=True,
        page_table=page_table,
        seqused_k=seqused_k,
        is_autotune=False,
    )
    ref = _reference_gated_decode(
        q,
        k_logical,
        v_logical,
        alpha,
        delta_logical,
        seqused_k,
        softmax_scale,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-2, rtol=2.5e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
