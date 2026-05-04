"""
Correctness and performance test for the SM100 Gluon dense forward attention kernel.

Requires: Blackwell GPU (compute capability >= 10.0)
Run: python -m pytest tests/test_gluon_sm100_dense_fwd.py -s -q
"""

import pytest
import torch
import time

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (SM100+)",
)


def reference_attention(q, k, v, softmax_scale, is_causal):
    """PyTorch reference implementation for correctness comparison."""
    # q: [B, S_q, H_q, D] -> [B, H_q, S_q, D]
    q_t = q.transpose(1, 2).float()
    k_t = k.transpose(1, 2).float()
    v_t = v.transpose(1, 2).float()

    # GQA: repeat KV heads
    if q_t.shape[1] != k_t.shape[1]:
        repeat = q_t.shape[1] // k_t.shape[1]
        k_t = torch.repeat_interleave(k_t, repeats=repeat, dim=1)
        v_t = torch.repeat_interleave(v_t, repeats=repeat, dim=1)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale

    if is_causal:
        S_q, S_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1,
        )
        scores.masked_fill_(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, v_t)
    return out.transpose(1, 2).contiguous().to(q.dtype)


# ===-----------------------------------------------------------------------===#
# Correctness Tests
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,num_heads_q,num_heads_kv,head_dim",
    [
        (2, 128, 128, 8, 8, 64),  # basic MHA
        (2, 256, 256, 8, 8, 128),  # larger head_dim
        (1, 512, 512, 16, 4, 64),  # GQA
        (2, 128, 256, 8, 8, 64),  # seqlen_q != seqlen_k
        (4, 64, 64, 4, 4, 64),  # small
    ],
)
def test_correctness(
    batch_size,
    seqlen_q,
    seqlen_k,
    num_heads_q,
    num_heads_kv,
    head_dim,
    is_causal,
    dtype,
):
    from flash_sparse_attn.ops.gluon.flash_dense_fwd import (
        _flash_dense_attn_base_forward,
    )

    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )

    out, lse, softmax_scale = _flash_dense_attn_base_forward(
        q,
        k,
        v,
        is_causal=is_causal,
    )
    out_ref = reference_attention(q, k, v, softmax_scale, is_causal)

    # bf16/fp16 tolerance
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 1e-2

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


# ===-----------------------------------------------------------------------===#
# Performance Benchmark
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen,num_heads,head_dim",
    [
        (4, 1024, 32, 64),
        (4, 2048, 32, 64),
        (4, 4096, 32, 128),
        (2, 8192, 32, 128),
    ],
)
def test_performance(batch_size, seqlen, num_heads, head_dim, is_causal):
    """Benchmark SM100 kernel vs PyTorch SDPA. Prints TFLOPS, does not assert."""
    from flash_sparse_attn.ops.gluon.flash_dense_fwd import (
        _flash_dense_attn_base_forward,
    )

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _flash_dense_attn_base_forward(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()

    # Benchmark gluon kernel
    start = time.perf_counter()
    N_ITER = 20
    for _ in range(N_ITER):
        _flash_dense_attn_base_forward(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    gluon_ms = (time.perf_counter() - start) / N_ITER * 1000

    # Benchmark PyTorch SDPA
    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k.transpose(1, 2).contiguous()
    v_sdpa = v.transpose(1, 2).contiguous()
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=is_causal
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_ITER):
        torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=is_causal
        )
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - start) / N_ITER * 1000

    # Compute TFLOPS
    flops = 4.0 * batch_size * num_heads * seqlen * seqlen * head_dim
    if is_causal:
        flops *= 0.5
    gluon_tflops = flops / (gluon_ms * 1e-3) * 1e-12
    sdpa_tflops = flops / (sdpa_ms * 1e-3) * 1e-12

    print(
        f"\n  B={batch_size} S={seqlen} H={num_heads} D={head_dim} causal={is_causal}"
        f"\n    Gluon SM100: {gluon_ms:.3f} ms ({gluon_tflops:.1f} TFLOPS)"
        f"\n    PyTorch SDPA: {sdpa_ms:.3f} ms ({sdpa_tflops:.1f} TFLOPS)"
        f"\n    Speedup: {sdpa_ms / gluon_ms:.2f}x"
    )
