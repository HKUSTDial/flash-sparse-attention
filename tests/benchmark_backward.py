from typing import List, Optional
import traceback

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.triton.interface import (
    flash_dense_attn_func,
    flash_sparse_attn_func,
    flash_gated_attn_func,
)
from test_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    format_tflops,
    generate_inputs,
    generate_train_configs,
)


def _bwd_flops(cfg: BenchmarkConfig) -> float:
    flops = (
        10.0
        * cfg.batch_size
        * cfg.num_heads
        * cfg.seqlen_q
        * cfg.seqlen_k
        * cfg.head_dim
    )
    if cfg.is_causal:
        flops *= 0.5
    return flops


def benchmark_triton_dense_backward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5

    out = flash_dense_attn_func(
        q,
        k,
        v,
        is_causal=cfg.is_causal,
        softmax_scale=softmax_scale,
        is_autotune=True,
        skip_checks=True,
    )
    dout = torch.randn_like(out)

    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        out.backward(dout, retain_graph=True)

    return do_bench(fn, warmup=20, rep=100)


def benchmark_triton_sparse_backward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 1.0

    out = flash_sparse_attn_func(
        q,
        k,
        v,
        is_causal=cfg.is_causal,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        is_autotune=True,
        skip_checks=True,
    )
    dout = torch.randn_like(out)

    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        out.backward(dout, retain_graph=True)

    return do_bench(fn, warmup=20, rep=100)


def benchmark_triton_gated_backward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    alpha = torch.randn(
        cfg.batch_size, cfg.seqlen_q, cfg.num_heads, device=device, dtype=dtype
    ).requires_grad_(True)
    delta = torch.randn(
        cfg.batch_size, cfg.seqlen_k, cfg.num_kv_heads, device=device, dtype=dtype
    ).requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 1.0
    gate_threshold = 1.0

    out = flash_gated_attn_func(
        q,
        k,
        v,
        alpha,
        delta,
        is_causal=cfg.is_causal,
        softmax_scale=softmax_scale,
        softmax_threshold=softmax_threshold,
        gate_threshold=gate_threshold,
        is_logsigmoid_gate=False,
        is_adapt_gate=False,
        is_autotune=True,
        skip_checks=True,
    )
    dout = torch.randn_like(out)

    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        alpha.grad = None
        delta.grad = None
        out.backward(dout, retain_graph=True)

    return do_bench(fn, warmup=20, rep=100)


def benchmark_fa_dense_backward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="random",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=cfg.is_causal,
            scale=softmax_scale,
            enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
        )
    dout = torch.randn_like(out)

    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out.backward(dout, retain_graph=True)

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cudnn_dense_backward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="random",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=cfg.is_causal,
            scale=softmax_scale,
            enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
        )
    dout = torch.randn_like(out)

    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out.backward(dout, retain_graph=True)

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    try:
        dense_ms = benchmark_triton_dense_backward(cfg)
        sparse_ms = benchmark_triton_sparse_backward(cfg)
        gated_ms = benchmark_triton_gated_backward(cfg)
        fa_ms = benchmark_fa_dense_backward(cfg)
        cudnn_ms = benchmark_cudnn_dense_backward(cfg)

        flops = _bwd_flops(cfg)
        dense_tflops = flops / dense_ms * 1e-9
        sparse_tflops = flops / sparse_ms * 1e-9
        gated_tflops = flops / gated_ms * 1e-9
        fa_tflops = flops / fa_ms * 1e-9 if fa_ms else None
        cudnn_tflops = flops / cudnn_ms * 1e-9 if cudnn_ms else None

        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=dense_ms,
            triton_sparse_ms=sparse_ms,
            triton_gated_ms=gated_ms,
            fa_dense_ms=fa_ms,
            cudnn_dense_ms=cudnn_ms,
            triton_dense_tflops=dense_tflops,
            triton_sparse_tflops=sparse_tflops,
            triton_gated_tflops=gated_tflops,
            fa_dense_tflops=fa_tflops,
            cudnn_dense_tflops=cudnn_tflops,
        )
    except Exception as exc:
        full_error = f"{exc}\n{traceback.format_exc()}"
        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=None,
            triton_sparse_ms=None,
            triton_gated_ms=None,
            fa_dense_ms=None,
            cudnn_dense_ms=None,
            triton_dense_tflops=None,
            triton_sparse_tflops=None,
            triton_gated_tflops=None,
            fa_dense_tflops=None,
            cudnn_dense_tflops=None,
            error_message=full_error,
        )


def print_results(results: List[BenchmarkResult]) -> None:
    ok = [r for r in results if r.error_message is None]
    if not ok:
        print("No successful benchmark results.")
        for r in results:
            print(f"Failed: {r.config}\n{r.error_message}")
        return

    rows = []
    for r in ok:
        rows.append(
            [
                r.config.batch_size,
                r.config.num_heads,
                r.config.head_dim,
                r.config.seqlen_q,
                r.config.seqlen_k,
                "causal" if r.config.is_causal else "non-causal",
                format_tflops(r.triton_dense_tflops),
                format_tflops(r.triton_sparse_tflops),
                format_tflops(r.triton_gated_tflops),
                format_tflops(r.fa_dense_tflops),
                format_tflops(r.cudnn_dense_tflops),
            ]
        )

    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    headers = [
        "B",
        "H",
        "D",
        "Seqlen_q",
        "Seqlen_k",
        "Mode",
        "Triton Dense TFLOPS",
        "Triton Sparse TFLOPS",
        "Triton Gated TFLOPS",
        "FA Dense TFLOPS",
        "cuDNN Dense TFLOPS",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)

    batch_sizes = [1]
    num_heads = [16]
    num_kv_heads = [8]
    seqlens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    head_dims = [128]
    is_causal = True

    configs = generate_train_configs(
        batch_sizes,
        num_heads,
        seqlens,
        head_dims,
        is_causal,
        num_kv_heads=num_kv_heads,
    )

    results: List[BenchmarkResult] = []
    print(
        f"Running {len(configs)} backward benchmark configurations on {device_name}..."
    )
    for cfg in tqdm(configs, desc="Benchmarking attn backward"):
        results.append(run_benchmark(cfg))

    print_results(results)


if __name__ == "__main__":
    main()
