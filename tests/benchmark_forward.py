from typing import List, Optional

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.triton.interface import (
    flash_attn_func,
    flash_sparse_attn_func,
)
from test_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    format_tflops,
    generate_train_configs,
)


def _fwd_flops(cfg: BenchmarkConfig) -> float:
    flops = (
        4.0
        * cfg.batch_size
        * cfg.num_heads
        * cfg.seqlen_q
        * cfg.seqlen_k
        * cfg.head_dim
    )
    if cfg.is_causal:
        flops *= 0.5
    return flops


def benchmark_triton_dense_forward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q = torch.randn(
        cfg.batch_size,
        cfg.seqlen_q,
        cfg.num_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        cfg.batch_size,
        cfg.seqlen_k,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        cfg.batch_size,
        cfg.seqlen_k,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=cfg.is_causal,
            window_size=(None, None),
        )

    return do_bench(fn, warmup=20, rep=100)


def benchmark_triton_sparse_forward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q = torch.randn(
        cfg.batch_size,
        cfg.seqlen_q,
        cfg.num_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        cfg.batch_size,
        cfg.seqlen_k,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        cfg.batch_size,
        cfg.seqlen_k,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    alpha = torch.randn(
        cfg.batch_size, cfg.num_heads, cfg.seqlen_q, device=device, dtype=dtype
    )
    delta = torch.randn(
        cfg.batch_size, cfg.num_kv_heads, cfg.seqlen_k, device=device, dtype=dtype
    )
    softmax_scale = cfg.head_dim**-0.5
    gate_scale = (cfg.seqlen_k + 1) ** -1

    def fn():
        flash_sparse_attn_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            gate_scale=gate_scale,
            is_causal=cfg.is_causal,
            is_logsigmoid_gate=True,
            is_adapt_gate=False,
            window_size=(None, None),
        )

    return do_bench(fn, warmup=20, rep=100)


def benchmark_fa_dense_forward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q = torch.randn(
        cfg.batch_size,
        cfg.num_heads,
        cfg.seqlen_q,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        cfg.batch_size,
        cfg.num_kv_heads,
        cfg.seqlen_k,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        cfg.batch_size,
        cfg.num_kv_heads,
        cfg.seqlen_k,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
            )

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cudnn_dense_forward(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q = torch.randn(
        cfg.batch_size,
        cfg.num_heads,
        cfg.seqlen_q,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        cfg.batch_size,
        cfg.num_kv_heads,
        cfg.seqlen_k,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        cfg.batch_size,
        cfg.num_kv_heads,
        cfg.seqlen_k,
        cfg.head_dim,
        device=device,
        dtype=dtype,
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
            )

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    try:
        triton_dense_ms = benchmark_triton_dense_forward(cfg)
        triton_sparse_ms = benchmark_triton_sparse_forward(cfg)
        fa_dense_ms = benchmark_fa_dense_forward(cfg)
        cudnn_dense_ms = benchmark_cudnn_dense_forward(cfg)

        flops = _fwd_flops(cfg)
        triton_dense_tflops = flops / triton_dense_ms * 1e-9
        triton_sparse_tflops = flops / triton_sparse_ms * 1e-9
        fa_dense_tflops = flops / fa_dense_ms * 1e-9 if fa_dense_ms else None
        cudnn_dense_tflops = flops / cudnn_dense_ms * 1e-9 if cudnn_dense_ms else None

        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=triton_dense_ms,
            triton_sparse_ms=triton_sparse_ms,
            fa_dense_ms=fa_dense_ms,
            cudnn_dense_ms=cudnn_dense_ms,
            triton_dense_tflops=triton_dense_tflops,
            triton_sparse_tflops=triton_sparse_tflops,
            fa_dense_tflops=fa_dense_tflops,
            cudnn_dense_tflops=cudnn_dense_tflops,
        )
    except Exception as exc:
        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=None,
            triton_sparse_ms=None,
            fa_dense_ms=None,
            cudnn_dense_ms=None,
            triton_dense_tflops=None,
            triton_sparse_tflops=None,
            fa_dense_tflops=None,
            cudnn_dense_tflops=None,
            error_message=str(exc),
        )


def print_results(results: List[BenchmarkResult]) -> None:
    ok = [r for r in results if r.error_message is None]
    if not ok:
        print("No successful benchmark results.")
        for r in results:
            print(f"Failed: {r.config} -> {r.error_message}")
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
    num_kv_heads = [16]
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
    print(f"Running {len(configs)} benchmark configurations on {device_name}...")
    for cfg in tqdm(configs, desc="Benchmarking attn forward"):
        results.append(run_benchmark(cfg))

    print_results(results)


if __name__ == "__main__":
    main()
