from typing import List, Optional
import traceback

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.triton.interface import (
    flash_dense_attn_with_kvcache_func,
    flash_sparse_attn_with_kvcache_func,
    flash_gated_attn_with_kvcache_func,
)
from test_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    format_tflops,
    generate_inputs,
    generate_decode_configs,
)


def _decode_fwd_flops(cfg: BenchmarkConfig) -> float:
    return (
        4.0
        * cfg.batch_size
        * cfg.num_heads
        * cfg.seqlen_q
        * cfg.seqlen_k
        * cfg.head_dim
    )


def benchmark_triton_dense_decode(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="llm",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        flash_dense_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            window_size=(None, None),
        )

    return do_bench(fn, warmup=20, rep=100)


def benchmark_triton_sparse_decode(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="llm",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 1.0

    def fn():
        flash_sparse_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            window_size=(None, None),
        )

    return do_bench(fn, warmup=20, rep=100)


def benchmark_triton_gated_decode(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="llm",
    )
    q = q.squeeze(1)
    alpha = torch.randn(cfg.batch_size, cfg.num_heads, device=device, dtype=dtype)
    delta = torch.randn(
        cfg.batch_size, cfg.num_kv_heads, cfg.seqlen_k, device=device, dtype=dtype
    )
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 1.0
    gate_threshold = 1.0

    def fn():
        flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            gate_threshold=gate_threshold,
            is_logsigmoid_gate=False,
            is_adapt_gate=False,
            window_size=(None, None),
        )

    return do_bench(fn, warmup=20, rep=100)


def benchmark_fa_decode(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="llm",
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
                enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
            )

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cudnn_decode(
    cfg: BenchmarkConfig, device: str = "cuda", dtype=torch.bfloat16
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="llm",
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
                enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
            )

    try:
        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    try:
        triton_dense_ms = benchmark_triton_dense_decode(cfg)
        triton_sparse_ms = benchmark_triton_sparse_decode(cfg)
        triton_gated_ms = benchmark_triton_gated_decode(cfg)
        fa_dense_ms = benchmark_fa_decode(cfg)
        cudnn_dense_ms = benchmark_cudnn_decode(cfg)

        flops = _decode_fwd_flops(cfg)
        triton_dense_tflops = flops / triton_dense_ms * 1e-9
        triton_sparse_tflops = flops / triton_sparse_ms * 1e-9
        triton_gated_tflops = flops / triton_gated_ms * 1e-9
        fa_dense_tflops = flops / fa_dense_ms * 1e-9 if fa_dense_ms else None
        cudnn_dense_tflops = flops / cudnn_dense_ms * 1e-9 if cudnn_dense_ms else None

        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=triton_dense_ms,
            triton_sparse_ms=triton_sparse_ms,
            triton_gated_ms=triton_gated_ms,
            fa_dense_ms=fa_dense_ms,
            cudnn_dense_ms=cudnn_dense_ms,
            triton_dense_tflops=triton_dense_tflops,
            triton_sparse_tflops=triton_sparse_tflops,
            triton_gated_tflops=triton_gated_tflops,
            fa_dense_tflops=fa_dense_tflops,
            cudnn_dense_tflops=cudnn_dense_tflops,
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
                r.config.num_kv_heads,
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

    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[5]))
    headers = [
        "B",
        "H",
        "H_kv",
        "D",
        "Seqlen_q",
        "Seqlen_k",
        "Mode",
        "Triton Dense TFLOPS",
        "Triton Sparse TFLOPS",
        "Triton Gated TFLOPS",
        "FA TFLOPS",
        "cuDNN TFLOPS",
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
    seqlens_k = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    head_dims = [128]
    is_causal = False

    configs = generate_decode_configs(
        batch_sizes,
        num_heads,
        num_kv_heads,
        seqlens_k,
        head_dims,
        is_causal,
    )

    results: List[BenchmarkResult] = []
    print(f"Running {len(configs)} decode benchmark configurations on {device_name}...")
    for cfg in tqdm(configs, desc="Benchmarking attn decode"):
        results.append(run_benchmark(cfg))

    print_results(results)


if __name__ == "__main__":
    main()
