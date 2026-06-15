import argparse
import traceback
from typing import Callable, List, Literal, Optional

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
from flash_sparse_attn.ops.triton import quant
from test_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    format_ms,
    generate_inputs,
    generate_decode_configs,
)
from benchmark_plot import plot_benchmark_results


BenchmarkMode = Literal["auto", "eager", "graph"]


def is_cuda_graph_available() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )


def _capture_cuda_graph(
    fn: Callable[[], object], capture_warmup: int = 3
) -> Callable[[], object]:
    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)

    static_output = None
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmup):
            static_output = fn()

    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        static_output = fn()

    current_stream.wait_stream(capture_stream)

    def replay():
        graph.replay()
        return static_output

    return replay


def do_bench_decode(
    fn: Callable[[], object],
    *,
    mode: BenchmarkMode = "auto",
    warmup: int = 20,
    rep: int = 100,
    label: str = "decode benchmark",
) -> float:
    if mode == "eager":
        return do_bench(fn, warmup=warmup, rep=rep)

    if not is_cuda_graph_available():
        if mode == "graph":
            raise RuntimeError("CUDA Graph is not available")
        return do_bench(fn, warmup=warmup, rep=rep)

    try:
        replay = _capture_cuda_graph(fn)
        return do_bench(replay, warmup=warmup, rep=rep)
    except Exception as exc:
        if mode == "graph":
            raise
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print(f"{label}: CUDA Graph benchmark failed, falling back to eager: {exc}")
        return do_bench(fn, warmup=warmup, rep=rep)


def allocate_decode_outputs(
    query: torch.Tensor, *, is_quant: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dtype = torch.bfloat16 if is_quant else query.dtype
    out = torch.empty(query.shape, dtype=out_dtype, device=query.device)
    lse = torch.empty(query.shape[:2], dtype=torch.float32, device=query.device)
    return out, lse


def benchmark_triton_dense_decode(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype=torch.bfloat16,
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)
    out, lse = allocate_decode_outputs(q)
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        return flash_dense_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton dense decode")


def benchmark_triton_dense_decode_quant(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)

    # Pre-quantize outside benchmark loop
    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)

    out, lse = allocate_decode_outputs(q_quant, is_quant=True)
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        return flash_dense_attn_with_kvcache_func(
            q_quant,
            k_quant,
            v_quant,
            softmax_scale=softmax_scale,
            query_scale=q_scale,
            key_scale=k_scale,
            value_scale=v_scale,
            is_quant=True,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton dense quant decode")


def benchmark_triton_sparse_decode(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype=torch.bfloat16,
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)
    out, lse = allocate_decode_outputs(q)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 16.0

    def fn():
        return flash_sparse_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton sparse decode")


def benchmark_triton_sparse_decode_quant(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)

    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)

    out, lse = allocate_decode_outputs(q_quant, is_quant=True)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 16.0

    def fn():
        return flash_sparse_attn_with_kvcache_func(
            q_quant,
            k_quant,
            v_quant,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            query_scale=q_scale,
            key_scale=k_scale,
            value_scale=v_scale,
            is_quant=True,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton sparse quant decode")


def benchmark_triton_gated_decode(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype=torch.bfloat16,
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)
    alpha = torch.randn(cfg.batch_size, cfg.num_heads, device=device, dtype=dtype)
    delta = torch.randn(
        cfg.batch_size, cfg.seqlen_k, cfg.num_kv_heads, device=device, dtype=dtype
    )
    out, lse = allocate_decode_outputs(q)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 16.0
    gate_threshold = 16.0

    def fn():
        return flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            gate_threshold=gate_threshold,
            is_logsigmoid_gate=False,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton gated decode")


def benchmark_triton_gated_decode_quant(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    benchmark_mode: BenchmarkMode = "auto",
) -> float:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
        input_source="random",
    )
    q = q.squeeze(1)

    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)

    alpha = torch.randn(
        cfg.batch_size, cfg.num_heads, device=device, dtype=torch.bfloat16
    )
    delta = torch.randn(
        cfg.batch_size,
        cfg.seqlen_k,
        cfg.num_kv_heads,
        device=device,
        dtype=torch.bfloat16,
    )
    out, lse = allocate_decode_outputs(q_quant, is_quant=True)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 16.0
    gate_threshold = 16.0

    def fn():
        return flash_gated_attn_with_kvcache_func(
            q_quant,
            k_quant,
            v_quant,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=softmax_threshold,
            gate_threshold=gate_threshold,
            is_logsigmoid_gate=False,
            query_scale=q_scale,
            key_scale=k_scale,
            value_scale=v_scale,
            is_quant=True,
            out=out,
            lse=lse,
            is_autotune=False,
            skip_checks=True,
        )

    return do_bench_decode(fn, mode=benchmark_mode, label="Triton gated quant decode")


def benchmark_fa_decode(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype=torch.bfloat16,
    benchmark_mode: BenchmarkMode = "auto",
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="random",
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
                enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
            )

    try:
        return do_bench_decode(
            fn, mode=benchmark_mode, label="FlashAttention SDPA decode"
        )
    except Exception:
        return None


def benchmark_cudnn_decode(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype=torch.bfloat16,
    benchmark_mode: BenchmarkMode = "auto",
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        device=device,
        dtype=dtype,
        layout="bhsd",
        input_source="random",
    )
    softmax_scale = cfg.head_dim**-0.5

    def fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                scale=softmax_scale,
                enable_gqa=True if cfg.num_heads != cfg.num_kv_heads else False,
            )

    try:
        return do_bench_decode(fn, mode=benchmark_mode, label="cuDNN SDPA decode")
    except Exception:
        return None


def run_benchmark(
    cfg: BenchmarkConfig, benchmark_mode: BenchmarkMode = "auto"
) -> BenchmarkResult:
    try:
        triton_dense_ms = benchmark_triton_dense_decode(
            cfg, benchmark_mode=benchmark_mode
        )
        triton_dense_quant_ms = benchmark_triton_dense_decode_quant(
            cfg, benchmark_mode=benchmark_mode
        )
        triton_sparse_ms = benchmark_triton_sparse_decode(
            cfg, benchmark_mode=benchmark_mode
        )
        triton_sparse_quant_ms = benchmark_triton_sparse_decode_quant(
            cfg, benchmark_mode=benchmark_mode
        )
        triton_gated_ms = benchmark_triton_gated_decode(
            cfg, benchmark_mode=benchmark_mode
        )
        triton_gated_quant_ms = benchmark_triton_gated_decode_quant(
            cfg, benchmark_mode=benchmark_mode
        )
        fa_dense_ms = benchmark_fa_decode(cfg, benchmark_mode=benchmark_mode)
        cudnn_dense_ms = benchmark_cudnn_decode(cfg, benchmark_mode=benchmark_mode)

        return BenchmarkResult(
            config=cfg,
            triton_dense_ms=triton_dense_ms,
            triton_sparse_ms=triton_sparse_ms,
            triton_gated_ms=triton_gated_ms,
            fa_dense_ms=fa_dense_ms,
            cudnn_dense_ms=cudnn_dense_ms,
            triton_dense_quant_ms=triton_dense_quant_ms,
            triton_sparse_quant_ms=triton_sparse_quant_ms,
            triton_gated_quant_ms=triton_gated_quant_ms,
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
                format_ms(r.triton_dense_ms),
                format_ms(r.triton_dense_quant_ms),
                format_ms(r.triton_sparse_ms),
                format_ms(r.triton_sparse_quant_ms),
                format_ms(r.triton_gated_ms),
                format_ms(r.triton_gated_quant_ms),
                format_ms(r.fa_dense_ms),
                format_ms(r.cudnn_dense_ms),
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
        "Triton Dense (ms)",
        "Triton Dense Quant (ms)",
        "Triton Sparse (ms)",
        "Triton Sparse Quant (ms)",
        "Triton Gated (ms)",
        "Triton Gated Quant (ms)",
        "FA Dense (ms)",
        "cuDNN Dense (ms)",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention decode kernels")
    parser.add_argument(
        "--benchmark-mode",
        choices=("auto", "eager", "graph"),
        default="auto",
        help=(
            "auto prefers CUDA Graph and falls back to eager; eager disables graph "
            "capture; graph requires graph capture to succeed"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    benchmark_mode: BenchmarkMode = args.benchmark_mode
    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)

    batch_sizes = [1]
    num_heads = [64]
    num_kv_heads = [4]
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
    print(f"Benchmark mode: {benchmark_mode}")
    for cfg in tqdm(configs, desc="Benchmarking attn decode"):
        results.append(run_benchmark(cfg, benchmark_mode=benchmark_mode))

    print_results(results)
    plot_benchmark_results(results, phase="decode")


if __name__ == "__main__":
    main()
