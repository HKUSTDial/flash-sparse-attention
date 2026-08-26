import argparse
import traceback
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.cute.interface import (
    flash_sparse_attn_func,
    window_sizes_heuristic,
)
from test_utils import (
    BenchmarkConfig,
    format_ms,
    generate_inputs,
    generate_decode_configs,
)
from benchmark_plot import plot_benchmark_results


BenchmarkMode = Literal["auto", "eager", "graph"]


@dataclass(frozen=True)
class DecCuteBenchmarkResult:
    config: BenchmarkConfig
    fa_ms: Optional[float] = None
    cudnn_ms: Optional[float] = None
    cute_base_ms: Optional[float] = None
    cute_window_ms: Optional[float] = None
    cute_threshold_ms: Optional[float] = None
    cute_split_ms: Optional[float] = None
    cute_all_ms: Optional[float] = None
    error_message: Optional[str] = None


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
) -> float:
    if mode == "eager":
        return do_bench(fn, warmup=warmup, rep=rep)
    if not is_cuda_graph_available():
        if mode == "graph":
            raise RuntimeError("CUDA Graph is not available")
        return do_bench(fn, warmup=warmup, rep=rep)
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        replay = _capture_cuda_graph(fn)
        return do_bench(replay, warmup=warmup, rep=rep)
    except Exception:
        if mode == "graph":
            raise
        return do_bench(fn, warmup=warmup, rep=rep)


def benchmark_fa_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bhsd",
    )
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

            def fn():
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=cfg.is_causal, enable_gqa=True
                )

            return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cudnn_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    cudnn_cfg = BenchmarkConfig(
        batch_size=cfg.batch_size,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        seqlen_q=2,
        seqlen_k=cfg.seqlen_k,
        is_causal=cfg.is_causal,
    )
    q, k, v = generate_inputs(
        cudnn_cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bhsd",
    )
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):

            def fn():
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=cfg.is_causal, enable_gqa=True
                )

            return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cute_base_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                is_local=False,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cute_window_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    softmax_scale = cfg.head_dim**-0.5
    num_kv_heads = v.shape[-2]
    ws = window_sizes_heuristic(cfg.seqlen_k, num_kv_heads, device=torch.device(device))
    try:

        def fn():
            return flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                window_sizes=ws,
                is_local=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cute_threshold_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                softmax_threshold=1.0,
                is_causal=cfg.is_causal,
                is_local=False,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cute_split_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                is_local=False,
                num_splits=-1,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_cute_all_decode(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    softmax_scale = cfg.head_dim**-0.5
    num_kv_heads = v.shape[-2]
    ws = window_sizes_heuristic(cfg.seqlen_k, num_kv_heads, device=torch.device(device))
    try:

        def fn():
            return flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                softmax_threshold=1.0,
                is_causal=cfg.is_causal,
                window_sizes=ws,
                is_local=True,
                num_splits=-1,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def run_benchmark(
    cfg: BenchmarkConfig, benchmark_mode: BenchmarkMode = "auto"
) -> DecCuteBenchmarkResult:
    try:
        return DecCuteBenchmarkResult(
            config=cfg,
            fa_ms=benchmark_fa_decode(cfg, mode=benchmark_mode),
            cudnn_ms=benchmark_cudnn_decode(cfg, mode=benchmark_mode),
            cute_base_ms=benchmark_cute_base_decode(cfg, mode=benchmark_mode),
            cute_window_ms=benchmark_cute_window_decode(cfg, mode=benchmark_mode),
            cute_threshold_ms=benchmark_cute_threshold_decode(cfg, mode=benchmark_mode),
            cute_split_ms=benchmark_cute_split_decode(cfg, mode=benchmark_mode),
            cute_all_ms=benchmark_cute_all_decode(cfg, mode=benchmark_mode),
        )
    except Exception as e:
        return DecCuteBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[DecCuteBenchmarkResult]) -> None:
    headers = [
        "B",
        "H",
        "H_kv",
        "D",
        "Sq",
        "Sk",
        "Causal",
        "FA",
        "cuDNN",
        "CuTe Base",
        "CuTe Window",
        "CuTe Thresh",
        "CuTe Split",
        "CuTe All",
    ]
    rows = []
    for r in results:
        if r.error_message:
            rows.append(
                [
                    r.config.batch_size,
                    r.config.num_heads,
                    r.config.num_kv_heads,
                    r.config.head_dim,
                    r.config.seqlen_q,
                    r.config.seqlen_k,
                    r.config.is_causal,
                    *["ERR"] * 7,
                ]
            )
        else:
            rows.append(
                [
                    r.config.batch_size,
                    r.config.num_heads,
                    r.config.num_kv_heads,
                    r.config.head_dim,
                    r.config.seqlen_q,
                    r.config.seqlen_k,
                    r.config.is_causal,
                    format_ms(r.fa_ms),
                    format_ms(r.cudnn_ms),
                    format_ms(r.cute_base_ms),
                    format_ms(r.cute_window_ms),
                    format_ms(r.cute_threshold_ms),
                    format_ms(r.cute_split_ms),
                    format_ms(r.cute_all_ms),
                ]
            )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description="CuTe decode benchmark")
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="auto",
        choices=["auto", "eager", "graph"],
        help="Benchmark mode: auto (try CUDA graph), eager, or graph",
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
    num_heads = [32]
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

    results: List[DecCuteBenchmarkResult] = []
    print(
        f"Running {len(configs)} decode (CuTe) benchmark configurations on {device_name}..."
    )
    print(f"Benchmark mode: {benchmark_mode}")
    for cfg in tqdm(configs, desc="Benchmarking decode (CuTe)"):
        results.append(run_benchmark(cfg, benchmark_mode=benchmark_mode))

    print_results(results)
    plot_benchmark_results(results, phase="decode_cute")


if __name__ == "__main__":
    main()
