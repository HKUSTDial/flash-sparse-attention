import argparse
import traceback
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_with_kvcache_func
from flash_sparse_attn.ops.triton import quant
from flash_sparse_attn.ops.triton.utils import window_sizes_heuristic
from test_utils import (
    BenchmarkConfig,
    format_ms,
    generate_inputs,
    generate_decode_configs,
)
from benchmark_plot import plot_benchmark_results


BenchmarkMode = Literal["auto", "eager", "graph"]


@dataclass(frozen=True)
class DecBenchmarkResult:
    config: BenchmarkConfig
    fa_ms: Optional[float] = None
    cudnn_ms: Optional[float] = None
    fsa_base_ms: Optional[float] = None
    fsa_local_ms: Optional[float] = None
    fsa_split_ms: Optional[float] = None
    fsa_quant_ms: Optional[float] = None
    fsa_threshold_ms: Optional[float] = None
    fsa_all_ms: Optional[float] = None
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


def benchmark_fsa_base_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=False,
                num_splits=1,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_fsa_local_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=window_sizes,
                softmax_threshold=0.0,
                is_local=True,
                is_quant=False,
                num_splits=1,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_fsa_split_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=False,
                num_splits=None,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_fsa_quant_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q_quant, q_scale = quant.quantize_fp8(q)
    q_quant = q_quant.squeeze(1)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q_quant,
                k_quant,
                v_quant,
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=True,
                num_splits=1,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_fsa_threshold_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.squeeze(1)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = cfg.head_dim / cfg.seqlen_k
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=softmax_threshold,
                is_local=False,
                is_quant=False,
                num_splits=1,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def benchmark_fsa_all_dec(
    cfg: BenchmarkConfig, device: str = "cuda", mode: BenchmarkMode = "auto"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q_quant, q_scale = quant.quantize_fp8(q)
    q_quant = q_quant.squeeze(1)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = cfg.head_dim / cfg.seqlen_k
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:

        def fn():
            return flash_sparse_attn_with_kvcache_func(
                q_quant,
                k_quant,
                v_quant,
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                window_sizes=window_sizes,
                softmax_threshold=softmax_threshold,
                is_local=True,
                is_quant=True,
                num_splits=None,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench_decode(fn, mode=mode)
    except Exception:
        return None


def run_benchmark(
    cfg: BenchmarkConfig, benchmark_mode: BenchmarkMode = "auto"
) -> DecBenchmarkResult:
    try:
        return DecBenchmarkResult(
            config=cfg,
            fa_ms=benchmark_fa_decode(cfg, mode=benchmark_mode),
            cudnn_ms=benchmark_cudnn_decode(cfg, mode=benchmark_mode),
            fsa_base_ms=benchmark_fsa_base_dec(cfg, mode=benchmark_mode),
            fsa_local_ms=benchmark_fsa_local_dec(cfg, mode=benchmark_mode),
            fsa_split_ms=benchmark_fsa_split_dec(cfg, mode=benchmark_mode),
            fsa_quant_ms=benchmark_fsa_quant_dec(cfg, mode=benchmark_mode),
            fsa_threshold_ms=benchmark_fsa_threshold_dec(cfg, mode=benchmark_mode),
            fsa_all_ms=benchmark_fsa_all_dec(cfg, mode=benchmark_mode),
        )
    except Exception as e:
        return DecBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[DecBenchmarkResult]) -> None:
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
        "FSA Base",
        "FSA +Local",
        "FSA +Split",
        "FSA +Quant",
        "FSA +Thresh",
        "FSA All",
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
                    *["ERR"] * 8,
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
                    format_ms(r.fsa_base_ms),
                    format_ms(r.fsa_local_ms),
                    format_ms(r.fsa_split_ms),
                    format_ms(r.fsa_quant_ms),
                    format_ms(r.fsa_threshold_ms),
                    format_ms(r.fsa_all_ms),
                ]
            )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description="FSA decode benchmark")
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

    results: List[DecBenchmarkResult] = []
    print(f"Running {len(configs)} decode benchmark configurations on {device_name}...")
    print(f"Benchmark mode: {benchmark_mode}")
    for cfg in tqdm(configs, desc="Benchmarking decode"):
        results.append(run_benchmark(cfg, benchmark_mode=benchmark_mode))

    print_results(results)
    plot_benchmark_results(results, phase="decode")


if __name__ == "__main__":
    main()
