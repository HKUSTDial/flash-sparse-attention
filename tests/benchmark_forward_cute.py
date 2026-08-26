from dataclasses import dataclass
from typing import List, Optional

import traceback

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
    generate_train_configs,
)
from benchmark_plot import plot_benchmark_results


@dataclass(frozen=True)
class FwdCuteBenchmarkResult:
    config: BenchmarkConfig
    fa_ms: Optional[float] = None
    cudnn_ms: Optional[float] = None
    cute_base_ms: Optional[float] = None
    cute_causal_ms: Optional[float] = None
    cute_window_ms: Optional[float] = None
    cute_threshold_ms: Optional[float] = None
    cute_split_ms: Optional[float] = None
    cute_all_ms: Optional[float] = None
    error_message: Optional[str] = None


def benchmark_fa_forward(cfg: BenchmarkConfig, device: str = "cuda") -> Optional[float]:
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
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=cfg.is_causal, enable_gqa=True
                )

            return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cudnn_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bhsd",
    )
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):

            def fn():
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=cfg.is_causal, enable_gqa=True
                )

            return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_base_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                is_local=False,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_causal_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                is_local=False,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_window_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                window_sizes=ws,
                is_local=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_threshold_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                softmax_threshold=1.0,
                is_causal=cfg.is_causal,
                is_local=False,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_split_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                is_causal=cfg.is_causal,
                is_local=False,
                num_splits=-1,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_all_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
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
            flash_sparse_attn_func(
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

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> FwdCuteBenchmarkResult:
    try:
        fa_ms = benchmark_fa_forward(cfg)
        cudnn_ms = benchmark_cudnn_forward(cfg)
        base_ms = benchmark_cute_base_forward(cfg)
        causal_ms = benchmark_cute_causal_forward(cfg)
        window_ms = benchmark_cute_window_forward(cfg)
        threshold_ms = benchmark_cute_threshold_forward(cfg)
        split_ms = benchmark_cute_split_forward(cfg)
        all_ms = benchmark_cute_all_forward(cfg)

        return FwdCuteBenchmarkResult(
            config=cfg,
            fa_ms=fa_ms,
            cudnn_ms=cudnn_ms,
            cute_base_ms=base_ms,
            cute_causal_ms=causal_ms,
            cute_window_ms=window_ms,
            cute_threshold_ms=threshold_ms,
            cute_split_ms=split_ms,
            cute_all_ms=all_ms,
        )
    except Exception as e:
        return FwdCuteBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[FwdCuteBenchmarkResult]) -> None:
    headers = [
        "B",
        "H",
        "H_kv",
        "D",
        "SeqQ",
        "SeqK",
        "Causal",
        "FA (ms)",
        "cuDNN (ms)",
        "CuTe Base",
        "CuTe Causal",
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
                    format_ms(r.cute_base_ms),
                    format_ms(r.cute_causal_ms),
                    format_ms(r.cute_window_ms),
                    format_ms(r.cute_threshold_ms),
                    format_ms(r.cute_split_ms),
                    format_ms(r.cute_all_ms),
                ]
            )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    torch.manual_seed(0)
    device_name = torch.cuda.get_device_name(0)

    batch_sizes = [1]
    num_heads = [32]
    num_kv_heads = [8]
    seqlens = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
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

    results: List[FwdCuteBenchmarkResult] = []
    print(
        f"Running {len(configs)} forward (CuTe) benchmark configurations on {device_name}..."
    )
    for cfg in tqdm(configs, desc="Benchmarking forward (CuTe)"):
        results.append(run_benchmark(cfg))

    print_results(results)
    plot_benchmark_results(results, phase="forward_cute")


if __name__ == "__main__":
    main()
