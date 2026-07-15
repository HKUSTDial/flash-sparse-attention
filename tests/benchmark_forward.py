from dataclasses import dataclass
from typing import List, Optional

import traceback

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from flash_sparse_attn.ops.triton.interface import flash_sparse_attn_func
from flash_sparse_attn.ops.triton import quant
from flash_sparse_attn.ops.triton.utils import window_sizes_heuristic
from test_utils import (
    BenchmarkConfig,
    format_ms,
    generate_inputs,
    generate_train_configs,
)
from benchmark_plot import plot_benchmark_results


@dataclass(frozen=True)
class FwdBenchmarkResult:
    config: BenchmarkConfig
    fa_ms: Optional[float] = None
    cudnn_ms: Optional[float] = None
    fsa_base_ms: Optional[float] = None
    fsa_window_ms: Optional[float] = None
    fsa_split_ms: Optional[float] = None
    fsa_quant_ms: Optional[float] = None
    fsa_skip_ms: Optional[float] = None
    fsa_all_ms: Optional[float] = None
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


def benchmark_fsa_base_forward(
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
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=False,
                is_split_kv=False,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_fsa_window_forward(
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
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:

        def fn():
            flash_sparse_attn_func(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=window_sizes,
                softmax_threshold=0.0,
                is_local=True,
                is_quant=False,
                is_split_kv=False,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_fsa_split_forward(
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
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=False,
                is_split_kv=True,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_fsa_quant_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    softmax_scale = cfg.head_dim**-0.5
    try:

        def fn():
            flash_sparse_attn_func(
                q_quant,
                k_quant,
                v_quant,
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                window_sizes=None,
                softmax_threshold=0.0,
                is_local=False,
                is_quant=True,
                is_split_kv=False,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_fsa_skip_forward(
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
    softmax_threshold = 1.0
    try:

        def fn():
            flash_sparse_attn_func(
                q,
                k,
                v,
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=None,
                key_scale=None,
                value_scale=None,
                window_sizes=None,
                softmax_threshold=softmax_threshold,
                is_local=False,
                is_quant=False,
                is_split_kv=False,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_fsa_all_forward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    softmax_scale = cfg.head_dim**-0.5
    softmax_threshold = 1.0
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:

        def fn():
            flash_sparse_attn_func(
                q_quant,
                k_quant,
                v_quant,
                is_causal=cfg.is_causal,
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                window_sizes=window_sizes,
                softmax_threshold=softmax_threshold,
                is_local=True,
                is_quant=True,
                is_split_kv=True,
                is_autotune=True,
                skip_checks=True,
            )

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> FwdBenchmarkResult:
    try:
        fa_ms = benchmark_fa_forward(cfg)
        cudnn_ms = benchmark_cudnn_forward(cfg)
        base_ms = benchmark_fsa_base_forward(cfg)
        local_ms = benchmark_fsa_window_forward(cfg)
        split_ms = benchmark_fsa_split_forward(cfg)
        quant_ms = benchmark_fsa_quant_forward(cfg)
        threshold_ms = benchmark_fsa_skip_forward(cfg)
        all_ms = benchmark_fsa_all_forward(cfg)

        return FwdBenchmarkResult(
            config=cfg,
            fa_ms=fa_ms,
            cudnn_ms=cudnn_ms,
            fsa_base_ms=base_ms,
            fsa_window_ms=local_ms,
            fsa_split_ms=split_ms,
            fsa_quant_ms=quant_ms,
            fsa_skip_ms=threshold_ms,
            fsa_all_ms=all_ms,
        )
    except Exception as e:
        return FwdBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[FwdBenchmarkResult]) -> None:
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
        "FSA Base",
        "FSA Window",
        "FSA Split",
        "FSA Quant",
        "FSA Skip",
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
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
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
                    format_ms(r.fsa_window_ms),
                    format_ms(r.fsa_split_ms),
                    format_ms(r.fsa_quant_ms),
                    format_ms(r.fsa_skip_ms),
                    format_ms(r.fsa_all_ms),
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

    results: List[FwdBenchmarkResult] = []
    print(
        f"Running {len(configs)} forward benchmark configurations on {device_name}..."
    )
    for cfg in tqdm(configs, desc="Benchmarking forward"):
        results.append(run_benchmark(cfg))

    print_results(results)
    plot_benchmark_results(results, phase="forward")


if __name__ == "__main__":
    main()
