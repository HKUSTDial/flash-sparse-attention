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
class BwdCuteBenchmarkResult:
    config: BenchmarkConfig
    fa_ms: Optional[float] = None
    cudnn_ms: Optional[float] = None
    cute_base_ms: Optional[float] = None
    cute_causal_ms: Optional[float] = None
    cute_window_ms: Optional[float] = None
    cute_threshold_ms: Optional[float] = None
    cute_all_ms: Optional[float] = None
    error_message: Optional[str] = None


def benchmark_fa_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bhsd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=cfg.is_causal, enable_gqa=True
            )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cudnn_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bhsd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=cfg.is_causal, enable_gqa=True
            )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        print(f"cuDNN backward failed for config {cfg}: {traceback.format_exc()}")
        return None


def benchmark_cute_base_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    try:
        out, _ = flash_sparse_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=cfg.is_causal,
            is_local=False,
        )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_causal_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    try:
        out, _ = flash_sparse_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=cfg.is_causal,
            is_local=False,
        )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_window_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    num_kv_heads = v.shape[-2]
    ws = window_sizes_heuristic(cfg.seqlen_k, num_kv_heads, device=torch.device(device))
    try:
        out, _ = flash_sparse_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=cfg.is_causal,
            window_sizes=ws,
            is_local=True,
        )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_threshold_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    try:
        out, _ = flash_sparse_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=1.0,
            is_causal=cfg.is_causal,
            is_local=False,
        )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def benchmark_cute_all_backward(
    cfg: BenchmarkConfig, device: str = "cuda"
) -> Optional[float]:
    q, k, v = generate_inputs(
        cfg,
        input_source="synthetic_llm",
        device=device,
        dtype=torch.bfloat16,
        layout="bshd",
    )
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    softmax_scale = cfg.head_dim**-0.5
    num_kv_heads = v.shape[-2]
    ws = window_sizes_heuristic(cfg.seqlen_k, num_kv_heads, device=torch.device(device))
    try:
        out, _ = flash_sparse_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=1.0,
            is_causal=cfg.is_causal,
            window_sizes=ws,
            is_local=True,
        )
        dout = torch.randn_like(out)

        def fn():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(dout, retain_graph=True)

        return do_bench(fn, warmup=20, rep=100)
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> BwdCuteBenchmarkResult:
    try:
        return BwdCuteBenchmarkResult(
            config=cfg,
            fa_ms=benchmark_fa_backward(cfg),
            cudnn_ms=benchmark_cudnn_backward(cfg),
            cute_base_ms=benchmark_cute_base_backward(cfg),
            cute_causal_ms=benchmark_cute_causal_backward(cfg),
            cute_window_ms=benchmark_cute_window_backward(cfg),
            cute_threshold_ms=benchmark_cute_threshold_backward(cfg),
            cute_all_ms=benchmark_cute_all_backward(cfg),
        )
    except Exception as e:
        return BwdCuteBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[BwdCuteBenchmarkResult]) -> None:
    headers = [
        "B",
        "H",
        "H_kv",
        "D",
        "Sq",
        "Sk",
        "Causal",
        "FA (ms)",
        "cuDNN (ms)",
        "CuTe Base",
        "CuTe Causal",
        "CuTe Window",
        "CuTe Thresh",
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
                    format_ms(r.cute_causal_ms),
                    format_ms(r.cute_window_ms),
                    format_ms(r.cute_threshold_ms),
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

    results: List[BwdCuteBenchmarkResult] = []
    print(
        f"Running {len(configs)} backward (CuTe) benchmark configurations on {device_name}..."
    )
    for cfg in tqdm(configs, desc="Benchmarking backward (CuTe)"):
        results.append(run_benchmark(cfg))

    print_results(results)
    plot_benchmark_results(results, phase="backward_cute")


if __name__ == "__main__":
    main()
