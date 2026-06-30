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
class BwdBenchmarkResult:
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
        return None


def benchmark_fsa_base_backward(
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
        out = flash_sparse_attn_func(
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
            is_split_qo=False,
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
    except Exception:
        return None


def benchmark_fsa_window_backward(
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
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:
        out = flash_sparse_attn_func(
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
            is_split_qo=False,
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
    except Exception:
        return None


def benchmark_fsa_split_backward(
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
        out = flash_sparse_attn_func(
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
            is_split_qo=True,
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
    except Exception:
        return None


def benchmark_fsa_quant_backward(
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
    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    try:
        out = flash_sparse_attn_func(
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
            is_split_qo=False,
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
    except Exception:
        return None


def benchmark_fsa_skip_backward(
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
    softmax_threshold = cfg.head_dim / cfg.seqlen_k
    try:
        out = flash_sparse_attn_func(
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
            is_split_qo=False,
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
    except Exception:
        return None


def benchmark_fsa_all_backward(
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
    q_quant, q_scale = quant.quantize_fp8(q)
    k_quant, k_scale = quant.quantize_fp8(k)
    v_quant, v_scale = quant.quantize_fp8(v)
    softmax_threshold = cfg.head_dim / cfg.seqlen_k
    window_sizes = window_sizes_heuristic(
        cfg.seqlen_k, cfg.num_kv_heads, device=torch.device(device)
    )
    try:
        out = flash_sparse_attn_func(
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
            is_split_kv=False,
            is_split_qo=True,
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
    except Exception:
        return None


def run_benchmark(cfg: BenchmarkConfig) -> BwdBenchmarkResult:
    try:
        return BwdBenchmarkResult(
            config=cfg,
            fa_ms=benchmark_fa_backward(cfg),
            cudnn_ms=benchmark_cudnn_backward(cfg),
            fsa_base_ms=benchmark_fsa_base_backward(cfg),
            fsa_window_ms=benchmark_fsa_window_backward(cfg),
            fsa_split_ms=benchmark_fsa_split_backward(cfg),
            fsa_quant_ms=benchmark_fsa_quant_backward(cfg),
            fsa_skip_ms=benchmark_fsa_skip_backward(cfg),
            fsa_all_ms=benchmark_fsa_all_backward(cfg),
        )
    except Exception as e:
        return BwdBenchmarkResult(
            config=cfg,
            error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def print_results(results: List[BwdBenchmarkResult]) -> None:
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

    results: List[BwdBenchmarkResult] = []
    print(
        f"Running {len(configs)} backward benchmark configurations on {device_name}..."
    )
    for cfg in tqdm(configs, desc="Benchmarking backward"):
        results.append(run_benchmark(cfg))

    print_results(results)
    plot_benchmark_results(results, phase="backward")


if __name__ == "__main__":
    main()
