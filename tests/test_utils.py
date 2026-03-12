import itertools
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    seqlen_q: int
    seqlen_k: int
    is_causal: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    config: BenchmarkConfig
    triton_dense_ms: Optional[float]
    triton_sparse_ms: Optional[float]
    fa_dense_ms: Optional[float]
    cudnn_dense_ms: Optional[float]
    triton_dense_tflops: Optional[float]
    triton_sparse_tflops: Optional[float]
    fa_dense_tflops: Optional[float]
    cudnn_dense_tflops: Optional[float]
    error_message: Optional[str] = None


def generate_train_configs(
    batch_sizes: List[int],
    num_heads: List[int],
    seqlens: List[int],
    head_dims: List[int],
    is_causal: bool,
    num_kv_heads: Optional[List[int]] = None,
) -> List[BenchmarkConfig]:
    """Generate benchmark configs for train-style attention where seqlen_q == seqlen_k."""
    kv_heads = num_kv_heads or num_heads
    cfgs: List[BenchmarkConfig] = []

    for bsz, h, h_kv, seqlen, hd in itertools.product(
        batch_sizes,
        num_heads,
        kv_heads,
        seqlens,
        head_dims,
    ):
        if h % h_kv != 0:
            continue
        cfgs.append(
            BenchmarkConfig(
                batch_size=bsz,
                num_heads=h,
                num_kv_heads=h_kv,
                head_dim=hd,
                seqlen_q=seqlen,
                seqlen_k=seqlen,
                is_causal=is_causal,
            )
        )
    return cfgs


def generate_decode_configs(
    batch_sizes: List[int],
    num_heads: List[int],
    num_kv_heads: List[int],
    seqlens_k: List[int],
    head_dims: List[int],
    is_causal: bool,
) -> List[BenchmarkConfig]:
    """Generate benchmark configs for decode-style attention where seqlen_q == 1."""
    cfgs: List[BenchmarkConfig] = []
    for bsz, h, h_kv, seqlen_k, hd in itertools.product(
        batch_sizes,
        num_heads,
        num_kv_heads,
        seqlens_k,
        head_dims,
    ):
        if h % h_kv != 0:
            continue
        cfgs.append(
            BenchmarkConfig(
                batch_size=bsz,
                num_heads=h,
                num_kv_heads=h_kv,
                head_dim=hd,
                seqlen_q=1,
                seqlen_k=seqlen_k,
                is_causal=is_causal,
            )
        )
    return cfgs


def format_tflops(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "N/A"
