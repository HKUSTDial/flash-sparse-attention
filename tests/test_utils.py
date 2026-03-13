import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_DATASET_ID = "SmallDoge/niah"

_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}
_TEXT_CACHE = {}


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


def _get_layers_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture: cannot locate transformer layers")


def _resolve_model_ref(model_path: Optional[str]) -> str:
    return (model_path or DEFAULT_MODEL_ID).strip()


def _load_text_from_hub(dataset_id: str, target_len: int) -> str:
    cache_key = (dataset_id, target_len)
    if cache_key in _TEXT_CACHE:
        return _TEXT_CACHE[cache_key]

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required to load text from hub") from exc

    preferred_splits = [
        f"len_{target_len}",
        "len_8192",
        "len_4096",
        "len_2048",
        "len_1024",
    ]

    for split in preferred_splits:
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
            first = next(iter(ds))
            text = str(first.get("input", "")).strip()
            if text:
                _TEXT_CACHE[cache_key] = text
                return text
        except Exception:
            continue

    raise RuntimeError(
        f"Failed to load a valid text sample from dataset '{dataset_id}'."
    )


def _build_input_ids(
    model_ref: str, batch_size: int, seq_in: int, device: str
) -> torch.Tensor:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers tokenizer is required to build input_ids"
        ) from exc

    tokenizer = _TOKENIZER_CACHE.get(model_ref)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        _TOKENIZER_CACHE[model_ref] = tokenizer
    base_text = _load_text_from_hub(DEFAULT_DATASET_ID, seq_in)
    encoded = tokenizer(base_text, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"][0]
    if ids.numel() == 0:
        raise RuntimeError("Tokenizer produced empty input_ids for base_text")

    # Repeat to reach target length, then truncate
    repeat = (seq_in + ids.numel() - 1) // ids.numel()
    seq_ids = ids.repeat(repeat)[:seq_in]

    # Create a batch by rolling the same sequence
    batched = []
    for b in range(batch_size):
        batched.append(torch.roll(seq_ids, shifts=b))
    input_ids = torch.stack(batched, dim=0).to(device=device, dtype=torch.long)
    return input_ids


def generate_inputs(
    cfg: BenchmarkConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    layout: str = "bshd",
    input_source: str = "random",
    model_path: Optional[str] = None,
):
    """
    Generate Q/K/V tensors for benchmarks.

    layout:
      - "bshd": [B, S, H, D] for Triton interfaces
      - "bhsd": [B, H, S, D] for PyTorch SDPA backends
    input_source:
      - "random": synthetic random tensors
      - "llm": tensors from a model layer q/k/v projections
    model_path:
        - Optional local folder path or HF repo id.
        - Defaults to Qwen/Qwen3-0.6B-Base when input_source is "llm"
    """
    if layout not in {"bshd", "bhsd"}:
        raise ValueError(f"Unsupported layout: {layout}")
    if input_source not in {"random", "llm"}:
        raise ValueError(f"Unsupported input_source: {input_source}")

    if input_source == "random":
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
    else:
        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for input_source='llm'"
            ) from exc

        resolved_model_ref = _resolve_model_ref(model_path)
        model_cache_key = (resolved_model_ref, str(dtype), str(device))
        model = _MODEL_CACHE.get(model_cache_key)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_ref,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.eval()
            model.to(device)
            _MODEL_CACHE[model_cache_key] = model

        layers = _get_layers_container(model)
        num_layers = len(layers)
        if num_layers < 2:
            raise ValueError(f"Model must have at least 2 layers, got: {num_layers}")

        attn_layer_idx = num_layers - 1

        attn = layers[attn_layer_idx].self_attn
        input_layernorm = getattr(layers[attn_layer_idx], "input_layernorm", None)
        q_proj = attn.q_proj
        k_proj = attn.k_proj
        v_proj = attn.v_proj

        num_heads_q = int(getattr(model.config, "num_attention_heads"))
        num_heads_kv = int(getattr(model.config, "num_key_value_heads", num_heads_q))
        if num_heads_q != cfg.num_heads or num_heads_kv != cfg.num_kv_heads:
            raise ValueError(
                "Config head mismatch with model: "
                f"cfg(H={cfg.num_heads}, H_kv={cfg.num_kv_heads}) vs "
                f"model(H={num_heads_q}, H_kv={num_heads_kv})"
            )

        seq_in = max(cfg.seqlen_q, cfg.seqlen_k)
        input_ids = _build_input_ids(
            model_ref=resolved_model_ref,
            batch_size=cfg.batch_size,
            seq_in=seq_in,
            device=device,
        )

        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_in = out.hidden_states[attn_layer_idx]
            if input_layernorm is not None:
                hidden_in = input_layernorm(hidden_in)
            q_lin = q_proj(hidden_in)
            k_lin = k_proj(hidden_in)
            v_lin = v_proj(hidden_in)

        head_dim_q = q_lin.shape[-1] // num_heads_q
        head_dim_k = k_lin.shape[-1] // num_heads_kv
        head_dim_v = v_lin.shape[-1] // num_heads_kv
        if head_dim_q != head_dim_k or head_dim_q != head_dim_v:
            raise ValueError(
                f"Head dim mismatch in model projections: q={head_dim_q}, k={head_dim_k}, v={head_dim_v}"
            )
        if head_dim_q != cfg.head_dim:
            raise ValueError(
                f"Config head_dim ({cfg.head_dim}) mismatch model head_dim ({head_dim_q})"
            )

        q = q_lin.view(cfg.batch_size, seq_in, num_heads_q, head_dim_q)
        k = k_lin.view(cfg.batch_size, seq_in, num_heads_kv, head_dim_q)
        v = v_lin.view(cfg.batch_size, seq_in, num_heads_kv, head_dim_q)

        q = q[:, : cfg.seqlen_q, :, :].contiguous()
        k = k[:, : cfg.seqlen_k, :, :].contiguous()
        v = v[:, : cfg.seqlen_k, :, :].contiguous()

    if layout == "bhsd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    return q, k, v
