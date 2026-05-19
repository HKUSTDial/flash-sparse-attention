import itertools
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import torch

from flash_sparse_attn.ops.triton import cache_utils, launch_template
from flash_sparse_attn.ops.triton.flash_dense_bwd import (
    _flash_dense_attn_backward,
    _flash_dense_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_dense_fwd import (
    _flash_dense_attn_forward,
    _flash_dense_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.flash_gated_bwd import (
    _flash_gated_attn_backward,
    _flash_gated_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_gated_fwd import (
    _flash_gated_attn_forward,
    _flash_gated_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.flash_sparse_bwd import (
    _flash_sparse_attn_backward,
    _flash_sparse_attn_varlen_backward,
)
from flash_sparse_attn.ops.triton.flash_sparse_fwd import (
    _flash_sparse_attn_forward,
    _flash_sparse_attn_varlen_forward,
)
from flash_sparse_attn.ops.triton.utils import window_sizes_heuristic
from flash_sparse_attn.ops.triton.interface import (
    flash_dense_attn_varlen_with_kvcache_func,
    flash_dense_attn_with_kvcache_func,
    flash_gated_attn_varlen_with_kvcache_func,
    flash_gated_attn_with_kvcache_func,
    flash_sparse_attn_varlen_with_kvcache_func,
    flash_sparse_attn_with_kvcache_func,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_DATASET_ID = "SmallDoge/niah"

_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}
_TEXT_CACHE = {}

CORRECTNESS_DTYPE = torch.bfloat16
KernelType = Literal["dense", "sparse", "gated"]

_DEFAULT_RTOL = {
    "dense": 8e-2,
    "sparse": 2.0e-1,
    "gated": 2.0e-1,
}
_DEFAULT_ATOL = {
    "dense": 8e-2,
    "sparse": 2.0e-1,
    "gated": 2.0e-1,
}
_DEFAULT_BWD_RTOL = {
    "dense": 1.0e-1,
    "sparse": 2.0e-1,
    "gated": 2.0e-1,
}
_DEFAULT_BWD_ATOL = {
    "dense": 1.0e-1,
    "sparse": 2.0e-1,
    "gated": 2.0e-1,
}


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
    triton_gated_ms: Optional[float]
    fa_dense_ms: Optional[float]
    cudnn_dense_ms: Optional[float]
    triton_dense_tflops: Optional[float]
    triton_sparse_tflops: Optional[float]
    triton_gated_tflops: Optional[float]
    fa_dense_tflops: Optional[float]
    cudnn_dense_tflops: Optional[float]
    triton_dense_fp8_ms: Optional[float] = None
    triton_dense_fp8_tflops: Optional[float] = None
    triton_sparse_fp8_ms: Optional[float] = None
    triton_sparse_fp8_tflops: Optional[float] = None
    triton_gated_fp8_ms: Optional[float] = None
    triton_gated_fp8_tflops: Optional[float] = None
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

        with torch.no_grad():
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


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cu_seqlens(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )


def _make_mask(
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    is_causal: bool,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    q_idx = torch.arange(seqlen_q, device=device)[:, None]
    k_idx = torch.arange(seqlen_k, device=device)[None, :]
    dist = q_idx + (seqlen_k - seqlen_q) - k_idx

    causal_mask = dist >= 0
    window_mask = (dist >= window_right) & (dist <= window_left)

    if is_causal:
        allowed = causal_mask & window_mask
    else:
        allowed = window_mask

    return ~allowed


def _reference_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    is_local: bool = False,
    local_seqlen_k: Optional[int] = None,
    kind: KernelType = "dense",
) -> torch.Tensor:
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    num_kv_heads = kh.shape[1]
    if qh.shape[1] != kh.shape[1]:
        if qh.shape[1] % kh.shape[1] != 0:
            raise ValueError(
                f"Q heads ({qh.shape[1]}) must be divisible by KV heads ({kh.shape[1]})"
            )
        repeat = qh.shape[1] // kh.shape[1]
        kh = torch.repeat_interleave(kh, repeats=repeat, dim=1)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    if is_causal and not is_local:
        mask = _make_mask(seqlen_q, seqlen_k, q.device, True, seqlen_k, 0)
        scores = scores.masked_fill(mask, float("-inf"))
    elif is_local:
        ws_seqlen = local_seqlen_k if local_seqlen_k is not None else seqlen_k
        ws = window_sizes_heuristic(ws_seqlen, num_kv_heads, q.device)
        num_heads_q = qh.shape[1]
        qpk = num_heads_q // num_kv_heads

        q_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        k_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        causal_offset = seqlen_k - seqlen_q
        dist = q_idx + causal_offset - k_idx

        arch = cache_utils.get_device_arch(q.device)
        tile_k = max(1 << (max(q.shape[-1], 16) - 1).bit_length(), 16)
        if seqlen_q > 1:
            tile_m, tile_n, _, _, _ = launch_template.get_fwd_dense_launch_config(
                is_split_kv=False,
                pack_gqa=False,
                qhead_per_kvhead=1,
                tile_k=tile_k,
                device=q.device,
                arch=arch,
            )
            m_block = q_idx // tile_m
            n_blk = k_idx // tile_n
            n_block_max_no_mask = (m_block * tile_m + causal_offset) // tile_n
            n_block_max_diag = (
                (m_block + 1) * tile_m + causal_offset + tile_n - 1
            ) // tile_n
            is_diagonal = (n_blk >= n_block_max_no_mask) & (n_blk < n_block_max_diag)
            causal_mask = dist >= 0
        else:
            if kind == "sparse":
                _, tile_n, _, _, _ = launch_template.get_dec_sparse_launch_config(
                    qhead_per_kvhead=qpk,
                    tile_k=tile_k,
                    device=q.device,
                    arch=arch,
                )
            elif kind == "gated":
                _, tile_n, _, _, _ = launch_template.get_dec_gated_launch_config(
                    qhead_per_kvhead=qpk,
                    tile_k=tile_k,
                    device=q.device,
                    arch=arch,
                )
            else:
                _, tile_n, _, _, _ = launch_template.get_dec_dense_launch_config(
                    qhead_per_kvhead=qpk,
                    tile_k=tile_k,
                    device=q.device,
                    arch=arch,
                )
            last_block_start = (seqlen_k - 1) // tile_n * tile_n
            is_diagonal = k_idx >= last_block_start

        for h in range(num_kv_heads):
            wl, wr = int(ws[h, 0].item()), int(ws[h, 1].item())
            window_mask = (dist >= wr) & (dist <= wl)
            if seqlen_q > 1:
                allowed = (is_diagonal & causal_mask) | window_mask
            else:
                allowed = is_diagonal | window_mask
            scores[:, h * qpk : (h + 1) * qpk] = scores[
                :, h * qpk : (h + 1) * qpk
            ].masked_fill(~allowed, float("-inf"))

    return scores


def reference_dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    is_local: bool = False,
    local_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    scores = _reference_scores(
        q, k, softmax_scale, is_causal, is_local, local_seqlen_k, kind="dense"
    )
    vh = v.transpose(1, 2).float()
    if scores.shape[1] != vh.shape[1]:
        if scores.shape[1] % vh.shape[1] != 0:
            raise ValueError(
                f"Q heads ({scores.shape[1]}) must be divisible by V heads ({vh.shape[1]})"
            )
        repeat = scores.shape[1] // vh.shape[1]
        vh = torch.repeat_interleave(vh, repeats=repeat, dim=1)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, vh)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def reference_sparse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    is_local: bool = False,
    local_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    scores = _reference_scores(
        q, k, softmax_scale, is_causal, is_local, local_seqlen_k, kind="sparse"
    )
    vh = v.transpose(1, 2).float()
    if scores.shape[1] != vh.shape[1]:
        if scores.shape[1] % vh.shape[1] != 0:
            raise ValueError(
                f"Q heads ({scores.shape[1]}) must be divisible by V heads ({vh.shape[1]})"
            )
        repeat = scores.shape[1] // vh.shape[1]
        vh = torch.repeat_interleave(vh, repeats=repeat, dim=1)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, vh)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def reference_gated_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    local_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    scores = _reference_scores(
        q, k, softmax_scale, is_causal, is_local, local_seqlen_k, kind="gated"
    )
    alpha_h = alpha.transpose(1, 2).float()
    delta_h = delta.transpose(1, 2).float()
    if alpha_h.shape[1] != delta_h.shape[1]:
        if alpha_h.shape[1] % delta_h.shape[1] != 0:
            raise ValueError(
                f"Q heads ({alpha_h.shape[1]}) must be divisible by Delta heads ({delta_h.shape[1]})"
            )
        repeat = alpha_h.shape[1] // delta_h.shape[1]
        delta_h = torch.repeat_interleave(delta_h, repeats=repeat, dim=1)
    raw_gate = alpha_h.unsqueeze(-1) * delta_h.unsqueeze(-2)
    gate = torch.nn.functional.logsigmoid(raw_gate) if is_logsigmoid_gate else raw_gate
    scores = scores + gate * softmax_scale
    vh = v.transpose(1, 2).float()
    if scores.shape[1] != vh.shape[1]:
        if scores.shape[1] % vh.shape[1] != 0:
            raise ValueError(
                f"Q heads ({scores.shape[1]}) must be divisible by V heads ({vh.shape[1]})"
            )
        repeat = scores.shape[1] // vh.shape[1]
        vh = torch.repeat_interleave(vh, repeats=repeat, dim=1)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, vh)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def _reference_varlen_forward(
    kind: KernelType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    is_local: bool = False,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    is_logsigmoid_gate: bool = True,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    batch_size = cu_seqlens_q.numel() - 1
    max_sk = max(
        int(cu_seqlens_k[i + 1].item()) - int(cu_seqlens_k[i].item())
        for i in range(batch_size)
    )
    for idx in range(batch_size):
        qs = int(cu_seqlens_q[idx].item())
        qe = int(cu_seqlens_q[idx + 1].item())
        ks = int(cu_seqlens_k[idx].item())
        ke = int(cu_seqlens_k[idx + 1].item())
        qi = q[qs:qe].unsqueeze(0)
        ki = k[ks:ke].unsqueeze(0)
        vi = v[ks:ke].unsqueeze(0)
        lsk = max_sk if is_local else None
        if kind == "gated":
            ai = alpha[qs:qe, :].unsqueeze(0)
            di = delta[ks:ke, :].unsqueeze(0)
            out_i = reference_gated_forward(
                qi,
                ki,
                vi,
                ai,
                di,
                softmax_scale=softmax_scale,
                is_causal=is_causal,
                is_local=is_local,
                is_logsigmoid_gate=is_logsigmoid_gate,
                local_seqlen_k=lsk,
            )
        elif kind == "sparse":
            out_i = reference_sparse_forward(
                qi,
                ki,
                vi,
                softmax_scale=softmax_scale,
                is_causal=is_causal,
                is_local=is_local,
                local_seqlen_k=lsk,
            )
        else:
            out_i = reference_dense_forward(
                qi,
                ki,
                vi,
                softmax_scale=softmax_scale,
                is_causal=is_causal,
                is_local=is_local,
                local_seqlen_k=lsk,
            )
        outs.append(out_i.squeeze(0))
    return torch.cat(outs, dim=0)


def _reference_varlen_decode(
    kind: KernelType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    is_local: bool = False,
    alpha: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    is_logsigmoid_gate: bool = True,
    max_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    batch_size = q.shape[0]
    outs: List[torch.Tensor] = []
    if max_seqlen_k is None:
        max_seqlen_k = max(
            int(cu_seqlens_k[i + 1].item()) - int(cu_seqlens_k[i].item())
            for i in range(batch_size)
        )

    for idx in range(batch_size):
        ks = int(cu_seqlens_k[idx].item())
        ke = int(cu_seqlens_k[idx + 1].item())

        qi = q[idx : idx + 1].unsqueeze(1)
        ki = k[ks:ke].unsqueeze(0)
        vi = v[ks:ke].unsqueeze(0)
        lsk = max_seqlen_k if is_local else None

        if kind == "gated":
            ai = alpha[idx : idx + 1].unsqueeze(1)
            di = delta[ks:ke, :].unsqueeze(0)
            out_i = reference_gated_forward(
                qi,
                ki,
                vi,
                ai,
                di,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
                is_logsigmoid_gate=is_logsigmoid_gate,
                local_seqlen_k=lsk,
            )
        elif kind == "sparse":
            out_i = reference_sparse_forward(
                qi,
                ki,
                vi,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
                local_seqlen_k=lsk,
            )
        else:
            out_i = reference_dense_forward(
                qi,
                ki,
                vi,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
                local_seqlen_k=lsk,
            )
        outs.append(out_i.squeeze(0).squeeze(0))
    return torch.stack(outs, dim=0)


def _assert_close(
    name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float
) -> None:
    try:
        torch.testing.assert_close(
            got.float(), ref.float(), rtol=rtol, atol=atol, msg=name
        )
    except AssertionError as exc:
        # allow larger max-diff if mean-diff stays in tolerance
        got_f = got.float()
        ref_f = ref.float()
        abs_diff = (got_f - ref_f).abs()
        mean_abs_diff = abs_diff.mean().item()
        ref_mean_abs = ref_f.abs().mean().item()
        mean_rel_diff = mean_abs_diff / max(ref_mean_abs, 1e-6)
        if mean_abs_diff <= atol or mean_rel_diff <= rtol:
            return
        max_abs_diff = abs_diff.max().item()
        raise AssertionError(
            f"{name}: mean-diff fallback failed "
            f"(mean_abs_diff={mean_abs_diff:.6f}, mean_rel_diff={mean_rel_diff:.6f}, "
            f"max_abs_diff={max_abs_diff:.6f}, atol={atol}, rtol={rtol})"
        ) from exc


def _run_forward_base(
    kind: KernelType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    is_local: bool,
    alpha: Optional[torch.Tensor],
    delta: Optional[torch.Tensor],
    is_logsigmoid_gate: bool,
):
    softmax_scale = q.shape[-1] ** -0.5
    threshold = q.shape[-1] / k.shape[1]
    if kind == "dense":
        out, _, _ = _flash_dense_attn_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
            pack_gqa=False,
        )
    elif kind == "sparse":
        out, _, _, _ = _flash_sparse_attn_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            pack_gqa=False,
        )
    else:
        out, _, _, _, _ = _flash_gated_attn_forward(
            q,
            k,
            v,
            alpha,
            delta,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_adapt_gate=False,
            pack_gqa=False,
        )
    return out, softmax_scale


def run_forward_base_case(
    kind: KernelType,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> None:
    device = torch.device("cuda")
    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    alpha = (
        torch.randn(batch_size, seqlen_q, num_heads_q, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(batch_size, seqlen_k, num_heads_kv, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )

    out, softmax_scale = _run_forward_base(
        kind,
        q,
        k,
        v,
        is_causal=is_causal,
        is_local=is_local,
        alpha=alpha,
        delta=delta,
        is_logsigmoid_gate=is_logsigmoid_gate,
    )
    if kind == "gated":
        out_ref = reference_gated_forward(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
            is_logsigmoid_gate=is_logsigmoid_gate,
        )
    elif kind == "sparse":
        out_ref = reference_sparse_forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
        )
    else:
        out_ref = reference_dense_forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
        )
    _assert_close(
        name=f"{kind}-base-forward",
        got=out,
        ref=out_ref,
        rtol=_DEFAULT_RTOL[kind],
        atol=_DEFAULT_ATOL[kind],
    )


def _run_forward_varlen(
    kind: KernelType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    is_local: bool,
    alpha: Optional[torch.Tensor],
    delta: Optional[torch.Tensor],
    is_logsigmoid_gate: bool,
):
    softmax_scale = q.shape[-1] ** -0.5
    threshold = q.shape[-1] / max_seqlen_k
    if kind == "dense":
        out, _, _ = _flash_dense_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
            pack_gqa=False,
        )
    elif kind == "sparse":
        out, _, _, _ = _flash_sparse_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            pack_gqa=False,
        )
    else:
        out, _, _, _, _ = _flash_gated_attn_varlen_forward(
            q,
            k,
            v,
            alpha,
            delta,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_adapt_gate=False,
            is_local=is_local,
            pack_gqa=False,
        )
    return out, softmax_scale


def run_forward_varlen_case(
    kind: KernelType,
    lens_q: Sequence[int],
    lens_k: Sequence[int],
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> None:
    device = torch.device("cuda")
    q = torch.cat(
        [
            torch.randn(seq_len, num_heads_q, head_dim, device=device, dtype=dtype)
            for seq_len in lens_q
        ],
        dim=0,
    )
    k = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    v = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    alpha = (
        torch.randn(sum(lens_q), num_heads_q, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(sum(lens_k), num_heads_kv, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )

    cu_seqlens_q = make_cu_seqlens(lens_q, device)
    cu_seqlens_k = make_cu_seqlens(lens_k, device)

    out, softmax_scale = _run_forward_varlen(
        kind,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(lens_q),
        max_seqlen_k=max(lens_k),
        is_causal=is_causal,
        is_local=is_local,
        alpha=alpha,
        delta=delta,
        is_logsigmoid_gate=is_logsigmoid_gate,
    )
    out_ref = _reference_varlen_forward(
        kind,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        is_local=is_local,
        alpha=alpha,
        delta=delta,
        is_logsigmoid_gate=is_logsigmoid_gate,
    )
    _assert_close(
        name=f"{kind}-varlen-forward",
        got=out,
        ref=out_ref,
        rtol=_DEFAULT_RTOL[kind],
        atol=_DEFAULT_ATOL[kind],
    )


def _quantize_per_tensor_fp8(x: torch.Tensor):
    """Quantize a tensor to float8_e5m2 with per-tensor scale."""
    amax = x.abs().amax().clamp(min=1e-12)
    scale = amax / torch.finfo(torch.float8_e5m2).max
    x_fp8 = (x / scale).to(torch.float8_e5m2)
    return x_fp8, scale.to(torch.float32)


def run_decode_base_case(
    kind: KernelType,
    batch_size: int,
    seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    use_output_buffers: bool = False,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
    is_quant: bool = False,
) -> None:
    device = torch.device("cuda")
    is_fp8 = dtype == torch.float8_e5m2

    # For FP8: generate in bf16, then quantize
    gen_dtype = torch.bfloat16 if is_fp8 else dtype
    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=gen_dtype)
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=gen_dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=gen_dtype
    )
    alpha = (
        torch.randn(batch_size, num_heads_q, device=device, dtype=gen_dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(batch_size, seqlen_k, num_heads_kv, device=device, dtype=gen_dtype)
        if kind == "gated"
        else None
    )
    softmax_scale = head_dim**-0.5
    threshold = head_dim / seqlen_k

    if is_fp8:
        q_fp8, q_scale = _quantize_per_tensor_fp8(q)
        k_fp8, k_scale = _quantize_per_tensor_fp8(k)
        v_fp8, v_scale = _quantize_per_tensor_fp8(v)

        out_buffer = (
            torch.empty(
                batch_size, num_heads_q, head_dim, device=device, dtype=torch.bfloat16
            )
            if use_output_buffers
            else None
        )
        lse_buffer = (
            torch.empty(batch_size, num_heads_q, device=device, dtype=torch.float32)
            if use_output_buffers
            else None
        )

        if kind == "dense":
            out, lse = flash_dense_attn_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )
        elif kind == "sparse":
            out, lse = flash_sparse_attn_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                softmax_scale=softmax_scale,
                softmax_threshold=threshold,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )
        else:
            out, lse = flash_gated_attn_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                alpha,
                delta,
                softmax_scale=softmax_scale,
                softmax_threshold=threshold,
                gate_threshold=threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )

        # Reference: dequantize then compute
        q_deq = q_fp8.to(torch.bfloat16) * q_scale
        k_deq = k_fp8.to(torch.bfloat16) * k_scale
        v_deq = v_fp8.to(torch.bfloat16) * v_scale
        if kind == "dense":
            out_ref = reference_dense_forward(
                q_deq.unsqueeze(1),
                k_deq,
                v_deq,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
            ).squeeze(1)
        elif kind == "sparse":
            out_ref = reference_sparse_forward(
                q_deq.unsqueeze(1),
                k_deq,
                v_deq,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
            ).squeeze(1)
        else:
            out_ref = reference_gated_forward(
                q_deq.unsqueeze(1),
                k_deq,
                v_deq,
                alpha.unsqueeze(1),
                delta,
                softmax_scale=softmax_scale,
                is_causal=False,
                is_local=is_local,
                is_logsigmoid_gate=is_logsigmoid_gate,
            ).squeeze(1)

        if use_output_buffers:
            assert out.data_ptr() == out_buffer.data_ptr()
            assert lse.data_ptr() == lse_buffer.data_ptr()

        _assert_close(
            name=f"{kind}-base-decode-fp8",
            got=out,
            ref=out_ref,
            rtol=1.5e-1,
            atol=1.5e-1,
        )
        return

    out_buffer = torch.empty_like(q) if use_output_buffers else None
    lse_buffer = (
        torch.empty(batch_size, num_heads_q, device=device, dtype=torch.float32)
        if use_output_buffers
        else None
    )

    if kind == "dense":
        out, lse = flash_dense_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )
    elif kind == "sparse":
        out, lse = flash_sparse_attn_with_kvcache_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )
    else:
        out, lse = flash_gated_attn_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )

    if kind == "dense":
        out_ref = reference_dense_forward(
            q.unsqueeze(1),
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=False,
            is_local=is_local,
        ).squeeze(1)
    elif kind == "sparse":
        out_ref = reference_sparse_forward(
            q.unsqueeze(1),
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=False,
            is_local=is_local,
        ).squeeze(1)
    else:
        out_ref = reference_gated_forward(
            q.unsqueeze(1),
            k,
            v,
            alpha.unsqueeze(1),
            delta,
            softmax_scale=softmax_scale,
            is_causal=False,
            is_local=is_local,
            is_logsigmoid_gate=is_logsigmoid_gate,
        ).squeeze(1)

    if use_output_buffers:
        assert out.data_ptr() == out_buffer.data_ptr()
        assert lse.data_ptr() == lse_buffer.data_ptr()

    _assert_close(
        name=f"{kind}-base-decode{'_auto_quant' if is_quant else ''}",
        got=out,
        ref=out_ref,
        rtol=1.5e-1 if is_quant else _DEFAULT_RTOL[kind],
        atol=1.5e-1 if is_quant else _DEFAULT_ATOL[kind],
    )


def run_decode_varlen_case(
    kind: KernelType,
    lens_k: Sequence[int],
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    use_output_buffers: bool = False,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
    is_quant: bool = False,
) -> None:
    device = torch.device("cuda")
    batch_size = len(lens_k)
    is_fp8 = dtype == torch.float8_e5m2

    gen_dtype = torch.bfloat16 if is_fp8 else dtype
    q = torch.randn(batch_size, num_heads_q, head_dim, device=device, dtype=gen_dtype)
    k = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=gen_dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    v = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=gen_dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    alpha = (
        torch.randn(batch_size, num_heads_q, device=device, dtype=gen_dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(sum(lens_k), num_heads_kv, device=device, dtype=gen_dtype)
        if kind == "gated"
        else None
    )

    cu_seqlens_k = make_cu_seqlens(lens_k, device)
    softmax_scale = head_dim**-0.5
    threshold = head_dim / max(lens_k)

    if is_fp8:
        q_fp8, q_scale = _quantize_per_tensor_fp8(q)
        k_fp8, k_scale = _quantize_per_tensor_fp8(k)
        v_fp8, v_scale = _quantize_per_tensor_fp8(v)

        out_buffer = (
            torch.empty(
                batch_size, num_heads_q, head_dim, device=device, dtype=torch.bfloat16
            )
            if use_output_buffers
            else None
        )
        lse_buffer = (
            torch.empty(batch_size, num_heads_q, device=device, dtype=torch.float32)
            if use_output_buffers
            else None
        )

        if kind == "dense":
            out, lse = flash_dense_attn_varlen_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max(lens_k),
                softmax_scale=softmax_scale,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )
        elif kind == "sparse":
            out, lse = flash_sparse_attn_varlen_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max(lens_k),
                softmax_scale=softmax_scale,
                softmax_threshold=threshold,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )
        else:
            out, lse = flash_gated_attn_varlen_with_kvcache_func(
                q_fp8,
                k_fp8,
                v_fp8,
                alpha,
                delta,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max(lens_k),
                softmax_scale=softmax_scale,
                softmax_threshold=threshold,
                gate_threshold=threshold,
                is_logsigmoid_gate=is_logsigmoid_gate,
                query_scale=q_scale,
                key_scale=k_scale,
                value_scale=v_scale,
                is_local=is_local,
                return_lse=True,
                out=out_buffer,
                lse=lse_buffer,
            )

        # Reference: dequantize then compute
        q_deq = q_fp8.to(torch.bfloat16) * q_scale
        k_deq = k_fp8.to(torch.bfloat16) * k_scale
        v_deq = v_fp8.to(torch.bfloat16) * v_scale
        out_ref = _reference_varlen_decode(
            kind,
            q_deq,
            k_deq,
            v_deq,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            is_local=is_local,
            alpha=alpha,
            delta=delta,
            is_logsigmoid_gate=is_logsigmoid_gate,
        )

        if use_output_buffers:
            assert out.data_ptr() == out_buffer.data_ptr()
            assert lse.data_ptr() == lse_buffer.data_ptr()

        _assert_close(
            name=f"{kind}-varlen-decode-fp8",
            got=out,
            ref=out_ref,
            rtol=1.5e-1,
            atol=1.5e-1,
        )
        return

    out_buffer = torch.empty_like(q) if use_output_buffers else None
    lse_buffer = (
        torch.empty(batch_size, num_heads_q, device=device, dtype=torch.float32)
        if use_output_buffers
        else None
    )

    if kind == "dense":
        out, lse = flash_dense_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lens_k),
            softmax_scale=softmax_scale,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )
    elif kind == "sparse":
        out, lse = flash_sparse_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lens_k),
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )
    else:
        out, lse = flash_gated_attn_varlen_with_kvcache_func(
            q,
            k,
            v,
            alpha,
            delta,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(lens_k),
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_quant=is_quant,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
        )

    out_ref = _reference_varlen_decode(
        kind,
        q,
        k,
        v,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        is_local=is_local,
        alpha=alpha,
        delta=delta,
        is_logsigmoid_gate=is_logsigmoid_gate,
    )

    if use_output_buffers:
        assert out.data_ptr() == out_buffer.data_ptr()
        assert lse.data_ptr() == lse_buffer.data_ptr()

    _assert_close(
        name=f"{kind}-varlen-decode{'_auto_quant' if is_quant else ''}",
        got=out,
        ref=out_ref,
        rtol=1.5e-1 if is_quant else _DEFAULT_RTOL[kind],
        atol=1.5e-1 if is_quant else _DEFAULT_ATOL[kind],
    )


def _collect_grads(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.grad.detach().clone() for name, tensor in params.items()}


def run_backward_base_case(
    kind: KernelType,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> None:
    device = torch.device("cuda")
    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype
    )
    alpha = (
        torch.randn(batch_size, seqlen_q, num_heads_q, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(batch_size, seqlen_k, num_heads_kv, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )
    softmax_scale = head_dim**-0.5
    threshold = head_dim / seqlen_k

    if kind == "dense":
        out, lse, _ = _flash_dense_attn_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
            pack_gqa=False,
        )
    elif kind == "sparse":
        out, lse, _, _ = _flash_sparse_attn_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            pack_gqa=False,
        )
    else:
        out, lse, _, _, _ = _flash_gated_attn_forward(
            q,
            k,
            v,
            alpha,
            delta,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_adapt_gate=False,
            pack_gqa=False,
        )

    dout = torch.randn_like(out)

    if kind == "dense":
        kernel_grads = _flash_dense_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
        )
    elif kind == "sparse":
        kernel_grads = _flash_sparse_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
        )
    else:
        kernel_grads = _flash_gated_attn_backward(
            q,
            k,
            v,
            alpha,
            delta,
            out,
            dout,
            lse,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_adapt_gate=False,
        )

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)
    ref_params = {"dq": rq, "dk": rk, "dv": rv}
    if kind == "gated":
        ra = alpha.detach().clone().requires_grad_(True)
        rd = delta.detach().clone().requires_grad_(True)
        ref_out = reference_gated_forward(
            rq,
            rk,
            rv,
            ra,
            rd,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
            is_logsigmoid_gate=is_logsigmoid_gate,
        )
        ref_params["da"] = ra
        ref_params["dd"] = rd
    elif kind == "sparse":
        ref_out = reference_sparse_forward(
            rq,
            rk,
            rv,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
        )
    else:
        ref_out = reference_dense_forward(
            rq,
            rk,
            rv,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
        )
    ref_out.backward(dout)
    ref_grads = _collect_grads(ref_params)

    names = ["dq", "dk", "dv"] if kind != "gated" else ["dq", "dk", "dv", "da", "dd"]
    for idx, name in enumerate(names):
        _assert_close(
            name=f"{kind}-base-backward-{name}",
            got=kernel_grads[idx],
            ref=ref_grads[name],
            rtol=_DEFAULT_BWD_RTOL[kind],
            atol=_DEFAULT_BWD_ATOL[kind],
        )


def run_backward_varlen_case(
    kind: KernelType,
    lens_q: Sequence[int],
    lens_k: Sequence[int],
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    is_causal: bool,
    is_local: bool = False,
    is_logsigmoid_gate: bool = True,
    dtype: torch.dtype = CORRECTNESS_DTYPE,
) -> None:
    device = torch.device("cuda")
    q = torch.cat(
        [
            torch.randn(seq_len, num_heads_q, head_dim, device=device, dtype=dtype)
            for seq_len in lens_q
        ],
        dim=0,
    )
    k = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    v = torch.cat(
        [
            torch.randn(seq_len, num_heads_kv, head_dim, device=device, dtype=dtype)
            for seq_len in lens_k
        ],
        dim=0,
    )
    alpha = (
        torch.randn(sum(lens_q), num_heads_q, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )
    delta = (
        torch.randn(sum(lens_k), num_heads_kv, device=device, dtype=dtype)
        if kind == "gated"
        else None
    )

    cu_seqlens_q = make_cu_seqlens(lens_q, device)
    cu_seqlens_k = make_cu_seqlens(lens_k, device)
    max_seqlen_k = max(lens_k)
    softmax_scale = head_dim**-0.5
    threshold = head_dim / max_seqlen_k

    if kind == "dense":
        out, lse, _ = _flash_dense_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max(lens_k),
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
            pack_gqa=False,
        )
    elif kind == "sparse":
        out, lse, _, _ = _flash_sparse_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
            pack_gqa=False,
        )
    else:
        out, lse, _, _, _ = _flash_gated_attn_varlen_forward(
            q,
            k,
            v,
            alpha,
            delta,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_adapt_gate=False,
            pack_gqa=False,
        )

    dout = torch.randn_like(out)
    if kind == "dense":
        kernel_grads = _flash_dense_attn_varlen_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max(lens_k),
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            is_local=is_local,
        )
    elif kind == "sparse":
        kernel_grads = _flash_sparse_attn_varlen_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            is_local=is_local,
        )
    else:
        kernel_grads = _flash_gated_attn_varlen_backward(
            q,
            k,
            v,
            alpha,
            delta,
            out,
            dout,
            lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(lens_q),
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            softmax_threshold=threshold,
            gate_threshold=threshold,
            is_logsigmoid_gate=is_logsigmoid_gate,
            is_local=is_local,
            is_adapt_gate=False,
        )

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)
    ref_params = {"dq": rq, "dk": rk, "dv": rv}
    if kind == "gated":
        ra = alpha.detach().clone().requires_grad_(True)
        rd = delta.detach().clone().requires_grad_(True)
        ref_out = _reference_varlen_forward(
            kind,
            rq,
            rk,
            rv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
            alpha=ra,
            delta=rd,
            is_logsigmoid_gate=is_logsigmoid_gate,
        )
        ref_params["da"] = ra
        ref_params["dd"] = rd
    else:
        ref_out = _reference_varlen_forward(
            kind,
            rq,
            rk,
            rv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            is_local=is_local,
        )
    ref_out.backward(dout)
    ref_grads = _collect_grads(ref_params)

    names = ["dq", "dk", "dv"] if kind != "gated" else ["dq", "dk", "dv", "da", "dd"]
    for idx, name in enumerate(names):
        _assert_close(
            name=f"{kind}-varlen-backward-{name}",
            got=kernel_grads[idx],
            ref=ref_grads[name],
            rtol=_DEFAULT_BWD_RTOL[kind],
            atol=_DEFAULT_BWD_ATOL[kind],
        )
