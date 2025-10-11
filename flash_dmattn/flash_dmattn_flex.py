from typing import Optional, Tuple
import math
import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers.integrations.flex_attention import compile_friendly_flex_attention


def flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    is_causal: Optional[bool] = None,
    softmax_scale: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen_q, nheads, dhead = query.shape
    _, seqlen_k, _, _ = key.shape
    query = query.transpose(1, 2).contiguous()  # [B, H, Q_LEN, D]
    key = key.transpose(1, 2).contiguous()      # [B, H, KV_LEN, D]
    value = value.transpose(1, 2).contiguous()  # [B, H, KV_LEN, D]
    if attn_mask is not None:
        attn_mask = attn_mask[:, :, :, : key.shape[-2]]
    else:
        attn_mask = torch.ones((batch, nheads, seqlen_q, seqlen_k), device=query.device, dtype=query.dtype)
    if attn_bias is not None:
        attn_bias = attn_bias[:, :, :, : key.shape[-2]]
    else:
        attn_bias = torch.zeros((batch, nheads, seqlen_q, seqlen_k), device=query.device, dtype=query.dtype)
    if is_causal is None:
        is_causal = True
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dhead)

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        score = score + attn_bias[batch_idx][head_idx][q_idx][kv_idx]
        return score

    def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        # It looks like you're attempting to use a Tensor in some data-dependent control flow.
        # We don't support that yet, please shout over at https://github.com/pytorch/functorch/issues/257 .
        # return q_idx >= kv_idx and attn_mask[batch_idx][head_idx][q_idx][kv_idx] > 0
        return q_idx >= kv_idx
    
    block_mask = create_block_mask(
        mask_mod=causal_mask_mod,
        B=query.shape[0],
        H=None,
        Q_LEN=query.shape[2],
        KV_LEN=key.shape[2],
        device=query.device,
        _compile=True,
    )

    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_DMODEL": 32,
        "num_stages": 1, 
        "num_warps": 8,
    }
    attn_output = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask if is_causal else None,
        scale=softmax_scale,
        kernel_options=kernel_options,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=False,
        training=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output

flex_dmattn_func = flex_attention_forward