# Copyright 2025 Jingze Shi and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, TypedDict
import torch
from torch.nn import functional as F
from .import_utils import is_flash_dmattn_available

from transformers.utils import logging


logger = logging.get_logger(__name__)


def _index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    reshaped = tensor.contiguous().reshape(-1, *tensor.shape[2:])
    return reshaped[indices]


def _fdma_unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    FDMA-compatible unpad_input function.
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        _index_first_axis(hidden_states, indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def _fdma_pad_input(hidden_states, indices, batch, seqlen):
    """
    FDMA-compatible pad_input function.
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.
    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # NOTE: Similar to the `.item()` in prepare_fa2_from_position_ids, with torch compile,
    # this might cause a graph break
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    bias_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    unpad_input_func,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.
    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.
    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        bias_layer (`torch.Tensor`):
            Attention bias tensor of shape (batch_size, num_key_value_heads, query_length, kv_seq_len).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        unpad_input_func:
            The function to use for unpadding the input tensors.
    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        bias_layer (`torch.Tensor`):
            Attention bias tensor without padding. Shape: (total_target_length, num_key_value_heads, query_length, kv_seq_len).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    # With static caches, the k/v states may be larger than the mask -> we need to slice them to avoid generating garbage
    # It's a bit of an anti-pattern, but otherwise we silently compute wrong attentions scores
    if key_layer.shape[1] > (seq_len := attention_mask.shape[-1]):
        key_layer, value_layer = key_layer[:, :seq_len, :, :], value_layer[:, :seq_len, :, :]

    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    key_lens_per_batch = attention_mask.sum(-1)

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)

    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k

        query_mask = attention_mask
        bias_view = bias_layer
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)

        query_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        bias_view = bias_layer[:, :, :1, :]
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input_func(query_layer, attention_mask)

        query_mask = attention_mask[:, -query_length:]
        bias_view = bias_layer[:, :, -query_length:, :]
    
    b_idx_q, pos_in_q = torch.nonzero(query_mask, as_tuple=True)
    bias_layer = bias_view[b_idx_q, :, pos_in_q, :]
    row_key_lens = key_lens_per_batch[b_idx_q]
    bias_layer = bias_layer[:, :, :max_seqlen_in_batch_k]
    col_idx = torch.arange(max_seqlen_in_batch_k, device=query_layer.device).view(1, 1, max_seqlen_in_batch_k)
    valid_cols = col_idx < row_key_lens.view(-1, 1, 1)
    bias_layer = bias_layer * valid_cols

    return (
        query_layer,
        key_layer,
        value_layer,
        bias_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def fdma_peft_integration_check(q, k, v, target_dtype: Optional[torch.dtype] = None):
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    return q, k, v


def _lazy_imports(impl: Optional[str]):
    # returns funcs and pad/unpad based on impl
    is_fdma = is_flash_dmattn_available()

    if impl == "flash_dmattn" or (impl is None and is_fdma):
        pad_input, unpad_input = _fdma_pad_input, _fdma_unpad_input
        from flash_dmattn import flash_dmattn_func, flash_dmattn_varlen_func
        return flash_dmattn_func, flash_dmattn_varlen_func, pad_input, unpad_input

    else:
        pad_input, unpad_input = _fdma_pad_input, _fdma_unpad_input
        return (
            getattr(impl, "flash_dmattn_func", None),
            getattr(impl, "flash_dmattn_varlen_func", None),
            pad_input,
            unpad_input,
        )


class FlashDynamicMaskAttentionKwargs(TypedDict, total=False):
    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    

def _flash_dynamic_mask_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attention_bias: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    
    if not all(k in globals() for k in ("_flash_fn", "_flash_varlen_fn", "_pad_fn", "_unpad_fn")):
        flash_fn, flash_varlen_fn, pad_fn, unpad_fn = _lazy_imports(implementation)
        globals()["_flash_fn"] = flash_fn
        globals()["_flash_varlen_fn"] = flash_varlen_fn
        globals()["_pad_fn"] = pad_fn
        globals()["_unpad_fn"] = unpad_fn
    else:
        flash_fn = globals()["_flash_fn"]
        flash_varlen_fn = globals()["_flash_varlen_fn"]
        pad_fn = globals()["_pad_fn"]
        unpad_fn = globals()["_unpad_fn"]

    is_causal = is_causal and not query_length == 1
    flash_kwargs = {}
    if deterministic is not None:
        flash_kwargs["deterministic"] = deterministic
    if softcap is not None:
        flash_kwargs["softcap"] = softcap
    query_states, key_states, value_states = fdma_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )
    if attention_mask is not None:
        q, k, v, bias, idx, (cu_q, cu_k), (mq, mk) = _upad_input(
            query_states, key_states, value_states, attention_bias, attention_mask, query_length, _fdma_unpad_input
        )
        if "mps" in str(q.device):
            cu_k = cu_k.clone()
        out_unpad = flash_varlen_fn(
            query=q,
            key=k,
            value=v,
            attn_bias=bias,
            cu_seqlens_q=cu_q.to(torch.int32),
            cu_seqlens_k=cu_k.to(torch.int32),
            max_seqlen_q=mq,
            max_seqlen_k=mk,
            scale=softmax_scale,
            is_causal=is_causal,
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]
        out = _fdma_pad_input(out_unpad, idx, query_states.shape[0], query_length)
    else:
        out = flash_fn(
            query_states, key_states, value_states, attn_bias=attention_bias, scale=softmax_scale, is_causal=is_causal
        )

    return out[0] if isinstance(out, tuple) else out
