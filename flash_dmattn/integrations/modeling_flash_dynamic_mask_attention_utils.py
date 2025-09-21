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
from .import_utils import is_flash_dmattn_available

from transformers.utils import logging
from transformers.integrations import flash_attention


logger = logging.get_logger(__name__)


def fdma_peft_integration_check(q, k, v, bias, target_dtype: Optional[torch.dtype] = None):
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-dmattn compatibility.")
        q, k, v, bias = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype), bias.to(target_dtype)
    return q, k, v, bias


def _lazy_imports(impl: Optional[str]):
    # returns funcs based on impl
    is_fdma = is_flash_dmattn_available()

    if impl == "flash_dmattn" or (impl is None and is_fdma):
        from flash_dmattn import flash_dmattn_func
        return flash_dmattn_func

    else:
        return getattr(impl, "flash_dmattn_func", None)


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
    key_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    keep_window_size: Optional[int] = None,
    deterministic: Optional[bool] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    dtype = query_states.dtype
    min_dtype = torch.finfo(dtype).min
    batch_size, _, num_kv_heads, _ = key_states.shape

    if not all(k in globals() for k in ("_flash_fn")):
        flash_fn = _lazy_imports(implementation)
        globals()["_flash_fn"] = flash_fn
    else:
        flash_fn = globals()["_flash_fn"]

    is_causal = is_causal and not query_length == 1
    flash_kwargs = {}
    if deterministic is not None:
        flash_kwargs["deterministic"] = deterministic
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    query_states, key_states, value_states, attention_bias = fdma_peft_integration_check(
        query_states, key_states, value_states, attention_bias, target_dtype
    )

    if attention_mask is not None and attention_mask.dim() == 4:
        if attention_bias.dim() == 3:
            attention_bias = attention_bias.unsqueeze(-2)
        attention_bias = attention_bias.masked_fill(
            ~attention_mask,
            min_dtype
        )

    if keep_window_size is not None and key_length > keep_window_size:
        topk_values, topk_indices = torch.topk(
            attention_bias, keep_window_size, dim=-1, largest=True, sorted=False
        )
        attention_mask = torch.zeros_like(attention_bias, dtype=torch.bool, device=attention_bias.device)
        attention_mask = attention_mask.scatter(-1, topk_indices, topk_values != min_dtype)
    else:
        attention_mask = None

    out = flash_fn(
        query_states, key_states, value_states, attn_mask=attention_mask, attn_bias=attention_bias, scale=softmax_scale, is_causal=is_causal
    )

    return out[0] if isinstance(out, tuple) else out
