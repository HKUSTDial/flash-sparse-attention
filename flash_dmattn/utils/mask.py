# Copyright 2025 Jingze Shi and Liangdong Wang. All rights reserved.
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

from typing import Optional

import torch


def dynamic_mask(
    attention_bias: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    window_size: int,
    min_dtype: float,
):
    r"""
    This function generates a dynamic mask based on the top-k attention bias.

    Args:
        attention_bias (torch.Tensor): The attention bias tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        attention_mask (Optional[torch.Tensor]): The attention mask boolean tensor of shape 
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        window_size (int): The number of top elements to consider for the mask.
        min_dtype (float): The minimum value to use for masking.
    
    Returns:
        attention_mask (Tensor): The attention mask tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
    """
    attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype) if attention_mask is not None else attention_bias
    topk_values, topk_indices = torch.topk(
        attention_bias.detach(),
        window_size, dim=-1, largest=True, sorted=False
    )
    attention_mask = torch.zeros_like(
        attention_bias, dtype=torch.bool, device=attention_bias.device
    ).scatter_(-1, topk_indices, topk_values != min_dtype)
    return attention_mask


def create_mask(
    attention_bias: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
    window_size: int,
    min_dtype: float,
) -> torch.Tensor:
    r"""
    This function creates a mask tensor for Flash Dynamic Mask Attention.

    If attention_mask is not of shape (batch_size, seq_len), it needs to match the shape of attention_bias.
    
    Args:
        Args:
        attention_bias (torch.Tensor): The attention bias tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, {key_len|1}).
        attention_mask (Optional[torch.Tensor]): The attention mask boolean tensor of shape 
            (batch_size, seq_len) or ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, {key_len|1}).
        batch_size (int): The batch size.
        query_len (int): The sequence length of the query.
        key_len (int): The sequence length of the key.
        window_size (int): The number of top elements to consider for the attention mask.
        min_dtype (float): The minimum value to use for masking.

    Returns:
        attention (Tensor): The attention mask tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, {key_len|1}).
    """

    # If attention_mask is of shape (batch_size, seq_len), reshape it to (batch_size, 1, 1, key_len)
    if attention_mask is not None and attention_mask.dim() == 2:
        if attention_mask.shape[-1] == key_len:
            attention_mask = attention_mask.view(batch_size, 1, 1, key_len)
        elif attention_mask.shape[-1] == query_len:
            pad_len = key_len - query_len
            if pad_len > 0:
                pad_mask = torch.ones(
                    (batch_size, 1, 1, pad_len),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    [pad_mask, attention_mask.view(batch_size, 1, 1, query_len)],
                    dim=-1,
                )
            else:
                attention_mask = attention_mask.view(batch_size, 1, 1, query_len)
        else:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} is not compatible with key_len {key_len} or query_len {query_len}."
            )
    
    attention_mask = dynamic_mask(attention_bias, attention_mask, window_size, min_dtype)

    return attention_mask
