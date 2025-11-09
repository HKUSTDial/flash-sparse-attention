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


def topk_indices(
    attention_bias: torch.Tensor,
    window_size: int,
    **kwargs,
) -> torch.Tensor:
    r"""
    This function generates top-k indices based on the attention bias.

    Args:
        attention_bias (torch.Tensor): The attention bias tensor of
            (batch_size, num_kv_heads, key_len).
        window_size (int): The number of top elements to consider for the mask.
        **kwargs: Additional keyword arguments.
    
    Returns:
        topk_indices (Tensor): The top-k indices tensor of shape
            (batch_size, num_kv_heads, window_size).
    """
    attention_bias = attention_bias.detach()
    topk_indices = torch.topk(
        attention_bias,
        window_size, dim=-1, largest=True, sorted=False
    ).indices
    topk_indices = torch.sort(topk_indices, dim=-1).values
    return topk_indices


def block_smooth(
    attention_mask: torch.Tensor,
    key_len: int,
    block_size: int,
):
    if block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}.")
 
    if block_size > 1:
        full_len = (key_len // block_size) * block_size

        if full_len:
            block_view = attention_mask[..., :full_len]
            block_shape = (*block_view.shape[:-1], full_len // block_size, block_size)
            blocks = block_view.view(*block_shape)
            block_counts = blocks.sum(dim=-1).to(torch.int64)
            block_keep = (block_counts * 2) > block_size
            blocks.copy_(block_keep.unsqueeze(-1).expand_as(blocks))

        if key_len > full_len:
            tail_slice = attention_mask[..., full_len:]
            tail_len = tail_slice.shape[-1]
            tail_counts = tail_slice.sum(dim=-1, keepdim=True).to(torch.int64)
            tail_keep = (tail_counts * 2) > tail_len
            tail_slice.copy_(tail_keep.expand_as(tail_slice))

    return attention_mask
    

def topk_mask(
    attention_bias: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    window_size: int,
    min_dtype: float,
    block_size: Optional[int] = None,
    **kwargs,
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
        block_size (Optional[int]): Optional size of aggregation blocks to smooth the
            resulting mask along the key dimension.
    
    Returns:
        attention_mask (Tensor): The attention mask tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
    """

    attention_bias = attention_bias.detach()
    attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype) if attention_mask is not None else attention_bias
    topk_values, topk_indices = torch.topk(
        attention_bias,
        window_size, dim=-1, largest=True, sorted=False
    )
    attention_mask = torch.zeros_like(
        attention_bias, dtype=torch.bool, device=attention_bias.device
    ).scatter_(-1, topk_indices, topk_values != min_dtype)

    if block_size is not None and block_size > 1:
        key_len = attention_mask.shape[-1]
        attention_mask = block_smooth(
            attention_mask=attention_mask,
            key_len=key_len,
            block_size=block_size
        )

    return attention_mask


def relu_mask(
    attention_bias: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    min_dtype: float,
    block_size: Optional[int] = None,
    **kwargs
):
    r"""
    This function generates a dynamic mask based on the ReLU of attention bias.

    Args:
        attention_bias (torch.Tensor): The attention bias tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        attention_mask (Optional[torch.Tensor]): The attention mask boolean tensor of shape 
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        min_dtype (float): The minimum value to use for masking.
        block_size (Optional[int]): Optional size of aggregation blocks to smooth the
            resulting mask along the key dimension.
    
    Returns:
        attention_mask (Tensor): The attention mask tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
    """

    attention_bias = attention_bias.detach()
    attention_bias = attention_bias.masked_fill(~attention_mask, min_dtype) if attention_mask is not None else attention_bias
    attention_mask = attention_bias > 0

    if block_size is not None and block_size > 1:
        key_len = attention_mask.shape[-1]
        attention_mask = block_smooth(
            attention_mask=attention_mask,
            key_len=key_len,
            block_size=block_size
        )

    return attention_mask



def create_mask(
    attention_bias: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
    window_size: int,
    min_dtype: float,
    block_size: Optional[int] = None,
    type: str = "topk",
) -> torch.Tensor:
    r"""
    This function creates a mask tensor for Flash Sparse Attention.

    If attention_mask is not of shape (batch_size, seq_len), it needs to match the shape of attention_bias.
    
    Args:
        attention_bias (torch.Tensor): The attention bias tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        attention_mask (Optional[torch.Tensor]): The attention mask boolean tensor of shape
            (batch_size, seq_len) or ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
        batch_size (int): The batch size.
        query_len (int): The sequence length of the query.
        key_len (int): The sequence length of the key.
        window_size (int): The number of top elements to consider for the attention mask.
        min_dtype (float): The minimum value to use for masking.
        block_size (Optional[int]): Optional size of aggregation blocks after top-k masking.
        type (str): The type of mask to create. Options are "topk" and "relu".

    Returns:
        attention (Tensor): The attention mask tensor of shape
            ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, key_len).
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
    
    # Generate dynamic mask based on attention_bias and attention_mask
    if type == "topk":
        attention_mask = topk_mask(
            attention_bias=attention_bias,
            attention_mask=attention_mask,
            window_size=window_size,
            min_dtype=min_dtype,
            block_size=block_size,
        )
    elif type == "relu":
        attention_mask = relu_mask(
            attention_bias=attention_bias,
            attention_mask=attention_mask,
            window_size=window_size,
            min_dtype=min_dtype,
            block_size=block_size,
        )
    else:
        raise ValueError(f"Unsupported mask type: {type}. Supported types are 'topk' and 'relu'.")

    return attention_mask
