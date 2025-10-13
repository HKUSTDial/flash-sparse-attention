from typing import Optional

import torch

from .modeling_flash_dynamic_mask_attention_utils import _flash_dynamic_mask_attention_forward
from transformers.utils import logging


logger = logging.get_logger(__name__)


def flash_dynamic_mask_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attention_bias: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    window_size: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    A wrapper around the _flash_dynamic_mask_attention_forward function to be used in
    the FlashDynamicMaskAttention class from HuggingFace Transformers.

    Args:
        module (torch.nn.Module): The attention module.
        query (torch.Tensor): The query tensor of shape (batch_size, num_heads, query_len, head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_kv_heads, key_len, head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_kv_heads, key_len, head_dim).
        attention_mask (Optional[torch.Tensor]): The attention mask boolean tensor of shape
        (batch_size, seq_len) or ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, {key_len|1}).
        attention_bias (Optional[torch.Tensor]): The attention bias float tensor of shape
        ({batch_size|1}, {num_heads|num_kv_heads|1}, {query_len|1}, {key_len|1}).
        scaling (Optional[float]): The scaling factor for the attention scores.
        window_size (Optional[int]): The size of the window to keep.
        softcap (Optional[float]): The softcap value for the attention scores.
        **kwargs: Additional keyword arguments.
            Includes:
                - is_causal (bool): Whether to apply a causal mask.
                - layer_idx (int): The index of the layer (for logging purposes).
                - implementation (str): The implementation to use ("flash_dmattn" or None).

    Returns:
        tuple[torch.Tensor, None]: The output tensor of shape (batch_size, seq_len, num_heads, head_dim)
        and None (for compatibility with other attention implementations).
    """

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_dynamic_mask_attention` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # This is before the transpose
    query_len = query.shape[2]
    key_len = key.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashDynamicMaskAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    # FDMA uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    attn_output = _flash_dynamic_mask_attention_forward(
        query,
        key,
        value,
        attention_mask,
        attention_bias,
        query_length=query_len,
        key_length=key_len,
        is_causal=is_causal,
        softmax_scale=scaling,
        softcap=softcap,
        window_size=window_size,
        target_dtype=target_dtype,
        implementation="flash_dmattn",
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    )

    return attn_output, None
