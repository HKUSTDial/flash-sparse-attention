from typing import Optional, Tuple
import torch
import torch.nn as nn


from flash_sparse_attn.flash_sparse_attn_interface import flash_sparse_attn_func
from flash_sparse_attn.utils.mask import create_mask


class DynamicMaskAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.g_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.num_key_value_heads,
            bias=False,
        )
        self.d_proj = nn.Linear(
            config.num_key_value_heads * self.head_dim,
            config.num_key_value_heads,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if past_key_values is not None:
            past_key, past_value = past_key_values
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)
        key_len = key_states.size(1)

        gate_states = self.g_proj(query_states)
        delta_states = self.d_proj(value_states)
        attn_bias = torch.sigmoid(gate_states) * delta_states

        query_states = query_states.view(bsz, seq_len, -1, self.head_dim)
        key_states = key_states.view(bsz, key_len, -1, self.head_dim)
        value_states = value_states.view(bsz, key_len, -1, self.head_dim)
        attn_bias = attn_bias.transpose(-1, -2).unsqueeze(-2)

        attn_mask = create_mask(
            attention_bias=attn_bias,
            query_len=query_states.shape[2],
            type="relu",
        )

        attn_output = flash_sparse_attn_func(
            query_states,
            key_states,
            value_states,
            attn_mask,
            attn_bias,
            softmax_scale=self.scaling,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, (key_states, value_states)
