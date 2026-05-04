from typing import Optional
import torch
import torch.nn as nn


from flash_sparse_attn.ops.triton.interface import flash_gated_attn_func


class FlashSparseAttentionConfig:
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        softmax_threshold: Optional[float] = None,
        gate_threshold: Optional[float] = None,
        max_position_embeddings: int = 32768,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.softmax_threshold = softmax_threshold
        self.gate_threshold = gate_threshold
        self.max_position_embeddings = max_position_embeddings


class FlashSparseAttention(nn.Module):
    def __init__(
        self, config: FlashSparseAttentionConfig, layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.softmax_threshold = config.softmax_threshold
        self.gate_threshold = config.gate_threshold

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.a_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads, bias=False
        )
        self.d_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )

        alpha_states = self.a_proj(hidden_states)
        delta_states = self.d_proj(hidden_states)

        attn_output = flash_gated_attn_func(
            query_states,
            key_states,
            value_states,
            alpha_states,
            delta_states,
            is_causal=self.is_causal,
            softmax_scale=self.scaling,
            softmax_threshold=self.softmax_threshold,
            gate_threshold=self.gate_threshold,
        )

        attn_output = attn_output.view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (key_states, value_states)
