"""Real-valued transformer layers with optional RoPE (Chronos-2 style)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with optional rotary position embeddings."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if use_rope:
            self.rope_embed = RotaryEmbedding(dim=self.head_dim, theta=rope_theta)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.nhead, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[batch, seq_len, d_model]``
            attention_mask: Additive mask ``[batch, 1, q_len, kv_len]`` or
                ``[batch, nhead, q_len, kv_len]`` (0 keep, ``-inf`` block).
            position_ids: ``[batch, seq_len]`` required when ``use_rope=True``.
        """
        if self.use_rope and position_ids is None:
            raise ValueError("position_ids must be provided when use_rope=True")

        query_states = self._split_heads(self.q_proj(hidden_states))
        key_states = self._split_heads(self.k_proj(hidden_states))
        value_states = self._split_heads(self.v_proj(hidden_states))

        if self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=scale,
        )
        return self.out_proj(self._merge_heads(attn_output))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with optional RoPE attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = MultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.ffn = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.self_attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + self.dropout(attn_out)
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer encoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return hidden_states
