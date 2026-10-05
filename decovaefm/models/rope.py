"""Rotary position embeddings (RoPE), adapted from Chronos-2.

Reference:
https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Applies rotary position embeddings (RoPE) to query/key tensors.

    Follows the Chronos-2 / transformers RotaryEmbedding pattern:
    inverse frequencies are registered as non-persistent buffers and cos/sin
    are computed from ``position_ids`` in the forward pass.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0, device: torch.device | None = None):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dimension must be even, got {dim}")

        self.dim = dim
        self.theta = theta
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for the given positions.

        Args:
            x: Reference tensor used for dtype/device (e.g. values), typically
                shaped ``[batch, n_heads, seq_len, head_dim]``.
            position_ids: Integer positions of shape ``[batch, seq_len]``.

        Returns:
            ``(cos, sin)`` each of shape ``[batch, seq_len, head_dim]``.
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor ``[batch, heads, seq, head_dim]``.
            k: Key tensor ``[batch, heads, seq, head_dim]``.
            cos: Cosine embedding ``[batch, seq, head_dim]``.
            sin: Sine embedding ``[batch, seq, head_dim]``.
            unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
                onto ``q``/``k`` (1 when heads are at dim 1).
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed
