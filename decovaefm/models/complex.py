import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


class ComplexLinear(nn.Module):
    """Complex-valued linear layer using real-imag decomposition."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_real = nn.Parameter(torch.empty(out_features))
        self.bias_imag = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        fan_in = self.weight_real.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias_real, -bound, bound)
        nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x.real
        xi = x.imag

        yr = F.linear(xr, self.weight_real, self.bias_real) - F.linear(
            xi, self.weight_imag
        )
        yi = F.linear(xi, self.weight_real, self.bias_imag) + F.linear(
            xr, self.weight_imag
        )
        return torch.complex(yr, yi)


class ComplexGELU(nn.Module):
    """Apply GELU independently to real and imaginary parts."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.gelu(x.real), F.gelu(x.imag))


class ComplexMLP(nn.Module):
    """Complex MLP with ``n_layers`` ComplexLinear layers and ComplexGELU between them."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if n_layers == 1:
            layers: list[nn.Module] = [ComplexLinear(in_dim, out_dim)]
        else:
            layers = [ComplexLinear(in_dim, hidden_dim), ComplexGELU()]
            for _ in range(n_layers - 2):
                layers.extend([ComplexLinear(hidden_dim, hidden_dim), ComplexGELU()])
            layers.append(ComplexLinear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexLayerNorm(nn.Module):
    """Apply layer norm to real and imaginary channels independently."""

    def __init__(self, dim: int):
        super().__init__()
        self.real_norm = nn.LayerNorm(dim)
        self.imag_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_norm(x.real), self.imag_norm(x.imag))


class ComplexMultiheadSelfAttention(nn.Module):
    """Complex-valued self-attention with real-valued attention weights."""

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
        self.use_rope = use_rope

        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

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

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same RoPE rotation to real and imaginary parts of Q/K."""
        cos, sin = self.rope_embed(q.real, position_ids)
        q_real, k_real = RotaryEmbedding.apply_rotary_pos_emb(q.real, k.real, cos, sin)
        q_imag, k_imag = RotaryEmbedding.apply_rotary_pos_emb(q.imag, k.imag, cos, sin)
        return torch.complex(q_real, q_imag), torch.complex(k_real, k_imag)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_rope and position_ids is None:
            raise ValueError("position_ids must be provided when use_rope=True")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_rope:
            q, k = self._apply_rope(q, k, position_ids)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Hermitian dot-product attention: scores[b,h,q,k] = Re(<q_{b,h,q}, k_{b,h,k}>).
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", q, torch.conj(k)).real * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context_real = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v.real)
        context_imag = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v.imag)
        context = torch.complex(context_real, context_imag)

        return self.out_proj(self._merge_heads(context))


class ComplexFeedForward(nn.Module):
    """Complex-valued feed-forward network."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = ComplexLinear(d_model, dim_feedforward)
        self.fc2 = ComplexLinear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = torch.complex(F.gelu(h.real), F.gelu(h.imag))
        h = torch.complex(self.dropout(h.real), self.dropout(h.imag))
        h = self.fc2(h)
        return torch.complex(self.dropout(h.real), self.dropout(h.imag))


class ComplexTransformerEncoderLayer(nn.Module):
    """Pre-norm complex transformer encoder layer."""

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
        self.norm1 = ComplexLayerNorm(d_model)
        self.norm2 = ComplexLayerNorm(d_model)
        self.attn = ComplexMultiheadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=use_rope,
            rope_theta=rope_theta,
        )
        self.ffn = ComplexFeedForward(d_model, dim_feedforward, dropout=dropout)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.ffn(self.norm2(x))
        return x


class ComplexTransformerEncoder(nn.Module):
    """Stack of complex transformer encoder layers."""

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
                ComplexTransformerEncoderLayer(
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
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, position_ids=position_ids)
        return x
