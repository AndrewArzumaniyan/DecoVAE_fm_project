import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex import ComplexTransformerEncoder, ComplexLinear, ComplexMLP
from .layers import TransformerEncoder


def _build_mlp(
    in_dim: int, hidden_dim: int, out_dim: int, dec_layers: int
) -> nn.Sequential:
    """Build an MLP with ``dec_layers`` linear layers and GELU between them."""
    if dec_layers < 1:
        raise ValueError(f"dec_layers must be >= 1, got {dec_layers}")
    if dec_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(dec_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class MultiScaleDecomposition(nn.Module):
    """Adaptive multi-scale decomposition for trend-seasonal separation."""

    def __init__(self, kernels: list[int] = None):
        super().__init__()
        if kernels is None:
            kernels = [5, 9, 13, 25]
        self.kernels = kernels
        self.M = len(kernels)
        self.weights = nn.Parameter(torch.randn(self.M))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input into trend and seasonal components.

        Args:
            x: Input tensor of shape (B, T) or (B, T, 1)

        Returns:
            Tuple of (trend, seasonal) components
        """
        # Ensure 2D input
        if x.dim() == 3:
            x = x.squeeze(-1)

        trend_components = []
        for k in self.kernels:
            padded = F.pad(x, (k // 2, k // 2), mode="reflect")
            trend = F.avg_pool1d(
                padded.unsqueeze(1), kernel_size=k, stride=1, padding=0
            )
            trend_components.append(trend.squeeze(1))

        # Stack components: (B, T, M)
        trend_stack = torch.stack(trend_components, dim=-1)

        # Weighted combination with softmax weights
        weights = F.softmax(self.weights, dim=0)
        trend = (trend_stack * weights).sum(dim=-1)
        seasonal = x - trend

        return trend, seasonal


class TrendEncoder(nn.Module):
    """Transformer-based trend encoder with patching and RoPE."""

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model

        # Patch embedding (positional information comes from RoPE in attention).
        self.patch_embed = nn.Linear(patch_size, d_model)

        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            use_rope=True,
            rope_theta=rope_theta,
        )

        # Latent distribution parameters
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    def _build_joint_attention_mask(
        self,
        n_context: int,
        n_target: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build additive prefix attention mask for [context, target] layout.

        Mask semantics for SDPA: 0 = attend, ``-inf`` = block.
        - Context tokens cannot attend to target tokens.
        - Target tokens can attend to all context tokens.
        - Target-to-target is non-causal (fully visible).
        """
        if n_target <= 0:
            return None

        total = n_context + n_target
        attn_mask = torch.zeros(1, 1, total, total, device=device)
        attn_mask[..., :n_context, n_context:] = float("-inf")
        return attn_mask

    def extract_patches(self, trend: torch.Tensor) -> torch.Tensor:
        """Extract fixed-size overlapping patches from the trend signal."""
        _, T = trend.shape
        if T < self.patch_size:
            pad = self.patch_size - T
            trend = F.pad(trend, (0, pad), mode="replicate")

        # Use unfold to avoid Python loops and keep patch extraction on-device.
        return trend.unfold(dimension=1, size=self.patch_size, step=self.stride)

    def forward(
        self, trend: torch.Tensor, n_target: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trend component and optionally produce target-horizon latent tokens.

        Args:
            trend: Input trend of shape (B, T)
            n_target: Number of target tokens for forecasting branch

        Returns:
            Tuple of (mu, logvar) over the joint [context, target] sequence
        """
        patches = self.extract_patches(trend)  # (B, N, P)
        B, N = patches.shape[0:2]
        seq_len = N + n_target

        # Embed patches
        h = self.patch_embed(patches)  # (B, N, d)
        joint_h = torch.cat(
            [h, torch.zeros(B, n_target, self.d_model, device=h.device, dtype=h.dtype)],
            dim=1,
        )

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0).expand(B, -1)
        attn_mask = self._build_joint_attention_mask(
            n_context=N,
            n_target=n_target,
            device=h.device,
        )

        joint_h = self.transformer(
            joint_h,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )  # (B, N + N_target, d)

        # Get distribution parameters
        mu = self.mu_proj(joint_h)
        logvar = self.logvar_proj(joint_h)

        return mu, logvar


class SeasonalEncoder(nn.Module):
    """FFT-based seasonal encoder in frequency domain with RoPE."""

    def __init__(
        self,
        d_model: int = 64,
        k_top: int = 16,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_top = k_top

        # Lift scalar FFT coeffs to d_model, then mix tokens with a tiny transformer.
        self.freq_embed = ComplexLinear(1, d_model)
        self.freq_token_proj = ComplexTransformerEncoder(
            d_model=d_model,
            nhead=1,
            num_layers=1,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            use_rope=True,
            rope_theta=rope_theta,
        )

        # Complex mean + real variance from concatenated real/imag states.
        self.mu_proj = ComplexLinear(d_model, d_model)
        self.logvar_proj = nn.Linear(2 * d_model, d_model)

    def forward(
        self, seasonal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode seasonal component in frequency domain.

        Args:
            seasonal: Input seasonal of shape (B, T)

        Returns:
            Tuple of (mu, logvar, rec_seasonal, topk_idx) where rec_seasonal
            is the top-K band-limited reconstruction via sparse irfft and
            topk_idx are the selected rFFT bin indices of shape (B, K).
        """
        # FFT
        fft = torch.fft.rfft(seasonal, dim=-1)
        amp = torch.abs(fft)

        # Select top K frequencies
        K = min(self.k_top, fft.shape[-1])
        _, topk_idx = torch.topk(amp, K, dim=-1)

        # Gather top-K complex coefficients.
        topk_complex = torch.gather(fft, dim=-1, index=topk_idx)  # (B, K) complex
        freq_tokens = topk_complex.unsqueeze(-1)  # (B, K, 1) complex

        # Band-limited seasonal reconstruction from top-K bins only.
        sparse_fft = torch.zeros_like(fft)
        sparse_fft.scatter_(-1, topk_idx, topk_complex)
        rec_seasonal = torch.fft.irfft(sparse_fft, n=seasonal.shape[-1], dim=-1)

        h = self.freq_embed(freq_tokens)
        h = self.freq_token_proj(h, position_ids=topk_idx)  # (B, K, d) complex

        # Complex mean and real log-variance.
        mu = self.mu_proj(h)
        h_real_imag = torch.cat([h.real, h.imag], dim=-1)
        logvar = self.logvar_proj(h_real_imag)

        return mu, logvar, rec_seasonal, topk_idx


class NoiseEncoder(nn.Module):
    """Transformer-based noise encoder with patching and RoPE (1 layer, 1 head)."""

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        nhead: int = 1,
        num_layers: int = 1,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size, d_model)

        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            use_rope=True,
            rope_theta=rope_theta,
        )

        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    def _build_joint_attention_mask(
        self,
        n_context: int,
        n_target: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build additive prefix attention mask for [context, target] layout.

        Mask semantics for SDPA: 0 = attend, ``-inf`` = block.
        - Context tokens cannot attend to target tokens.
        - Target tokens can attend to all context tokens.
        - Target-to-target is non-causal (fully visible).
        """
        if n_target <= 0:
            return None

        total = n_context + n_target
        attn_mask = torch.zeros(1, 1, total, total, device=device)
        attn_mask[..., :n_context, n_context:] = float("-inf")
        return attn_mask

    def extract_patches(self, noise: torch.Tensor) -> torch.Tensor:
        """Extract fixed-size overlapping patches from the noise signal."""
        _, T = noise.shape
        if T < self.patch_size:
            pad = self.patch_size - T
            noise = F.pad(noise, (0, pad), mode="replicate")

        return noise.unfold(dimension=1, size=self.patch_size, step=self.stride)

    def forward(
        self, noise: torch.Tensor, n_target: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode noise component and optionally produce target-horizon latent tokens.

        Args:
            noise: Input residual noise of shape (B, T)
            n_target: Number of target tokens for forecasting branch

        Returns:
            Tuple of (mu, logvar) over the joint [context, target] sequence
        """
        patches = self.extract_patches(noise)  # (B, N, P)
        B, N = patches.shape[0:2]
        seq_len = N + n_target

        h = self.patch_embed(patches)  # (B, N, d)
        joint_h = torch.cat(
            [h, torch.zeros(B, n_target, self.d_model, device=h.device, dtype=h.dtype)],
            dim=1,
        )

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0).expand(B, -1)
        attn_mask = self._build_joint_attention_mask(
            n_context=N,
            n_target=n_target,
            device=h.device,
        )

        joint_h = self.transformer(
            joint_h,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )  # (B, N + N_target, d)

        mu = self.mu_proj(joint_h)
        logvar = self.logvar_proj(joint_h)

        return mu, logvar


class DecoVAEFM(nn.Module):
    """Simplified DecoVAE-FM model for time series forecasting."""

    def __init__(
        self,
        pred_len: int = 24,
        d_model: int = 64,
        patch_size: int = 16,
        stride: int = 8,
        n_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        k_top: int = 32,
        rope_theta: float = 10000.0,
        kernels: list[int] | None = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # Decomposition
        self.decomposition = MultiScaleDecomposition(kernels=kernels)

        # Encoders
        self.trend_encoder = TrendEncoder(
            patch_size, stride, d_model, n_heads, enc_layers, rope_theta=rope_theta
        )
        self.seasonal_encoder = SeasonalEncoder(
            d_model, k_top=k_top, rope_theta=rope_theta
        )
        self.noise_encoder = NoiseEncoder(
            patch_size, stride, d_model, nhead=1, num_layers=1, rope_theta=rope_theta
        )

        # Trend mu decoder; logvar decoder fuses z_trend, pooled z_seas, and z_noise.
        self.trend_decoder = _build_mlp(d_model, d_model, patch_size, dec_layers)
        self.logvar_decoder = _build_mlp(4 * d_model, d_model, patch_size, dec_layers)
        self.seasonal_decoder = ComplexMLP(d_model, d_model, 1, dec_layers)
        # Bias logvar head toward unit variance to avoid early precision collapse.
        logvar_out = self.logvar_decoder[-1]
        nn.init.zeros_(logvar_out.bias)
        with torch.no_grad():
            logvar_out.weight.mul_(0.01)

    def _fold_trend_patches(
        self, patch_values: torch.Tensor, output_length: int
    ) -> torch.Tensor:
        """Average overlapping decoded patches back into a full-length sequence."""
        batch_size, num_patches, patch_size = patch_values.shape
        stride = self.trend_encoder.stride

        # Fold 1D patches via 2D fold with height=1, then normalize overlaps.
        patch_values_2d = patch_values.transpose(1, 2)
        recon = F.fold(
            patch_values_2d,
            output_size=(1, output_length),
            kernel_size=(1, patch_size),
            stride=(1, stride),
        ).squeeze(2).squeeze(1)

        overlap = torch.ones(
            (batch_size, patch_size, num_patches),
            device=patch_values.device,
            dtype=patch_values.dtype,
        )
        counts = F.fold(
            overlap,
            output_size=(1, output_length),
            kernel_size=(1, patch_size),
            stride=(1, stride),
        ).squeeze(2).squeeze(1)

        return recon / counts.clamp_min(1.0)

    def _reconstruct_time_from_freq_tokens(
        self,
        seasonal_freq_tokens: torch.Tensor,
        topk_idx: torch.Tensor,
        context_len: int,
        output_length: int,
    ) -> torch.Tensor:
        """Scatter decoded top-K coeffs onto rFFT bins and irfft to time domain.

        ``topk_idx`` are bin indices from the context-length rFFT. When
        ``output_length != context_len``, bins are remapped so physical
        frequency ``k / context_len`` is preserved on the longer grid.
        """
        freq_tokens = seasonal_freq_tokens.squeeze(-1)  # (B, K) complex
        n_freq = output_length // 2 + 1
        if output_length == context_len:
            scatter_idx = topk_idx
        else:
            scatter_idx = (
                (topk_idx.float() * output_length / context_len)
                .round()
                .long()
                .clamp(0, n_freq - 1)
            )

        sparse_fft = torch.zeros(
            freq_tokens.shape[0],
            n_freq,
            device=freq_tokens.device,
            dtype=freq_tokens.dtype,
        )
        sparse_fft.scatter_(-1, scatter_idx, freq_tokens)
        return torch.fft.irfft(sparse_fft, n=output_length, dim=-1)

    def encode(self, x: torch.Tensor, n_target: int = 0) -> dict:
        """Encode input into latent distributions for context and optional target horizon."""
        # Decompose
        trend, seasonal = self.decomposition(x)

        # Encode components
        trend_mu, trend_logvar = self.trend_encoder(trend, n_target=n_target)
        seasonal_mu, seasonal_logvar, rec_seasonal, topk_idx = self.seasonal_encoder(
            seasonal
        )

        noise = x - trend - rec_seasonal
        noise_mu, noise_logvar = self.noise_encoder(noise, n_target=n_target)

        return {
            "trend_mu": trend_mu,
            "trend_logvar": trend_logvar,
            "seasonal_mu": seasonal_mu,
            "seasonal_logvar": seasonal_logvar,
            "noise_mu": noise_mu,
            "noise_logvar": noise_logvar,
            "topk_idx": topk_idx,
            "target_trend": trend,
            "target_seasonal": rec_seasonal,
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        logvar = logvar.clamp(-20.0, 10.0)
        var = torch.exp(logvar)

        if mu.is_complex():
            scale = torch.sqrt(var / 2.0)
            eps_real = torch.randn_like(scale)
            eps_imag = torch.randn_like(scale)
            eps = torch.complex(eps_real, eps_imag)
            return mu + scale * eps

        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> dict:
        """
        Forward pass for training (reconstruction + forecasting).

        Args:
            x: Input tensor of shape (B, T)
            pred_len: Optional forecasting horizon. If None, uses self.pred_len.

        Returns:
            Dictionary with observation Gaussian params (sum of trend/seasonal)
            for reconstruction and forecast, plus latent encoder parameters.
        """
        B, T = x.shape
        L = self.pred_len if pred_len is None else pred_len

        # Number of trend tokens needed for target horizon.
        n_target = max(1, math.ceil(L / self.trend_encoder.stride))

        # Encode context and target tokens jointly for trend branch.
        encodings = self.encode(x, n_target=n_target)

        # Sample from latent
        z_trend = self.reparameterize(encodings["trend_mu"], encodings["trend_logvar"])
        z_seasonal = self.reparameterize(
            encodings["seasonal_mu"], encodings["seasonal_logvar"]
        )
        z_noise = self.reparameterize(
            encodings["noise_mu"], encodings["noise_logvar"]
        )

        decoded_trend = self._fold_trend_patches(self.trend_decoder(z_trend), T + L)

        seasonal_freq = self.seasonal_decoder(z_seasonal)
        decoded_seasonal = self._reconstruct_time_from_freq_tokens(
            seasonal_freq,
            encodings["topk_idx"],
            context_len=T,
            output_length=T + L,
        )

        z_seas = torch.cat(
            [z_seasonal.real, z_seasonal.imag], dim=-1
        ).mean(dim=1, keepdim=True)
        z_seas = z_seas.expand(-1, z_trend.size(1), -1)
        logvar = self._fold_trend_patches(
            self.logvar_decoder(torch.cat([z_noise, z_seas, z_trend], dim=-1)), T + L
        ).clamp(-20.0, 10.0)

        mu = decoded_trend + decoded_seasonal

        return {
            "recon_mu": mu[:, :T],
            "recon_logvar": logvar[:, :T],
            "forecast_mu": mu[:, T:],
            "forecast_logvar": logvar[:, T:],
            "decoded_trend": decoded_trend,
            "decoded_seasonal": decoded_seasonal,
            "decoded_logvar": logvar,
            "target_trend": encodings["target_trend"],
            "target_seasonal": encodings["target_seasonal"],
            "trend_mu": encodings["trend_mu"],
            "trend_logvar": encodings["trend_logvar"],
            "seasonal_mu": encodings["seasonal_mu"],
            "seasonal_logvar": encodings["seasonal_logvar"],
            "noise_mu": encodings["noise_mu"],
            "noise_logvar": encodings["noise_logvar"],
        }
