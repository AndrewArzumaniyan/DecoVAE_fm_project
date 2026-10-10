"""DecoVAE-FM (the original code, unchanged modules) with switchable structural options.

Options:
  input_decomp   'original' : the original MultiScaleDecomposition + Top-K (as in decovaefm)
                 'period'  : precomputed per-window period-aware decomposition (experiments/decomposition/period_decomp)
  seas_decoder   'remap'   : the original ComplexMLP + rFFT-bin remap to T+L (as in decovaefm)
                 'fourier' : HARD periodic decoder -- MLP(pooled z_s, log p) -> Fourier coefficients
                             at the estimated period p (exactly periodic, zero-mean by construction)
                 'det_resid': deterministic seasonal + learned periodic Fourier correction (zero-init)
                 'det'     : no learning: the deterministic seasonal extrapolation is passed through
Everything else (encoders, trend decoder, sigma head, VAE sampling) is the original code.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decovaefm.models.decovae_fm import DecoVAEFM, _build_mlp  # noqa: E402

H_MAX = 8


class StructDecoVAE(DecoVAEFM):
    def __init__(self, *, input_decomp="original", seas_decoder="remap", **kw):
        super().__init__(**kw)
        self.input_decomp = input_decomp
        self.seas_decoder_kind = seas_decoder
        self.sample_latents = True
        if seas_decoder in ("fourier", "det_resid"):
            self.fourier_head = _build_mlp(2 * self.d_model + 1, self.d_model, 2 * H_MAX, 2)
        if seas_decoder == "det_resid":          # start exactly at the deterministic seasonal
            nn.init.zeros_(self.fourier_head[-1].weight)
            nn.init.zeros_(self.fourier_head[-1].bias)

    def _rep(self, mu, logvar):
        return self.reparameterize(mu, logvar) if self.sample_latents else mu

    @staticmethod
    def fourier_basis(period: torch.Tensor, n: int) -> torch.Tensor:
        """(B, n, 2*H_MAX) basis; harmonics above p/2 and p == 0 (no season) are masked."""
        t = torch.arange(n, device=period.device, dtype=torch.float32)[None, :, None]
        h = torch.arange(1, H_MAX + 1, device=period.device, dtype=torch.float32)[None, None, :]
        p = period.clamp_min(1.0)[:, None, None]
        ang = 2 * math.pi * h * t / p
        mask = ((h <= period[:, None, None] / 2) & (period[:, None, None] > 0)).float()
        return torch.cat([torch.cos(ang) * mask, torch.sin(ang) * mask], dim=-1)

    def forward(self, x, pred_len, trend_in=None, seas_in=None, period=None, seas_det=None):
        B, T = x.shape
        L = pred_len
        n_target = max(1, math.ceil(L / self.trend_encoder.stride))
        if self.input_decomp == "original":
            trend, seasonal = self.decomposition(x)
        else:
            trend, seasonal = trend_in, seas_in
        t_mu, t_lv = self.trend_encoder(trend, n_target=n_target)
        s_mu, s_lv, rec_seas, topk_idx = self.seasonal_encoder(seasonal)
        noise = x - trend - (rec_seas if self.input_decomp == "original" else seasonal)
        n_mu, n_lv = self.noise_encoder(noise, n_target=n_target)

        z_t, z_s, z_n = self._rep(t_mu, t_lv), self._rep(s_mu, s_lv), self._rep(n_mu, n_lv)
        dec_trend = self._fold_trend_patches(self.trend_decoder(z_t), T + L)
        if self.seas_decoder_kind == "remap":
            dec_seas = self._reconstruct_time_from_freq_tokens(
                self.seasonal_decoder(z_s), topk_idx, context_len=T, output_length=T + L)
        elif self.seas_decoder_kind in ("fourier", "det_resid"):
            pooled = torch.cat([z_s.real, z_s.imag], -1).mean(1)
            coef = self.fourier_head(torch.cat([pooled, torch.log1p(period)[:, None]], -1))
            dec_seas = (self.fourier_basis(period, T + L) @ coef[:, :, None]).squeeze(-1)
            if self.seas_decoder_kind == "det_resid":   # hard periodic: det. seasonal + learned correction
                dec_seas = seas_det + dec_seas
        else:
            dec_seas = seas_det
        z_seas = torch.cat([z_s.real, z_s.imag], -1).mean(1, keepdim=True).expand(-1, z_t.size(1), -1)
        logvar = self._fold_trend_patches(
            self.logvar_decoder(torch.cat([z_n, z_seas, z_t], -1)), T + L).clamp(-20, 10)
        mu = dec_trend + dec_seas
        return dict(mu=mu, logvar=logvar, trend=dec_trend, seasonal=dec_seas,
                    kl_terms=[(t_mu, t_lv), (s_mu, s_lv), (n_mu, n_lv)])
