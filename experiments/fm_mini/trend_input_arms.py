"""Trend-branch input variants, built on Df (period-aware input, Fourier decoder, no noise branch).

- Dd: the trend branch sees the deseasonalised context d = x - seasonal (trend + residual) instead of the HP trend, so
  the transformer does its own smoothing and right-edge level estimate, and a predictable residual could reach the mean.
- Dda: Dd whose decoded trend is added to a level anchor, the exponential smoothing (alpha 0.05) of d at the last point
  (step 0: the best simple anchor). The network learns the deviation from that level.
The computed seasonal path and everything else are Df's. With no noise branch the residual x - trend - seasonal (zero
for Dd) is not needed.
"""
import torch
from fourier_arm import FourierArm
from data import L

D_ARMS = ('Dd', 'Dda')
ALPHA = 0.05


class DeseasonalizedArm(FourierArm):
    def __init__(self, arm):
        super().__init__('Df'); self.name = arm; self.anchor = arm == 'Dda'

    def _encode(self, x, n_target):
        trend, seasonal = self.components
        d = x-seasonal
        tm, tl = self.trend_encoder(d, n_target=n_target)
        sm, sl, rec, idx = self.seasonal_encoder(seasonal)
        return dict(trend_mu=tm, trend_logvar=tl, seasonal_mu=sm, seasonal_logvar=sl, topk_idx=idx, target_trend=d, target_seasonal=seasonal)

    def level_anchor(self, d):
        n = d.shape[-1]; k = torch.arange(n-1, -1, -1, device=d.device, dtype=d.dtype)  # age of each point
        w = ALPHA*(1-ALPHA)**k; w[0] = (1-ALPHA)**(n-1)  # exponential smoothing started at d[0]
        return (d*w).sum(-1)

    def forward(self, x, pred_len=None, noise_permutation=None):
        length = x.shape[-1]; e = self.encode(x, n_target=L//16)
        zt = self.reparameterize(e['trend_mu'], e['trend_logvar']); zs = self.reparameterize(e['seasonal_mu'], e['seasonal_logvar'])
        trend = self._fold_trend_patches(self.trend_decoder(zt), length+L)
        if self.anchor: trend = trend+self.level_anchor(e['target_trend'])[:, None]
        seasonal = self.seasonal_path(zs, length+L)
        pooled = torch.cat([zs.real, zs.imag], -1).mean(1, keepdim=True).expand(-1, zt.size(1), -1)
        lv = self._fold_trend_patches(self.logvar_decoder(torch.cat([pooled, zt], -1)), length+L).clamp(-20, 10)
        mu = trend+seasonal
        return dict(e, recon_mu=mu[:, :length], recon_logvar=lv[:, :length], forecast_mu=mu[:, length:], forecast_logvar=lv[:, length:],
                    decoded_trend=trend, decoded_seasonal=seasonal)


class CombinedArm(DeseasonalizedArm):
    """Dcomb: Dda (deseasonalised trend input + ES level anchor) plus Dfs's sigma correction
    h(t) = MLP([S(t), |S(t)|, 8 window features]) (zero-initialised); run with --recon-weight 0.2."""
    def __init__(self, arm='Dcomb'):
        super().__init__('Dda'); self.name = arm
        from seasonal_arms import N_FEAT
        self.sig = torch.nn.Sequential(torch.nn.Linear(N_FEAT+2, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        torch.nn.init.zeros_(self.sig[-1].weight); torch.nn.init.zeros_(self.sig[-1].bias)
        self.s12 = None

    def forward(self, x, pred_len=None, noise_permutation=None):
        out = super().forward(x, pred_len, noise_permutation)
        s = out['decoded_seasonal']; feats = self.s12[1]; n = x.shape[-1]
        h = self.sig(torch.cat([s[..., None], s.abs()[..., None], feats[:, None, :].expand(-1, s.shape[1], -1)], -1)).squeeze(-1)
        out['recon_logvar'] = (out['recon_logvar']+h[:, :n]).clamp(-20, 10); out['forecast_logvar'] = (out['forecast_logvar']+h[:, n:]).clamp(-20, 10)
        return out


COMB_ARMS = ('Dcomb',)
