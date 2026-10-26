"""Conditional-VAE latents for the forecast, built on Dcomb.

The encoder of the original model sees only the context, so its latents cannot carry information about the future
and the NLL averaged over latent draws penalises any spread they cause (they collapse, 0.05% of the variance).
Here a new latent z gets a posterior q(z | context, future) during training and a prior p(z | context) at test:
ELBO = E_q log p(y | x, z) - KL(q || p), weight 1, with the KL divided by L = 96 to match the mean NLL per point.
At evaluation (model.eval()) z is drawn from the prior, so the predictive is a mixture with real spread.
z changes only the forecast; the context reconstruction is Dcomb's.

- Dzs (structured): per period slot s, an amplitude factor exp(0.3 z_amp,s) and a time shift of 0.1 z_ph,s cycles of
  the computed seasonal continuation in the horizon (rotation of each harmonic's cos/sin pair), plus a level shift
  0.5 z_lvl of the horizon (normalised units); 2 S + 1 = 7 dims.
- Dzg (general): 8 dims, a zero-initialised MLP([z, context embedding]) adds a correction to the whole horizon mean.
- Dzghc (17.10): Dzgh whose hybrid term also adds 1 x the almost fair CRPS of the prior-draw mixture (Dmc1's loss).
- Dzsh / Dzgh (hybrid, as the hybrid objective of Sohn et al. 2015): the same plus the log score of the test-time
  predictive, a mixture over K = 8 prior draws per point, -log (1/K sum_k N(y_t; mu_k,t, sd_t^2)), added with weight 1.
  The smoke runs showed a large posterior-prior gap (val CRPS 0.49 with posterior draws vs 0.57 with prior draws);
  unlike the plain NLL of a prior draw (which would again penalise spread), the mixture log score is proper for the
  mixture and rewards the right spread.
Prior and posterior are small MLPs with zero-initialised outputs (both start at N(0, 1), KL 0). Context embedding:
mean of the trend-encoder tokens (64) and the 8 window features of the sigma head.
"""
import math
import torch
from torch import nn
from data import L
from fourier_arm import fourier_basis, harmonics, N_SEASONS
from trend_input_arms import CombinedArm

Z_ARMS = ('Dzs', 'Dzg', 'Dzsh', 'Dzgh', 'Dzghc')
K_PRIOR = 8
CTX = 64+8


def _mlp(i, h, o, layers=2):
    mods = []; d = i
    for _ in range(layers-1): mods += [nn.Linear(d, h), nn.GELU()]; d = h
    mods.append(nn.Linear(d, o)); nn.init.zeros_(mods[-1].weight); nn.init.zeros_(mods[-1].bias)
    return nn.Sequential(*mods)


class CVAEArm(CombinedArm):
    def __init__(self, arm):
        super().__init__('Dcomb'); self.name = arm; self.structured = arm.startswith('Dzs'); self.hybrid = arm.endswith('h') or arm == 'Dzghc'
        self.crps_lam = 1. if arm == 'Dzghc' else 0.
        self.dz = 2*N_SEASONS+1 if self.structured else 8
        extra = N_SEASONS*L if self.structured else 0
        self.prior = _mlp(CTX, 64, 2*self.dz)
        self.post = _mlp(CTX+2*L+extra, 128, 2*self.dz, layers=3)
        if not self.structured: self.corr = _mlp(self.dz+CTX, 64, L)
        self.future = None; self._coefs = None

    def seasonal_path(self, zs, length):
        periods, coefs = self.fourier
        pooled = torch.cat([zs.real, zs.imag], -1).mean(1)
        head_in = torch.cat([pooled[:, None, :].expand(-1, N_SEASONS, -1), torch.log1p(periods.float())[..., None]], -1)
        delta = self.fourier_head(head_in)
        if not self.use_correction: delta = torch.zeros_like(delta)
        _, mask = harmonics(periods.double())
        delta = delta*mask.repeat_interleave(2, -1).float()
        self._coefs = coefs.float()+delta  # (B, S, 2H), kept for the horizon modulation
        return torch.einsum('bsnk,bsk->bn', fourier_basis(periods, length), self._coefs)

    def _slots(self, coefs, n):
        """Per-slot horizon seasonal (B, S, L) for given coefficients."""
        periods = self.fourier[0]
        return torch.einsum('bsnk,bsk->bsn', fourier_basis(periods, n+L)[:, :, n:], coefs)

    def _modulated(self, z, n):
        periods = self.fourier[0]; S = N_SEASONS
        amp = torch.exp(.3*z[:, :S]); shift = .1*z[:, S:2*S]; level = .5*z[:, 2*S]
        c = self._coefs.reshape(*self._coefs.shape[:2], -1, 2); a, b = c[..., 0], c[..., 1]   # (B, S, H)
        h = torch.arange(1, a.shape[-1]+1, device=z.device, dtype=a.dtype)
        th = 2*math.pi*h[None, None, :]*shift[..., None]
        a2 = a*torch.cos(th)+b*torch.sin(th); b2 = -a*torch.sin(th)+b*torch.cos(th)
        rot = torch.stack([a2, b2], -1).flatten(-2)*amp[..., None]
        return self._slots(rot, n).sum(1)+level[:, None]

    def forward(self, x, pred_len=None, noise_permutation=None):
        out = super().forward(x, pred_len, noise_permutation); n = x.shape[-1]
        ctx = torch.cat([out['trend_mu'].mean(1), self.s12[1]], -1)
        pm, plv = self.prior(ctx).chunk(2, -1); plv = plv.clamp(-10, 5)
        base = out['forecast_mu']; seas_h = out['decoded_seasonal'][:, n:]
        if self.training and self.future is not None:
            q_in = [ctx, self.future, base]
            if self.structured: q_in.append(self._slots(self._coefs, n).flatten(1))
            qm, qlv = self.post(torch.cat(q_in, -1)).chunk(2, -1); qlv = qlv.clamp(-10, 5)
            z = qm+torch.exp(.5*qlv)*torch.randn_like(qm)
            out['cvae_kl'] = .5*(plv-qlv+(qlv.exp()+(qm-pm)**2)/plv.exp()-1).sum(-1)
            if self.hybrid:  # log score of the K-draw prior mixture (sd does not depend on z)
                lv = out['forecast_logvar']; ll = []; mks = []
                for _ in range(K_PRIOR):
                    zk = pm+torch.exp(.5*plv)*torch.randn_like(pm); mk = self._mean(base, seas_h, ctx, zk, n); mks.append(mk)
                    ll.append(-.5*(lv+(self.future-mk)**2*torch.exp(-lv)))
                out['cvae_mix'] = -(torch.logsumexp(torch.stack(ll), 0)-math.log(K_PRIOR)).mean()
                if self.crps_lam:  # Dzghc: + lambda * almost fair CRPS of the same prior-draw mixture (as Dmc1)
                    from score_arms import af_crps
                    out['cvae_mix'] = out['cvae_mix']+self.crps_lam*af_crps(self.future, torch.stack(mks), lv[None].expand(len(mks), -1, -1))
        else:
            z = pm+torch.exp(.5*plv)*torch.randn_like(pm)
        out['forecast_mu'] = self._mean(base, seas_h, ctx, z, n); out['cvae_z'] = z
        return out

    def _mean(self, base, seas_h, ctx, z, n):
        if self.structured: return base-seas_h+self._modulated(z, n)
        return base+self.corr(torch.cat([z, ctx], -1))


def wrap_loss(base_loss):
    def loss(out, x, y):
        value = base_loss(out, x, y)
        if 'cvae_kl' in out: value = value+out['cvae_kl'].mean()/L
        if 'cvae_mix' in out: value = value+out['cvae_mix']
        return value
    return loss
