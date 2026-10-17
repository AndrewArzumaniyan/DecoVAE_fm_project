"""Dcomb variants: HP trend regulariser and a capped sd for asinh space.

HP regulariser: penalty on the second differences of the decoded trend over the whole window (context + horizon),
weighted as in the Hodrick-Prescott objective sum (x - tau)^2 + lambda sum (D2 tau)^2. Our data term is the Gaussian NLL,
(x - mu)^2 / (2 sd^2) per point, so the penalty per window is lambda * mean_t (D2 tau_t)^2 / (2 sd_w^2), with sd_w^2 the
window's mean predicted context variance (detached); it is added to the loss averaged over windows.
- DhpP: lambda per window from the HP gain formula lambda = 1 / (4 (1 - cos(2 pi / P))^2) with cutoff P = 1.5 p_max
  (the estimator's own HP cutoff; P = T / 2 when the window has no season);
- DhpT: cutoff P = T / 2 for every window (stiff trend; lambda about 2.8e6);
- DhpS / DhpM: fixed lambda = 1e-3 / 1e-1 for every window (not tied to the period). The trained Dcomb trend is far from
  HP-smooth (median |D2 tau| = 0.22 in normalised units: the trend branch also fits the residual), so HP-equivalent
  lambdas (1e3..3e6) give penalties 1e5..1e9 times the loss and force an HP-smooth trend; the fixed values are chosen so
  that the penalty of the trained Dcomb is about 0.5 (soft) and 50 (medium), i.e. comparable to the loss.
DcombC: Dcomb with the log-variances capped at 2 (sd <= e times the context scale of the working space). For the asinh
cache, where a huge sd in u-space becomes an exponential tail in x (a single window can dominate the mean).
"""
import math
import torch
from data import T
from trend_input_arms import CombinedArm

HP_ARMS = ('DhpP', 'DhpT', 'DhpS', 'DhpM')
CAP_ARMS = ('DcombC',)
LV_MAX = 2.


def hp_lambda(cutoff):
    return 1/(4*(1-torch.cos(2*math.pi/cutoff))**2)


class HPArm(CombinedArm):
    def __init__(self, arm):
        super().__init__('Dcomb'); self.name = arm

    def forward(self, x, pred_len=None, noise_permutation=None):
        out = super().forward(x, pred_len, noise_permutation)
        if self.training:
            tau = out['decoded_trend']; d2 = tau[:, 2:]-2*tau[:, 1:-1]+tau[:, :-2]
            if self.name == 'DhpP':
                pmax = self.fourier[0].float().max(1).values
                lam = hp_lambda(torch.where(pmax >= 2, 1.5*pmax, torch.full_like(pmax, T/2)))
            elif self.name == 'DhpT': lam = hp_lambda(torch.full((tau.shape[0],), T/2, device=tau.device))
            else: lam = torch.full((tau.shape[0],), 1e-3 if self.name == 'DhpS' else 1e-1, device=tau.device)
            var = out['recon_logvar'].detach().exp().mean(1)
            out['hp_pen'] = (lam*(d2**2).mean(1)/(2*var)).mean()
        return out


class CappedArm(CombinedArm):
    def __init__(self, arm):
        super().__init__('Dcomb'); self.name = arm

    def forward(self, x, pred_len=None, noise_permutation=None):
        out = super().forward(x, pred_len, noise_permutation)
        out['recon_logvar'] = out['recon_logvar'].clamp(max=LV_MAX); out['forecast_logvar'] = out['forecast_logvar'].clamp(max=LV_MAX)
        return out


def wrap_loss(base_loss):
    def loss(out, x, y):
        value = base_loss(out, x, y)
        if 'hp_pen' in out: value = value+out['hp_pen']
        return value
    return loss
