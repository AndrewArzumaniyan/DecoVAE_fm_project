"""Learnable seasonal corrections on top of the computed Fourier continuation (arms Cs1, Cs2, Cs12).

Base: Cf0 (period-aware input, continuous-frequency Fourier decoder, the old correction Δ switched off). The context
part of the seasonal stays the least-squares fit; corrections act on the HORIZON only, so the context reconstruction
term (where the LS fit is already optimal; both NLL terms are means, equal total weight) does not pull them to zero -
the suspected reason Δ stayed 0.
- S1 (Cs1): horizon coefficients c = c_inf + sum_k beta_k (c_hl_k - c_inf), hl in {4, 2, 1} cycles, where c_hl are
  recency-weighted (half-life hl * p_max) ridge fits to the context residual; beta = tanh(MLP(features)).
- S2 (Cs2): horizon seasonal multiplied by a trust gate g = 1 + tanh(MLP(features)) in (0, 2).
- Cs12: both. The MLP's last layer is zero-initialised, so every arm starts exactly at Cf0.
Features (8, per window, from the context only): has_season, hold-out R2 of a first-half fit on the second half,
cycle-amplitude CV, last-cycle amplitude - 1, cycles in the window / 10, explained share, log1p(p1)/5, log1p(p2)/5.
Cache: <cache>/<split>_s12_<tag>.npz with coefs_hl (N, 2, 3, 24) and feats (N, 8); build with
`python seasonal_arms.py <cache> --workers 4` after the Fourier cache exists.
"""
import argparse
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from torch import nn

from fm_paths import ROOT, VER0_DIR, RUNS, CACHE  # noqa: E402,F401  (also puts the shared benchmark modules on sys.path)
from data import T, L, normalize
from fourier_arm import FourierArm, H_MAX, N_SEASONS, cache_tag, fourier_basis

S_ARMS = ('Cs1', 'Cs2', 'Cs12')
SIG_ARMS = ('Dfs',)  # Df with an informed sigma head (see SigmaInfoArm)
HLS = (4., 2., 1.)
N_FEAT = 8
RIDGE = 1.0


def _basis(p, H, t):
    h = np.arange(1, H+1); ang = 2*np.pi*np.mod(np.outer(t, h)/p, 1.0)
    B = np.empty((len(t), 2*H)); B[:, 0::2] = np.cos(ang); B[:, 1::2] = np.sin(ang); return B


def window_extras(zc, trend, s_full, periods, coefs):
    """coefs_hl (2, 3, 2*H_MAX) and features (8,) for one window (normalised, clipped context)."""
    out_c = np.zeros((N_SEASONS, len(HLS), 2*H_MAX)); feats = np.zeros(N_FEAT)
    slots = []
    for j in range(N_SEASONS):
        p = periods[j]; nz = np.flatnonzero(coefs[j])
        if p > 0 and len(nz): slots.append((j, float(p), nz.max()//2+1))
    if not slots: return out_c, feats
    t = np.arange(T); B = np.concatenate([_basis(p, H, t) for _, p, H in slots], 1)
    c0 = np.concatenate([coefs[j, :2*H] for j, _, H in slots]); r = zc-trend; pmax = max(p for _, p, _ in slots)
    for k, hl in enumerate(HLS):
        w = 0.5**((T-1-t)/(hl*pmax)); Bw = B*w[:, None]
        c = np.linalg.solve(B.T@Bw+RIDGE*np.eye(len(c0)), Bw.T@r+RIDGE*c0); o = 0
        for j, _, H in slots: out_c[j, k, :2*H] = c[o:o+2*H]; o += 2*H
    half = T//2; c1 = np.linalg.lstsq(B[:half], r[:half], rcond=None)[0]; e2 = r[half:]-B[half:]@c1
    hold_r2 = 1-np.sum(e2**2)/max(np.sum((r[half:]-r[half:].mean())**2), 1e-12)
    s_ctx = s_full[:T]; q = int(round(pmax)); amps = []
    for k in range(T//q):
        seg = slice(T-(k+1)*q, T-k*q); den = np.sum(s_ctx[seg]**2)
        if den > 1e-9: amps.append(np.sum(r[seg]*s_ctx[seg])/den)
    amp_cv = float(np.std(amps)/max(abs(np.mean(amps)), 1e-6)) if len(amps) >= 2 else 0.
    a_last = amps[0]-1 if amps else 0.
    explained = float(np.var(s_ctx)/max(np.var(r), 1e-12))
    ps = [p for _, p, _ in slots]+[0.]
    feats[:] = [1., np.clip(hold_r2, -1, 1), np.clip(amp_cv, 0, 3), np.clip(a_last, -2, 2), T/pmax/10, np.clip(explained, 0, 2),
                np.log1p(ps[0])/5, np.log1p(ps[1])/5]
    return out_c, feats


def _work(job):
    lo, x, trend, s_full, periods, coefs = job  # arrays are sliced in the parent: npz members decompress on every access
    zc = np.clip(normalize(x.astype(np.float64))[0], -20, 20)
    res = [window_extras(zc[i, :T], trend[i].astype(np.float64), s_full[i].astype(np.float64), periods[i], coefs[i]) for i in range(len(x))]
    return lo, np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def build(cache, workers=4, splits=('train', 'val', 'test_id', 'test_ood')):
    cache = Path(cache)
    for split in splits:
        path = cache/f'{split}_s12_{cache_tag()}.npz'
        if path.exists(): continue
        x = np.load(cache/f'{split}.npz')['x']; f = np.load(cache/f'{split}_fourier_{cache_tag()}.npz')
        arrs = [f[k] for k in ('trend', 'seasonal_full', 'periods', 'coefs')]; n = len(x); step = 512
        jobs = [(lo, x[lo:lo+step], *[a_[lo:lo+step] for a_ in arrs]) for lo in range(0, n, step)]
        with get_context('spawn').Pool(workers) as pool:
            parts = sorted(pool.map(_work, jobs), key=lambda q: q[0])
        np.savez_compressed(path, coefs_hl=np.concatenate([q[1] for q in parts]), feats=np.concatenate([q[2] for q in parts]).astype('float32'))
        print('saved', path.name, n, flush=True)


def attach_s12(raw, cache, split):
    d = np.load(Path(cache)/f'{split}_s12_{cache_tag()}.npz'); raw['s12_coefs'] = d['coefs_hl']; raw['s12_feats'] = d['feats']
    return raw


class AdaptiveSeasonalArm(FourierArm):
    def __init__(self, arm):
        super().__init__('Cf'); self.name = arm; self.use_correction = False
        self.s1, self.s2 = arm in ('Cs1', 'Cs12'), arm in ('Cs2', 'Cs12')
        n_out = len(HLS)*self.s1+self.s2
        self.adapt = nn.Sequential(nn.Linear(N_FEAT, 32), nn.GELU(), nn.Linear(32, n_out))
        nn.init.zeros_(self.adapt[-1].weight); nn.init.zeros_(self.adapt[-1].bias)
        self.s12 = None  # (coefs_hl (B, S, 3, 2H), feats (B, 8)), attached per batch

    def seasonal_path(self, zs, length):
        periods, coefs = self.fourier; coefs_hl, feats = self.s12
        basis = fourier_basis(periods, length); c = coefs.float()
        s_full = torch.einsum('bsnk,bsk->bn', basis, c)
        ctx = length-L; out = self.adapt(feats); c_h = c
        if self.s1:
            beta = torch.tanh(out[:, :len(HLS)])
            c_h = c+torch.einsum('bk,bskh->bsh', beta, coefs_hl.float()-c[:, :, None, :])
        s_h = torch.einsum('bsnk,bsk->bn', basis[:, :, ctx:, :], c_h)
        if self.s2: s_h = (1+torch.tanh(out[:, -1:]))*s_h
        self.last_adapt = out.detach()
        return torch.cat([s_full[:, :ctx], s_h], 1)


class SigmaInfoArm(FourierArm):
    """Dfs: Df whose log-variance gets an additive per-step correction h(t) = MLP([S(t), |S(t)|, 8 window features]),
    S = the computed seasonal (context fit and continuation). Zero-initialised last layer: starts exactly at Df. Lets sigma
    follow the phase of the cycle (wider at peaks) and the reliability of the seasonal fit."""
    def __init__(self, arm='Dfs'):
        super().__init__('Df'); self.name = arm
        self.sig = nn.Sequential(nn.Linear(N_FEAT+2, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.sig[-1].weight); nn.init.zeros_(self.sig[-1].bias)
        self.s12 = None

    def forward(self, x, pred_len=None, noise_permutation=None):
        out = super().forward(x, pred_len, noise_permutation)
        s = out['decoded_seasonal']; feats = self.s12[1]; n = x.shape[-1]
        h = self.sig(torch.cat([s[..., None], s.abs()[..., None], feats[:, None, :].expand(-1, s.shape[1], -1)], -1)).squeeze(-1)
        out['recon_logvar'] = (out['recon_logvar']+h[:, :n]).clamp(-20, 10); out['forecast_logvar'] = (out['forecast_logvar']+h[:, n:]).clamp(-20, 10)
        return out


def set_s12(model, raw, idx, device):
    if hasattr(model, 's12'):  # every arm that takes the S1/S2 window features
        model.s12 = (torch.from_numpy(raw['s12_coefs'][idx]).to(device), torch.from_numpy(raw['s12_feats'][idx]).to(device))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--workers', type=int, default=4); a = p.parse_args()
    build(a.cache, a.workers)
