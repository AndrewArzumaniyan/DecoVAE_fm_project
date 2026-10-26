"""Synthetic windows with known horizon perturbations of the cycle, in the mini-FM cache format.

Purpose: check that the structured CVAE latent (cvae_arms.py) learns what it is meant to: amplitude, phase and level
changes of the cycle in the horizon, with a predictive spread that matches the true one. Each window (T + L = 608):
- trend: level + slope t + a slow wave longer than the window (period 1200-3000 steps; reading (b): it is trend);
- seasonal: one period p (train/val/test_id: {12, 24, 48, 96, 168}; test_ood: {18, 36, 72, 120}), 1-3 harmonics with
  random coefficients, unit amplitude scale;
- volatility class v (half the windows each): sd of the per-cycle amplitude drift s_a = 0.05 (low) or 0.25 (high);
  the drift is visible in the context (each cycle's amplitude is multiplied by exp(s_a * N(0, 1)));
- horizon: amplitude factor A = exp(s_a * xi_a), time shift phi = 0.05 xi_ph cycles, level shift l = 0.3 xi_l
  (xi ~ N(0, 1)), applied from t = T on; noise N(0, 0.2^2) everywhere.
Stored per window: x, the clean trend and seasonal (with the horizon perturbation), v, log A, phi, l, p; dataset is
'syn_low' / 'syn_high' so metrics are reported per volatility class.
"""
import argparse
import json
from pathlib import Path
import numpy as np

T, L = 512, 96
W = T+L
PERIODS = {'id': (12, 24, 48, 96, 168), 'ood': (18, 36, 72, 120)}


def window(rng, periods):
    t = np.arange(W, dtype=float); p = float(rng.choice(periods)); high = rng.random() < .5; sa = .25 if high else .05
    trend = rng.normal(0, 1)+rng.normal(0, 1)*t/W+rng.uniform(.2, 1.)*np.sin(2*np.pi*t/rng.uniform(1200, 3000)+rng.uniform(0, 2*np.pi))
    nh = rng.integers(1, 4); a = rng.normal(0, 1, nh)/np.arange(1, nh+1); b = rng.normal(0, 1, nh)/np.arange(1, nh+1)
    norm = np.sqrt((a**2+b**2).sum()/2); a, b = a/norm, b/norm
    la, phi, lev = sa*rng.normal(), .05*rng.normal(), .3*rng.normal()
    tt = np.where(t >= T, t+phi*p, t)
    seas = sum(a[h]*np.cos(2*np.pi*(h+1)*tt/p)+b[h]*np.sin(2*np.pi*(h+1)*tt/p) for h in range(nh))
    cyc = np.floor(t/p).astype(int); drift = np.exp(sa*rng.normal(size=cyc.max()+1))[cyc]
    amp = np.where(t >= T, drift[T-1]*np.exp(la), drift)  # the horizon continues the last context cycle's amplitude, times A
    seas = seas*amp; trend = trend+np.where(t >= T, lev, 0.)
    x = trend+seas+.2*rng.normal(size=W)
    return dict(x=x.astype('float32'), trend=trend.astype('float32'), seasonal=seas.astype('float32'), high=high, log_amp=la, phi=phi, level=lev, period=p)


def main():
    p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('--n-train', type=int, default=8000); p.add_argument('--seed', type=int, default=1015)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    for split, n, kind in (('train', a.n_train, 'id'), ('val', 1000, 'id'), ('test_id', 1000, 'id'), ('test_ood', 1000, 'ood')):
        rows = [window(rng, PERIODS[kind]) for _ in range(n)]
        arr = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        np.savez_compressed(out/f'{split}.npz', x=arr['x'], trend_true=arr['trend'], seasonal_true=arr['seasonal'], high=arr['high'],
                            log_amp=arr['log_amp'], phase_shift=arr['phi'], level_shift=arr['level'], period=arr['period'],
                            dataset=np.where(arr['high'], 'syn_high', 'syn_low'), series_id=np.arange(n), start=np.zeros(n, int), split=np.array([split]*n))
        print(split, n, flush=True)
    (out/'config.json').write_text(json.dumps(dict(T=T, L=L, generator='build_synth_cvae.py', seed=a.seed, periods=PERIODS), indent=2))


if __name__ == '__main__':
    main()
