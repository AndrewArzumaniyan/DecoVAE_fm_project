"""Synthetic decomposition laboratory in the mini-FM cache format.

Every window (T + L = 608 points) is x = base trend + short cycle + long wave + noise, all components known:
- short cycle (seasonality): period p in {12, 24, 48, 96, 168}, 1-3 harmonics, RMS 1;
- long wave: one sinusoid, period P in {128, 192, 256, 384, 512, 1024, 2048}, RMS = r * 1 with r in {0.5, 1, 2, 4};
  under reading (b) with the estimator's boundary T/3 (= 170.7) it is seasonality when P <= T/3 (P = 128: 4 cycles in
  the context) and trend when P > T/3 (192 and 256 are the borderline 2-2.7 cycles);
- base trend: level + slope, half of the windows with a kink (slope change at a random point);
- noise: sd sigma_n in {0.1, 0.3}; half of the windows homoscedastic, half with sd tied to the phase of the short cycle,
  sd_t = sigma_n * (0.3 + 0.7 |c_t| / max|c|) rescaled to mean sigma_n (visible in the context, continues in the horizon).
The true predictive is N(trend_t + seasonal_t, sd_t^2); oracle CRPS and widths follow from it.
Splits: train (random factors), val, test_id stratified over the 28 cells (P, r) with 64 windows each, test_ood the same
grid with short periods {18, 36, 72} not seen in training. dataset = 'P<P>_r<r>' (the cell), so per-source metrics are
per cell. Stored: x, trend_true, seasonal_true (by the T/3 rule), long_wave, short_cycle, base_trend, sd, and the factors.
"""
import argparse
import json
from pathlib import Path
import numpy as np

T, L = 512, 96
W = T+L
P_GRID = (128, 192, 256, 384, 512, 1024, 2048)
R_GRID = (.5, 1., 2., 4.)
SHORT = {'id': (12, 24, 48, 96, 168), 'ood': (18, 36, 72)}


def window(rng, P, r, shorts):
    t = np.arange(W, dtype=float); p = float(rng.choice(shorts)); nh = int(rng.integers(1, 4))
    a, b = rng.normal(0, 1, nh)/np.arange(1, nh+1), rng.normal(0, 1, nh)/np.arange(1, nh+1)
    c = sum(a[h]*np.cos(2*np.pi*(h+1)*t/p)+b[h]*np.sin(2*np.pi*(h+1)*t/p) for h in range(nh)); c /= c.std()
    lw = np.sin(2*np.pi*t/P+rng.uniform(0, 2*np.pi)); lw *= r/lw[:T].std() if lw[:T].std() > 1e-9 else r
    kink = rng.random() < .5; base = rng.normal(0, 1)+rng.normal(0, 1)*t/W
    if kink: t0 = rng.integers(T//4, T+L//2); base = base+rng.normal(0, 1)*np.maximum(t-t0, 0)/W
    sn = float(rng.choice((.1, .3))); phase = rng.random() < .5
    sd = sn*(.3+.7*np.abs(c)/np.abs(c).max()) if phase else np.full(W, sn)
    if phase: sd *= sn/sd.mean()
    seasonal = c+(lw if P <= T/3 else 0.); trend = base+(lw if P > T/3 else 0.)
    x = trend+seasonal+sd*rng.standard_normal(W)
    return dict(x=x, trend_true=trend, seasonal_true=seasonal, long_wave=lw, short_cycle=c, base_trend=base, sd=sd,
                P=P, r=r, p=p, nh=nh, sigma_n=sn, phase_noise=phase, kink=kink)


def build(out, n_train, seed):
    out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    plan = [('train', n_train, 'id', False), ('val', 1000, 'id', False), ('test_id', 64, 'id', True), ('test_ood', 64, 'ood', True)]
    for split, n, kind, strat in plan:
        cells = [(P, r) for P in P_GRID for r in R_GRID]
        rows = ([window(rng, P, r, SHORT[kind]) for P, r in cells for _ in range(n)] if strat else
                [window(rng, P_GRID[rng.integers(len(P_GRID))], R_GRID[rng.integers(len(R_GRID))], SHORT[kind]) for _ in range(n)])
        arr = {k: np.stack([w[k] for w in rows]) for k in rows[0]}
        f32 = {k: v.astype('float32') for k, v in arr.items() if v.ndim == 2}
        np.savez_compressed(out/f'{split}.npz', **f32, **{k: v for k, v in arr.items() if v.ndim == 1},
                            dataset=np.array([f'P{int(P)}_r{r:g}' for P, r in zip(arr['P'], arr['r'])]), series_id=np.arange(len(rows)),
                            start=np.zeros(len(rows), int), split=np.array([split]*len(rows)))
        print(split, len(rows), flush=True)
    (out/'config.json').write_text(json.dumps(dict(T=T, L=L, generator='build_lab.py', seed=seed, P=P_GRID, r=R_GRID, short=SHORT, boundary='T/3'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('--n-train', type=int, default=20000); p.add_argument('--seed', type=int, default=1017)
    a = p.parse_args(); build(Path(a.out), a.n_train, a.seed)
