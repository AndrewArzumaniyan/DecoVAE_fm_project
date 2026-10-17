"""Evaluate models on the decomposition laboratory (build_lab.py).

For a trained run (any arm; zero-shot from real data or trained on the lab) or the pseudo-run `estimator` (period_decomp
alone, from the cache components), per window of a lab split:
- decomposition on the context (models with a seasonal branch and the estimator), in the window's normalised units:
  mse_trend, mse_seasonal vs the true components (the T/3 rule decides where the long wave belongs), and lw_to_seasonal =
  least-squares coefficient of the long wave when the model's seasonal is regressed on [true short cycle, true long wave]
  (1 = the wave went to the seasonal, 0 = it did not);
- forecast (models, 64 samples): CRPS / oracle CRPS, where the oracle is the true N(trend + seasonal, sd^2); mse_mean of the
  predictive mean vs the noise-free truth; coverage of the central 50/80/98% intervals (from the PIT); w80_ratio = model
  80% width / true 80% width.
Output: <out>/<name>_<split>.csv.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train import runner, T  # noqa: E402
from metrics import distribution  # noqa: E402


def lw_share(seas, short, lw):
    """Per-window LS coefficient of lw in seas ~ a short + b lw (context)."""
    X = np.stack([short, lw], -1); XtX = np.einsum('ntk,ntj->nkj', X, X)+1e-9*np.eye(2); Xty = np.einsum('ntk,nt->nk', X, seas)
    return np.linalg.solve(XtX, Xty[..., None])[:, 1, 0]


def truth(raw):
    m, s = raw['mean'], raw['scale']
    return dict(trend=(raw['trend_true']-m)/s, seasonal=raw['seasonal_true']/s, lw=raw['long_wave']/s, short=raw['short_cycle']/s)


def decomp_rows(tr_hat, se_hat, tru):
    c = slice(0, T)
    return dict(mse_trend=((tr_hat[:, c]-tru['trend'][:, c])**2).mean(1), mse_seasonal=((se_hat[:, c]-tru['seasonal'][:, c])**2).mean(1),
                lw_to_seasonal=lw_share(se_hat[:, c], tru['short'][:, c], tru['lw'][:, c]))


def run_model(run, cache, split, device):
    cfg = json.loads((run/'provenance.json').read_text())['config']; train.SIZE.clear(); train.SIZE.update(cfg.get('size') or {})
    model = train.make_model(cfg['arm']).to(device)
    model.load_state_dict(torch.load(run/'best.pt', weights_only=False, map_location=device)['model'])
    raw = train.load_split(cache, split, cfg['arm'], cfg['clip']); _, outputs = runner.evaluate(model, raw, device, 64)
    idx = np.concatenate([o['indices'] for o in outputs]); assert (idx == np.arange(len(raw['x']))).all()
    smp = np.concatenate([o['samples'] for o in outputs], 1); dec = np.concatenate([o['decoded_normalized'] for o in outputs]); tru = truth(raw)
    x = raw['x'][idx].astype(float); y = x[:, T:]; s = raw['scale'][idx, 0]
    mu_t = (raw['trend_true']+raw['seasonal_true'])[idx, T:]; sd = raw['sd'][idx, T:]
    w = (y-mu_t)/sd; oracle = (sd*(w*(2*norm.cdf(w)-1)+2*norm.pdf(w)-1/np.sqrt(np.pi))).mean(1)
    crps = distribution(y, smp)['crps']; pit = (smp < y[None]).mean(0); q = np.quantile(smp, [.1, .9], axis=0)
    rows = dict(crps_ratio=crps/oracle, ncrps=crps/s, mse_mean=((smp.mean(0)-mu_t)**2).mean(1)/s**2,
                c50=(np.abs(pit-.5) <= .25).mean(1), c80=(np.abs(pit-.5) <= .4).mean(1), c98=(np.abs(pit-.5) <= .49).mean(1),
                w80_ratio=(q[1]-q[0]).mean(1)/(2*1.2816*sd).mean(1))
    if np.abs(dec[:, 1]).max() > 0: rows.update(decomp_rows(dec[:, 0], dec[:, 1], tru))
    return raw, idx, rows, cfg['arm'], cfg['seed']


def run_estimator(cache, split):
    raw = train.load_split(cache, split, 'Df', 20.); idx = np.arange(len(raw['x'])); tru = truth(raw)
    comp = raw['components']  # (N, 2, T): HP trend and Fourier seasonal of the normalised context
    pad = lambda a: np.concatenate([a, np.zeros((len(a), tru['trend'].shape[1]-a.shape[1]))], 1)
    return raw, idx, decomp_rows(pad(comp[:, 0]), pad(comp[:, 1]), tru), 'estimator', 0


def main():
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+', help='run dirs or "estimator"'); p.add_argument('--cache', required=True)
    p.add_argument('--splits', default='test_id,test_ood'); p.add_argument('--out', required=True); p.add_argument('--device', default='cuda')
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache)
    for r in a.runs:
        for split in a.splits.split(','):
            raw, idx, rows, arm, seed = run_estimator(cache, split) if r == 'estimator' else run_model(Path(r), cache, split, a.device)
            meta = {k: np.asarray(raw[k])[idx] for k in ('dataset', 'P', 'r', 'p', 'sigma_n', 'phase_noise', 'kink')}
            name = 'estimator' if r == 'estimator' else Path(r).name
            pd.DataFrame(dict(**meta, **rows, arm=arm, seed=seed, run=name)).to_csv(out/f'{name}_{split}.csv', index=False)
            print(name, split, len(idx), {k: round(float(np.mean(v)), 3) for k, v in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
