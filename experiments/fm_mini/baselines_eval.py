"""Statistical baselines on the mini-FM evaluation splits, Gaussian predictive with closed-form CRPS.

All work in context-normalised units (z), so CRPS(z) = CRPS(x)/scale = the models' ncrps.
Inputs are the clipped contexts the models see; truth is the unclipped horizon.
- naive: last value, sigma_h = sd(diff) sqrt(h)
- seasonal_naive: p = round(longest estimated period) (naive if none), sigma_h = sd(x_t - x_{t-p}) sqrt(ceil(h/p))
- period: HP trend level + Fourier extrapolation at estimated periods; sigma = sd(context residual)
- ar: AutoReg (lags by BIC, const+trend) on the context, dynamic forecast and its variance
"""
import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from fm_paths import ROOT, VER0_DIR, RUNS, CACHE  # noqa: E402,F401  (also puts the shared benchmark modules on sys.path)
from data import T, L, normalize
from fourier_arm import cache_tag

OUT = RUNS


def gaussian(y, mu, sd):
    sd = np.maximum(sd, 1e-6); w = (y-mu)/sd
    crps = sd*(w*(2*norm.cdf(w)-1)+2*norm.pdf(w)-1/np.sqrt(np.pi))
    lo, hi = mu-1.6449*sd, mu+1.6449*sd
    return dict(ncrps=crps.mean(), nmae=np.abs(y-mu).mean(), coverage90=((y >= lo) & (y <= hi)).mean(), width90=(hi-lo).mean(),
                ncrps_h1=crps[:32].mean(), ncrps_h3=crps[64:].mean())


def ar(x):
    from statsmodels.tsa.ar_model import AutoReg
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fits = [AutoReg(x, lags=p, trend='ct', hold_back=20).fit() for p in (1, 2, 5, 10, 20)]
        f = min(fits, key=lambda a: a.bic); pr = f.get_prediction(start=T, end=T+L-1, dynamic=True)
    return np.asarray(pr.predicted_mean), np.sqrt(np.asarray(pr.var_pred_mean))


def main():
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--clip', type=float, default=20.)
    p.add_argument('--splits', default='test_id,test_ood')
    p.add_argument('--out', default=str(OUT/'baseline_rows.csv')); a = p.parse_args(); cache = Path(a.cache)
    rows = []; h = np.arange(1, L+1)
    for split in a.splits.split(','):
        d = np.load(cache/f'{split}.npz'); z = normalize(d['x'].astype(np.float64))[0]
        c = np.load(cache/f'{split}_fourier_{cache_tag()}.npz')
        for i in range(len(z)):
            x = np.clip(z[i, :T], -a.clip, a.clip); y = z[i, T:]
            base = dict(split=split, dataset=d['dataset'][i], series_id=i)
            dx = np.diff(x)
            rows.append(dict(base, method='naive', **gaussian(y, np.full(L, x[-1]), dx.std()*np.sqrt(h))))
            periods = c['periods'][i]; pmax = periods.max()
            if pmax >= 2:
                q = int(round(pmax)); r = x[q:]-x[:-q]
                mu = x[T-q+np.arange(L) % q]; sd = r.std()*np.sqrt(np.ceil(h/q))
            else: q = 0; mu = np.full(L, x[-1]); sd = dx.std()*np.sqrt(h)
            rows.append(dict(base, method='seasonal_naive', period=q, **gaussian(y, mu, sd)))
            resid = x-c['trend'][i]-c['seasonal'][i]
            rows.append(dict(base, method='period', period=pmax, **gaussian(y, c['trend'][i][-1]+c['seasonal_full'][i][T:], np.full(L, resid.std()))))
            mu, sd = ar(x); rows.append(dict(base, method='ar', **gaussian(y, mu, sd)))
            if i % 256 == 0: print(split, i, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print(df.groupby(['split', 'method']).ncrps.mean())


if __name__ == '__main__':
    main()
