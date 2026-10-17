"""ES+Fourier baseline (no training): exponential smoothing (alpha 0.05) of the deseasonalised context plus the
estimator's Fourier continuation of the seasonal part; Gaussian predictive with sigma = sd of the context residual
(context - trend - seasonal), closed-form CRPS in context-normalised units (= the models' ncrps). Rows in the format of
baselines_eval.py (method 'es_fourier'), appended to the output CSV.

Usage: python es_fourier_baseline.py <cache> [--out baseline_rows_es.csv]   (after cache_fourier.py)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from fm_paths import RUNS
from data import T, normalize
from fourier_arm import cache_tag


def es(v, a=.05):
    lev = v[0]
    for x in v[1:]: lev = a*x+(1-a)*lev
    return lev


def main():
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--clip', type=float, default=20.)
    p.add_argument('--splits', default='test_id,test_ood'); p.add_argument('--out', default=str(RUNS/'baseline_rows_es.csv'))
    a = p.parse_args(); cache = Path(a.cache); rows = []
    for split in a.splits.split(','):
        d = np.load(cache/f'{split}.npz', allow_pickle=True); f = np.load(cache/f'{split}_fourier_{cache_tag()}.npz')
        z = normalize(d['x'].astype(np.float64))[0]; zc = np.clip(z[:, :T], -a.clip, a.clip); y = z[:, T:]
        for i in range(len(z)):
            x = zc[i]; mu = es(x-f['seasonal'][i])+f['seasonal_full'][i][T:]; sd = max((x-f['trend'][i]-f['seasonal'][i]).std(), 1e-6)
            w = (y[i]-mu)/sd; crps = sd*(w*(2*norm.cdf(w)-1)+2*norm.pdf(w)-1/np.sqrt(np.pi))
            rows.append(dict(split=split, dataset=d['dataset'][i], series_id=i, method='es_fourier', ncrps=crps.mean(), nmae=np.abs(y[i]-mu).mean()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), 'rows ->', a.out)


if __name__ == '__main__':
    main()
