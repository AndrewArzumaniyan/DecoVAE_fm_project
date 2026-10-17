"""Data of the report's forecast figure: results/fig_forecasts.npz.

Window rule (fixed before plotting): for each of four OOD sources, the window with the median nCRPS ratio
PD-MixC / TF-6L-MixC (seed 0), i.e. a typical window. Takes mean and 5% / 95% quantiles from the ood_outputs.npz that
train.py wrote for the two runs, and the raw window from the cache.

Usage: python figure_data.py --cache <cache> --tf <TF-6L-MixC run> --pd <PD-MixC run>
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES = ['Electricity', 'Pedestrian_Counts', 'Solar', 'Uber_TLC']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cache', required=True); p.add_argument('--tf', required=True); p.add_argument('--pd', required=True)
    p.add_argument('--out', default=str(Path(__file__).resolve().parent/'results'/'fig_forecasts.npz')); a = p.parse_args()
    c = np.load(Path(a.cache)/'test_ood.npz', allow_pickle=True); ds = c['dataset']
    rows = {k: pd.read_csv(Path(r)/'ood_rows.csv') for k, r in (('tf', a.tf), ('pd', a.pd))}
    ratio = rows['pd'].ncrps.values/rows['tf'].ncrps.values; idx = []
    for src in SOURCES:
        w = np.flatnonzero(ds == src); idx.append(int(w[np.argsort(ratio[w])[len(w)//2]]))
    out = dict(index=np.array(idx), dataset=ds[idx], x=c['x'][idx].astype(np.float32))
    for k, r in (('tf', a.tf), ('pd', a.pd)):
        o = np.load(Path(r)/'ood_outputs.npz')
        for q in ('mean', 'q05', 'q95'): out[f'{k}_{q}'] = o[q][idx].astype(np.float32)
        out[f'{k}_ncrps'] = rows[k].ncrps.values[idx]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out, **out); print('windows', idx, '->', a.out)


if __name__ == '__main__':
    main()
