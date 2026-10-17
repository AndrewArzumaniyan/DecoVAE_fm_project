"""Collect the per-run outputs of finished mini-FM runs into the small tables in results/.

Run directories (written by train.py and eval_space.py) hold checkpoints, sample dumps and per-window tables; together
they are tens of gigabytes. This script keeps only what the tables and figures need:

- runs.csv               one row per run: id, family, arm, seed, cache, training settings, epochs, parameters, versions
- history.csv            validation curve of every run
- window_scores.parquet  per-window scores of eval_space.py (x-space, 128 samples): nCRPS, guarded nCRPS (stored only
                         where the guard changed the value, NaN elsewhere), nMAE, energy and variogram scores, latent share
- train_eval_scores.parquet  per-window scores of the evaluation inside train.py, for runs without eval_space output
- calibration.parquet    per (run, split, source): PCE, CCE, c50 / c80 / c98, share above q95, and a 20-bin PIT histogram
- baselines.parquet      per-window scores of the statistical baselines (baselines_eval.py)
- lab_scores.parquet     per-window rows of lab_eval.py (synthetic lab, incl. the estimator alone)

Usage: python collect_results.py --runs <dir with run folders> --out results [--lab <dir>] [--baselines csv ...]
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from calib_metrics import window_metrics

EXCLUDE = re.compile(r'(smoke|^srv2b0_|^r[123]d?_|^lseas|_loso_)')  # smoke tests, learning-rate probe, laptop runs,
# unfinished leave-one-source-out folds
SPLITS = ('test_id', 'test_ood')
SCORE_COLS = ['ncrps', 'ncrps_g', 'nmae', 'nes', 'nvs', 'latent_share']  # coverage comes from the PIT (calibration.parquet)
PIT_BINS = 20


def family(run_id):
    """Run id without the seed suffix: srv10_Dmc1_s3 -> srv10_Dmc1 (a trailing 'r' marks a resumed copy)."""
    return re.sub(r'_s\d+r?$', '', run_id)


def run_row(run):
    prov = json.loads((run/'provenance.json').read_text()); cfg = prov['config']
    status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    size = cfg.get('size') or {}
    return dict(run=run.name, family=family(run.name), arm=cfg['arm'], seed=cfg['seed'], cache=Path(cfg['cache']).name,
                recon_weight=cfg.get('recon_weight', 1.0), batch=cfg.get('batch', 32), lr=cfg.get('lr', 1e-3), patience=cfg.get('patience', 10),
                val_every=cfg.get('val_every', 1), clip=cfg.get('clip', 0.), d_model=size.get('d_model'), n_layers=size.get('n_layers'),
                enc_layers=size.get('enc_layers'), status=status.get('status'), epochs=status.get('epochs'),
                early_stopped=status.get('early_stopped'), parameters=status.get('parameters'),
                torch=prov['versions'].get('torch'), numpy=prov['versions'].get('numpy'), gpu=prov.get('gpu'),
                lock_sha256=(prov.get('lock_sha256') or '')[:12], time_utc=prov['time_utc'])


def calibration(pit, ds):
    """Same quantities as the report tables: PIT pooled within a source, window CCE averaged within a source."""
    cce = window_metrics(pit)['cce']; rows = []
    for d in np.unique(ds):
        s = ds == d; u = pit[s].ravel()
        hist = np.histogram(u, bins=PIT_BINS, range=(0, 1))[0]
        rows.append(dict(dataset=d, n_windows=int(s.sum()), pce=np.mean([abs(q-(u <= q).mean()) for q in np.arange(.1, 1., .1)]),
                         cce=cce[s].mean(), c50=(np.abs(u-.5) <= .25).mean(), c80=(np.abs(u-.5) <= .4).mean(),
                         c98=(np.abs(u-.5) <= .49).mean(), up95=(u > .95).mean(),
                         **{f'pit{k:02d}': int(v) for k, v in enumerate(hist)}))
    return rows


def compact(df):
    for c in df.columns:
        if df[c].dtype == np.float64: df[c] = df[c].astype('float32')
        elif df[c].dtype == object and c != 'series_id': df[c] = df[c].astype('category')
    return df


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', required=True); p.add_argument('--out', required=True)
    p.add_argument('--lab'); p.add_argument('--baselines', nargs='*', default=[]); a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in sorted(Path(a.runs).iterdir()) if (r/'status.json').exists() and not EXCLUDE.search(r.name)]  # no status = never started
    meta, hist, scores, early, calib = [], [], [], [], []
    for run in runs:
        row = run_row(run); meta.append(row)
        if (run/'history.csv').exists():
            h = pd.read_csv(run/'history.csv'); h.insert(0, 'run', run.name); hist.append(h)
        has_space = all((run/f'space_eval_{s}.csv').exists() for s in SPLITS)
        for split in SPLITS:
            if has_space:
                df = pd.read_csv(run/f'space_eval_{split}.csv', dtype={'series_id': str})
                df = df[['dataset', 'series_id', *[c for c in SCORE_COLS if c in df]]]
                df['ncrps_g'] = df.ncrps_g.where(df.ncrps_g != df.ncrps)
                df.insert(0, 'window', np.arange(len(df))); df.insert(0, 'split', split); df.insert(0, 'run', run.name); scores.append(df)
                pit = np.load(run/f'space_eval_pit_{split}.npz')['pit']
                calib += [dict(run=run.name, split=split, **r) for r in calibration(pit, df.dataset.values)]
            else:
                f = run/('fm_rows.csv' if split == 'test_id' else 'ood_rows.csv')
                if not f.exists(): continue
                df = pd.read_csv(f, dtype={'series_id': str})
                keep = [c for c in ('dataset', 'series_id', 'ncrps', 'nmae') if c in df]
                df = df[keep]; df.insert(0, 'window', np.arange(len(df))); df.insert(0, 'split', split); df.insert(0, 'run', run.name); early.append(df)
        print(run.name, 'space' if has_space else 'train-eval', flush=True)
    pd.DataFrame(meta).to_csv(out/'runs.csv', index=False)
    pd.concat(hist).to_csv(out/'history.csv', index=False, float_format='%.6g')
    compact(pd.concat(scores, ignore_index=True)).to_parquet(out/'window_scores.parquet', compression='zstd', index=False)
    compact(pd.concat(early, ignore_index=True)).to_parquet(out/'train_eval_scores.parquet', compression='zstd', index=False)
    compact(pd.DataFrame(calib)).to_parquet(out/'calibration.parquet', compression='zstd', index=False)
    if a.baselines:
        b = []
        for f in a.baselines:
            df = pd.read_csv(f, dtype={'series_id': str}); df.insert(0, 'protocol', 'loso' if 'loso' in Path(f).stem else 'main'); b.append(df)
        compact(pd.concat(b, ignore_index=True)).to_parquet(out/'baselines.parquet', compression='zstd', index=False)
    if a.lab:
        lab = []
        for f in sorted(Path(a.lab).glob('*_test_*.csv')):
            df = pd.read_csv(f); df.insert(0, 'split', 'test_id' if f.stem.endswith('test_id') else 'test_ood'); lab.append(df)
        compact(pd.concat(lab, ignore_index=True)).to_parquet(out/'lab_scores.parquet', compression='zstd', index=False)
    print(len(meta), 'runs ->', out)


if __name__ == '__main__':
    main()
