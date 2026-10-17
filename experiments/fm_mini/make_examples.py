"""Example forecasts for the overview notebook: results/examples.npz.

For a fixed rule of windows (the first `--per-source` windows of every source in the split, in cache order, so that the
choice does not depend on how the forecasts look) and a few trained runs, store the raw context and horizon, predictive
quantiles from 128 samples, and the decoded trend / seasonal components in data units. Sampling is the evaluation of
train.py (runner.evaluate, evaluation seed 1729).

Usage: python make_examples.py --runs <run dir> ... --names PD-MixC ... [--split test_ood] [--per-source 2]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import train
from train import runner, T

QUANTILES = np.array([.05, .1, .25, .5, .75, .9, .95])


def subset(raw, idx):
    n = len(raw['x'])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v) for k, v in raw.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--split', default='test_ood'); p.add_argument('--per-source', type=int, default=2); p.add_argument('--device', default='cuda')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent/'results'/'examples.npz')); a = p.parse_args()
    assert len(a.runs) == len(a.names)
    out, idx, base = {}, None, None
    for run, name in zip(map(Path, a.runs), a.names):
        cfg = json.loads((run/'provenance.json').read_text())['config']
        train.SIZE.clear(); train.SIZE.update(cfg.get('size') or {})
        model = train.make_model(cfg['arm']).to(a.device)
        model.load_state_dict(torch.load(run/'best.pt', weights_only=False, map_location=a.device)['model'])
        raw = train.load_split(train.ROOT/cfg['cache'], a.split, cfg['arm'], cfg['clip'])
        if idx is None:
            ds = np.asarray(raw['dataset']); idx = np.concatenate([np.flatnonzero(ds == d)[:a.per_source] for d in dict.fromkeys(ds)])
            base = dict(x=raw['x'][idx].astype(np.float32), dataset=ds[idx], series_id=np.asarray(raw['series_id'])[idx].astype(str))
        else:
            assert np.array_equal(raw['x'][idx], base['x']), f'{run}: different windows in its cache'
        r = subset(raw, idx); _, outputs = runner.evaluate(model, r, a.device, 128)
        smp = np.concatenate([o['samples'] for o in outputs], axis=1)                       # (128, N, L), data units
        dec = np.concatenate([o['decoded_normalized'] for o in outputs])                  # (N, 2, T+L), normalised
        m, s = r['mean'][:, :1], r['scale'][:, :1]
        out[f'{name}/quantiles'] = np.quantile(smp, QUANTILES, axis=0).astype(np.float32)
        out[f'{name}/trend'] = (dec[:, 0]*s+m).astype(np.float32)
        out[f'{name}/seasonal'] = (dec[:, 1]*s).astype(np.float32)                        # seasonal part has no level
        out[f'{name}/run'] = np.array(run.name)
        print(name, run.name, len(idx), 'windows', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **base, quantile_levels=QUANTILES, context_length=T, models=np.array(a.names), **out)
    print('->', a.out)


if __name__ == '__main__':
    main()
