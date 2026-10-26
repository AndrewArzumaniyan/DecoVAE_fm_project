"""Check a transferred checkpoint against its original saved scores, without rewriting it."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--tolerance', type=float, default=1e-7)
    a = p.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root / 'code/fm_mini'))
    import train
    torch.set_num_threads(1)
    run = root / 'notes/analysis/fm_mini' / a.run
    cfg = json.loads((run / 'provenance.json').read_text())['config']
    train.SIZE.clear()
    train.SIZE.update(cfg.get('size') or {})
    model = train.make_model(cfg['arm']).to('cuda')
    model.load_state_dict(torch.load(run / 'best.pt', weights_only=False, map_location='cuda')['model'])
    cache = root / cfg['cache']
    raw = train.load_split(cache, 'test_ood', cfg['arm'], cfg['clip'])
    metrics, _ = train.runner.evaluate(model, raw, 'cuda', 128, limit=a.limit)
    values = metrics.crps.values / raw['scale'][:len(metrics), 0]
    reference = run / 'space_eval_test_ood.csv'
    if not reference.exists():
        reference = run / 'ood_rows.csv'
    expected = pd.read_csv(reference).ncrps.to_numpy()[:len(values)]
    delta = np.abs(values - expected)
    result = dict(run=a.run, windows=len(values), maximum_absolute_difference=float(delta.max()),
                  mean_absolute_difference=float(delta.mean()), tolerance=a.tolerance,
                  passed=bool(np.isfinite(delta).all() and delta.max() <= a.tolerance),
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),
                  versions={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'statsmodels', 'pandas', 'pyarrow']})
    output = root / a.output
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit('Transferred checkpoint did not reproduce the saved scores')


if __name__ == '__main__':
    main()
