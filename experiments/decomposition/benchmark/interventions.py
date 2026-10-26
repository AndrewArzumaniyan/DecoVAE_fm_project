"""Paired interventions: how the predicted variance responds to a controlled change of the input.

Each of 64 synthetic series is evaluated twice: as generated (base) and with one component changed
(modified). The ratio of predicted variances, modified / base, is reported per series.

Protocol
- Each window, base or modified, is normalised by its own context statistics, as in training.
  Normalising the modified window with the base statistics pushes the context out of the training
  range and produces spurious variance ratios of 100x and more.
- Predicted variance is the forecast variance in original units, averaged over the horizon.
- Latents are set to their posterior means, so the pass is deterministic.
- Summaries are the median, the geometric mean and the share of series with |log ratio| > log 2.
  The mean of ratios is reported too, but it is dominated by a few series.

Ideal ratio: 1 for a trend or seasonal change, (1 + value)^2 for a noise change (2.25 for noise x1.5).
Usage: python interventions.py RUN [RUN ...] --out NAME     (RUN = folder in RUNS_DIR with best.pt)
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from data import T, L, SCENARIOS, sample, normalize
from fourier_arm import FourierArm, decompose_window, make_model
from runner import OUT, save_json

INTERVENTIONS = [('slope', 3e-3), ('seasonal', .5), ('noise', .5)]
SEED, N = 990000, 64
LOG2 = np.log(2.)
_cache = {}


def load(run):
    cfg = json.loads((OUT/run/'provenance.json').read_text())['config']
    model = make_model(cfg['arm'])
    model.load_state_dict(torch.load(OUT/run/'best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval()
    model.reparameterize = lambda mu, logvar: mu  # posterior means
    return model, cfg


def records(n=N):
    return [sample(SEED + i, SCENARIOS[(i*7) % len(SCENARIOS)]) for i in range(n)]


def delta(record, kind, value):
    if kind == 'slope':
        return value*np.arange(T + L)
    return value*record[kind]  # 'seasonal' or 'noise': scale that true component by 1 + value


def components(z):
    out = []
    for row in z:
        key = row[:T].tobytes()
        if key not in _cache:
            _cache[key] = decompose_window(row[:T])
        out.append(_cache[key])
    trend, seasonal, periods, coefs, _ = (np.stack(v) for v in zip(*out))
    return trend, seasonal, periods, coefs


@torch.no_grad()
def horizon_variance(model, z, scale):
    if model.arm in 'CD':
        trend, seasonal, periods, coefs = components(z)
        model.components = (torch.tensor(trend, dtype=torch.float32), torch.tensor(seasonal, dtype=torch.float32))
        if isinstance(model, FourierArm):
            model.fourier = (torch.from_numpy(periods), torch.from_numpy(coefs))
    out = model(torch.from_numpy(z[:, :T].astype('float32')))
    return (out['forecast_logvar'].exp().numpy()*scale**2).mean(-1)


def run_rows(run, recs):
    model, cfg = load(run)
    xb = np.stack([r['x'] for r in recs]); zb, _, sb = normalize(xb)
    vb = horizon_variance(model, zb, sb)
    rows = []
    for kind, value in INTERVENTIONS:
        xm = xb + np.stack([delta(r, kind, value) for r in recs]); zm, _, sm = normalize(xm)
        ratio = horizon_variance(model, zm, sm)/np.maximum(vb, 1e-12)
        ideal = (1 + value)**2 if kind == 'noise' else 1.
        rows += [dict(run=run, arm=cfg['arm'], seed=int(cfg['seed']), series_id=i, scenario=json.dumps(recs[i]['scenario']),
                      kind=kind, value=value, ideal=ideal, ratio=float(r), log_ratio=float(np.log(r)))
                 for i, r in enumerate(ratio)]
    return rows


def summarize(df, keys):
    return df.groupby(keys).agg(
        n=('ratio', 'size'), ideal=('ideal', 'first'), median=('ratio', 'median'),
        geomean=('log_ratio', lambda v: float(np.exp(v.mean()))), mean=('ratio', 'mean'), max=('ratio', 'max'),
        frac_off_2x=('log_ratio', lambda v: float((v.abs() > LOG2).mean()))).reset_index()


def main():
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--out', default='interventions')
    a = p.parse_args(); torch.set_num_threads(1)
    out = OUT/a.out; out.mkdir(parents=True, exist_ok=True); recs = records(); rows = []
    for k, run in enumerate(a.runs, 1):
        rows += run_rows(run, recs); print(f'{k}/{len(a.runs)} {run}', flush=True)
    df = pd.DataFrame(rows); df.to_csv(out/'intervention_rows.csv', index=False)
    per_run = summarize(df, ['run', 'arm', 'seed', 'kind', 'value']); per_run.to_csv(out/'intervention_summary.csv', index=False)
    by_arm = per_run.groupby(['arm', 'kind', 'value']).agg(ideal=('ideal', 'first'), median_of_run_medians=('median', 'median'),
                                                          min_run_median=('median', 'min'), max_run_median=('median', 'max'),
                                                          runs=('run', 'size')).reset_index()
    by_arm.to_csv(out/'intervention_by_arm.csv', index=False)
    save_json(out/'provenance.json', dict(runs=a.runs, data_seed=SEED, n=N, interventions=INTERVENTIONS,
                                          normalisation='per window', latents='posterior means'))
    print(by_arm.to_string(index=False))


if __name__ == '__main__':
    main()
