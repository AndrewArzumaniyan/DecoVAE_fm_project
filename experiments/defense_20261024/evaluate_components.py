"""Extend frozen laboratory checkpoints with horizon-component and interval diagnostics.

No training, no oracle inputs, and no change to the original T/3 lab labels.
Additional simple/negative cases have explicit generator-defined components.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr


CASES = ['noise_only', 'trend_only', 'single_24', 'slow_1024_plus_24',
         'two_seasons_24_64', 'nearby_seasons_24_27', 'amplitude_drift', 'weak_seasonality']


def diagnostic_cache(root, train):
    cache = root / 'code/fm_mini/cache/defense_diagnostics_20261024'
    cache.mkdir(exist_ok=True)
    path = cache / 'test_ood.npz'
    if not path.exists():
        rows = []
        t = np.arange(608, dtype=float)
        for k, case in enumerate(CASES):
            for i in range(32):
                rng = np.random.default_rng(np.random.SeedSequence([20261024, k, i]))
                phase1, phase2 = rng.uniform(0, 2*np.pi, 2)
                trend = rng.normal(0, .5) + rng.uniform(-.002, .002)*t
                seasonal = np.zeros_like(t)
                if case == 'noise_only':
                    trend = np.zeros_like(t)
                elif case not in ['trend_only']:
                    seasonal = np.sin(2*np.pi*t/24 + phase1)
                if case == 'slow_1024_plus_24':
                    trend = trend + 1.5*np.sin(2*np.pi*t/1024 + phase2)
                if case == 'two_seasons_24_64':
                    seasonal = seasonal + .6*np.sin(2*np.pi*t/64 + phase2)
                if case == 'nearby_seasons_24_27':
                    seasonal = seasonal + .6*np.sin(2*np.pi*t/27 + phase2)
                if case == 'amplitude_drift':
                    seasonal = seasonal*(.5 + .6*t/512)
                if case == 'weak_seasonality':
                    seasonal = seasonal*.1
                innovation = np.random.default_rng(np.random.SeedSequence([20261025, k, i])).normal(size=608)
                for sigma in [.1, .3, .6]:
                    rows.append(dict(x=trend+seasonal+sigma*innovation, trend_true=trend,
                                     seasonal_true=seasonal, sd=np.full(608, sigma),
                                     dataset=f'{case}_sigma{sigma:g}', series_id=f'diag-{k}-{i}',
                                     case=case, sigma_n=sigma))
        arrays = {key: np.stack([r[key] for r in rows]).astype('float32')
                  for key in ['x', 'trend_true', 'seasonal_true', 'sd']}
        arrays.update({key: np.array([r[key] for r in rows]) for key in ['dataset', 'series_id', 'case', 'sigma_n']})
        arrays.update(start=np.zeros(len(rows), dtype=int), split=np.array(['test_ood']*len(rows)))
        np.savez_compressed(path, **arrays)
        (cache / 'config.json').write_text(json.dumps(dict(seed=20261024, cases=CASES, rows=len(rows),
                                                         target_rule='Explicit generator-defined components; no period cutoff',
                                                         common_noise='same innovation for three noise levels of a clean curve'), indent=2))
    # Everything estimated here uses only each context, exactly as in the training cache.
    from fourier_arm import attach, cache_tag
    import seasonal_arms as sa
    raw = dict(np.load(path))
    z, mean, scale = train.normalize(raw['x'].astype('float64'))
    z64 = np.clip(z, -20, 20)
    raw.update(z=z64.astype('float32'), mean=mean, scale=scale)
    attach(raw, cache, 'test_ood')
    features = cache / f'test_ood_s12_{cache_tag()}.npz'
    if not features.exists():
        extras = [sa.window_extras(z64[i, :512], raw['components'][i, 0].astype(float),
                                   raw['seasonal_full'][i].astype(float), raw['periods'][i], raw['coefs'][i])
                  for i in range(len(z64))]
        np.savez_compressed(features, coefs_hl=np.stack([x[0] for x in extras]),
                            feats=np.stack([x[1] for x in extras]).astype('float32'))
    return cache


def score(root, run_name, cache, split, label, cohort, draws, train):
    run = root / 'notes/analysis/fm_mini' / run_name
    cfg = json.loads((run / 'provenance.json').read_text())['config']
    train.SIZE.clear()
    train.SIZE.update(cfg.get('size') or {})
    model = train.make_model(cfg['arm']).to('cuda')
    model.load_state_dict(torch.load(run / 'best.pt', weights_only=False, map_location='cuda')['model'])
    raw = train.load_split(cache, split, cfg['arm'], cfg['clip'])
    _, outputs = train.runner.evaluate(model, raw, 'cuda', draws)
    idx = np.concatenate([o['indices'] for o in outputs])
    assert np.array_equal(idx, np.arange(len(raw['x'])))
    samples = np.concatenate([o['samples'] for o in outputs], axis=1)
    mean = np.concatenate([o['mean'] for o in outputs])
    decoded = np.concatenate([o['decoded_normalized'] for o in outputs])
    tr, se = raw['trend_true'].astype(float), raw['seasonal_true'].astype(float)
    sd = raw['sd'].astype(float)[:, 512:]
    y = raw['x'].astype(float)[:, 512:]
    truth = (tr + se)[:, 512:]
    scale = raw['scale'][:, 0]
    from metrics import distribution
    metrics = distribution(y, samples)
    u = (y-truth)/sd
    oracle = (sd*(u*(2*ndtr(u)-1)+2*np.exp(-u*u/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))).mean(1)
    data = dict(dataset=raw['dataset'], series_id=raw['series_id'], model=label,
                seed=cfg['seed'], run=run_name, cohort=cohort, split=split,
                mean_horizon_mse=((mean-truth)**2).mean(1)/scale**2,
                ncrps=metrics['crps']/scale, crps_over_process_oracle=metrics['crps']/oracle)
    for coverage in [.5, .8, .9, .98]:
        q = np.quantile(samples, [(1-coverage)/2, (1+coverage)/2], axis=0)
        data[f'coverage{round(100*coverage)}'] = ((y >= q[0]) & (y <= q[1])).mean(1)
        data[f'width{round(100*coverage)}'] = (q[1]-q[0]).mean(1)/scale
    # The plain and feature Transformers do not claim an identifiable output decomposition.
    if label in ['A', 'Dmc1']:
        tr_true = (tr-raw['mean'])/raw['scale']
        se_true = se/raw['scale']
        for region, sl in [('context', slice(0, 512)), ('horizon', slice(512, 608))]:
            data[f'trend_{region}_mse'] = ((decoded[:, 0, sl]-tr_true[:, sl])**2).mean(1)
            data[f'seasonal_{region}_mse'] = ((decoded[:, 1, sl]-se_true[:, sl])**2).mean(1)
        denom = (se_true[:, :512]**2).sum(1)
        projection = ((decoded[:, 0, :512]-tr_true[:, :512])*se_true[:, :512]).sum(1)/np.maximum(denom, 1e-12)
        projection[denom < 1e-10] = np.nan
        data['seasonal_in_trend_gain'] = projection
        data['decoded_seasonal_rms_context'] = np.sqrt((decoded[:, 1, :512]**2).mean(1))
    for key in ['P', 'r', 'p', 'case', 'sigma_n', 'phase_noise']:
        if key in raw:
            data[key] = raw[key]
    return pd.DataFrame(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--draws', type=int, default=128)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root/'code/fm_mini'))
    import train
    torch.set_num_threads(1)
    cache = diagnostic_cache(root, train)
    if a.prepare_only:
        print('diagnostic cache prepared:',cache,flush=True)
        return
    out = root/'final_results'
    fragments = out/'component_fragments'
    fragments.mkdir(parents=True, exist_ok=True)
    frames = []
    groups = {'A': 'lab_A', 'N6mc1': 'srv13_lab_N6mc1',
              'Dmc1': 'lab_Dmc1', 'N6s': 'srv13_lab_N6s'}
    for label, family in groups.items():
        for seed in range(3):
            run = f'{family}_s{seed}'
            best = root/'notes/analysis/fm_mini'/run/'best.pt'
            fingerprint = hashlib.sha256(best.read_bytes()).hexdigest()
            for cohort, ca, split in [('lab', root/'code/fm_mini/cache/lab_1017', 'test_id'),
                                       ('lab', root/'code/fm_mini/cache/lab_1017', 'test_ood'),
                                       ('simple_negative', cache, 'test_ood')]:
                target = fragments/f'{label}_s{seed}_{cohort}_{split}.csv'
                marker = target.with_suffix('.json')
                expected = dict(draws=a.draws, best_sha256=fingerprint)
                if not (marker.exists() and json.loads(marker.read_text()) == expected and target.exists()):
                    print('evaluate',run,cohort,split,flush=True)
                    df = score(root, run, ca, split, label, cohort, a.draws, train)
                    temporary = target.with_suffix('.csv.tmp')
                    df.to_csv(temporary, index=False)
                    temporary.replace(target)
                    marker.write_text(json.dumps(expected, indent=2)+'\n')
                frames.append(pd.read_csv(target))
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out/'component_windows.csv', index=False)
    metrics = [c for c in all_rows.select_dtypes('number').columns if c not in ['seed', 'P', 'r', 'p', 'sigma_n', 'phase_noise']]
    summary = all_rows.groupby(['cohort','split','dataset','model','seed'], observed=True)[metrics].mean()
    summary.to_csv(out/'component_by_seed.csv')
    summary.groupby(['cohort','split','dataset','model'], observed=True)[metrics].agg(['mean','std']).to_csv(out/'component_summary.csv')
    intervals = []
    rng = np.random.default_rng(20261024)
    for key, group in all_rows.groupby(['cohort','split','dataset','model'], observed=True):
        available = [m for m in metrics if group[m].notna().all()]
        ss = sorted(group.seed.unique())
        parts = [group[group.seed == s].sort_values('series_id') for s in ss]
        ids = parts[0].series_id.tolist()
        if any(d.series_id.tolist() != ids for d in parts):
            raise ValueError('Component bootstrap requires paired seed evaluations')
        values = np.stack([d[available].to_numpy(float) for d in parts])
        ns, nw, nm = values.shape
        sw = rng.multinomial(ns, np.full(ns, 1/ns), size=2000)/ns
        ww = rng.multinomial(nw, np.full(nw, 1/nw), size=2000)/nw
        simulated = np.einsum('bs,snm,bn->bm', sw, values, ww, optimize=True)
        low, high = np.quantile(simulated, [.025,.975], axis=0)
        mean = values.mean((0,1))
        for j, metric in enumerate(available):
            intervals.append(dict(zip(['cohort','split','dataset','model'],key),
                                  metric=metric, mean=mean[j], ci_low=low[j], ci_high=high[j],
                                  seeds=ns, series=nw, bootstrap=2000))
    pd.DataFrame(intervals).to_csv(out/'component_intervals.csv',index=False)
    (out/'COMPONENT_PROTOCOL.md').write_text(
        '# Frozen-checkpoint component diagnostics\n\n'
        'No retraining and no test-based checkpoint selection. Three training seeds, 128 predictive draws.\n'
        'The original lab retains its T/3 target convention; this convention is not a universal definition of trend.\n'
        'Additional cases have explicit generator-defined components, including no-season controls.\n'
        'True components and noise levels are used only for evaluation, never as model inputs.\n'
        'Forecast MSE is against the clean generator curve. The process oracle knows the generator trajectory.\n'
        'Only A and Dmc1 claim an explicit output decomposition; N6/N6s component errors are undefined.\n'
        'Intervals jointly resample paired training seeds and synthetic series within each case (2,000 draws).\n'
        'Widths are in context-standard-deviation units. Inspect widths together with coverage.\n'
        'Simple/negative cases are a new diagnostic distribution, not a claim that the lab training covered them.\n')
    (out/'components_complete.json').write_text(json.dumps(dict(passed=True, windows=len(all_rows),
                                                               models=list(groups), seeds=3, draws=a.draws),indent=2)+'\n')
    print('component diagnostics complete',len(all_rows),flush=True)


if __name__ == '__main__':
    main()
