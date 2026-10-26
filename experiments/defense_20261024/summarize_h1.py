"""Summarize the five-seed extension; preserve individual seeds, blocks and uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_rows(root, profile, arm, seed, real_control='published'):
    r = root / 'notes/analysis/fm_mini'
    if profile == 'real':
        if arm == 'Dmc1':
            run = (f'srv10_Dmc1_s{seed}' if real_control == 'published' else
                   f'srv14r_Dmc1_w0.2_s{seed}' if seed < 3 else f'defense_h1_real_Dmc1_s{seed}')
        elif seed < 3:
            run = f'srv13_{arm}_s{seed}'
        else:
            run = f'defense_h1_real_{arm}_s{seed}'
        path = r / run / 'space_eval_test_ood.csv'
    else:
        run = f'srv13_h1_{arm}_s{seed}' if seed < 3 else f'defense_h1_strong_{arm}_s{seed}'
        split = 'test_id' if profile == 'strong' else 'test_ood'
        folder = 'h1' if seed < 3 else 'defense_h1_metrics'
        path = r / folder / f'{run}_{split}.csv'
    d = pd.read_csv(path, dtype={'series_id': str})
    if 'block' in d:
        d = d.rename(columns={'block': 'dataset'})
    d['occurrence'] = d.groupby(['dataset', 'series_id'], sort=False).cumcount()
    d['run'] = run
    return d


def paired_interval(a, b, datasets, series, samples=2000, contrast='ratio'):
    """Paired bootstrap of training seeds and series within each fixed source/block."""
    rng = np.random.default_rng(20261024)
    groups = []
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        ids = series[mask]
        unique = sorted(set(ids))
        am = np.stack([a[:, mask][:, ids == sid].sum(1) for sid in unique], 1)
        bm = np.stack([b[:, mask][:, ids == sid].sum(1) for sid in unique], 1)
        counts = np.array([(ids == sid).sum() for sid in unique])
        groups.append((am, bm, counts))
    def combine(effects):
        return np.exp(np.log(effects).mean()) if contrast == 'ratio' else np.mean(effects)
    def effect(am, bm, counts):
        return (am.sum()/bm.sum() if contrast == 'ratio' else
                (am-bm).sum()/(len(am)*counts.sum()))
    point = float(combine([effect(am,bm,c) for am,bm,c in groups]))
    bootstrap = []
    for _ in range(samples):
        seeds = rng.integers(0, a.shape[0], a.shape[0])
        effects = []
        for am, bm, counts in groups:
            clusters = rng.integers(0, am.shape[1], am.shape[1])
            effects.append(effect(am[seeds][:,clusters],bm[seeds][:,clusters],counts[clusters]))
        bootstrap.append(combine(effects))
    low, high = np.quantile(bootstrap, [.025, .975])
    return point, float(low), float(high)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--real-control', choices=['published','matched'], default='published')
    a = p.parse_args()
    root = a.root.resolve()
    out = root / 'final_results'
    out.mkdir(exist_ok=True)
    comparisons, rows, blocks = [], [], []
    for profile in ['real', 'strong', 'base']:
        frames = {arm: [find_rows(root, profile, arm, seed, a.real_control) for seed in range(5)]
                  for arm in ['Dmc1', 'Dmc1Zn', 'Dmc1B4']}
        reference = frames['Dmc1'][0]
        keys = list(zip(reference.dataset, reference.series_id, reference.occurrence))
        for arm, fs in frames.items():
            for seed, d in enumerate(fs):
                if list(zip(d.dataset, d.series_id, d.occurrence)) != keys:
                    raise ValueError(f'Unpaired evaluation windows: {profile}, {arm}, {seed}')
        metrics = ['ncrps'] if profile == 'real' else ['crps', 'crps_h1', 'lv_err']
        for arm, fs in frames.items():
            for seed, d in enumerate(fs):
                cols = [c for c in [*metrics, 'coverage50', 'coverage80', 'coverage90', 'w80', 'c50', 'c80', 'c98'] if c in d]
                per = d.groupby('dataset')[cols].mean()
                for ds, s in per.iterrows():
                    blocks.append(dict(profile=profile, arm=arm, seed=seed, dataset=ds, **s.to_dict()))
                rows.append(dict(profile=profile, arm=arm, seed=seed, **per.mean().to_dict()))
        for arm in ['Dmc1Zn', 'Dmc1B4']:
            for metric in metrics:
                x = np.stack([d[metric].values for d in frames[arm]])
                y = np.stack([d[metric].values for d in frames['Dmc1']])
                point, low, high = paired_interval(x, y, reference.dataset.values, reference.series_id.values)
                for seed in range(5):
                    per_seed = pd.DataFrame(dict(ds=reference.dataset.values, a=x[seed], b=y[seed])).groupby('ds')[['a', 'b']].mean()
                    ratio = float(np.exp(np.log(per_seed.a / per_seed.b).mean()))
                    rows.append(dict(profile=profile, arm=arm, seed=seed, metric=metric, ratio_to_Dmc1=ratio))
                comparisons.append(dict(profile=profile, arm=arm, metric=metric, contrast='ratio', effect=point,
                                        ratio=point, ci_low=low, ci_high=high, seeds=5))
            # Signed changes in coverage are not automatically improvements. Inspect the nominal target.
            for candidates in [('c50','coverage50'),('c80','coverage80'),('c98','coverage98'),('coverage90',),('w80',)]:
                name=next((c for c in candidates if all(c in d for fs in frames.values() for d in fs)),None)
                if name is None:
                    continue
                x=np.stack([d[name].values for d in frames[arm]])
                y=np.stack([d[name].values for d in frames['Dmc1']])
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    continue
                point,low,high=paired_interval(x,y,reference.dataset.values,reference.series_id.values,contrast='difference')
                comparisons.append(dict(profile=profile,arm=arm,metric=name,contrast='difference',effect=point,
                                        ci_low=low,ci_high=high,seeds=5))
    pd.DataFrame(rows).to_csv(out / 'h1_by_seed.csv', index=False)
    pd.DataFrame(blocks).to_csv(out / 'h1_by_block.csv', index=False)
    table = pd.DataFrame(comparisons)
    table.to_csv(out / 'h1_comparisons.csv', index=False)
    lines = ['# Final-model H1 extension', '',
             'Real-data control: '+('matched current-environment reruns (srv14r w0.2 seeds 0–2; defense seeds 3–4).'
                                   if a.real_control=='matched' else
                                   'historical published srv10. Different training environment; this is a secondary comparison, not the matched primary H1 test.'),
             'Five paired training seeds (0–4). Lower ratios favor the added mechanism.',
             'Intervals resample paired training seeds and whole series within fixed sources/blocks (2,000 draws).',
             'They do not measure uncertainty over new sources. B4 is handcrafted volatility information, not a raw-input encoder.',
             'Real = main six-source OOD; strong = h1fm_strong/test_id; base = h1fm_strong/test_ood.', '',
             'Coverage/width differences are arm minus Dmc1, averaged equally over fixed sources/blocks.',
             'A positive coverage change alone does not establish better calibration; compare against nominal coverage and width.',
             'These multiple diagnostic contrasts are exploratory, without multiplicity correction.', '',
             '| Profile | Arm | Metric | Contrast | Effect | 95% interval |',
             '|---|---|---|---|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['profile']} | {row['arm']} | {row['metric']} | {row['contrast']} | {row['effect']:.5f} | [{row['ci_low']:.5f}; {row['ci_high']:.5f}] |")
    lines += ['', 'A non-significant result is not proof of equivalence. Inspect interval width, practical effect,',
              'per-seed results, coverage and variance error together. No test result selected a checkpoint or hyperparameter.', '']
    (out / 'H1_FINDINGS.md').write_text('\n'.join(lines))
    (out / 'h1_summary_complete.json').write_text(json.dumps({'passed': True, 'real_control':a.real_control,
                                                            'comparisons': comparisons}, indent=2) + '\n')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
