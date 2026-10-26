"""Frozen Chronos reference and all-model evaluation with the same nine-quantile score.

This is an external reference with potential pretraining overlap, not a clean
zero-shot leaderboard. No tuning or checkpoint selection uses these scores.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

Q = np.round(np.arange(.1, 1., .1), 1)
MODEL = 'amazon/chronos-bolt-small'


def qcrps(y, quantiles):
    error = y[..., None] - quantiles
    return (2*np.maximum(Q*error, (Q-1)*error).mean(-1)).mean(-1)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)


def raw_split(root, split):
    cache = root/'code/fm_mini/cache/fm_mini_644d2b51896d'
    with np.load(cache/f'{split}.npz') as f:
        raw = {k:f[k] for k in f.files}
    x = raw['x'].astype(float)
    mean, scale = x[:, :512].mean(1, keepdims=True), x[:, :512].std(1, keepdims=True)
    scale = np.where(scale < 1e-8, 1., scale)
    z = (x-mean)/scale
    with np.load(cache/f'{split}_fourier_83a0a46efb39.npz') as f:
        periods = f['periods'].copy()
    return raw, x, mean, scale, z, periods


def chronos(root, out):
    os.environ['HF_HOME'] = str(root/'.hf-cache')
    from huggingface_hub import model_info
    from chronos import BaseChronosPipeline
    provenance = out/'external_model_revision.json'
    if provenance.exists():
        info = json.loads(provenance.read_text())
    else:
        info = dict(model=MODEL, revision=model_info(MODEL).sha,
                    context=512, horizon=96, quantiles=Q.tolist(), native_horizon=64,
                    extension='Chronos pipeline autoregressive extension beyond native horizon',
                    overlap='Potential pretraining-source overlap; external reference only')
        provenance.write_text(json.dumps(info, indent=2)+'\n')
    pipe = None
    for split in ['test_id', 'test_ood']:
        path = out/f'external_quantiles_{split}.npz'
        if path.exists():
            with np.load(path) as f:
                if str(f['revision']) == info['revision'] and np.array_equal(f['levels'], Q):
                    continue
        _, x, _, _, _, _ = raw_split(root, split)
        if pipe is None:
            pipe = BaseChronosPipeline.from_pretrained(MODEL, revision=info['revision'],
                    device_map='cuda', torch_dtype=torch.float32)
        quantiles = []
        for start in range(0, len(x), 64):
            with torch.inference_mode():
                q, _ = pipe.predict_quantiles(inputs=torch.tensor(x[start:start+64, :512], dtype=torch.float32),
                        prediction_length=96, quantile_levels=Q.tolist())
            quantiles.append(q.float().cpu().numpy())
            print('chronos', split, min(start+64, len(x)), '/', len(x), flush=True)
        values = np.concatenate(quantiles)
        if values.shape != (len(x), 96, len(Q)) or not np.isfinite(values).all():
            raise ValueError('Unexpected/nonfinite external quantiles')
        atomic_npz(path, quantiles=values, levels=Q, revision=np.array(info['revision']))
    (out/'external_chronos_complete.json').write_text(json.dumps(dict(passed=True, **info),indent=2)+'\n')


def snaive_quantiles(z, periods):
    context, horizon = z[:, :512].clip(-20, 20), 96
    mu = np.empty((len(z), horizon)); sd = np.empty_like(mu)
    h = np.arange(1, horizon+1)
    for i in range(len(z)):
        period = periods[i].max()
        if period >= 2:
            k = int(round(period))
            if k >= 512:
                raise ValueError('Seasonal-naive period exceeds context')
            residual = context[i, k:]-context[i, :-k]
            mu[i] = context[i, 512-k+np.arange(horizon)%k]
            sd[i] = residual.std()*np.sqrt(np.ceil(h/k))
        else:
            mu[i] = context[i, -1]
            sd[i] = np.diff(context[i]).std()*np.sqrt(h)
    return mu[..., None]+np.maximum(sd, 1e-6)[..., None]*norm.ppf(Q)


def score(root, out):
    sys.path.insert(0, str(root/'code/fm_mini'))
    import train
    frames = []
    fragments = out/'quantile_fragments'; fragments.mkdir(exist_ok=True)
    families = {'PD':'srv10_Dmc1', 'Transformer':'srv13_N6mc1',
                'Transformer_period_features':'srv13_N6s',
                'Transformer_d128_KS1to1':'srv15k_N6s_d128_ks100'}
    for split in ['test_id', 'test_ood']:
        raw, x, mean, scale, z, periods = raw_split(root, split)
        common = dict(split=split, dataset=raw['dataset'].astype(str), series_id=raw['series_id'].astype(str),
                      window=np.arange(len(x)))
        reference = snaive_quantiles(z, periods)
        sn = qcrps(z[:, 512:], reference)
        with np.load(out/f'external_quantiles_{split}.npz') as f:
            values = (f['quantiles']-mean[..., None])/scale[..., None]
        def rows(label, seed, quantiles):
            return pd.DataFrame(dict(**common, model=label, seed=seed,
                qcrps9=qcrps(z[:, 512:], quantiles), snaive_qcrps9=sn,
                coverage80=((z[:, 512:]>=quantiles[..., 0])&(z[:, 512:]<=quantiles[..., -1])).mean(1)))
        frames.extend([rows('Chronos_Bolt_small', 0, values), rows('seasonal_naive', 0, reference)])
        for label, family in families.items():
            runs = sorted((root/'notes/analysis/fm_mini').glob(family+'_s[0-9]'))
            if not runs:
                raise ValueError('Missing frozen family '+family)
            for run in runs:
                cfg = json.loads((run/'provenance.json').read_text())['config']
                path = fragments/f'{run.name}_{split}.csv'; marker = path.with_suffix('.json')
                fingerprint = dict(best_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(), draws=128,
                                   quantiles=Q.tolist(), protocol='quantile_linear_samples_v1')
                if path.exists() and marker.exists() and json.loads(marker.read_text()) == fingerprint:
                    frame = pd.read_csv(path, dtype={'series_id':str})
                else:
                    print('matched score', run.name, split, flush=True)
                    train.SIZE.clear(); train.SIZE.update(cfg.get('size') or {})
                    model = train.make_model(cfg['arm']).to('cuda')
                    model.load_state_dict(torch.load(run/'best.pt', map_location='cuda', weights_only=False)['model'])
                    loaded = train.load_split(root/cfg['cache'], split, cfg['arm'], cfg['clip'])
                    if not np.array_equal(loaded['x'], raw['x']):
                        raise ValueError('Model and external test windows are not identical')
                    _, outputs = train.runner.evaluate(model, loaded, 'cuda', 128)
                    indices = np.concatenate([o['indices'] for o in outputs])
                    if not np.array_equal(indices, np.arange(len(x))):
                        raise ValueError('Unexpected prediction order')
                    samples = np.concatenate([o['samples'] for o in outputs], axis=1)
                    quantiles = np.moveaxis(np.quantile(samples, Q, axis=0), 0, -1)
                    quantiles = (quantiles-mean[..., None])/scale[..., None]
                    frame = rows(label, cfg['seed'], quantiles)
                    temporary = path.with_suffix('.csv.tmp'); frame.to_csv(temporary, index=False); temporary.replace(path)
                    marker.write_text(json.dumps(fingerprint,indent=2)+'\n')
                    del model, outputs, samples, quantiles
                    gc.collect(); torch.cuda.empty_cache()
                frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out/'matched_quantile_windows.csv', index=False)
    by_source = all_rows.groupby(['split','model','seed','dataset'])[['qcrps9','snaive_qcrps9','coverage80']].mean()
    by_source['ratio'] = by_source.qcrps9/by_source.snaive_qcrps9
    by_source.to_csv(out/'matched_quantile_by_source.csv')
    result = []; rng = np.random.default_rng(20261024)
    for (split, label), group in all_rows.groupby(['split','model']):
        seeds = sorted(group.seed.unique()); datasets = sorted(group.dataset.unique())
        blocks = []
        for ds in datasets:
            parts = [group[(group.dataset==ds)&(group.seed==s)].sort_values('window') for s in seeds]
            if any(p.window.tolist()!=parts[0].window.tolist() for p in parts):
                raise ValueError('Unpaired windows')
            values = np.stack([p[['qcrps9','snaive_qcrps9']].to_numpy() for p in parts])
            ids, inverse = np.unique(parts[0].series_id, return_inverse=True)
            blocks.append((values, inverse, len(ids)))
        seed_skills = by_source.loc[(split,label),'ratio'].groupby('seed').apply(lambda x:np.exp(np.log(x).mean()))
        boot=[]
        for _ in range(2000):
            sidx=rng.integers(0,len(seeds),len(seeds)); ratios=[]
            for values, inverse, n in blocks:
                multiplicity=np.bincount(rng.integers(0,n,n),minlength=n)[inverse]
                v=(values[sidx]*multiplicity[None,:,None]).sum((0,1))
                ratios.append(v[0]/v[1])
            boot.append(np.exp(np.log(ratios).mean()))
        low, high=np.quantile(boot,[.025,.975])
        pooled=float(np.exp(np.mean([np.log(v[...,0].sum()/v[...,1].sum()) for v,_,_ in blocks])))
        result.append(dict(split=split, model=label, seeds=len(seeds), skill_mean=seed_skills.mean(),skill_pooled=pooled,
            skill_sd=seed_skills.std(ddof=1) if len(seeds)>1 else np.nan, ci_low=low, ci_high=high,
            coverage80=by_source.loc[(split,label),'coverage80'].mean()))
    table=pd.DataFrame(result); table.to_csv(out/'matched_quantile_results.csv',index=False)
    lines=['# External reference with a matched nine-quantile score','',
        'All models use the same frozen windows (512 context, 96 horizon), context scaling, seasonal-naive reference, and qCRPS9.',
        'qCRPS9 is twice the mean pinball loss at quantiles 0.1 through 0.9. It is an approximation, not the MC128 CRPS in the main results.',
        'Own-model quantiles are estimated from the existing 128 predictive draws; Chronos uses its predicted quantiles.',
        'Chronos-Bolt-small extends its native 64 horizon autoregressively to 96. No tuning is performed.',
        'Potential pretraining-source overlap prevents interpreting Chronos as a clean zero-shot comparison.',
        'Pooled skill averages errors over seeds within a source before taking the geometric source ratio; its bootstrap uses the same estimator. Mean per-seed skill and SD are also retained in CSV.',
        '95% intervals jointly resample training seeds and whole series within each fixed source; they do not quantify uncertainty across new sources.', '',
        '| Split | Model | Seeds | Pooled skill | 95% interval | Coverage 80% |','|---|---|---:|---:|---|---:|']
    for row in result:
        lines.append(f"| {row['split']} | {row['model']} | {row['seeds']} | {row['skill_pooled']:.4f} | [{row['ci_low']:.4f}; {row['ci_high']:.4f}] | {row['coverage80']:.3f} |")
    (out/'EXTERNAL_REFERENCE.md').write_text('\n'.join(lines)+'\n')
    (out/'external_qcrps_complete.json').write_text(json.dumps(dict(passed=True,draws=128,quantiles=Q.tolist(),
                                                                 models=list(families),bootstrap=2000),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['chronos','score'],required=True);a=p.parse_args()
    root=a.root.resolve()
    if root.stat().st_uid!=os.getuid():
        raise SystemExit('Owned root required')
    torch.set_num_threads(1)
    out=root/'final_results';out.mkdir(exist_ok=True)
    (chronos if a.mode=='chronos' else score)(root,out)


if __name__=='__main__':
    main()
