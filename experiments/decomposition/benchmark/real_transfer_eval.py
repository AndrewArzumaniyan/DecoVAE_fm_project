"""Zero-shot evaluation of completed synthetic checkpoints on fixed real windows."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fourier_arm import FOURIER_ARMS, make_model, attach
from data import L, T, normalize
from runner import OUT, cache_components, evaluate


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def training_checkpoints(expected,allow_partial=False):
    incomplete=[]; available=[]
    for run_id in expected:
        run=OUT/run_id
        status_path=run/'status.json'
        complete=(status_path.exists() and json.loads(status_path.read_text()).get('status')=='complete'
                  and (run/'best.pt').exists())
        if complete:
            available.append(run_id)
        else:
            incomplete.append(run_id)
            if allow_partial and (run/'best.pt').exists(): available.append(run_id)
    if incomplete and not allow_partial:
        raise RuntimeError('Real transfer starts only after every synthetic checkpoint is complete: '+', '.join(incomplete))
    if not available:
        raise RuntimeError('No synthetic checkpoints are available for real transfer')
    return expected,available,incomplete


def main():
    p=argparse.ArgumentParser()
    p.add_argument('runs',nargs='+',help='synthetic training runs to evaluate')
    p.add_argument('--out',required=True)
    p.add_argument('--windows-run',required=True,help='training run made with --real; it stores the fixed real windows')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--allow-partial',action='store_true',help='Evaluate available checkpoints and clearly report missing/incomplete training runs')
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run this evaluation in the GPU-enabled execution context')
    torch.set_num_threads(1)
    expected_run_ids,run_ids,incomplete_runs=training_checkpoints(a.runs,a.allow_partial)
    partial_training_runs=[]
    for run_id in run_ids:
        status_path=OUT/run_id/'status.json'
        if not status_path.exists() or json.loads(status_path.read_text()).get('status')!='complete':
            partial_training_runs.append(run_id)

    windows_dir=OUT/a.windows_run
    windows_path=windows_dir/'real_forecast_outputs.npz'
    index_path=windows_dir/'real_windows.json'
    if not windows_path.exists() or not index_path.exists():
        raise FileNotFoundError(f'Fixed real windows not found under {windows_dir}')
    index=json.loads(index_path.read_text())
    counts=Counter(row['dataset'] for row in index)
    required={'Electricity','London_Smart_Meters','Q-TRAFFIC','Solar','M4_Daily'}
    if set(counts)!=required or any(row.get('split')!='test' for row in index):
        raise ValueError(f'Unexpected real-window index: counts={dict(counts)}')
    with np.load(windows_path,allow_pickle=True) as saved:
        x=np.asarray(saved['data'],dtype=np.float32)
    if x.shape!=(len(index),T+L):
        raise ValueError(f'Window shape mismatch: {x.shape}, index rows={len(index)}, expected length={T+L}')
    z,mean,scale=normalize(x)
    raw=dict(x=x,z=z.astype('float32'),mean=mean,scale=scale)

    output=OUT/(a.out+('_partial' if a.allow_partial else ''))
    output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    window_hash=digest(windows_path); index_hash=digest(index_path)
    component_cache=output/'real_period_components.npz'
    raw['components']=cache_components(raw['z'],component_cache)
    fourier=attach(dict(z=raw['z']),output,'real')  # periods/coefs for Cf/Df; C/D components stay as above
    raw.update(periods=fourier['periods'],coefs=fourier['coefs'])

    rows=[]
    for position,run_id in enumerate(run_ids,1):
        model_dir=OUT/run_id
        checkpoint_path=model_dir/'best.pt'; checkpoint_hash=digest(checkpoint_path)
        model_out=output/run_id
        model_status=model_out/'status.json'
        if model_status.exists():
            prior=json.loads(model_status.read_text())
            rows_path=model_out/'real_fm_rows.csv'
            outputs_path=model_out/'real_forecast_outputs.npz'
            if (prior.get('status')=='complete' and prior.get('checkpoint_sha256')==checkpoint_hash
                    and prior.get('windows_sha256')==window_hash and rows_path.exists() and outputs_path.exists()):
                rows.append(pd.read_csv(rows_path))
                print(f'real transfer {position}/{len(run_ids)}: {run_id} already complete; reusing outputs',flush=True)
                continue
        checkpoint=torch.load(checkpoint_path,weights_only=False,map_location=a.device)
        arm=checkpoint.get('config',{}).get('arm')
        seed=int(checkpoint.get('config',{}).get('seed',-1))
        if arm not in {'A','B','C','D',*FOURIER_ARMS} or not run_id.endswith(f'_{arm}_s{seed}'):
            raise ValueError(f'Checkpoint identity/config mismatch for {run_id}: arm={arm}, seed={checkpoint.get("config",{}).get("seed")}')
        model=make_model(arm).to(a.device)
        model.load_state_dict(checkpoint['model'])
        frame,outputs=evaluate(model,raw,a.device,a.samples)
        window_ids=frame.series_id.astype(int).to_numpy()
        frame['dataset']=[index[int(i)]['dataset'] for i in window_ids]
        frame['series_id']=[index[int(i)]['series_id'] for i in window_ids]
        frame['start']=[index[int(i)]['start'] for i in window_ids]
        frame['source']=[index[int(i)]['source'] for i in window_ids]
        frame['arm']=arm
        frame['seed']=seed
        frame['ncrps']=frame.crps.to_numpy()/scale[:,0]
        frame['nrmse']=frame.rmse.to_numpy()/scale[:,0]
        model_out.mkdir(parents=True,exist_ok=True)
        frame.to_csv(model_out/'real_fm_rows.csv',index=False)
        np.savez_compressed(model_out/'real_forecast_outputs.npz',outputs=np.array(outputs,dtype=object),data=x)
        (model_out/'real_windows.json').write_text(json.dumps(index,indent=2))
        (model_out/'status.json').write_text(json.dumps(dict(status='complete',run_id=run_id,arm=arm,seed=seed,rows=len(frame),samples=a.samples,checkpoint_sha256=checkpoint_hash,windows_sha256=window_hash),indent=2))
        rows.append(frame)
        print(f'real transfer {position}/{len(run_ids)}: {run_id} ({len(frame)} windows)',flush=True)

    result=pd.concat(rows,ignore_index=True)
    result.to_csv(output/'real_transfer_rows.csv',index=False)
    summary=result.groupby(['dataset','arm','seed'])[['ncrps','nrmse','coverage90','width90']].mean().reset_index()
    summary.to_csv(output/'real_transfer_summary.csv',index=False)
    rng=np.random.default_rng(20261009); paired=[]
    for (dataset,seed),group in result.groupby(['dataset','seed'],sort=True):
        for metric in ('ncrps','nrmse'):
            pivot=group.pivot(index=['series_id','start'],columns='arm',values=metric)
            contrasts=[('G_original','B','A'),('G_period','D','C'),
                       ('period_with_noise_minus_original_with_noise','C','A'),
                       ('period_without_noise_minus_original_without_noise','D','B')]
            for label,left,right in contrasts:
                if left not in pivot or right not in pivot: continue
                diff=(pivot[left]-pivot[right]).dropna().to_numpy(dtype=float)
                draws=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
                lo,hi=np.quantile(draws,[.025,.975])
                paired.append(dict(dataset=dataset,seed=int(seed),metric=metric,effect=label,
                                   mean=float(diff.mean()),ci_low=float(lo),ci_high=float(hi),n=len(diff)))
            if set('ABCD').issubset(pivot.columns):
                diff=((pivot.D-pivot.C)-(pivot.B-pivot.A)).dropna().to_numpy(dtype=float)
                draws=rng.choice(diff,(2000,len(diff)),replace=True).mean(axis=1)
                lo,hi=np.quantile(draws,[.025,.975])
                paired.append(dict(dataset=dataset,seed=int(seed),metric=metric,effect='interaction',
                                   mean=float(diff.mean()),ci_low=float(lo),ci_high=float(hi),n=len(diff)))
    paired_frame=pd.DataFrame(paired)
    paired_frame.to_csv(output/'real_transfer_paired_effects.csv',index=False)
    counts_json={k:int(v) for k,v in sorted(counts.items())}
    provenance=dict(status='complete' if not incomplete_runs else 'partial',runs=a.runs,
                    expected_run_ids=expected_run_ids,run_ids=run_ids,
                    incomplete_training_runs=incomplete_runs,partial_training_runs=partial_training_runs,
                    device=a.device,
                    torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
                    samples=a.samples,windows_run=a.windows_run,window_count=len(index),window_counts=counts_json,
                    windows_sha256=window_hash,window_index_sha256=index_hash,
                    component_cache_sha256=digest(component_cache) if component_cache.exists() else None,
                    checkpoint_sha256={run_id:digest(OUT/run_id/'best.pt') for run_id in run_ids},
                    started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
                    note='Zero-shot: checkpoints trained only on synthetic v2; no real fitting or tuning.')
    (output/'status.json').write_text(json.dumps(provenance,indent=2))
    lines=['# Zero-shot real transfer', '',
           f'Evaluated {len(run_ids)} of {len(expected_run_ids)} planned synthetic v2 checkpoints. All evaluated models were trained on synthetic data only; metrics remain separate by source, with no mixed score. This is an interim transfer report.' if incomplete_runs else 'All planned checkpoints were trained on synthetic v2 only. Metrics remain separate by source; no mixed score is calculated.',
           '', 'Incomplete/missing synthetic training runs: '+(', '.join(incomplete_runs) if incomplete_runs else 'none')+'.',
           'Checkpoints included despite interrupted training: '+(', '.join(partial_training_runs) if partial_training_runs else 'none')+'.',
           '', '## Window counts', '', *[f'- {name}: {count}' for name,count in sorted(counts.items())], '',
           '## Mean normalized metrics by dataset, arm, and seed', '', summary.to_string(index=False), '',
           '## Paired effects by source and seed', '',
           'Positive `G_original`/`G_period` means the noise branch improves normalized CRPS or NRMSE. Positive period-versus-original differences mean the period-aware model is worse. Intervals bootstrap independent real series within each source; Solar has only five series. Interactions are included only when all four arms are present. No values are pooled across sources.', '',
           paired_frame.to_string(index=False), '']
    (output/'report.md').write_text('\n'.join(lines))
    print(f'Wrote {len(result)} rows to {output}',flush=True)


if __name__=='__main__':
    main()
