"""Combine the 12 held-out stress evaluations without pooling scenarios."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from runner import OUT

METRICS = ('crps','rmse','coverage90','width90','variance_mae','mean_bias_mse')

def ci(values, rng, repeats=2000):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if not len(values): return (float('nan'),float('nan'))
    draws=rng.choice(values,(repeats,len(values)),replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws,[.025,.975]))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-prefix',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    paths=sorted(path for path in OUT.glob(a.run_prefix+'*') if (path/'stress_fm_rows.csv').exists())
    frames=[]; interventions=[]
    for path in paths:
        row=pd.read_csv(path/'stress_fm_rows.csv')
        row['model_run']=path.name.removeprefix(a.run_prefix); row['arm']=row.model_run.str[0]; row['seed']=row.model_run.str.split('_s').str[-1].astype(int)
        frames.append(row)
        iv=pd.read_csv(path/'fm_interventions.csv'); iv['model_run']=path.name.removeprefix(a.run_prefix); iv['arm']=iv.model_run.str[0]; iv['seed']=iv.model_run.str.split('_s').str[-1].astype(int); interventions.append(iv)
    if len(frames)!=12: raise SystemExit(f'Expected 12 complete stress outputs; found {len(frames)}')
    data=pd.concat(frames,ignore_index=True); iv=pd.concat(interventions,ignore_index=True)
    out=OUT/a.out; out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'stress_rows_all.csv',index=False); iv.to_csv(out/'intervention_rows_all.csv',index=False)
    rows=[]; rng=np.random.default_rng(20261007)
    for (model,scenario),g in data.groupby(['model_run','scenario'],sort=True):
        rec=dict(model_run=model,arm=g.arm.iloc[0],seed=int(g.seed.iloc[0]),scenario=scenario,n=len(g))
        for metric in METRICS:
            vals=g[metric].to_numpy(dtype=float); rec[metric+'_mean']=float(np.nanmean(vals))
            rec[metric+'_ci_low'],rec[metric+'_ci_high']=ci(vals,rng)
        rows.append(rec)
    summary=pd.DataFrame(rows); summary.to_csv(out/'by_model_scenario.csv',index=False)
    effects=[]
    for (seed,scenario),g in data.groupby(['seed','scenario'],sort=True):
        pivot=g.pivot(index='series_id',columns='arm',values='crps')
        for label,off,on in [('G_original','B','A'),('G_period','D','C')]:
            if off in pivot and on in pivot:
                d=(pivot[off]-pivot[on]).dropna().to_numpy(); lo,hi=ci(d,rng)
                effects.append(dict(seed=int(seed),scenario=scenario,effect=label,mean=float(d.mean()),ci_low=lo,ci_high=hi,n=len(d)))
        if all(arm in pivot for arm in 'ABCD'):
            d=((pivot.D-pivot.C)-(pivot.B-pivot.A)).dropna().to_numpy(); lo,hi=ci(d,rng)
            effects.append(dict(seed=int(seed),scenario=scenario,effect='interaction',mean=float(d.mean()),ci_low=lo,ci_high=hi,n=len(d)))
    effect_frame=pd.DataFrame(effects); effect_frame.to_csv(out/'paired_branch_effects.csv',index=False)
    effect_summary=effect_frame.groupby(['scenario','effect'],sort=True).agg(mean=('mean','mean'),positive_seeds=('mean',lambda v:int((v>0).sum()))).reset_index()
    intervention_summary=iv.groupby(['arm','seed','changed'],sort=True)[['observation_variance_ratio','forecast_mean_response','decoded_trend_response_error','decoded_seasonal_response_error']].mean()
    intervention_summary.to_csv(out/'intervention_summary.csv')
    lines=['# Synthetic stress evaluation','',f'{len(paths)} checkpoints; eight held-out stress cases. Results remain split by model, seed, and scenario. Confidence intervals resample rows within each scenario (2,000 replicates).','','## Stress metrics by checkpoint and case','','See `by_model_scenario.csv` for CRPS, RMSE, coverage, interval width, variance error, and mean-bias error with row-bootstrap intervals.','','## Paired noise-branch effects','','Positive G means lower CRPS with the noise branch. Interactions are G_period minus G_original. Seed-level estimates remain separate. Mean effect and number of positive seeds are shown per case; no cross-case aggregate is reported.','',effect_summary.to_string(index=False),'','## Paired interventions','','Responses remain split by arm, seed, and intervention in `intervention_summary.csv`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {len(summary)} model-scenario rows and {len(effects)} paired effects to {out}')

if __name__=='__main__': main()
