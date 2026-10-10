"""Scenario-wise summary for the extended 29-case checkpoint evaluation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from runner import OUT

METRICS=('crps','rmse','coverage90','width90','variance_mae','mean_bias_mse')

def interval(v,rng,repeats=2000):
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return np.nan,np.nan
    return tuple(float(x) for x in np.quantile(rng.choice(v,(repeats,len(v)),replace=True).mean(axis=1),[.025,.975]))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prefix',required=True); p.add_argument('--out',required=True); p.add_argument('--allow-partial',action='store_true'); a=p.parse_args()
    folders=sorted(x for x in OUT.glob(a.prefix+'*') if (x/'diagnostic_fm_rows.csv').exists())
    if len(folders)!=12 and not (a.allow_partial and 1<=len(folders)<12): raise SystemExit(f'Expected 12 complete outputs, found {len(folders)}; use --allow-partial to report a clearly incomplete set')
    data=pd.concat([pd.read_csv(x/'diagnostic_fm_rows.csv') for x in folders],ignore_index=True)
    expected_runs={f'{arm}_s{seed}' for seed in range(3) for arm in 'ABCD'}
    found_runs={f"{row.arm.iloc[0]}_s{int(row.seed.iloc[0])}" for _,row in data.groupby('model_run')}
    missing_runs=sorted(expected_runs-found_runs)
    out=OUT/a.out; out.mkdir(parents=True,exist_ok=True); data.to_csv(out/'all_rows.csv',index=False)
    rng=np.random.default_rng(20261007); rows=[]
    for (model,case),g in data.groupby(['model_run','case'],sort=True):
        r=dict(model_run=model,arm=g.arm.iloc[0],seed=int(g.seed.iloc[0]),case=case,predictable=bool(g.predictable.iloc[0]),n=len(g))
        for metric in METRICS:
            v=g[metric].to_numpy(dtype=float); r[metric+'_mean']=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            r[metric+'_ci_low'],r[metric+'_ci_high']=interval(v,rng)
        rows.append(r)
    by_model=pd.DataFrame(rows); by_model.to_csv(out/'by_model_case.csv',index=False)
    effects=[]
    for (seed,case),g in data.groupby(['seed','case'],sort=True):
        pivot=g.pivot(index='series_id',columns='arm',values='crps')
        for label,off,on in [('G_original','B','A'),('G_period','D','C')]:
            if off not in pivot or on not in pivot: continue
            v=(pivot[off]-pivot[on]).dropna().to_numpy(); lo,hi=interval(v,rng)
            effects.append(dict(seed=int(seed),case=case,effect=label,mean=float(v.mean()),ci_low=lo,ci_high=hi,n=len(v)))
        if set('ABCD').issubset(pivot.columns):
            v=((pivot.D-pivot.C)-(pivot.B-pivot.A)).dropna().to_numpy(); lo,hi=interval(v,rng)
            effects.append(dict(seed=int(seed),case=case,effect='interaction',mean=float(v.mean()),ci_low=lo,ci_high=hi,n=len(v)))
    effect_df=pd.DataFrame(effects); effect_df.to_csv(out/'paired_branch_effects.csv',index=False)
    effect_summary=effect_df.groupby(['case','effect']).agg(mean=('mean','mean'),positive_seeds=('mean',lambda v:int((v>0).sum()))).reset_index()
    decomposition=[]
    for (seed,case),g in data.groupby(['seed','case'],sort=True):
        pivot=g.pivot(index='series_id',columns='arm',values='crps')
        for label,period,original in [('period_with_noise_minus_original_with_noise','C','A'),('period_without_noise_minus_original_without_noise','D','B')]:
            if period not in pivot or original not in pivot: continue
            values=(pivot[period]-pivot[original]).dropna().to_numpy(); lo,hi=interval(values,rng)
            decomposition.append(dict(seed=int(seed),case=case,comparison=label,mean=float(values.mean()),ci_low=lo,ci_high=hi,n=len(values)))
    decomp_df=pd.DataFrame(decomposition); decomp_df.to_csv(out/'decomposition_effects.csv',index=False)
    case_summary=by_model.groupby(['arm','case','predictable'],sort=True)[[x+'_mean' for x in METRICS]].mean().reset_index()
    case_summary.to_csv(out/'case_summary_by_arm.csv',index=False)
    missing_text=', '.join(missing_runs) if missing_runs else 'none'
    lines=[
        '# Extended synthetic diagnostics','',
        f'{len(folders)} evaluated checkpoints; missing: {missing_text}. Results are split by case, model, and seed. Row-bootstrap confidence intervals use 2,000 resamples within each case.','',
        'The `sigma_unseen_future_jump` case is an unpredictable future stress. Its variance error is descriptive and is not a forecastability criterion.','',
        '## Per-checkpoint results','',
        'See `by_model_case.csv` for each checkpoint, case, confidence interval, calibration, and mean error. `case_summary_by_arm.csv` averages available trained seeds within each arm while keeping cases separate. `data_audit.json` records generated-suite counts and truth checks.','',
        '## Decomposition-method comparisons','',
        '`decomposition_effects.csv` reports paired CRPS differences for period-aware versus original input decomposition, with noise-branch status held fixed. Positive values mean the period-aware model has higher CRPS. Seed-level rows are kept separate.','',
        '## Noise-branch effects','',
        'Positive G means lower CRPS with the noise branch. Interactions are G_period minus G_original and appear only when all four arms are available for a seed. Means and the number of positive training seeds are shown per case; no pooled case score is used.','',
        effect_summary.to_string(index=False)
    ]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {len(by_model)} checkpoint-case rows and {len(effect_df)} paired effects to {out}')

if __name__=='__main__': main()
