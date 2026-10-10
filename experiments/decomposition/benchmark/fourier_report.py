"""Experiment 1 report: C' (Cf) and D' (Df) against the synthetic_v2 A/B/C/D checkpoints.

Paired differences resample independent series within a block/case and seed (2,000 replicates);
training seeds stay separate. Criteria K1-K3 were fixed before the runs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import T
from runner import OUT,save_json
from fourier_arm import cache_tag

BASE=DIAG=COMP=COMP_FOURIER=None  # set from the command line
PAIRS=[('Cf','A'),('Cf','C'),('Df','B'),('Df','D'),('Cf0','A'),('Cf','Df')]
KEY_CASES=['period_37_1','dual_17_3_96_ratio_1','dual_17_3_96_ratio_3']


def ci(values,rng,repeats=2000):
    v=np.asarray(values,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return np.nan,np.nan
    draws=rng.choice(v,(repeats,len(v)),replace=True).mean(1)
    return tuple(float(q) for q in np.quantile(draws,[.025,.975]))


def paired(df,group,rng):
    rows=[]
    for (seed,g),s in df.groupby(['seed',group]):
        pivot=s.pivot_table(index='series_id',columns='arm',values='crps')
        for left,right in PAIRS:
            if left in pivot and right in pivot:
                d=(pivot[left]-pivot[right]).dropna(); lo,hi=ci(d,rng)
                rows.append({'seed':int(seed),group:g,'pair':f'{left}-{right}','mean':float(d.mean()),'ci_low':lo,'ci_high':hi,'n':len(d)})
    return pd.DataFrame(rows)


def load_runs(queue):
    runs={}
    for s in range(3):
        for arm in 'ABCD':
            if (OUT/f'{BASE}_{arm}_s{s}'/'fm_rows.csv').exists(): runs[(arm,s)]=f'{BASE}_{arm}_s{s}'
        for arm in ('Cf','Df'):
            name=f'{queue}_{arm}_s{s}'
            if (OUT/name/'status.json').exists() and json.loads((OUT/name/'status.json').read_text())['status']=='complete': runs[(arm,s)]=name
    return runs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--queue',required=True,help='prefix of the Cf/Df training runs')
    p.add_argument('--base',required=True,help='prefix of the A/B/C/D training runs')
    p.add_argument('--diag-prefix',required=True,help='prefix of the A/B/C/D diagnostic-suite evaluations')
    p.add_argument('--components',required=True,help='component diagnostics of A/B/C/D')
    p.add_argument('--fourier-components',required=True,help='component diagnostics of Cf/Df (+ _nocorr)')
    p.add_argument('--out',default='fourier_report'); a=p.parse_args()
    global BASE,DIAG,COMP,COMP_FOURIER; BASE,DIAG,COMP,COMP_FOURIER=a.base,a.diag_prefix,a.components,a.fourier_components
    out=OUT/a.out; out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(20261010); runs=load_runs(a.queue); lines=['# Experiment 1: continuous-frequency Fourier seasonal decoder','']
    lines+=['Runs: '+', '.join(f'{k[0]}_s{k[1]}' for k in sorted(runs)),'']
    # v2 test split
    frames=[]
    for (arm,s),run in runs.items():
        df=pd.read_csv(OUT/run/'fm_rows.csv'); df['arm']=arm; df['seed']=s; frames.append(df)
    test=pd.concat(frames); test.to_csv(out/'v2_test_rows.csv',index=False)
    by_block=test.groupby(['block','arm','seed'])[['crps','rmse','coverage90','width90','mean_bias_mse']].mean().reset_index(); by_block.to_csv(out/'v2_test_by_block.csv',index=False)
    eff_block=paired(test,'block',rng); eff_block.to_csv(out/'v2_test_paired.csv',index=False)
    lines+=['## v2 test: CRPS by block (mean over available seeds)','',test.groupby(['block','arm']).crps.mean().unstack().round(4).to_string(),'']
    # diagnostic suite
    frames=[]
    for (arm,s),run in runs.items():
        sources=[(arm,OUT/f'{DIAG}_{arm}_s{s}')] if arm in 'ABCD' else [(arm,OUT/f'diagnostic_eval_fourier_{run}'),(arm+'0',OUT/f'diagnostic_eval_fourier_{run}_nocorr')]
        for label,path in sources:
            if (path/'diagnostic_fm_rows.csv').exists():
                df=pd.read_csv(path/'diagnostic_fm_rows.csv'); df['arm']=label; df['seed']=s; frames.append(df)
    diag=pd.concat(frames); diag.to_csv(out/'diagnostic_rows.csv',index=False)
    eff_case=paired(diag,'case',rng); eff_case.to_csv(out/'diagnostic_paired.csv',index=False)
    case_table=diag.groupby(['case','arm']).crps.mean().unstack().round(4)
    # deterministic reference: last trend value + continuous-frequency seasonal extrapolation
    suite=Path(json.loads((OUT/f'{DIAG}_A_s0'/'provenance.json').read_text())['suite'])
    if not suite.is_absolute(): suite=ROOT/suite
    raw=dict(np.load(suite/'test.npz')); cache=np.load(suite/f'suite_fourier_{cache_tag()}.npz')
    m=raw['x'][:,:T].mean(1,keepdims=True); s=raw['x'][:,:T].std(1,keepdims=True)
    det=(cache['trend'][:,-1:]+cache['seasonal_full'][:,T:])*s+m
    ref=pd.DataFrame(dict(case=raw['case'],det_mse=((det-raw['forecast_mean'])**2).mean(1))).groupby('case').det_mse.mean()
    mse=diag.groupby(['case','arm']).mean_bias_mse.mean().unstack(); mse['period_det']=ref; mse.round(4).to_csv(out/'diagnostic_mean_mse.csv')
    lines+=['## Diagnostic suite: CRPS by case (mean over seeds; Cf0 = Cf with the learned correction zeroed at evaluation)','',case_table.to_string(),'',
            '## Mean error against the true conditional mean (MSE); period_det = deterministic trend-last + Fourier extrapolation','',mse.round(4).to_string(),'']
    # components
    comp=[]
    if (OUT/COMP/'component_rows.csv').exists(): comp.append(pd.read_csv(OUT/COMP/'component_rows.csv'))
    for name in (COMP_FOURIER,COMP_FOURIER+'_nocorr'):
        if (OUT/name/'component_rows.csv').exists(): comp.append(pd.read_csv(OUT/name/'component_rows.csv'))
    if comp:
        comp=pd.concat(comp); ct=comp.groupby(['case','arm'])[['input_seasonal_nrmse','decoded_seasonal_nrmse','decoded_trend_nrmse']].mean().reset_index(); ct.to_csv(out/'component_by_case.csv',index=False)
        lines+=['## Seasonal component NRMSE, input vs decoded (context, posterior means)','',ct[ct.case.isin(KEY_CASES+['negative_season_noise','period_19_7','period_144','dual_24_168_ratio_1'])].round(3).to_string(index=False),'']
    # criteria
    checks=[]
    if comp is not None and len(comp):
        for case in KEY_CASES:
            r=ct[(ct.case==case)&(ct.arm=='Cf')]
            if len(r): checks.append(dict(criterion='K1',case=case,value=float(r.decoded_seasonal_nrmse.iloc[0]/r.input_seasonal_nrmse.iloc[0]),passed=bool(r.decoded_seasonal_nrmse.iloc[0]<=2*r.input_seasonal_nrmse.iloc[0])))
    k2=eff_case[(eff_case.pair=='Cf-A')]
    for case in KEY_CASES:
        r=k2[k2.case==case]; checks.append(dict(criterion='K2',case=case,value=int((r.ci_high<0).sum()),passed=bool((r.ci_high<0).sum()>=2)))
    r=eff_block[(eff_block.pair=='Cf-A')&(eff_block.block=='seasonal')]; checks.append(dict(criterion='K2',case='v2 seasonal block',value=int((r.ci_high<0).sum()),passed=bool((r.ci_high<0).sum()>=2)))
    worse=k2.groupby('case').apply(lambda g:bool(len(g)==3 and (g.ci_low>0).all())); checks.append(dict(criterion='K2 no-harm',case=';'.join(worse[worse].index) or 'none',value=int(worse.sum()),passed=bool(worse.sum()==0)))
    seasonal=test[(test.block=='seasonal')&(test.arm=='Cf')].crps.mean(); checks.append(dict(criterion='K3',case='v2 seasonal block CRPS',value=float(seasonal),passed=bool(seasonal<=.21)))
    checks=pd.DataFrame(checks); checks.to_csv(out/'criteria.csv',index=False)
    lines+=['## Pre-registered criteria','',checks.to_string(index=False),'',
            'K1: decoded seasonal NRMSE <= 2x input. K2: Cf-A paired CRPS interval below 0 in >= 2 of 3 seeds; no case with Cf worse than A in all 3 seeds. K3: v2 seasonal-block CRPS <= 0.21.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'provenance.json',dict(runs={f'{k[0]}_s{k[1]}':v for k,v in runs.items()},suite=str(suite),bootstrap=2000,pairs=PAIRS))
    print('\n'.join(lines))

if __name__=='__main__': main()
