"""Export the completed three-seed reference and supplementary report tables.

Reads an already downloaded snapshot; does not import model code or use a GPU.
Five-seed H1 is handled separately by summarize_h1.py after training completes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_h1 import paired_interval


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=a.root/'notes/analysis/fm_mini';out=a.out
    out.mkdir(parents=True,exist_ok=True)
    families={'PD_NLL':'srv5_Dcomb','PD_fresh':'srv14r_Dmc1_w0.2',
              'Zn':'srv13_Dmc1Zn','B4':'srv13_Dmc1B4'}
    sn=pd.read_csv(r/'baseline_rows.csv').query("method=='seasonal_naive'").groupby(['split','dataset']).ncrps.mean()
    extra=[]
    for label,fam in families.items():
        for split in ['test_ood','test_id']:
            vals=[]
            for run in sorted(r.glob(fam+'_s[0-9]')):
                d=pd.read_csv(run/f'space_eval_{split}.csv')
                vals.append(float(np.exp(np.log(d.groupby('dataset').ncrps.mean()/sn.loc[split]).mean())))
            if not vals:raise ValueError('Missing family '+fam)
            extra.append(dict(model=label,split=split,seeds=len(vals),skill=np.mean(vals),sd=np.std(vals,ddof=1)))
    pd.DataFrame(extra).to_csv(out/'additional_models.csv',index=False)
    comparisons=[]
    for profile in ['real','strong','base']:
        frames={}
        for arm in ['Dmc1','Dmc1Zn','Dmc1B4']:
            parts=[]
            for seed in range(3):
                if profile=='real':
                    family='srv14r_Dmc1_w0.2' if arm=='Dmc1' else 'srv13_'+arm
                    path=r/f'{family}_s{seed}/space_eval_test_ood.csv'
                else:
                    split='test_id' if profile=='strong' else 'test_ood'
                    path=r/f'h1/srv13_h1_{arm}_s{seed}_{split}.csv'
                parts.append(pd.read_csv(path,dtype={'series_id':str}).rename(columns={'block':'dataset'}))
            frames[arm]=parts
        reference=frames['Dmc1'][0]
        keys=list(zip(reference.dataset,reference.series_id))
        for parts in frames.values():
            if any(list(zip(d.dataset,d.series_id))!=keys for d in parts):
                raise ValueError('Unpaired H1 windows')
        for arm in ['Dmc1Zn','Dmc1B4']:
            for metric in (['ncrps'] if profile=='real' else ['crps','crps_h1','lv_err']):
                x=np.stack([d[metric] for d in frames[arm]])
                y=np.stack([d[metric] for d in frames['Dmc1']])
                effect,lo,hi=paired_interval(x,y,reference.dataset.to_numpy(),reference.series_id.to_numpy())
                comparisons.append(dict(profile=profile,arm=arm,metric=metric,ratio=effect,ci_low=lo,ci_high=hi,seeds=3))
    pd.DataFrame(comparisons).to_csv(out/'h1_three_seed_reference.csv',index=False)
    loso=pd.read_csv(out/'loso_sources.csv')
    grouped=loso.groupby(['comparison','seasonal_report_group']).ratio.apply(lambda x:np.exp(np.log(x).mean())).reset_index()
    grouped.to_csv(out/'loso_seasonality_groups.csv',index=False)
    source=pd.read_csv(out/'main_source_ratios.csv');logs=np.log(source.Dmc1_over_N6mc1.to_numpy())
    rng=np.random.default_rng(20261025)
    lo,hi=np.quantile(np.exp(rng.choice(logs,size=(10000,len(logs))).mean(1)),[.025,.975])
    uncertainty=dict(ratio=float(np.exp(logs.mean())),ci_low=float(lo),ci_high=float(hi),sources=len(logs),
                     resamples=10000,kind='source bootstrap after seed averaging; exploratory six-source set')
    (out/'main_source_uncertainty.json').write_text(json.dumps(uncertainty,indent=2)+'\n')
    lab=[]
    for label,fam in {'A':'lab_A','PD':'lab_Dmc1','TF':'srv13_lab_N6mc1','TF_hint':'srv13_lab_N6s',
                      'TF_phase':'srv13_lab_N6p','TF_add_output':'srv13_lab_N6o'}.items():
        for split in ['test_id','test_ood']:
            scores=[pd.read_csv(path)[['crps_ratio','c80','w80_ratio','mse_mean']].mean()
                    for path in sorted((r/'lab').glob(fam+'_s[012]_'+split+'.csv'))]
            if len(scores)!=3:raise ValueError('Incomplete laboratory reference '+fam)
            lab.append(dict(model=label,split=split,seeds=3,**pd.DataFrame(scores).mean().to_dict()))
    pd.DataFrame(lab).to_csv(out/'lab_reference_summary.csv',index=False)
    print('Supplementary report tables exported:',out)


if __name__=='__main__':main()
