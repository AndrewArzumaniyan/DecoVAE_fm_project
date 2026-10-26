"""Export final H1 and frozen-component tables/figures for the local report."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root,out=a.root,a.out;out.mkdir(parents=True,exist_ok=True)
    r=root/'notes/analysis/fm_mini'
    h=pd.read_csv(root/'final_results/h1_comparisons.csv')
    if json.loads((root/'final_results/h1_summary_complete.json').read_text()).get('real_control')!='matched':
        raise ValueError('Matched H1 control required')
    sn=pd.read_csv(r/'baseline_rows.csv').query("method=='seasonal_naive'").groupby(['split','dataset']).ncrps.mean()
    rows=[]
    families=[('Dmc1_matched','srv14r_Dmc1_w0.2','defense_h1_real_Dmc1'),
              ('Dmc1Zn','srv13_Dmc1Zn','defense_h1_real_Dmc1Zn'),
              ('Dmc1B4','srv13_Dmc1B4','defense_h1_real_Dmc1B4')]
    for label,old,new in families:
        for split in ['test_id','test_ood']:
            values=[]
            for seed in range(5):
                fam=old if seed<3 else new
                d=pd.read_csv(r/f'{fam}_s{seed}/space_eval_{split}.csv')
                values.append(np.exp(np.log(d.groupby('dataset').ncrps.mean()/sn.loc[split]).mean()))
            rows.append(dict(model=label,split=split,seeds=5,skill=np.mean(values),sd=np.std(values,ddof=1)))
    pd.DataFrame(rows).to_csv(out/'h1_five_seed_skills.csv',index=False)
    d=pd.read_csv(root/'final_results/component_intervals.csv')
    select=d[d.cohort.eq('simple_negative')&d.metric.isin([
        'mean_horizon_mse','seasonal_context_mse','seasonal_horizon_mse',
        'trend_context_mse','trend_horizon_mse','seasonal_in_trend_gain','coverage90','width90'])]
    select.to_csv(out/'new_diagnostics_selected_intervals.csv',index=False)
    mid=select[select.dataset.str.endswith('_sigma0.3')]
    mid.pivot_table(index=['dataset','model'],columns='metric',values='mean').to_csv(out/'new_diagnostics_sigma03.csv')
    scores=pd.read_csv(root/'final_results/component_by_seed.csv')
    scores.groupby(['cohort','split','model']).mean(numeric_only=True).to_csv(out/'component_cohort_summary.csv')
    trainings=[]
    for path in (r).glob('defense_h1_*/status.json'):
        s=json.loads(path.read_text())
        trainings.append(dict(run=path.parent.name,status=s.get('status'),ood_evaluated=s.get('ood_evaluated')))
    if len(trainings)!=12 or any(s['status']!='complete' or s['ood_evaluated'] is not True for s in trainings):
        raise ValueError('All twelve trainings must be complete')
    telemetry=[json.loads(x) for x in (root/'ops/telemetry.jsonl').read_text().splitlines()]
    audit=dict(trainings=trainings,
        peaks={k:max((row.get(k,0) or 0) for row in telemetry) for k in ['cpu_c','gpu_c','gpu_used_mib','load1','cpu_pressure10']},
        foreign_gpu_samples=sum(bool(row.get('foreign_gpu_pids')) for row in telemetry),
        monitor_errors=sum(bool(row.get('gpu_error')) for row in telemetry))
    (out/'completed_experiment_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

    os.environ.setdefault('MPLCONFIGDIR','/tmp/decovae-report-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,tests,title in [
        (axes[0],[('real','ncrps','Real CRPS'),('strong','crps_h1','Strong volatility, steps 1–32')],'Forecast error'),
        (axes[1],[('strong','lv_err','Strong volatility'),('base','lv_err','Base volatility')],'Log-variance error')]:
        labels=[];i=0
        for profile,metric,label in tests:
            for arm,color in [('Dmc1Zn','#b74b35'),('Dmc1B4','#2676a3')]:
                row=h[(h.profile==profile)&(h.metric==metric)&(h.arm==arm)].iloc[0]
                ax.errorbar(100*(row.effect-1),i,xerr=[[100*(row.effect-row.ci_low)],[100*(row.ci_high-row.effect)]],
                            fmt='o',capsize=4,color=color)
                labels.append(label+' / '+('Zn' if arm.endswith('Zn') else 'B4'));i+=1
        ax.axvline(0,color='gray',linestyle=':');ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis();ax.set_xlabel('Change vs matched Dmc1, % (lower is better)');ax.set_title(title)
    fig.suptitle('Final H1: 5 seeds; paired seed/series 95% intervals (exploratory)')
    fig.tight_layout();fig.savefig(out/'h1_five_seed_intervals.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    case='slow_1024_plus_24_sigma0.3'
    for ax,region in zip(axes,['context','horizon']):
        for i,model in enumerate(['A','Dmc1']):
            row=d[(d.dataset==case)&(d.model==model)&(d.metric=='seasonal_'+region+'_mse')].iloc[0]
            ax.errorbar(row['mean'],i,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],fmt='o',capsize=4)
        ax.set_xscale('log');ax.set_yticks([0,1],['Original A','PD']);ax.set_title(region.capitalize())
        ax.set_xlabel('Seasonal MSE / context variance, 95% interval')
    fig.suptitle('Slow 1024 wave + fast 24 cycle; sigma=0.3; 3 frozen lab seeds')
    fig.tight_layout();fig.savefig(out/'component_recovery_intervals.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4))
    for model,label in [('A','Original A'),('Dmc1','PD')]:
        part=d[d.dataset.str.startswith('weak_seasonality_sigma')&d.model.eq(model)&d.metric.eq('seasonal_in_trend_gain')].copy()
        part['sigma']=part.dataset.str.split('sigma').str[-1].astype(float);part=part.sort_values('sigma')
        ax.errorbar(part.sigma,part['mean'],yerr=[part['mean']-part.ci_low,part.ci_high-part['mean']],fmt='o-',capsize=4,label=label)
    ax.axhline(0,color='gray',linestyle=':');ax.axhline(1,color='gray',linestyle=':')
    ax.set_xlabel('Noise standard deviation');ax.set_ylabel('Seasonal projection in trend error')
    ax.set_title('Weak cycle: decomposition depends on signal-to-noise ratio')
    ax.legend();fig.tight_layout();fig.savefig(out/'weak_seasonality_leakage.png',dpi=180);plt.close(fig)
    print('Completed-result tables, audit and figures exported:',out)


if __name__=='__main__':main()
