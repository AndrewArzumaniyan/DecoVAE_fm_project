"""Build a self-contained results packet and an explicitly qualified review-response draft."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, help='Separate report directory; preserve downloaded source results')
    a = p.parse_args()
    root = a.root.resolve()
    runs = root/'notes/analysis/fm_mini'
    source = root/'final_results'
    out = a.out.resolve() if a.out else source
    out.mkdir(parents=True,exist_ok=True)
    def passed(name):
        try:return json.loads((source/name).read_text()).get('passed') is True
        except (OSError,ValueError):return False
    h1_matched=passed('matched_h1_complete.json')
    if h1_matched:
        summary=json.loads((source/'h1_summary_complete.json').read_text())
        h1_matched=(summary.get('real_control')=='matched' and bool(summary.get('comparisons')) and
                    all(row.get('seeds')==5 for row in summary.get('comparisons',[])))
    baseline = pd.read_csv(runs/'baseline_rows.csv')
    sn = baseline[baseline.method.eq('seasonal_naive')].groupby(['split','dataset']).ncrps.mean()

    @lru_cache(None)
    def rows(run, split='test_ood'):
        return pd.read_csv(runs/run/f'space_eval_{split}.csv',dtype={'series_id':str})

    def members(fam):
        return sorted(p.name for p in runs.glob(fam+'_s[0-9]') if (p/'space_eval_test_ood.csv').exists())

    def source_mean(fam, split='test_ood'):
        return pd.concat([rows(r,split).groupby('dataset').ncrps.mean() for r in members(fam)],axis=1).mean(1)

    def skills(fam, split):
        reference=sn.loc[split]
        return [float(np.exp(np.log(rows(r,split).groupby('dataset').ncrps.mean()/reference).mean())) for r in members(fam)]

    families={'A':'srv9_Ar02','N6mc1':'srv13_N6mc1','Dmc1':'srv10_Dmc1','N6s':'srv13_N6s',
              'Dmc1hg':'srv14h_Dmc1hg','Dmc1_KS1to1':'srv15k_Dmc1_ks100',
              'N6s_KS1to1':'srv15k_N6s_ks100','N6s_d128_KS1to1':'srv15k_N6s_d128_ks100'}
    records=[]
    for model,fam in families.items():
        row=dict(model=model,family=fam)
        for split,tag in [('test_ood','ood'),('test_id','id')]:
            values=skills(fam,split)
            row[tag+'_mean']=np.mean(values);row[tag+'_sd']=np.std(values,ddof=1);row['seeds']=len(values)
        records.append(row)
    table=pd.DataFrame(records)
    table.to_csv(out/'main_results.csv',index=False)
    source_ratio=source_mean('srv10_Dmc1')/source_mean('srv13_N6mc1')
    source_ratio.rename('Dmc1_over_N6mc1').to_csv(out/'main_source_ratios.csv')
    remaining=float(np.exp(np.log(source_ratio.drop(['Electricity','Solar'])).mean()))
    concentration=float(np.log(source_ratio.loc[['Electricity','Solar']]).sum()/np.log(source_ratio).sum())

    # Reproduce the completed LOSO experiment from saved scores; exclude the L175 intervention.
    fold_data={name:{} for name in ['Dmc1','N6mc1','N6s']}
    for pth in runs.glob('*/provenance.json'):
        name=pth.parent.name
        if not name.startswith(('srv12_loso_','srv13_loso_')):
            continue
        cfg=json.loads(pth.read_text())['config'];arm=cfg['arm']
        if arm not in fold_data or str(cfg.get('cache','')).endswith('L175'):
            continue
        for ds,v in rows(name).groupby('dataset').ncrps.mean().items():
            fold_data[arm].setdefault(ds,[]).append(float(v))
    seasonal={'Solar','weatherbench_weekly','Electricity','Q-TRAFFIC','London_Smart_Meters','mexico_city_bikes','Pedestrian_Counts'}
    folds=[];contrasts=[];rng=np.random.default_rng(991)
    for numerator,denominator in [('Dmc1','N6mc1'),('N6s','N6mc1'),('Dmc1','N6s')]:
        ds=sorted(set(fold_data[numerator])&set(fold_data[denominator]))
        if len(ds)!=15:
            raise ValueError('Incomplete LOSO evidence')
        rat=np.array([np.mean(fold_data[numerator][d])/np.mean(fold_data[denominator][d]) for d in ds])
        logs=np.log(rat)
        draws=np.exp(rng.choice(logs,size=(10000,len(ds)),replace=True).mean(1))
        lo,hi=np.quantile(draws,[.025,.975])
        contrasts.append(dict(comparison=numerator+'/'+denominator,ratio=np.exp(logs.mean()),ci_low=lo,ci_high=hi,sources=15))
        for d,v in zip(ds,rat):
            folds.append(dict(comparison=numerator+'/'+denominator,dataset=d,ratio=v,
                              seasonal_report_group=d in seasonal,
                              numerator_seeds=len(fold_data[numerator][d]),denominator_seeds=len(fold_data[denominator][d])))
    pd.DataFrame(folds).to_csv(out/'loso_sources.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'loso_comparisons.csv',index=False)

    # Use exactly the report's pooled-within-source definition, not mean per-window PCE.
    calibration=[]
    for model,fam in families.items():
        points=[]
        for name in members(fam):
            file=runs/name/'space_eval_pit_test_ood.npz'
            if not file.exists():
                continue
            with np.load(file) as z:pit=z['pit']
            ds=rows(name).dataset.values
            per=[]
            for d in np.unique(ds):
                u=pit[ds==d].ravel()
                per.append([np.mean([abs(q-(u<=q).mean()) for q in np.arange(.1,1.,.1)]),
                            (np.abs(u-.5)<=.25).mean(),(np.abs(u-.5)<=.4).mean(),(np.abs(u-.5)<=.49).mean()])
            points.append(np.mean(per,axis=0))
        if points:
            v=np.mean(points,axis=0)
            calibration.append(dict(model=model,seeds=len(points),PCE=v[0],c50=v[1],c80=v[2],c98=v[3]))
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for i,row in enumerate(contrasts):
        ax.errorbar(row['ratio'],i,xerr=[[row['ratio']-row['ci_low']],[row['ci_high']-row['ratio']]],fmt='o',capsize=4)
    ax.axvline(1,color='gray',linestyle=':');ax.set_yticks(range(len(contrasts)),[r['comparison'] for r in contrasts])
    ax.set_xlabel('LOSO CRPS ratio; source-bootstrap 95% interval (lower is better)')
    fig.tight_layout();fig.savefig(out/'loso_intervals.png',dpi=180);plt.close(fig)
    if passed('components_complete.json'):
        component=pd.read_csv(source/'component_intervals.csv')
        ds='slow_1024_plus_24_sigma0.3'
        sub=component[(component.cohort=='simple_negative')&(component.dataset==ds)&(component.metric=='mean_horizon_mse')]
        fig,ax=plt.subplots(figsize=(8,4))
        for i,(_,row) in enumerate(sub.iterrows()):
            ax.errorbar(row['mean'],i,xerr=[[max(0,row['mean']-row.ci_low)],[max(0,row.ci_high-row['mean'])]],fmt='o',capsize=4)
        ax.set_yticks(range(len(sub)),sub.model);ax.set_xlabel('Clean-mean horizon MSE / context variance, 95% interval')
        ax.set_title('Slow wave + fast seasonality; sigma=0.3; frozen lab checkpoints')
        fig.tight_layout();fig.savefig(out/'synthetic_intervals.png',dpi=180);plt.close(fig)

    lines=['# Defense evidence status','','## Main results','',
           'Skill is geometric over sources; mean and SD below are over training seeds. SD is not a confidence interval.',
           'Additional KernelSynth 1:1 means one synthetic window per real training window.', '',
           '| Model | Seeds | OOD mean ± SD | ID mean ± SD |','|---|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['model']} | {row['seeds']} | {row['ood_mean']:.4f} ± {row['ood_sd']:.4f} | {row['id_mean']:.4f} ± {row['id_sd']:.4f} |")
    lines += ['',f'Electricity and Solar account for {concentration:.1%} of the main log gain; without them Dmc1/N6mc1={remaining:.4f}.',
              '', '## Completed LOSO', '', '| Comparison | Ratio | 95% source interval |','|---|---:|---|']
    for row in contrasts:
        lines.append(f"| {row['comparison']} | {row['ratio']:.4f} | [{row['ci_low']:.4f}; {row['ci_high']:.4f}] |")
    lines += ['', 'LOSO uses 15 folds, usually two seeds; Solar and Pedestrian have three. The source intervals resample folds after seed averaging.',
              'The strongest interpretation is that the period estimator supplies transferable information; hard additive decomposition has no established accuracy advantage over N6s.',
              '', '## Final H1', '']
    h1=source/'H1_FINDINGS.md'
    lines.append(h1.read_text() if h1_matched else
                 'Matched five-seed H1 summary is not available. Any historical-control comparison is secondary; do not claim the matched experiment is complete.')
    component_description='The new frozen-checkpoint component diagnostics are not complete.'
    if passed('components_complete.json'):
        component=pd.read_csv(source/'component_intervals.csv')
        def comp_value(dataset, model, metric):
            return float(component[(component.cohort=='simple_negative')&(component.dataset==dataset)&
                                   (component.model==model)&(component.metric==metric)].iloc[0]['mean'])
        slow='slow_1024_plus_24_sigma0.3';weak='weak_seasonality_sigma0.3'
        component_description=(
            'Frozen-checkpoint diagnostics are complete: four model families, three seeds, 128 predictive draws; '
            '4,352 windows per model/seed and 52,224 model-seed-window rows. '
            f'For a slow 1024 wave plus a fast 24 cycle at noise SD 0.3, seasonal projection in trend error is '
            f"{comp_value(slow,'A','seasonal_in_trend_gain'):.4f} for A and {comp_value(slow,'Dmc1','seasonal_in_trend_gain'):.5f} for PD. "
            f"PD seasonal horizon MSE is {comp_value(slow,'Dmc1','seasonal_horizon_mse'):.5f}, "
            f"but its 90% interval coverage is {comp_value(slow,'Dmc1','coverage90'):.4f}. "
            f"For a weak cycle at noise SD 0.3, PD leakage returns to {comp_value(weak,'Dmc1','seasonal_in_trend_gain'):.4f}. "
            'No-season controls correctly suppress PD seasonality but retain forecast errors. '
            'These are new diagnostic cases for frozen lab-trained models, not universal real-data guarantees.')
        lines += ['', '## Frozen-checkpoint component diagnostics', '', component_description,
                  'See component_intervals.csv and COMPONENT_PROTOCOL.md for per-case seed/series intervals.']
    fft=root/'reference_local/fft_reconstruction/code/FINDINGS.md'
    if fft.exists():
        lines += ['', '## Independent decoder-only study', '', fft.read_text()]
    if passed('external_qcrps_complete.json'):
        checked=out/'external_report_validation.json'
        external_dir=out if checked.exists() and json.loads(checked.read_text()).get('passed') else source
        lines += ['',(external_dir/'EXTERNAL_REFERENCE.md').read_text()]
    lines += ['', '## Reproducibility and limits', '',
              '- Transfer hashes and two numerical checkpoint-reproduction checks are in ops/.',
              '- Source-held-out does not mean the original six sources were never inspected during development.',
              '- KernelSynth/model-size configurations were explored on the same OOD set. The lowest skill is a candidate, not an independently confirmed universal winner.',
              '- PCE pools PIT within each source before taking absolute deviations; conditional/per-window calibration is a different quantity.',
              '- External results use the matched nine-quantile score only when EXTERNAL_REFERENCE.md is complete. Reported pretraining overlap remains; this is not a clean zero-shot comparison.',
              '- The tested HP penalties on decoded outputs do not establish the effect of the intended latent-trajectory regularizer.',
              '- Public release/tagging has not been performed by these jobs.', '']
    (out/'DEFENSE_STATUS.md').write_text('\n'.join(lines))
    external_description=('A new external reference uses identical frozen windows, context scaling, seasonal-naive reference and the same nine-quantile score for Chronos and our models. See EXTERNAL_REFERENCE.md. Potential pretraining overlap still prevents a clean zero-shot claim.'
                          if passed('external_qcrps_complete.json') else
                          'A clean external zero-shot comparison remains a limitation; the old Chronos reference used a different CRPS approximation and reported training overlap. The new matched reference is not complete.')
    if passed('external_qcrps_complete.json'):
        ext=pd.read_csv(external_dir/'matched_quantile_results.csv')
        ext=ext[ext.split=='test_ood'].copy()
        field='skill_pooled' if 'skill_pooled' in ext else 'skill_mean'
        chronos=ext[ext.model=='Chronos_Bolt_small'].iloc[0]
        own=ext[~ext.model.isin(['Chronos_Bolt_small','seasonal_naive'])].sort_values(field).iloc[0]
        external_description += (f' Under qCRPS9, Chronos pooled skill is {chronos[field]:.4f}; '
                                 f'the best tested own-model reference is {own.model} at {own[field]:.4f}. '
                                 'These are a separate metric table, not the MC128 headline results.')
    h1_description=('The matched extension is complete: real-data Dmc1 uses current-environment srv14r w0.2 seeds 0–2 and new seeds 3–4, while strong-volatility controls use their matching runs. The published srv10 baseline is preserved only as a secondary historical comparison.'
                    if h1_matched else 'The matched five-seed extension is not complete. Do not report the historical-control comparison as the primary H1 result.')
    if h1_matched:
        results=pd.read_csv(source/'h1_comparisons.csv')
        def h1_interval(profile,arm,metric):
            row=results[(results.profile==profile)&(results.arm==arm)&(results.metric==metric)].iloc[0]
            return f'{row.effect:.5f} [{row.ci_low:.5f}; {row.ci_high:.5f}]'
        h1_description += (
            f" On real OOD, Zn/Dmc1 CRPS is {h1_interval('real','Dmc1Zn','ncrps')}; "
            f"on strong volatility, its log-variance error ratio is {h1_interval('strong','Dmc1Zn','lv_err')}. "
            f"B4 first-32-step CRPS on strong volatility is {h1_interval('strong','Dmc1B4','crps_h1')}, "
            'a small targeted effect of handcrafted features. All intervals resample paired seeds and series '
            'within fixed sources/blocks; multiple diagnostic contrasts are exploratory without multiplicity correction.')
    response=f'''# Review response — evidence-based draft

1. **Concentration and uncertainty.** We agree. Electricity and Solar explain {concentration:.1%} of the six-source log improvement; excluding them the ratio is {remaining:.4f}. We now report every source and a completed 15-source LOSO analysis with source-level intervals. We restrict the generalization claim to the tested regime and especially seasonal sources.

2. **Estimator-information control and baselines.** N6s receives the estimated seasonal continuation, phases, and reliability features as inputs, with the same mixture objective, without deseasonalizing the input or adding seasonality to the output. Dmc1/N6s in LOSO is {contrasts[2]['ratio']:.4f} [{contrasts[2]['ci_low']:.4f}; {contrasts[2]['ci_high']:.4f}]. This changes our interpretation: estimator information, rather than mandatory additive decomposition, explains most accuracy gains. The main comparison uses training-source validation; an additional source-specific tuning study is not claimed. OOD metrics were inspected during development, as disclosed below. {external_description}

3. **Development exposure.** The original OOD sources were inspected during iterative development. Validation stopping uses training-source validation, but that does not make all architectural choices independent of OOD diagnostics. We explicitly disclose this and do not relabel the old test as untouched. LOSO is a broader robustness check, not a retroactively pristine final test.

4. **LOSO and seasonality.** All 15 folds are complete for Dmc1, N6mc1, and N6s. Dmc1/N6mc1 is {contrasts[0]['ratio']:.4f} [{contrasts[0]['ci_low']:.4f}; {contrasts[0]['ci_high']:.4f}]. The source grouping is the report's estimator-based seasonal grouping, not ground-truth annotation. We provide per-source results and qualify the small number of sources and seeds.

5. **Estimator failures.** We report natural estimator-failure cases, feature ablations, the long-period intervention, and hybrid trust gating. Hard addition is sensitive to errors; N6s may ignore an unreliable hint. {component_description} We do not claim an oracle annotation of real-data decomposition or a universally safe fallback.

6. **Original decoder.** The independent FFT-only study tests original reconstruction, amplitude correction, and exact original frequencies with unchanged native encoders/decomposition/loss, at horizons 96/128/256 and three seeds. Neither correction restores learned decomposition or consistently improves mean CRPS in that profile. Its CPU/runtime/training-budget differences are disclosed, and it is not numerically pooled with the real-data experiment.

7. **Final H1.** {h1_description} See H1_FINDINGS.md only when the matched completion marker is present. B4 is handcrafted volatility information, not a capacity-matched raw encoder. Do not infer equivalence solely from a non-significant result.

8. **Reproducibility.** We have a hash-verified private working snapshot, configurations, per-window scores, checkpoints, scripts, and numerical checks reproducing two reference checkpoints. Public packaging and a tagged release must still be completed; this draft does not claim that a release has already been published.
'''
    (out/'REVIEW_RESPONSE_DRAFT.md').write_text(response)
    (out/'evidence_complete.json').write_text(json.dumps({'passed':True,'main_results':records,'LOSO':contrasts,
                                                        'H1_available':h1_matched,
                                                        'components_available':passed('components_complete.json'),
                                                        'matched_external_available':passed('external_qcrps_complete.json')},indent=2)+'\n')
    print('Evidence packet assembled',out,flush=True)


if __name__=='__main__':
    main()
