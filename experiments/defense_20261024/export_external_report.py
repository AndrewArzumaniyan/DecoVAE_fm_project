"""Validate frozen external-score rows and label the bootstrap estimand explicitly.

The evaluator's bootstrap pools errors over resampled seeds inside each source.
The primary table therefore shows pooled skill, retaining mean per-seed skill/SD
as separate CSV columns. No predictions are recomputed by this export.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.root/'final_results';out=a.out;out.mkdir(parents=True,exist_ok=True)
    marker=json.loads((raw/'external_qcrps_complete.json').read_text())
    if marker.get('passed') is not True:raise ValueError('Completed external evaluation required')
    frame=pd.read_csv(raw/'matched_quantile_windows.csv',dtype={'series_id':str})
    table=pd.read_csv(raw/'matched_quantile_results.csv')
    if frame.duplicated(['split','model','seed','window']).any():raise ValueError('Duplicate prediction rows')
    values=frame[['qcrps9','snaive_qcrps9','coverage80']].to_numpy()
    if not np.isfinite(values).all() or (values[:,:2]<0).any():raise ValueError('Invalid score')
    if not frame.coverage80.between(0,1).all():raise ValueError('Invalid coverage')
    for split,part in frame.groupby('split'):
        reference=part[part.model=='seasonal_naive'].sort_values('window')
        keys=reference[['window','dataset','series_id']].reset_index(drop=True)
        for _,g in part.groupby(['model','seed']):
            g=g.sort_values('window')
            pd.testing.assert_frame_equal(g[['window','dataset','series_id']].reset_index(drop=True),keys)
            if not np.allclose(g.snaive_qcrps9,reference.snaive_qcrps9,rtol=1e-12,atol=1e-12):
                raise ValueError('Different seasonal-naive reference')
    by_source=frame.groupby(['split','model','seed','dataset'])[['qcrps9','snaive_qcrps9','coverage80']].mean()
    by_source['ratio']=by_source.qcrps9/by_source.snaive_qcrps9
    expected=pd.read_csv(raw/'matched_quantile_by_source.csv').set_index(['split','model','seed','dataset'])
    pd.testing.assert_frame_equal(by_source.sort_index(),expected.sort_index(),rtol=1e-11,atol=1e-12)
    points=[]
    for row in table.itertuples():
        g=by_source.loc[(row.split,row.model)]
        pooled=g.groupby('dataset')[['qcrps9','snaive_qcrps9']].mean()
        point=float(np.exp(np.log(pooled.qcrps9/pooled.snaive_qcrps9).mean()))
        per_seed=g.ratio.groupby('seed').apply(lambda x:np.exp(np.log(x).mean()))
        if not np.isclose(per_seed.mean(),row.skill_mean,rtol=1e-11,atol=1e-12):
            raise ValueError('Mean skill does not match saved rows')
        if not row.ci_low-1e-12<=point<=row.ci_high+1e-12:raise ValueError('Inspect bootstrap interval')
        points.append(point)
    table['skill_pooled']=points
    table.to_csv(out/'matched_quantile_results.csv',index=False)
    by_source.to_csv(out/'matched_quantile_by_source.csv')
    revision=json.loads((raw/'external_model_revision.json').read_text())
    lines=['# External reference: common nine-quantile score','',
        'Same frozen ID/OOD windows, context 512, horizon 96, context scaling and seasonal-naive reference.',
        'qCRPS9 = twice the mean pinball loss at quantiles 0.1 through 0.9. Do not pool with the main MC128 CRPS table.',
        'Pooled skill averages scores over training seeds within a source, then takes the geometric source ratio. '
        'The bootstrap estimates this same quantity; mean per-seed skill and SD are retained separately in CSV.',
        '95% intervals resample training seeds and whole series within each fixed source, not new sources. '
        'Chronos uses one pretrained checkpoint; its interval reflects test-series sampling only.',
        'Own models use quantiles estimated from 128 predictive samples; Chronos uses native predicted quantiles. '
        'Its native 64-step horizon is extended by the pipeline to 96. No extra tuning.',
        'Potential pretraining-source overlap prevents a clean source-zero-shot or SOTA claim.',
        f"Pinned external model: {revision['model']} at revision {revision['revision']}.", '',
        '| Split | Model | Seeds/checkpoints | Pooled skill | 95% interval | Coverage 80% |',
        '|---|---|---:|---:|---|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.split} | {row.model} | {row.seeds} | {row.skill_pooled:.4f} | '
                     f'[{row.ci_low:.4f}; {row.ci_high:.4f}] | {row.coverage80:.4f} |')
    (out/'EXTERNAL_REFERENCE.md').write_text('\n'.join(lines)+'\n')
    reference=frame[frame.model=='seasonal_naive']
    audit=dict(passed=True,rows=len(frame),groups=len(table),
        windows_per_split={split:len(g) for split,g in reference.groupby('split')},
        windows_by_source={split:{name:len(g) for name,g in part.groupby('dataset')}
                           for split,part in reference.groupby('split')},
        estimand='Geometric source ratio after pooling errors across training seeds; matches saved bootstrap',
        checks=['paired window identities','same seasonal-naive scores','finite nonnegative scores',
                'source aggregation','per-seed means','pooled point within saved interval'],
        raw_source_sha256={name:hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in
            ['matched_quantile_windows.csv','matched_quantile_results.csv','matched_quantile_by_source.csv','external_model_revision.json']})
    (out/'external_report_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('External report table validated and exported:',out)


if __name__=='__main__':main()
