"""Post-run diagnostics, paired interventions, and paired series bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import ROOT,PERIOD_DECOMP_DIR,T,L,SCENARIOS,STRESS,sample,normalize
from adapters import algorithm,Arm
from fourier_arm import FOURIER_ARMS,make_model,attach
from metrics import components,bootstrap
from runner import OUT,save_json,set_batch


def component_diagnostics(path):
    ref=json.loads((path/'data_reference.json').read_text()); raw=dict(np.load(Path(ref['cache'])/'test.npz'))
    z,m,s=normalize(raw['x'])
    data=np.load(path/'forecast_outputs.npz',allow_pickle=True)['outputs']
    decoded=np.concatenate([o['decoded_normalized'] for o in data]); n=len(decoded)
    config=json.loads((path/'provenance.json').read_text())['config']; model=make_model(config['arm'])
    model.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=False)['model']); model.eval(); torch.set_num_threads(2)
    input_trend=[]; input_seas=[]
    cached=None; fourier=None
    if config['arm'] in FOURIER_ARMS:
        fourier=attach(dict(z=z),ref['cache'],'test'); cached=fourier['components']
    elif config['arm'] in 'CD':
        digest=hashlib.sha256((PERIOD_DECOMP_DIR/'period_decomp.py').read_bytes()+(Path(__file__).parent/'adapters.py').read_bytes()).hexdigest()[:12]
        cache_path=Path(ref['cache'])/f'test_period_{digest}.npz'
        if cache_path.exists(): cached=np.load(cache_path)['components']
    for i in range(0,n,32):
        x=torch.tensor(z[i:i+32,:T])
        if fourier is not None: model.fourier=(torch.from_numpy(fourier['periods'][i:i+32]),torch.from_numpy(fourier['coefs'][i:i+32]))
        if config['arm'][0] in 'CD':
            if cached is not None: model.components=(torch.from_numpy(cached[i:i+32,0]),torch.from_numpy(cached[i:i+32,1]))
            else:
                comps=[algorithm(v,'period')[0] for v in z[i:i+32,:T]]
                model.components=(torch.tensor(np.stack([c['trend'] for c in comps]),dtype=torch.float32),torch.tensor(np.stack([c['seasonal'] for c in comps]),dtype=torch.float32))
        with torch.no_grad(): enc=model.encode(x,n_target=L//16)
        input_trend.extend(enc['target_trend'].numpy()); input_seas.extend(enc['target_seasonal'].numpy())
    input_trend=np.array(input_trend); input_seas=np.array(input_seas); rows=[]
    for i in range(n):
        if 'params_json' in raw:
            no_trend=not bool(raw['has_trend'][i])
            rec=dict(x=raw['x'][i],trend=raw['trend'][i],seasonal=raw['seasonal'][i],noise=raw['noise'][i],
                     params=json.loads(str(raw['params_json'][i])),scenario=('none' if no_trend else 'linear',0,float(np.sqrt(np.mean(raw['variance'][i]))),'iid'))
        else:
            rec=sample(719+200000+i,SCENARIOS[int(raw['scenario'][i])])
        decoded_original={'trend':decoded[i,0,:T]*s[i]+m[i], 'seasonal':decoded[i,1,:T]*s[i]}
        decoded_original['noise']=raw['x'][i,:T]-decoded_original['trend']-decoded_original['seasonal']
        input_original={'trend':input_trend[i]*s[i]+m[i], 'seasonal':input_seas[i]*s[i]}
        input_original['noise']=raw['x'][i,:T]-input_original['trend']-input_original['seasonal']
        row=dict(series_id=i,noise_level=rec['scenario'][2],scenario=json.dumps(rec['scenario']))
        row.update({'input_'+k:v for k,v in components(rec,input_original).items()})
        row.update({'decoded_'+k:v for k,v in components(rec,decoded_original).items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path/'component_rows.csv',index=False)


def algorithm_bootstrap(path):
    df=pd.read_csv(path/'algorithm_rows.csv'); rows=[]
    for dataset,group in df.groupby('dataset'):
        for metric in ('nrmse','noise_nrmse','residual_seasonal_energy'):
            if metric not in group: continue
            pivot=group.pivot(index='series_id',columns='method',values=metric)
            if 'period' not in pivot or 'original' not in pivot: continue
            diff=(pivot['period']-pivot['original']).dropna()
            if len(diff): rows.append(dict(dataset=dataset,metric=metric,effect_period_minus_original=float(diff.mean()),ci95=bootstrap(diff),n=len(diff)))
    save_json(path/'paired_bootstrap.json',rows)


def interventions(path,n=32):
    methods=('original','period'); rows=[]; samples=[]
    for i in range(n):
        rec=sample(990000+i,SCENARIOS[(i*7)%len(SCENARIOS)])
        base=rec['x']; _,m,s=normalize(base)
        for changed in ('seasonal','trend','noise'):
            other=base.copy()
            delta=.5*rec[changed] if changed!='trend' else .003*np.arange(T+L)
            other+=delta
            for method in methods:
                a,_,_=algorithm((base[:T]-m)/s,method); b,_,_=algorithm((other[:T]-m)/s,method)
                row=dict(series_id=i,method=method,changed=changed)
                for component in ('trend','seasonal','noise'):
                    response=(b[component]-a[component])*s
                    expected=delta[:T] if component==changed else np.zeros(T)
                    row[component+'_response_error']=float(np.sqrt(np.mean((response-expected)**2))/s)
                    row[component+'_response_rms']=float(np.sqrt(np.mean(response**2))/s)
                rows.append(row)
    pd.DataFrame(rows).to_csv(path/'intervention_rows.csv',index=False)


def factor_report(paths,out):
    frames=[]; smoke=[]
    for path in paths:
        status=json.loads((path/'status.json').read_text())
        if status['smoke']: smoke.append(path.name); continue
        if status['status']!='complete' or not (path/'fm_rows.csv').exists(): continue
        df=pd.read_csv(path/'fm_rows.csv'); frames.append(df)
    out.mkdir(parents=True,exist_ok=True)
    if not frames: return
    df=pd.concat(frames); df.to_csv(out/'all_fm_rows.csv',index=False); effects=[]
    for seed,g in df.groupby('seed'):
        for scenario,s in g.groupby('scenario'):
            pivot=s.pivot(index='series_id',columns='arm',values='crps')
            for label,on,off in [('G_original','A','B'),('G_period','C','D'),('G_fourier','Cf','Df')]:
                if on in pivot and off in pivot:
                    diff=(pivot[off]-pivot[on]).dropna(); effects.append(dict(seed=int(seed),scenario=scenario,effect=label,mean=float(diff.mean()),ci95=bootstrap(diff),n=len(diff)))
            if all(a in pivot for a in 'ABCD'):
                v=(pivot.D-pivot.C)-(pivot.B-pivot.A); effects.append(dict(seed=int(seed),scenario=scenario,effect='interaction',mean=float(v.mean()),ci95=bootstrap(v),n=len(v)))
    save_json(out/'factor_effects.json',effects)
    save_json(out/'excluded_smoke.json',smoke)
    df.groupby(['arm','seed'])[['crps','coverage90','width90','rmse','variance_mae']].mean().to_csv(out/'factor_summary.csv')
    text=['# Factorial comparison','', 'Smoke runs excluded: '+', '.join(smoke), '',df.groupby(['arm','seed'])[['crps','coverage90','width90','rmse','variance_mae']].mean().to_string(),'', 'Positive G means lower CRPS with the noise branch. Interaction = G_period - G_original. Confidence intervals resample paired independent series, separately by scenario and seed.']
    (out/'report.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['components','algorithms','factor']); parser.add_argument('paths',nargs='+'); parser.add_argument('--out',default='factor_report'); a=parser.parse_args(); paths=[OUT/p for p in a.paths]
    if a.command=='components':
        for path in paths: component_diagnostics(path)
    elif a.command=='algorithms':
        for path in paths: algorithm_bootstrap(path); interventions(path)
    else: factor_report(paths,OUT/a.out)
if __name__=='__main__': main()
