"""Evaluate one saved checkpoint on the versioned mechanism-level diagnostic suite."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adapters import Arm, algorithm
from fourier_arm import FOURIER_ARMS,make_model,attach
from data import ROOT,PERIOD_DECOMP_DIR,T,L,generate_diagnostics,normalize
from runner import OUT,evaluate,save_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--device',default='cuda'); p.add_argument('--n-samples',type=int,default=128); p.add_argument('--suite'); p.add_argument('--output-dir'); p.add_argument('--no-correction',action='store_true'); a=p.parse_args()
    model_path=OUT/a.run; out=OUT/a.output_dir if a.output_dir else OUT/('diagnostics_'+a.run)
    out.mkdir(parents=True,exist_ok=True)
    device=a.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; select cpu explicitly or restore the driver')
    torch.set_num_threads(1)
    cfg=json.loads((model_path/'provenance.json').read_text())['config']; arm=cfg['arm']
    model=make_model(arm).to(device); checkpoint=torch.load(model_path/'best.pt',weights_only=False,map_location=device); model.load_state_dict(checkpoint['model']); model.eval()
    if a.no_correction: model.use_correction=False
    suite=Path(a.suite) if a.suite else generate_diagnostics(n_per_case=64)
    raw={k:v for k,v in np.load(suite/'test.npz').items()}
    z,m,s=normalize(raw['x']); raw.update(z=z.astype('float32'),mean=m,scale=s)
    if arm in FOURIER_ARMS: attach(raw,suite,'suite')
    elif arm in 'CD':
        digest=hashlib.sha256((PERIOD_DECOMP_DIR/'period_decomp.py').read_bytes()+(Path(__file__).parent/'adapters.py').read_bytes()).hexdigest()[:12]
        cache=suite/f'period_{digest}.npz'
        if cache.exists(): raw['components']=np.load(cache)['components']
        else:
            comps=[algorithm(v[:T],'period')[0] for v in raw['z']]
            raw['components']=np.stack([[c['trend'],c['seasonal']] for c in comps]).astype('float32')
            np.savez_compressed(cache,components=raw['components'])
    rows,outputs=evaluate(model,raw,device,a.n_samples)
    rows['case']=raw['case'][:len(rows)]; rows['tail']=raw['tail'][:len(rows)]
    rows['arm']=arm+('0' if a.no_correction else ''); rows['seed']=cfg['seed']; rows['model_run']=a.run
    predicted_mean=np.concatenate([o['mean'] for o in outputs]); predicted_var=np.concatenate([o['observation_variance'] for o in outputs])
    true_mean=raw['forecast_mean'][:len(rows)]; true_var=raw['forecast_variance'][:len(rows)]
    rows['variance_mae']=np.abs(predicted_var-true_var).mean(axis=1)
    rows['mean_bias_mse']=np.mean((predicted_mean-true_mean)**2,axis=1)
    rows['predictable']=rows.case!='sigma_unseen_future_jump'
    rows.to_csv(out/'diagnostic_fm_rows.csv',index=False)
    np.savez_compressed(out/'diagnostic_predictions.npz',mean=predicted_mean,observation_variance=predicted_var,
                        true_mean=true_mean,true_variance=true_var,case=raw['case'][:len(rows)])
    save_json(out/'provenance.json',dict(model_run=a.run,checkpoint=str(model_path/'best.pt'),suite=str(suite),n_per_case=int(json.loads((suite/'config.json').read_text())['n_per_case']),n_samples=a.n_samples,device=device,fourier_correction=not a.no_correction,ar_truth='conditional mean and full multi-step residual variance',unpredictable_case='sigma_unseen_future_jump; variance scores are descriptive'))
    print(json.dumps(dict(output=str(out),rows=len(rows),cases=int(rows.case.nunique()),device=device)),flush=True)

if __name__=='__main__': main()
