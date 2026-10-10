"""Reproducible benchmark and training CLI. Run through uv run --locked."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from data import ROOT,PERIOD_DECOMP_DIR,BENCHMARK_DIR,RUNS_DIR,T,L,SCENARIOS,STRESS,sample,normalize,generate,generate_real,generate_diagnostics,real_windows
from adapters import Arm,algorithm,ar_forecast
from fourier_arm import FOURIER_ARMS,make_model,attach,set_fourier
from metrics import components,distribution,bootstrap

OUT=RUNS_DIR
STOP_REQUESTED = False

def stop_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str))

def provenance(path,args):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'provenance.json').exists():
        archive=path/'previous_attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        shutil.copyfile(path/'provenance.json',archive/'provenance.json')
        if (path/'source_snapshot').exists(): shutil.copytree(path/'source_snapshot',archive/'source_snapshot')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in (BENCHMARK_DIR,PERIOD_DECOMP_DIR,ROOT/'decovaefm'/'models') for p in directory.glob('*.py')}
    save_json(path/'provenance.json',dict(config=vars(args),time_utc=datetime.now(timezone.utc).isoformat(),python=os.sys.version,versions={k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','statsmodels','pandas','pyarrow')},gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,cuda=torch.version.cuda,source_hashes=hashes,data_seed=719,evaluation_seed=1729,training_seed=args.seed,lock_sha256=hashlib.sha256((ROOT/'uv.lock').read_bytes()).hexdigest()))
    for name in hashes:
        target=path/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    shutil.copyfile(ROOT/'uv.lock',path/'uv.lock')

def benchmark(args,path):
    records=[]
    for si,sc in enumerate(SCENARIOS+STRESS):
        for i in range(args.n):
            records.append(dict(sample(900000+si*1000+i,sc),series_id=f'syn-{si}-{i}',dataset='synthetic',stress=si>=len(SCENARIOS)))
    if args.real: records+=real_windows('test',args.n)
    rows=[]; saved=[]
    for i,r in enumerate(records):
        z,m,s=normalize(r['x']); x=z[:T]; y=r['x'][T:]; methods=['original','period','stl','fourier_ar','naive','seasonal_naive','ar']
        if r['dataset']=='synthetic': methods+=['stl_oracle']
        periods=None
        for method in methods:
            start=time.monotonic(); d=None
            if method=='naive': pred=np.full(L,x[-1])
            elif method=='ar': pred=ar_forecast(x)
            elif method=='seasonal_naive':
                p=max(2,int(round(max(periods or [1])))) if periods else 1
                pred=x[T-p+np.arange(L)%p]
            else:
                oracle=sorted(set(p[0] for p in r.get('params',[])))
                d,pred,estimated=algorithm(x,method,oracle)
                if method=='period': periods=estimated
            forecast=pred*s+m
            row={k:r[k] for k in ('dataset','series_id')}; row.update(method=method,rmse=float(np.sqrt(np.mean((forecast-y)**2))),mae=float(np.abs(forecast-y).mean()),nrmse=float(np.sqrt(np.mean((forecast-y)**2))/s.item()),seconds=time.monotonic()-start)
            if 'scenario' in r: row.update(scenario=json.dumps(r['scenario']),stress=r['stress'],noise_level=r['scenario'][2])
            if d is not None:
                original={k:d[k]*s+(m if k=='trend' else 0) for k in ('trend','seasonal','noise')}
                saved.append(dict(index=i,method=method,forecast=forecast,**original))
                if r['dataset']=='synthetic': row.update(components(r,original))
                row['estimated_periods']=json.dumps(estimated)
            rows.append(row)
        if i%32==0:
            pd.DataFrame(rows).to_csv(path/'algorithm_rows.csv',index=False)
            print(f'benchmark {i+1}/{len(records)}',flush=True)
    pd.DataFrame(rows).to_csv(path/'algorithm_rows.csv',index=False)
    np.savez_compressed(path/'algorithm_outputs.npz',outputs=np.array(saved,dtype=object))
    np.savez_compressed(path/'algorithm_data.npz',records=np.array(records,dtype=object))
    report(path)
    save_json(path/'status.json',dict(status='complete',series=len(records),rows=len(rows)))


def cache_components(z,cache_path):
    if cache_path.exists(): return np.load(cache_path)['components']
    start=time.monotonic(); partial=cache_path.with_name(cache_path.stem+'.partial.npz')
    out=list(np.load(partial)['components']) if partial.exists() else []
    if len(out)>len(z): raise ValueError('Component cache is longer than its input split')
    def save_partial():
        if not out: return
        temporary=partial.with_name(partial.stem+'.tmp.npz')
        np.savez_compressed(temporary,components=np.stack(out).astype('float32'))
        os.replace(temporary,partial)
    for i in range(len(out),len(z)):
        if STOP_REQUESTED:
            save_partial(); raise InterruptedError('Period-component cache interrupted; partial rows saved')
        x=z[i,:T]
        d,_,_=algorithm(x,'period'); out.append(np.stack([d['trend'],d['seasonal']]))
        if (i+1)%64==0:
            save_partial()
            if (i+1)%256==0: print(f'period cache {i+1}/{len(z)} {time.monotonic()-start:.1f}s',flush=True)
    c=np.stack(out).astype('float32'); np.savez_compressed(cache_path,components=c)
    if partial.exists(): partial.unlink()
    return c


def load_splits(args):
    cache=generate_real(stop=lambda:STOP_REQUESTED) if getattr(args,'data','synthetic')=='real' else generate(args.profile); loaded={}
    for split in ('train','val','test'):
        raw=dict(np.load(cache/f'{split}.npz')); z,m,s=normalize(raw['x']); raw.update(z=z.astype('float32'),mean=m,scale=s)
        if args.arm in FOURIER_ARMS: attach(raw,cache,split,stop=lambda:STOP_REQUESTED)
        elif args.arm in 'CD':
            adapter_hash=hashlib.sha256((PERIOD_DECOMP_DIR/'period_decomp.py').read_bytes()+(Path(__file__).parent/'adapters.py').read_bytes()).hexdigest()[:12]
            raw['components']=cache_components(z,cache/f'{split}_period_{adapter_hash}.npz')
        loaded[split]=raw
    return loaded,cache


def set_batch(model,raw,idx,device):
    if model.arm in 'CD':
        c=torch.from_numpy(raw['components'][idx]).to(device)
        model.components=(c[:,0],c[:,1])
    set_fourier(model,raw,idx,device)
    return torch.from_numpy(raw['z'][idx,:T]).to(device)


def loss(out,x,y):
    terms=[]
    for prefix,target in [('recon',x),('forecast',y)]:
        lv=out[prefix+'_logvar']; terms.append(.5*(lv+(target-out[prefix+'_mu'])**2*torch.exp(-lv)).mean())
    kl=0.
    for name in ('trend','seasonal','noise'):
        if name+'_mu' not in out: continue
        mu=out[name+'_mu']; lv=out[name+'_logvar'].clamp(-20,10)
        kl+= ((mu.abs()**2+lv.exp()-1-lv).mean()*(1 if mu.is_complex() else .5))
    return sum(terms)+.01*kl


@torch.no_grad()
def evaluate(model,raw,device,n_samples,limit=None):
    model.eval(); torch.manual_seed(1729); n=len(raw['z']) if limit is None else min(limit,len(raw['z'])); rows=[]; outputs=[]
    for start in range(0,n,16):
        if STOP_REQUESTED: raise InterruptedError('Evaluation stopped for resource recovery')
        idx=np.arange(start,min(start+16,n)); x=set_batch(model,raw,idx,device)
        mus=[]; variances=[]; obs=[]; decoded=[]
        for _ in range(n_samples):
            o=model(x); mu=o['forecast_mu']; var=o['forecast_logvar'].exp()
            mus.append(mu.cpu().numpy()); variances.append(var.cpu().numpy()); obs.append((mu+var.sqrt()*torch.randn_like(mu)).cpu().numpy())
            decoded.append(np.stack([o['decoded_trend'].cpu().numpy(),o['decoded_seasonal'].cpu().numpy()],1))
        means=np.stack(mus); var=np.stack(variances); samples=np.stack(obs)
        scale=raw['scale'][idx]; center=raw['mean'][idx]
        means=means*scale+center; samples=samples*scale+center; var=var*scale**2
        true_variance=raw.get('variance',np.zeros_like(raw['x']))[idx,T:] if 'variance' in raw else None
        if true_variance is not None and 'phi' in raw:
            phi=raw['phi'][idx,None]
            powers=2*np.arange(L,dtype=float)[None,:]
            multistep=true_variance[:,0,None]*np.cumsum(phi**powers,axis=1)
            true_variance=np.where(phi!=0,multistep,true_variance)
        metrics=distribution(raw['x'][idx,T:],samples,means.mean(0),var.mean(0),means.var(0),true_variance)
        if 'trend' in raw and 'seasonal' in raw:
            truth=raw['trend'][idx,T:]+raw['seasonal'][idx,T:]
            if 'phi' in raw and 'noise' in raw:
                horizons=np.arange(1,L+1,dtype=float)[None,:]
                truth=truth+raw['phi'][idx,None]**horizons*raw['noise'][idx,T-1,None]
            metrics['mean_bias_mse']=np.mean((means.mean(0)-truth)**2,axis=-1)
        rows.extend([dict(series_id=int(i),**{k:float(v[j]) for k,v in metrics.items()}) for j,i in enumerate(idx)])
        outputs.append(dict(indices=idx,samples=samples,mean=means.mean(0),observation_variance=var.mean(0),latent_mean_variance=means.var(0),decoded_normalized=np.mean(decoded,axis=0)))
    return pd.DataFrame(rows),outputs


def train(args,path):
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    start=time.monotonic()
    device=args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; execute outside sandbox, no silent CPU fallback')
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    try: raw,cache=load_splits(args)
    except InterruptedError:
        save_json(path/'status.json',dict(status='stopped_before_training',epochs=0,smoke=False,test_evaluated=False))
        return
    save_json(path/'data_reference.json',dict(cache=str(cache),sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.glob('*.npz') if '_period_' not in p.name}))
    model=make_model(args.arm).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(args.seed); best=float('inf'); stale=0; first_epoch=0; history=[]; resume_order=None; resume_batch=0; resume_losses=[]
    if args.resume:
        checkpoint=torch.load(args.resume,weights_only=False,map_location=device)
        if checkpoint['config']['arm']!=args.arm or checkpoint['config']['profile']!=args.profile:
            raise ValueError('Resume requires the same arm and data profile')
        if checkpoint['config'].get('data','synthetic')!=getattr(args,'data','synthetic'):
            raise ValueError('Resume requires the same synthetic/real dataset mode')
        previous_best=Path(args.resume).parent/'best.pt'
        if previous_best.exists() and previous_best.resolve()!=(path/'best.pt').resolve():
            shutil.copyfile(previous_best,path/'best.pt')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); first_epoch=checkpoint['epoch']+(0 if checkpoint.get('partial_epoch') else 1)
        if checkpoint.get('partial_epoch'):
            resume_order=checkpoint['order']; resume_batch=checkpoint['next_batch']; resume_losses=checkpoint['losses']
        torch.set_rng_state(checkpoint['torch_rng'].cpu()); np.random.set_state(checkpoint['numpy_rng']); random.setstate(checkpoint['python_rng']); rng.bit_generator.state=checkpoint['shuffle_rng']
        if device=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
        best=checkpoint['best']; stale=checkpoint['stale']; history=checkpoint['history']
    epochs=args.epochs or (10 if args.profile=='pilot' else 60); status='complete'
    for epoch in (range(first_epoch,epochs) if stale<10 else []):
        tick=time.monotonic(); model.train(); order=resume_order if resume_order is not None else rng.permutation(len(raw['train']['z'])); losses=resume_losses if resume_order is not None else []; first_batch=resume_batch if resume_order is not None else 0
        resume_order=None; resume_batch=0; resume_losses=[]
        for b in range(first_batch,len(order),32):
            idx=order[b:b+32]; x=set_batch(model,raw['train'],idx,device); y=torch.from_numpy(raw['train']['z'][idx,T:]).to(device)
            optimizer.zero_grad(set_to_none=True); value=loss(model(x),x,y)
            if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
            value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(value.item())
            if STOP_REQUESTED or time.monotonic()-start>args.minutes*60:
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],numpy_rng=np.random.get_state(),python_rng=random.getstate(),shuffle_rng=rng.bit_generator.state,config=vars(args),status='time_limit',partial_epoch=True,order=order,next_batch=b+32,losses=losses)
                torch.save(checkpoint,path/'last.pt')
                save_json(path/'status.json',dict(status='time_limit',epochs=len(history),elapsed_seconds=time.monotonic()-start,smoke=args.epochs is not None and args.epochs<10,partial_epoch=True,test_evaluated=False))
                print('Time limit: checkpoint and exact minibatch/RNG position saved',flush=True)
                return
        # Evaluation uses a fixed independent RNG state, without perturbing training.
        state=torch.get_rng_state(); cuda_state=torch.cuda.get_rng_state_all() if device=='cuda' else []
        try:
            try: metrics,_=evaluate(model,raw['val'],device,32,args.val_limit)
            finally:
                torch.set_rng_state(state)
                if device=='cuda': torch.cuda.set_rng_state_all(cuda_state)
        except InterruptedError:
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history,torch_rng=torch.get_rng_state(),cuda_rng=cuda_state,numpy_rng=np.random.get_state(),python_rng=random.getstate(),shuffle_rng=rng.bit_generator.state,config=vars(args),status='interrupted',partial_epoch=True,order=order,next_batch=len(order),losses=losses)
            torch.save(checkpoint,path/'last.pt')
            save_json(path/'status.json',dict(status='interrupted',epochs=len(history),smoke=False,partial_epoch=True,test_evaluated=False))
            return
        if getattr(args,'data','synthetic')=='real':
            validation=metrics.assign(dataset=raw['val']['dataset'][:len(metrics)],ncrps=metrics.crps.to_numpy()/raw['val']['scale'][:len(metrics),0])
            score=validation.groupby('dataset').ncrps.mean().mean()
        elif 'block' in raw['val']:
            metrics['block']=raw['val']['block'][:len(metrics)]
            score=metrics.groupby('block').crps.mean().mean()
        else: score=metrics.crps.mean()
        improved=score<best
        if improved: best=score; stale=0
        else: stale+=1
        record=dict(epoch=epoch,loss=np.mean(losses),validation_crps=score,seconds=time.monotonic()-tick,status=status); history.append(record)
        pd.DataFrame(history).to_csv(path/'history.csv',index=False)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,stale=stale,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],numpy_rng=np.random.get_state(),python_rng=random.getstate(),shuffle_rng=rng.bit_generator.state,config=vars(args),status=status)
        torch.save(checkpoint,path/'last.pt')
        if improved: torch.save(checkpoint,path/'best.pt')
        print(json.dumps(record),flush=True)
        if status=='time_limit' or stale>=10: break
    model.load_state_dict(torch.load(path/'best.pt',weights_only=False,map_location=device)['model'])
    metrics,outputs=evaluate(model,raw['test'],device,128,args.test_limit)
    metrics['arm']=args.arm; metrics['seed']=args.seed
    if 'scenario' in raw['test']:
        if 'block' in raw['test']:
            metrics['block']=raw['test']['block'][:len(metrics)]
            metrics['scenario']=metrics['block']
            metrics['noise_level']=[float(np.mean(raw['test']['variance'][int(i),:T]))**.5 for i in metrics.series_id]
        else:
            metrics['noise_level']=[SCENARIOS[int(raw['test']['scenario'][i])][2] for i in metrics.series_id]
            metrics['scenario']=[json.dumps(SCENARIOS[int(raw['test']['scenario'][i])]) for i in metrics.series_id]
    else:
        metrics['dataset']=raw['test']['dataset'][:len(metrics)]
        metrics['ncrps']=metrics.crps.to_numpy()/raw['test']['scale'][:len(metrics),0]
        metrics['nrmse']=metrics.rmse.to_numpy()/raw['test']['scale'][:len(metrics),0]
    metrics.to_csv(path/'fm_rows.csv',index=False); np.savez_compressed(path/'forecast_outputs.npz',outputs=np.array(outputs,dtype=object))
    if args.real and getattr(args,'data','synthetic')!='real':
        records=real_windows('test',args.n); real=dict(x=np.stack([r['x'] for r in records])); z,m,s=normalize(real['x']); real.update(z=z.astype('float32'),mean=m,scale=s)
        if args.arm in FOURIER_ARMS: attach(real,path,'real')
        elif args.arm in 'CD': real['components']=cache_components(z,path/'real_period.npz')
        df,outputs=evaluate(model,real,device,128)
        for k in ('dataset','series_id','start','split','source'): df[k]=[r[k] for r in records]
        df.to_csv(path/'real_fm_rows.csv',index=False); np.savez_compressed(path/'real_forecast_outputs.npz',outputs=np.array(outputs,dtype=object),data=real['x'])
        save_json(path/'real_windows.json',[{k:v for k,v in r.items() if k!='x'} for r in records])
    save_json(path/'status.json',dict(status=status,epochs=len(history),elapsed_seconds=time.monotonic()-start,stale=stale,early_stopped=stale>=10,smoke=args.epochs is not None and args.epochs<10,partial_validation=args.val_limit,partial_test=args.test_limit))
    report(path)


def report(path):
    texts=['# Decomposition benchmark','', 'Synthetic component truth and real forecast diagnostics are reported separately. Real variance truth is unavailable; this does not establish correct noise estimation.','']
    for filename,groups in [('algorithm_rows.csv',['dataset','method']),('fm_rows.csv',['arm','noise_level']),('real_fm_rows.csv',['dataset'])]:
        if not (path/filename).exists(): continue
        df=pd.read_csv(path/filename); numeric=df.select_dtypes('number').columns.difference(['series_id','seed','noise_level','stress'])
        if 'noise_level' not in df and filename=='fm_rows.csv': groups=['arm','dataset']
        table=df.groupby(groups)[numeric].agg(['mean','median',lambda v:v.quantile(.9)])
        table.to_csv(path/(filename.replace('_rows','_summary')))
        texts+=['## '+filename,'',df.groupby(groups)[[c for c in ('nrmse','rmse','crps','coverage90','width90','variance_mae','noise_nrmse') if c in df]].mean().to_string(), '']
    (path/'report.md').write_text('\n'.join(texts))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['validate','generate','generate-diagnostics','benchmark','train','report']); p.add_argument('--profile',choices=['pilot','night','v2pilot','v2night'],default='pilot'); p.add_argument('--arm',choices=list('ABCD')+list(FOURIER_ARMS),default='A'); p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cuda'); p.add_argument('--real',action='store_true'); p.add_argument('--n',type=int,default=32); p.add_argument('--epochs',type=int); p.add_argument('--minutes',type=float,default=120); p.add_argument('--val-limit',type=int); p.add_argument('--test-limit',type=int); p.add_argument('--resume'); p.add_argument('--run-id'); p.add_argument("--data",choices=["synthetic","real"],default="synthetic"); args=p.parse_args()
    if args.command=='validate':
        import pytest
        raise SystemExit(pytest.main([str(Path(__file__).parent/'test_checks.py'),'-q']))
    if args.command=='generate': print(generate_real() if args.data=='real' else generate(args.profile)); return
    if args.command=='generate-diagnostics': print(generate_diagnostics(n_per_case=args.n,seed=307190)); return
    run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+args.command+'_'+args.arm+f'_s{args.seed}'
    path=OUT/run_id
    if args.command=='report': report(path); return
    provenance(path,args)
    if args.command=='benchmark': benchmark(args,path)
    elif args.command=='train':
        try: train(args,path)
        except InterruptedError:
            save_json(path/'status.json',dict(status='interrupted',smoke=False,test_evaluated=False))

if __name__=='__main__': main()
