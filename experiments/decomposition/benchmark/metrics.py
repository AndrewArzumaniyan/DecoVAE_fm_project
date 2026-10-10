import numpy as np
from data import T,L

def components(record,d):
    scale=max(record['x'][:T].std(),1e-8); result={}
    for name in ('trend','seasonal','noise'):
        err=np.asarray(d[name])-record[name][:T]
        result[name+'_nmae']=float(np.abs(err).mean()/scale)
        result[name+'_nrmse']=float(np.sqrt(np.mean(err**2))/scale)
        for edge,sl in [('left',slice(0,32)),('right',slice(-32,None))]: result[name+'_'+edge+'_nrmse']=float(np.sqrt(np.mean(err[sl]**2))/scale)
    result['trend_diff_nrmse']=float(np.sqrt(np.mean((np.diff(d['trend'])-np.diff(record['trend'][:T]))**2))/scale)
    result['false_seasonal_rms']=float(np.sqrt(np.mean(d['seasonal']**2))) if not record['params'] else np.nan
    result['false_trend_rms']=float(np.sqrt(np.mean(d['trend']**2))) if record['scenario'][0]=='none' else np.nan
    basis=[]
    for p,h,amp,phase in record['params']:
        t=np.arange(T); basis.extend([np.sin(2*np.pi*h*t/p),np.cos(2*np.pi*h*t/p)])
    if basis:
        X=np.array(basis).T; error=d['noise']-record['noise'][:T]
        proj=X@np.linalg.lstsq(X,error,rcond=None)[0]
        result['residual_seasonal_energy']=float(np.mean(proj**2)/scale**2)
        coefficients=np.linalg.lstsq(X,d['seasonal'],rcond=None)[0].reshape(-1,2)
        true=np.array(record['params'])
        amplitudes=np.sqrt(np.sum(coefficients**2,axis=1))
        phases=np.arctan2(coefficients[:,1],coefficients[:,0])
        result['seasonal_amplitude_nmae']=float(np.abs(amplitudes-true[:,2]).mean()/scale)
        result['seasonal_phase_mae']=float(np.abs(np.angle(np.exp(1j*(phases-true[:,3])))).mean())
    else: result['residual_seasonal_energy']=0.
    return result

def distribution(y,samples,mu=None,observation_var=None,latent_var=None,true_var=None):
    # Empirical CRPS in O(S log S), exact for the empirical mixture samples.
    s=np.sort(samples,axis=0); n=len(s)
    weights=(2*np.arange(1,n+1)-n-1)[:,None,None]
    crps=np.abs(samples-y).mean(0)-(weights*s).sum(0)/n**2
    mean=samples.mean(0) if mu is None else mu
    out={'crps':crps.mean(-1),'mae':np.abs(mean-y).mean(-1),'rmse':np.sqrt(np.mean((mean-y)**2,-1))}
    for c in (.5,.8,.9):
        low,high=np.quantile(samples,[(1-c)/2,(1+c)/2],axis=0)
        out[f'coverage{int(c*100)}']=((y>=low)&(y<=high)).mean(-1)
        out[f'width{int(c*100)}']=(high-low).mean(-1)
    for j,sl in enumerate((slice(0,32),slice(32,64),slice(64,96))):
        out[f'crps_h{j+1}']=crps[:,sl].mean(-1)
        low,high=np.quantile(samples, [.05,.95],axis=0)
        out[f'coverage90_h{j+1}']=((y[:,sl]>=low[:,sl])&(y[:,sl]<=high[:,sl])).mean(-1)
        out[f'width90_h{j+1}']=(high[:,sl]-low[:,sl]).mean(-1)
    if observation_var is not None:
        out['observation_variance']=observation_var.mean(-1)
        out['latent_mean_variance']=latent_var.mean(-1)
        if true_var is not None: out['variance_mae']=np.abs(observation_var-true_var).mean(-1)
    return out

def bootstrap(values,seed=991,n=2000):
    rng=np.random.default_rng(seed); v=np.asarray(values); means=[]
    for _ in range(n): means.append(rng.choice(v,len(v),replace=True).mean())
    return list(np.quantile(means,[.025,.975]))
