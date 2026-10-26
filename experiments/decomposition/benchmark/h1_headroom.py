"""Before training: does the H1 data reward tracking the volatility state at all?

All forecasters use the true conditional mean; they differ only in the noise distribution:
oracle (simulated from the true state), Gaussian with the oracle variance, a constant sigma from the
whole context noise (the best a window-level sigma can do), a sigma from the last 32 noise values, and
the unconditional sigma. Headroom = relative CRPS gain of the oracle over the context-constant sigma.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from data import T,L
from h1_data import generate_h1,oracle_noise
from runner import OUT,save_json

BUCKETS=dict(h1=slice(0,32),h2=slice(32,64),h3=slice(64,96))


def crps_gauss(y,mu,sd):
    z=(y-mu)/sd
    return sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--profile',default='h1check'); p.add_argument('--paths',type=int,default=512); p.add_argument('--out',default='h1_headroom'); a=p.parse_args()
    cache=generate_h1(a.profile); raw=dict(np.load(cache/'test.npz')); rng=np.random.default_rng(7)
    y=raw['x'][:,T:].astype(float); mu=raw['forecast_mean'].astype(float); noise=raw['noise'].astype(float)
    scale=raw['x'][:,:T].std(1,keepdims=True)
    crps={}
    for i in range(len(y)):
        row=dict(state=json.loads(str(raw['state_json'][i])),variance=raw['variance'][i])
        samples=mu[i][None,:]+oracle_noise(row,a.paths,rng)
        s=np.sort(samples,0); n=len(s); w=(2*np.arange(1,n+1)-n-1)[:,None]
        crps.setdefault('oracle',[]).append(np.abs(samples-y[i]).mean(0)-(w*s).sum(0)/n**2)
    crps['oracle']=np.stack(crps['oracle'])
    sd=dict(gauss_oracle_var=np.sqrt(raw['forecast_variance'].astype(float)),
            context_const=np.sqrt((noise[:,:T]**2).mean(1,keepdims=True))*np.ones((1,L)),
            recent32=np.sqrt((noise[:,T-32:T]**2).mean(1,keepdims=True))*np.ones((1,L)),
            unconditional=np.sqrt(np.mean(raw['variance'][:,:T],1,keepdims=True))*np.ones((1,L)))
    for k,v in sd.items(): crps[k]=crps_gauss(y,mu,v)
    rows=[]
    for block in np.unique(raw['block']):
        m=raw['block']==block
        for bucket,sl in BUCKETS.items():
            rec=dict(block=block,bucket=bucket,n=int(m.sum()))
            for k,v in crps.items(): rec[k]=float((v[m][:,sl]/scale[m]).mean())
            rec['headroom_vs_context_const']=1-rec['oracle']/rec['context_const']
            rec['headroom_vs_recent32']=1-rec['oracle']/rec['recent32']
            lv=np.log(raw['forecast_variance'][m][:,sl].mean(1)); lc=np.log(sd['context_const'][m][:,0]**2)
            rec['corr_log_oracle_var_vs_context']=float(np.corrcoef(lv,lc)[0,1])
            rec['sd_log_oracle_var']=float(lv.std())
            rows.append(rec)
    df=pd.DataFrame(rows); out=OUT/a.out; out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'headroom.csv',index=False)
    save_json(out/'provenance.json',dict(cache=str(cache),paths=a.paths,crps_units='divided by context std of each row',
              forecasters={'oracle':'simulated from the true volatility state at T','gauss_oracle_var':'Gaussian, exact conditional variance',
                           'context_const':'Gaussian, mean square of true context noise','recent32':'Gaussian, last 32 true noise values','unconditional':'Gaussian, mean true context variance'}))
    pd.set_option('display.width',220); print(df.round(4).to_string(index=False))

if __name__=='__main__': main()
