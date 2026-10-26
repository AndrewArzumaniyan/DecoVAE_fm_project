"""H1 synthetic data: noise volatility that is independent of the level, trend, and seasonality.

Blocks: garch, sv (log-AR(1) stochastic volatility), regime (two-state Markov sigma), homo (constant
sigma; negative control), level (sigma a deterministic function of the level; H1 predicts no gain).
Every row stores the latent volatility state at T, so oracle predictive samples can be simulated.
Kept apart from data.py so existing cache digests do not change.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from data import T,L

H1_BLOCKS=('garch','sv','regime','homo','level')
H1_MIXTURE={'garch':.25,'sv':.25,'regime':.25,'homo':.125,'level':.125}
BURN=1000


def _structure(sel,srng,t):
    trend=np.zeros(len(t)); seasonal=np.zeros(len(t)); params=[]
    if sel.random()<.5: trend=sel.uniform(-1,1)+sel.uniform(-.004,.004)*t
    if sel.random()<.7:
        p=float(sel.choice([17.3,24.,37.1,48.,96.])); amp=float(srng.uniform(.4,.9)); phase=float(srng.uniform(-np.pi,np.pi))
        seasonal=amp*np.sin(2*np.pi*t/p+phase); params.append([p,1,amp,phase])
    return trend,seasonal,params


def _garch_path(rng,n,omega,alpha,beta,z):
    """Variance path s2[t] (variance of eps[t]) and eps; s2[n] is the next-step variance."""
    s2=np.empty(n+1); eps=np.empty(n); var=omega/(1-alpha-beta)
    for _ in range(BURN):
        e=np.sqrt(var)*rng.normal(); var=omega+alpha*e*e+beta*var
    s2[0]=var
    for i in range(n):
        eps[i]=np.sqrt(s2[i])*z[i]; s2[i+1]=omega+alpha*eps[i]**2+beta*s2[i]
    return s2,eps


STRENGTH=dict(base=dict(garch_ab=(.05,.15,.80,.90,.97),sv=(.97,.25),regime=(.6,1.4,.98,.995)),
              strong=dict(garch_ab=(.10,.25,.72,.88,.99),sv=(.985,.40),regime=(.4,1.8,.99,.998)))


def sample_h1(seed,block,strength='base'):
    sel,srng,nrng,vrng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    n=T+L; t=np.arange(n,dtype=float)
    trend,seasonal,params=_structure(sel,srng,t)
    sigma_bar=float(sel.uniform(.2,.4)); z=nrng.normal(size=n); state={}; cfg=STRENGTH[strength]
    if block=='garch':
        a0,a1,b0,b1,cap=cfg['garch_ab']; alpha=float(sel.uniform(a0,a1)); beta=float(sel.uniform(b0,b1)); beta=min(beta,cap-alpha)
        omega=sigma_bar**2*(1-alpha-beta)
        s2,eps=_garch_path(vrng,n,omega,alpha,beta,z); sigma=np.sqrt(s2[:n])
        state=dict(kind='garch',omega=omega,alpha=alpha,beta=beta,next_var=float(s2[T]),eps_T=float(eps[T-1]))
    elif block=='sv':
        phi,eta=cfg['sv']; mu=np.log(sigma_bar**2)-.5*eta**2/(1-phi**2)
        h=mu+vrng.normal()*eta/np.sqrt(1-phi**2); path=np.empty(n)
        for i in range(n): h=mu+phi*(h-mu)+eta*vrng.normal(); path[i]=h
        sigma=np.exp(path/2); eps=sigma*z
        state=dict(kind='sv',mu=float(mu),phi=phi,eta=eta,h_T=float(path[T-1]))
    elif block=='regime':
        lo,hi,s0,s1=cfg['regime']; stay=float(sel.uniform(s0,s1)); levels=np.sqrt(2/(lo**2+hi**2))*np.array([lo,hi])*sigma_bar
        s=int(vrng.integers(0,2)); states=np.empty(n,dtype=int)
        for i in range(n):
            if vrng.random()>stay: s=1-s
            states[i]=s
        sigma=levels[states]; eps=sigma*z
        state=dict(kind='regime',stay=stay,levels=levels.tolist(),state_T=int(states[T-1]))
    elif block=='homo':
        sigma=np.full(n,sigma_bar); eps=sigma*z; state=dict(kind='homo',sigma=sigma_bar)
    elif block=='level':
        if not np.any(trend): trend=sel.uniform(-1,1)+sel.uniform(-.004,.004)*t
        level=trend+seasonal; u=(level-level[:T].min())/max(np.ptp(level[:T]),1e-8)
        sigma=sigma_bar*(.5+np.clip(u,-.4,1.6)); eps=sigma*z; state=dict(kind='level')
    else: raise ValueError(block)
    return dict(x=trend+seasonal+eps,trend=trend,seasonal=seasonal,noise=eps,variance=sigma**2,params=params,
                block=block,sigma_bar=sigma_bar,state=state,forecast_mean=trend[T:]+seasonal[T:])


def oracle_noise(row,n_paths,rng):
    """Future noise paths (n_paths, L) from the true latent volatility state at T."""
    st=row['state']; z=rng.normal(size=(n_paths,L))
    if st['kind']=='garch':
        var=np.full(n_paths,st['next_var']); out=np.empty((n_paths,L))
        for k in range(L):
            out[:,k]=np.sqrt(var)*z[:,k]; var=st['omega']+st['alpha']*out[:,k]**2+st['beta']*var
        return out
    if st['kind']=='sv':
        h=np.full(n_paths,st['h_T']); out=np.empty((n_paths,L))
        for k in range(L):
            h=st['mu']+st['phi']*(h-st['mu'])+st['eta']*rng.normal(size=n_paths); out[:,k]=np.exp(h/2)*z[:,k]
        return out
    if st['kind']=='regime':
        s=np.full(n_paths,st['state_T']); levels=np.array(st['levels']); out=np.empty((n_paths,L))
        for k in range(L):
            s=np.where(rng.random(n_paths)>st['stay'],1-s,s); out[:,k]=levels[s]*z[:,k]
        return out
    return np.sqrt(row['variance'][T:])[None,:]*z


def oracle_variance(row):
    """Exact conditional variance of the future noise given the true state at T."""
    st=row['state']; k=np.arange(L)
    if st['kind']=='garch':
        bar=st['omega']/(1-st['alpha']-st['beta'])
        return bar+(st['alpha']+st['beta'])**k*(st['next_var']-bar)
    if st['kind']=='sv':
        kk=k+1; m=st['mu']+st['phi']**kk*(st['h_T']-st['mu']); v=st['eta']**2*(1-st['phi']**(2*kk))/(1-st['phi']**2)
        return np.exp(m+v/2)
    if st['kind']=='regime':
        lo,hi=np.array(st['levels'])**2; d=2*st['stay']-1; p_hi=.5+(st['state_T']-.5)*d**(k+1)
        return p_hi*hi+(1-p_hi)*lo
    return row['variance'][T:]


def generate_h1(profile='h1pilot'):
    sizes={'h1pilot':(4096,512,1024),'h1night':(8192,1024,2048),'h1check':(0,0,1024)}[profile.split('_')[0]]
    strength=profile.split('_')[1] if '_' in profile else 'base'
    cfg=dict(profile=profile,sizes=sizes,T=T,L=L,mixture=H1_MIXTURE,seed=4242,strength=strength)
    digest=hashlib.sha256((json.dumps(cfg,sort_keys=True)+Path(__file__).read_text()).encode()).hexdigest()[:16]
    path=Path(__file__).parent/'cache'/('h1_'+digest); path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(cfg,indent=2))
    for split,size in zip(('train','val','test'),sizes):
        target=path/f'{split}.npz'
        if target.exists() or not size: continue
        split_seed=4242+100000*('train','val','test').index(split)
        labels=np.random.default_rng(split_seed).choice(list(H1_MIXTURE),size=size,p=list(H1_MIXTURE.values()))
        rows=[sample_h1(split_seed+i,str(b),strength) for i,b in enumerate(labels)]
        arrays={k:np.stack([r[k] for r in rows]).astype('float32') for k in ('x','trend','seasonal','noise','variance','forecast_mean')}
        arrays['forecast_variance']=np.stack([oracle_variance(r) for r in rows]).astype('float32')
        arrays['block']=labels.astype('U16'); arrays['phi']=np.zeros(size,dtype='float32')
        arrays['params_json']=np.asarray([json.dumps(r['params']) for r in rows],dtype='U512')
        arrays['state_json']=np.asarray([json.dumps(r['state']) for r in rows],dtype='U512')
        arrays['has_trend']=np.asarray([np.any(np.abs(r['trend'])>1e-12) for r in rows],dtype=bool)
        np.savez_compressed(target,**arrays)
    return path
