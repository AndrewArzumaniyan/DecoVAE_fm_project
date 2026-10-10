"""Independent synthetic streams; context-only affine normalization and real windows."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT as ROOT, PERIOD_DECOMP_DIR, BENCHMARK_DIR, VER0_DIR, RUNS_DIR  # noqa: E402,F401
T, L = 512, 96
SCENARIOS = [(t, p, s, mode) for t in ('none','linear','kink') for p in (0,12,24,48,96) for s in (.1,.3,.6) for mode in ('iid',)]
STRESS = [('linear', p, .3, 'iid') for p in (17.3,30.4,168)] + [('kink',24,.3,m) for m in ('ar05','ar08','dual','amplitude','smooth')]

V2_MIXTURE = {'iid': .40, 'seasonal': .20, 'heteroskedastic': .20, 'ar': .10, 'amplitude_drift': .10}

DIAGNOSTIC_CASES = (
    ('negative_noise_only','negative', 'noise'),
    ('negative_trend_noise','negative', 'trend'),
    ('negative_season_noise','negative', 'seasonal'),
    ('period_19_7','period',19.7), ('period_37_1','period',37.1), ('period_144','period',144.),
    ('dual_24_168_ratio_1','dual',(24.,168.,1.)), ('dual_24_168_ratio_3','dual',(24.,168.,3.)),
    ('dual_17_3_96_ratio_1','dual',(17.3,96.,1.)), ('dual_17_3_96_ratio_3','dual',(17.3,96.,3.)),
    ('amplitude_growth','amplitude',.5), ('amplitude_decay','amplitude',-.5),
    ('frequency_drift_plus_10','frequency',.10), ('frequency_drift_minus_10','frequency',-.10),
    ('trend_curve','trend','curve'), ('trend_kink_256','trend',256), ('trend_kink_448','trend',448),
    ('sigma_smooth_up','sigma','smooth_up'), ('sigma_smooth_down','sigma','smooth_down'),
    ('sigma_switch_256','sigma',('switch',256)), ('sigma_switch_448','sigma',('switch',448)),
    ('sigma_unseen_future_jump','sigma','future_jump'),
    ('ar_phi_02','ar',.2), ('ar_phi_05','ar',.5), ('ar_phi_08','ar',.8),
    ('student_t_df5','tail','student_t5'), ('gaussian_rare_outliers','tail','mixture'),
    ('combined_dual_ar05','combined','dual_ar05'), ('combined_amplitude_smooth_sigma','combined','amplitude_sigma'),
)

def diagnostic_sample(seed, case):
    """Generate one diagnostic row with explicit component and forecast truth."""
    name,kind,spec=case
    select,trng,srng,nrng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    n=T+L; t=np.arange(n,dtype=float); trend=np.zeros(n); seasonal=np.zeros(n); phi=0.; params=[]
    sigma=float(select.uniform(.22,.38)); sigma_path=np.full(n,sigma); tail='gaussian'
    p=None; freq_drift=0.; amp_drift=0.
    if kind=='negative':
        if spec=='trend': trend=select.uniform(-1,1)+select.uniform(-.005,.005)*t
        if spec=='seasonal': p=float(select.choice([17.3,24.,37.1,96.]))
    elif kind=='period': p=float(spec)
    elif kind=='dual': p1,p2,ratio=spec
    elif kind=='amplitude': p=float(select.choice([24.,48.,96.])); amp_drift=float(spec)
    elif kind=='frequency': p=float(select.choice([19.7,30.4,37.1])); freq_drift=float(spec)
    elif kind=='trend':
        if spec=='curve': trend=select.uniform(-.4,.4)+select.uniform(-.004,.004)*t+select.uniform(-.00002,.00002)*(t-T/2)**2
        else:
            knot=int(spec); trend=select.uniform(-1,1)+select.uniform(-.004,.004)*t+select.uniform(-.008,.008)*np.maximum(t-knot,0)
    elif kind=='sigma':
        if spec=='smooth_up': sigma_path=sigma*(.7+.6*t/(n-1))
        elif spec=='smooth_down': sigma_path=sigma*(1.3-.6*t/(n-1))
        elif isinstance(spec,tuple):
            knot=int(spec[1]); sigma_path=np.where(t<knot,.65*sigma,1.35*sigma)
        else:
            sigma_path[T:]=sigma*select.uniform(1.6,2.4)
    elif kind=='ar': phi=float(spec)
    elif kind=='tail': tail=str(spec)
    elif kind=='combined':
        if spec=='dual_ar05': p1,p2,ratio=24.,168.,1.; phi=.5
        else: p=float(select.choice([24.,48.,96.])); amp_drift=.5; sigma_path=sigma*(.7+.6*t/(n-1))
    if kind in ('period','amplitude','frequency') or (kind=='negative' and spec=='seasonal') or (kind=='combined' and spec=='amplitude_sigma'):
        p=float(p)
        phase=float(srng.uniform(-np.pi,np.pi)); amp=float(srng.uniform(.5,1.))
        if freq_drift:
            phase_t=2*np.pi/p*(t+freq_drift*t*t/(2*(n-1)))
        else: phase_t=2*np.pi*t/p
        envelope=1+amp_drift*t/(n-1)
        seasonal=amp*envelope*np.sin(phase_t+phase); params=[[p,1,amp,phase]]
    if kind=='dual' or (kind=='combined' and spec=='dual_ar05'):
        for j,period in enumerate((float(p1),float(p2))):
            amp=float(srng.uniform(.5,1.))*(1. if j==0 else 1./ratio); phase=float(srng.uniform(-np.pi,np.pi))
            seasonal += amp*np.sin(2*np.pi*t/period+phase); params.append([period,1,amp,phase])
    eps=nrng.normal(size=n)*sigma_path
    if tail=='student_t5': eps=nrng.standard_t(5,size=n)*sigma_path*np.sqrt(3/5)
    elif tail=='mixture':
        outlier=nrng.random(n)<.005; base_scale=np.sqrt((1-.005*25)/.995)
        eps=nrng.normal(size=n)*sigma_path*base_scale; eps[outlier]=nrng.normal(size=int(outlier.sum()))*sigma_path[outlier]*5.
    noise=eps.copy(); conditional=np.zeros(n)
    if phi:
        prev=0.; burn_sigma=float(np.mean(sigma_path[:T]))
        for value in nrng.normal(size=1024)*burn_sigma: prev=phi*prev+value
        for i in range(n): prev=phi*prev+eps[i]; noise[i]=prev
        conditional[1:]=phi*noise[:-1]
    forecast_mean=trend[T:]+seasonal[T:]+(phi**np.arange(1,L+1))*noise[T-1]
    innovation_var=sigma_path[T:]**2
    forecast_var=innovation_var if not phi else np.full(L,sigma_path[T]**2)*np.cumsum(phi**(2*np.arange(L)))
    return dict(case=name,x=trend+seasonal+noise,trend=trend,seasonal=seasonal,noise=noise,
                variance=sigma_path**2,conditional=conditional,phi=phi,params=params,
                forecast_mean=forecast_mean,forecast_variance=forecast_var,tail=tail)

def generate_diagnostics(n_per_case=64,seed=307190):
    """Create the held-out, one-mechanism-at-a-time diagnostic suite."""
    cfg=dict(profile='diagnostics_v1',n_per_case=n_per_case,seed=seed,T=T,L=L,cases=DIAGNOSTIC_CASES)
    digest=hashlib.sha256((json.dumps(cfg,sort_keys=True)+Path(__file__).read_text()).encode()).hexdigest()[:16]
    path=Path(__file__).parent/'cache'/('diagnostics_'+digest); path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(cfg,indent=2))
    target=path/'test.npz'
    if not target.exists():
        records=[diagnostic_sample(seed+ci*100000+i,case) for ci,case in enumerate(DIAGNOSTIC_CASES) for i in range(n_per_case)]
        arrays={key:np.stack([r[key] for r in records]).astype('float32') for key in ('x','trend','seasonal','noise','variance','conditional','forecast_mean','forecast_variance')}
        arrays['phi']=np.asarray([r['phi'] for r in records],dtype='float32')
        arrays['case']=np.asarray([r['case'] for r in records],dtype='U64')
        arrays['tail']=np.asarray([r['tail'] for r in records],dtype='U16')
        arrays['params_json']=np.asarray([json.dumps(r['params']) for r in records],dtype='U512')
        np.savez_compressed(target,**arrays)
    return path

def sample_v2(seed, block):
    """One synthetic_v2 row. Component and noise RNG streams are independent."""
    sel, trng, srng, nrng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    t = np.arange(T+L, dtype=float); trend=np.zeros(T+L); seasonal=np.zeros(T+L)
    params=[]; phi=0.; kind=block
    if block == 'iid':
        trend_kind=sel.choice(['none','linear','kink']); period=int(sel.choice([0,12,24,48,96])); sigma=float(sel.choice([.1,.3,.6]))
        base=sample(seed, (trend_kind,period,sigma,'iid'))
        return dict(base, block=kind)
    if block == 'seasonal':
        dual=bool(sel.integers(0,2)); periods=([24,168] if dual else [17.3,30.4,168,19.7,37.1,144])
        if not dual: periods=[float(sel.choice(periods))]
        ratio=float(sel.choice([1.,3.])); amps=[]
        for j,p in enumerate(periods):
            amp=(1. if j==0 else 1./ratio)*float(srng.uniform(.45,.9)); phase=float(srng.uniform(-np.pi,np.pi))
            seasonal += amp*np.sin(2*np.pi*t/p+phase); params.append([p,1,amp,phase]); amps.append(amp)
        if sel.integers(0,2): trend=sel.uniform(-1,1)+sel.uniform(-.005,.005)*t
        sigma=float(sel.uniform(.2,.45))
    elif block == 'heteroskedastic':
        sigma0=float(sel.uniform(.2,.4)); slope=float(sel.uniform(-.45,.45))
        sigma=sigma0*(1+slope*(t/(T+L)-.5)); sigma=np.maximum(sigma,sigma0*.3)
        if sel.integers(0,2): seasonal=.7*np.sin(2*np.pi*t/24+srng.uniform(-np.pi,np.pi))
        if sel.integers(0,2): trend=sel.uniform(-1,1)+sel.uniform(-.005,.005)*t
    elif block == 'ar':
        phi=float(sel.choice([.2,.5,.8])); sigma=float(sel.uniform(.2,.4))
        if sel.integers(0,2): trend=sel.uniform(-1,1)+sel.uniform(-.005,.005)*t
        if sel.integers(0,2): seasonal=.7*np.sin(2*np.pi*t/24+srng.uniform(-np.pi,np.pi))
    else:
        amp=float(sel.uniform(.4,1.)); phase=float(srng.uniform(-np.pi,np.pi)); p=float(sel.choice([24,48,96]))
        seasonal=amp*(1+.5*t/(T+L))*np.sin(2*np.pi*t/p+phase); params=[[p,1,amp,phase]]
        sigma=float(sel.uniform(.2,.4))
        if sel.integers(0,2): trend=sel.uniform(-1,1)+sel.uniform(-.005,.005)*t
    std=np.full(T+L,sigma) if np.isscalar(sigma) else sigma
    eps=nrng.normal(size=T+L)*std; noise=eps.copy(); conditional=np.zeros(T+L)
    if phi:
        prev=0.
        for value in nrng.normal(size=1024)*float(np.mean(std)): prev=phi*prev+value
        for i in range(T+L): prev=phi*prev+eps[i]; noise[i]=prev
        conditional[1:]=phi*noise[:-1]
    return dict(x=trend+seasonal+noise,trend=trend,seasonal=seasonal,noise=noise,
                variance=(std**2 if not phi else np.full(T+L,np.mean(std)**2)),
                conditional=conditional,params=params,phi=phi,scenario=(kind,0,float(np.mean(std)),'iid'),block=kind)

def sample(seed, scenario):
    tr, period, sigma, mode = scenario
    a,b,c = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    t = np.arange(T+L, dtype=float)
    trend = np.zeros(T+L)
    if tr != 'none':
        trend = a.uniform(-1,1) + a.uniform(-.005,.005)*t
        if tr == 'kink':
            trend += a.uniform(-.008,.008)*np.maximum(t-a.integers(128,480),0)
    seas = np.zeros(T+L); basis = []; params=[]
    if period:
        for p in ([period,168] if mode == 'dual' else [period]):
            hmax = min(int(b.integers(1,5)),int(p//2)-1)
            for h in range(1,hmax+1):
                amp, phase = b.uniform(.3,1.5)/h, b.uniform(-np.pi,np.pi)
                w = 2*np.pi*h*t/p
                component=amp*np.sin(w+phase)
                if mode == 'amplitude': component *= 1+.5*t/(T+L)
                seas += component
                basis.extend([np.sin(w),np.cos(w)]); params.append([p,h,amp,phase])
    std = sigma*(1+.5*np.sin(2*np.pi*t/128)) if mode == 'smooth' else np.full(T+L,sigma)
    phi = {'ar05':.5,'ar08':.8}.get(mode,0)
    noise=c.normal(size=T+L)*std
    if phi:
        prev=0.
        for v in c.normal(size=1024)*sigma: prev=phi*prev+v
        for i in range(T+L): prev=phi*prev+noise[i]; noise[i]=prev
    conditional = np.zeros(T+L)
    if phi: conditional[1:]=phi*noise[:-1]
    return dict(x=trend+seas+noise, trend=trend, seasonal=seas, noise=noise, variance=std**2, conditional=conditional, params=params, phi=phi, scenario=scenario)

def normalize(x):
    mean=x[..., :T].mean(axis=-1,keepdims=True)
    scale=x[..., :T].std(axis=-1,keepdims=True)
    scale=np.where(scale<1e-8,1.,scale)
    return (x-mean)/scale,mean,scale

def generate(profile='pilot'):
    sizes={'pilot':(2048,256,512),'night':(8192,1024,2048),
           'v2pilot':(4096,512,1024),'v2night':(8192,1024,2048)}[profile]
    cfg={'profile':profile,'sizes':sizes,'T':T,'L':L,'scenarios':SCENARIOS,'seed':719,
         'v2_mixture':V2_MIXTURE if profile.startswith('v2') else None}
    digest=hashlib.sha256((json.dumps(cfg,sort_keys=True)+Path(__file__).read_text()).encode()).hexdigest()[:16]
    path=Path(__file__).parent/'cache'/digest
    path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(cfg,indent=2))
    for split,n in zip(('train','val','test'),sizes):
        target=path/f'{split}.npz'
        if target.exists(): continue
        split_seed=719+100000*('train','val','test').index(split)
        if profile.startswith('v2'):
            labels=np.random.default_rng(split_seed).choice(list(V2_MIXTURE),size=n,p=list(V2_MIXTURE.values()))
            records=[sample_v2(split_seed+i,str(labels[i])) for i in range(n)]
        else:
            labels=np.array([i%len(SCENARIOS) for i in range(n)])
            records=[sample(split_seed+i,SCENARIOS[int(labels[i])]) for i in range(n)]
        arrays={k:np.stack([r[k] for r in records]).astype('float32') for k in ('x','trend','seasonal','noise','variance','conditional')}
        arrays['phi']=np.asarray([r['phi'] for r in records],dtype='float32')
        arrays['scenario']=labels.astype('U32') if profile.startswith('v2') else labels
        if profile.startswith('v2'):
            arrays['block']=labels.astype('U32')
            arrays['params_json']=np.asarray([json.dumps(r['params']) for r in records],dtype='U512')
            arrays['has_trend']=np.asarray([np.any(np.abs(r['trend'])>1e-12) for r in records],dtype=bool)
        np.savez_compressed(target,**arrays)
    return path

def real_windows(split, n_per=32, datasets=('Electricity','London_Smart_Meters','Q-TRAFFIC','Solar','M4_Daily'), seed=719, stop=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng=np.random.default_rng(seed); counts={d:0 for d in datasets}; out=[]
    for path in sorted((VER0_DIR/split).glob('*.parquet')):
        if stop and stop(): raise InterruptedError('Real data preparation stopped')
        pf=pq.ParquetFile(path,pre_buffer=False)
        present=set(pf.read(columns=['dataset']).column('dataset').to_pylist())
        if not any(ds in present and count<n_per for ds,count in counts.items()): continue
        for batch in pf.iter_batches(batch_size=8,columns=['dataset','series_id','values'],use_threads=False):
            if stop and stop(): raise InterruptedError('Real data preparation stopped')
            # Do not materialize list<float> as Python floats for the whole batch.
            names=batch.column(0).to_pylist(); identities=batch.column(1).to_pylist()
            for j,ds in enumerate(names):
                if ds not in counts or counts[ds]>=n_per: continue
                x=np.asarray(batch.column(2)[j].values.to_numpy(zero_copy_only=False),dtype=float)
                if len(x)<T+L: continue
                start=int(rng.integers(0,len(x)-T-L+1)); w=x[start:start+T+L]
                if not np.isfinite(w).all() or w[:T].std()<1e-8: continue
                out.append(dict(x=w.copy(),dataset=ds,series_id=identities[j],start=start,split=split,source=str(path.relative_to(VER0_DIR))))
                counts[ds]+=1
            if all(v>=n_per for v in counts.values()):
                batch=None; x=None; pf=None
                pa.default_memory_pool().release_unused()
                return out
        batch=None; x=None; pf=None
        pa.default_memory_pool().release_unused()
    return out

def generate_real(stop=None):
    cfg={'per_dataset':{'train':128,'val':32,'test':32},'seed':719,'datasets':['Electricity','London_Smart_Meters','Q-TRAFFIC','Solar','M4_Daily'],'T':T,'L':L}
    digest=hashlib.sha256((json.dumps(cfg,sort_keys=True)+Path(__file__).read_text()+(VER0_DIR/'manifest.json').read_text()).encode()).hexdigest()[:16]
    path=Path(__file__).parent/'cache'/('real_'+digest); path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(cfg,indent=2)); identities={}
    for split,n in cfg['per_dataset'].items():
        target=path/f'{split}.npz'; index=path/f'{split}_windows.json'
        if not target.exists():
            records=real_windows(split,n,datasets=tuple(cfg['datasets']),stop=stop)
            np.savez_compressed(target,x=np.stack([r['x'] for r in records]).astype('float32'),dataset=np.array([r['dataset'] for r in records]))
            index.write_text(json.dumps([{k:v for k,v in r.items() if k!='x'} for r in records],indent=2))
        identities[split]={(r['dataset'],r['series_id']) for r in json.loads(index.read_text())}
    for left,right in [('train','val'),('train','test'),('val','test')]:
        if identities[left]&identities[right]: raise ValueError(f'Series overlap across {left}/{right}')
    return path
