import numpy as np
import torch
from data import sample,sample_v2,diagnostic_sample,DIAGNOSTIC_CASES,normalize,T,L
from metrics import components,distribution
from adapters import Arm,algorithm
from runner import loss


def test_generator_and_truth():
    a=sample(5,('linear',24,.3,'iid')); b=sample(5,('linear',24,.3,'iid'))
    assert np.array_equal(a['x'],b['x'])
    assert np.allclose(a['x'],a['trend']+a['seasonal']+a['noise'])
    d={k:a[k][:T] for k in ('trend','seasonal','noise')}
    assert components(a,d)['noise_nrmse']==0
    corrupt=dict(d,seasonal=2*d['seasonal'])
    assert components(a,corrupt)['seasonal_nrmse']>0
    assert np.corrcoef(corrupt['seasonal'],d['seasonal'])[0,1]>.999
    moved=dict(d,trend=d['trend']+d['seasonal'],seasonal=np.zeros(T))
    assert np.allclose(sum(moved.values()),a['x'][:T])
    assert components(a,moved)['trend_nrmse']>0


def test_extended_diagnostic_truth():
    assert len(DIAGNOSTIC_CASES)==29
    for name in ('negative_noise_only','negative_trend_noise','negative_season_noise','ar_phi_08','sigma_unseen_future_jump','student_t_df5','gaussian_rare_outliers'):
        case=next(c for c in DIAGNOSTIC_CASES if c[0]==name)
        a=diagnostic_sample(1234,case); b=diagnostic_sample(1234,case)
        assert np.array_equal(a['x'],b['x'])
        assert np.allclose(a['x'],a['trend']+a['seasonal']+a['noise'])
        assert np.allclose(a['forecast_mean'],a['trend'][T:]+a['seasonal'][T:]+a['phi']**np.arange(1,L+1)*a['noise'][T-1])
        assert np.allclose(a['forecast_variance'],a['variance'][T:] if not a['phi'] else a['variance'][T]*np.cumsum(a['phi']**(2*np.arange(L))))
    noise=diagnostic_sample(7,next(c for c in DIAGNOSTIC_CASES if c[0]=='negative_noise_only'))
    assert np.all(noise['trend']==0) and np.all(noise['seasonal']==0)
    seasonal=diagnostic_sample(7,next(c for c in DIAGNOSTIC_CASES if c[0]=='negative_season_noise'))
    assert np.all(seasonal['trend']==0) and np.any(seasonal['seasonal']!=0)
    ar=sample_v2(21,'ar')
    assert np.isfinite(ar['variance']).all() and np.isfinite(ar['conditional']).all()


def test_normalization_and_future():
    r=sample(4,('kink',48,.1,'iid')); altered=r['x'].copy(); altered[T:]+=100
    z,m,s=normalize(r['x']); z2,m2,s2=normalize(altered)
    assert np.array_equal(m,m2) and np.array_equal(s,s2)
    for method in ('original','period'):
        a,_,_=algorithm(z[:T],method); b,_,_=algorithm(z2[:T],method)
        assert all(np.array_equal(a[k],b[k]) for k in ('trend','seasonal','noise'))
    assert np.isfinite(normalize(np.ones(T+L))[0]).all()
    for method in ('original','period','stl','fourier_ar'):
        d,forecast,_=algorithm(np.ones(T),method)
        assert np.isfinite(forecast).all()


def test_noise_off_and_backward():
    torch.set_num_threads(2)
    for arm in 'ABCD':
        torch.manual_seed(8); model=Arm(arm); x=torch.randn(2,T)
        if arm in 'CD': model.components=(x*.2,x*.5)
        out=model(x); value=loss(out,x,torch.randn(2,L)); value.backward()
        assert torch.isfinite(value)
        if arm in 'BD':
            assert not hasattr(model,'noise_encoder')
            assert not any(k.startswith('noise_') for k in out)
            assert model.logvar_decoder[0].in_features==192
        assert out['forecast_mu'].shape==(2,L)


def test_gaussian_distribution():
    rng=np.random.default_rng(2); y=rng.normal(size=(1000,L))*2
    samples=rng.normal(size=(128,1000,L))*2
    m=distribution(y,samples,observation_var=np.full_like(y,4),latent_var=np.zeros_like(y),true_var=np.full_like(y,4))
    assert abs(m['coverage90'].mean()-.9)<.025
    assert m['variance_mae'].max()==0
    assert abs(samples.var()-4)<.04


def test_exact_minibatch_resume(tmp_path,monkeypatch):
    from argparse import Namespace
    import pandas as pd
    import runner
    rng=np.random.default_rng(4); raw={'x':rng.normal(size=(64,T+L)).astype('float32')}
    z,m,s=normalize(raw['x']); raw.update(z=z.astype('float32'),mean=m,scale=s,scenario=np.zeros(64,dtype=int),variance=np.ones_like(raw['x']))
    cache=tmp_path/'cache'; cache.mkdir()
    monkeypatch.setattr(runner,'load_splits',lambda args:({'train':raw,'val':raw,'test':raw},cache))
    monkeypatch.setattr(runner,'evaluate',lambda *args:(pd.DataFrame({'series_id':[0],'crps':[1.]}),[]))
    full=tmp_path/'full'; resumed=tmp_path/'resumed'; full.mkdir(); resumed.mkdir()
    config=dict(device='cpu',seed=0,arm='B',profile='pilot',epochs=1,minutes=10,resume=None,val_limit=None,test_limit=None,real=False,n=0)
    runner.train(Namespace(**config),full)
    runner.train(Namespace(**dict(config,minutes=0)),resumed)
    cp=torch.load(resumed/'last.pt',weights_only=False)
    assert cp['partial_epoch'] and cp['next_batch']==32
    runner.train(Namespace(**dict(config,resume=str(resumed/'last.pt'))),resumed)
    expected=torch.load(full/'last.pt',weights_only=False)['model']; actual=torch.load(resumed/'last.pt',weights_only=False)['model']
    assert expected.keys()==actual.keys()
    assert all(torch.equal(expected[k],actual[k]) for k in expected)




def test_real_windows_do_not_retain_full_series(tmp_path,monkeypatch):
    import data
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/'ver0'/'test'; root.mkdir(parents=True)
    original=np.arange(100000,dtype='float32')
    table=pa.table({'dataset':['chosen','skip'],'series_id':['a','b'],'values':[original,original]})
    pq.write_table(table,root/'part-00000.parquet')
    monkeypatch.setattr(data,'VER0_DIR',tmp_path/'ver0')
    records=data.real_windows('test',1,datasets=('chosen',))
    assert len(records)==1 and records[0]['x'].shape==(T+L,)
    assert records[0]['x'].base is None
    assert records[0]['x'].nbytes==(T+L)*8


def test_fourier_decoder_continuous_frequency():
    from fourier_arm import fourier_basis,decompose_window,H_MAX
    t=np.arange(T+L)
    for periods in ((37.1,0.),(17.3,96.)):
        coefs=np.zeros((1,2,2*H_MAX)); truth=np.zeros(T+L)
        for j,p in enumerate(periods):
            if not p: continue
            coefs[0,j,:4]=[.3,.8,-.2,.1]
            truth+=.3*np.cos(2*np.pi*t/p)+.8*np.sin(2*np.pi*t/p)-.2*np.cos(4*np.pi*t/p)+.1*np.sin(4*np.pi*t/p)
        basis=fourier_basis(torch.tensor([periods],dtype=torch.float64),T+L)
        decoded=torch.einsum('bsnk,bsk->bn',basis,torch.tensor(coefs).float()).numpy()[0]
        assert np.abs(decoded-truth).max()<1e-4
    # Harmonics above p/2 and absent seasons contribute nothing.
    basis=fourier_basis(torch.tensor([[5.,0.]],dtype=torch.float64),T+L)
    assert basis[0,0,:,4:].abs().max()==0 and basis[0,1].abs().max()==0
    x=sample(11,('linear',24,.3,'iid'))['x']; z,_,_=normalize(x)
    trend,seasonal,periods,coefs,full=decompose_window(z[:T])
    assert np.allclose(full[:T],seasonal) and periods[0]>0
    d,_,_=algorithm(z[:T],'period')
    assert np.array_equal(d['trend'],trend) and np.array_equal(d['seasonal'],seasonal)


def test_fourier_arm_starts_at_deterministic_seasonal():
    from fourier_arm import FourierArm,decompose_window
    torch.set_num_threads(2)
    rows=[normalize(sample(20+i,('kink',17.3 if i else 24,.3,'iid'))['x'])[0] for i in range(2)]
    parts=[decompose_window(z[:T]) for z in rows]
    for arm in ('Cf','Df'):
        torch.manual_seed(3); model=FourierArm(arm); x=torch.tensor(np.stack(rows)[:,:T],dtype=torch.float32)
        model.components=(torch.tensor(np.stack([p[0] for p in parts]),dtype=torch.float32),torch.tensor(np.stack([p[1] for p in parts]),dtype=torch.float32))
        model.fourier=(torch.tensor(np.stack([p[2] for p in parts])),torch.tensor(np.stack([p[3] for p in parts])))
        out=model(x); value=loss(out,x,torch.randn(2,L)); value.backward()
        assert torch.isfinite(value) and out['forecast_mu'].shape==(2,L)
        assert np.abs(out['decoded_seasonal'].detach().numpy()-np.stack([p[4] for p in parts])).max()<1e-4
        assert model.fourier_head[-1].weight.grad.abs().sum()>0
        assert hasattr(model,'noise_encoder')==(arm=='Cf')
