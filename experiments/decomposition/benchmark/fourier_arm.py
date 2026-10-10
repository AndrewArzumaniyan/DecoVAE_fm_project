"""C'/D' (CLI arms Cf/Df): arm C/D with the original Top-K remap seasonal decoder replaced by a
continuous-frequency Fourier decoder at the per-window estimated periods.

seasonal(t) = sum_j sum_h (a_jh + da_jh) cos(2 pi h t / p_j) + (b_jh + db_jh) sin(2 pi h t / p_j), t in [0, T+L)
(a, b, p) come from period_decomp.decompose on the context; da/db = MLP(pooled z_s, log1p p_j) with a
zero-initialised last layer, so training starts exactly at the deterministic extrapolation.
Everything else (encoders, trend decoder, sigma head, loss) is unchanged from Arm.
"""
import hashlib
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import ROOT,PERIOD_DECOMP_DIR,T,L
from adapters import Arm,pdm,_build_mlp

H_MAX = 12          # largest entry of period_decomp.HARMONIC_GRID
N_SEASONS = 2       # period_decomp.EstimatorConfig.max_seasons
CACHE_VERSION = 'fourier-v1'
FOURIER_ARMS = ('Cf','Df')


def cache_tag():
    return hashlib.sha256((PERIOD_DECOMP_DIR/'period_decomp.py').read_bytes()+CACHE_VERSION.encode()).hexdigest()[:12]


def decompose_window(x):
    """Context components plus fixed-size period/coefficient arrays for one normalized window."""
    d=pdm.decompose(np.asarray(x,dtype=np.float64))
    periods=np.zeros(N_SEASONS); coefs=np.zeros((N_SEASONS,2*H_MAX))
    for j,(s,c) in enumerate(zip(d.seasons,d.coefs)):
        periods[j]=s.period; coefs[j,:len(c)]=c
    full=d.seasonal_at(np.arange(T+L))
    return d.trend,d.seasonal,periods,coefs,full


def fourier_components(z,cache_path,stop=lambda:False):
    """Resumable cache of trend[T], seasonal[T], periods[2], coefs[2,2H], seasonal_full[T+L]."""
    keys=('trend','seasonal','periods','coefs','seasonal_full')
    if cache_path.exists():
        data=np.load(cache_path)
        return {k:data[k] for k in keys}
    partial=cache_path.with_name(cache_path.stem+'.partial.npz')
    out={k:list(np.load(partial)[k]) for k in keys} if partial.exists() else {k:[] for k in keys}
    def save(path):
        temporary=path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temporary,**{k:np.stack(v).astype('float64' if k in ('periods','coefs') else 'float32') for k,v in out.items()})
        os.replace(temporary,path)
    start=time.monotonic()
    for i in range(len(out['trend']),len(z)):
        if stop():
            save(partial); raise InterruptedError('Fourier component cache interrupted; partial rows saved')
        for k,v in zip(keys,decompose_window(z[i,:T])): out[k].append(v)
        if (i+1)%256==0:
            save(partial); print(f'fourier cache {cache_path.name} {i+1}/{len(z)} {time.monotonic()-start:.0f}s',flush=True)
    save(cache_path)
    if partial.exists(): partial.unlink()
    return {k:np.stack(v) for k,v in out.items()}


def attach(raw,cache_dir,prefix,stop=lambda:False):
    """Add components (as used by arms C/D) and the Fourier inputs to a loaded split."""
    c=fourier_components(raw['z'],Path(cache_dir)/f'{prefix}_fourier_{cache_tag()}.npz',stop)
    raw['components']=np.stack([c['trend'],c['seasonal']],1).astype('float32')
    raw['periods']=c['periods']; raw['coefs']=c['coefs']; raw['seasonal_full']=c['seasonal_full']
    return raw


def harmonics(periods):
    """(B, S, H) harmonic numbers with a mask for h <= floor(p/2) and p > 0."""
    h=torch.arange(1,H_MAX+1,device=periods.device,dtype=periods.dtype)
    mask=(h[None,None,:]<=torch.floor(periods[...,None]/2-1e-9))&(periods[...,None]>0)
    return h,mask


def fourier_basis(periods,length):
    """(B, S, length, 2H) basis in the period_decomp layout [cos h1, sin h1, cos h2, ...].

    The phase h t / p is reduced modulo 1 in float64 before the trigonometric functions, so the
    long horizon does not lose frequency precision in float32.
    """
    p=periods.double(); h,mask=harmonics(p)
    t=torch.arange(length,device=p.device,dtype=torch.float64)
    cycles=h[None,None,None,:]*t[None,None,:,None]/p.clamp_min(1.)[...,None,None]
    angle=(2*math.pi*torch.remainder(cycles,1.)).float()
    m=mask[:,:,None,:].float()
    return torch.stack([torch.cos(angle)*m,torch.sin(angle)*m],-1).flatten(-2)


class FourierArm(Arm):
    def __init__(self,name):
        if name not in FOURIER_ARMS: raise ValueError(name)
        super().__init__(name[0])          # self.arm stays 'C'/'D' so every C/D branch still applies
        self.name=name; self.use_correction=True
        del self.seasonal_decoder          # unused: replaced by the Fourier decoder
        self.fourier_head=_build_mlp(2*64+1,64,2*H_MAX,2)
        nn.init.zeros_(self.fourier_head[-1].weight); nn.init.zeros_(self.fourier_head[-1].bias)
        self.fourier=None                  # (periods (B,S), coefs (B,S,2H)), attached per batch

    def seasonal_path(self,zs,length):
        periods,coefs=self.fourier
        pooled=torch.cat([zs.real,zs.imag],-1).mean(1)
        head_in=torch.cat([pooled[:,None,:].expand(-1,N_SEASONS,-1),torch.log1p(periods.float())[...,None]],-1)
        delta=self.fourier_head(head_in)
        if not self.use_correction: delta=torch.zeros_like(delta)
        _,mask=harmonics(periods.double())
        delta=delta*mask.repeat_interleave(2,-1).float()
        return torch.einsum('bsnk,bsk->bn',fourier_basis(periods,length),coefs.float()+delta)

    def forward(self,x,pred_len=None,noise_permutation=None):
        length=x.shape[-1]; e=self.encode(x,n_target=L//16)
        zt=self.reparameterize(e['trend_mu'],e['trend_logvar']); zs=self.reparameterize(e['seasonal_mu'],e['seasonal_logvar'])
        trend=self._fold_trend_patches(self.trend_decoder(zt),length+L)
        seasonal=self.seasonal_path(zs,length+L)
        pooled=torch.cat([zs.real,zs.imag],-1).mean(1,keepdim=True).expand(-1,zt.size(1),-1)
        inputs=[pooled,zt]
        if self.arm in 'AC':
            zn=self.reparameterize(e['noise_mu'],e['noise_logvar'])
            if noise_permutation is not None: zn=zn[noise_permutation]
            inputs.insert(0,zn)
        lv=self._fold_trend_patches(self.logvar_decoder(torch.cat(inputs,-1)),length+L).clamp(-20,10)
        mu=trend+seasonal
        return dict(e,recon_mu=mu[:,:length],recon_logvar=lv[:,:length],forecast_mu=mu[:,length:],forecast_logvar=lv[:,length:],decoded_trend=trend,decoded_seasonal=seasonal)


def make_model(arm):
    return FourierArm(arm) if arm in FOURIER_ARMS else Arm(arm)


def set_fourier(model,raw,idx,device):
    if isinstance(model,FourierArm):
        model.fourier=(torch.from_numpy(raw['periods'][idx]).to(device),torch.from_numpy(raw['coefs'][idx]).to(device))
