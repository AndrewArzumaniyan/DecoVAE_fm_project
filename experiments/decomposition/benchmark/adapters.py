"""Adapters preserve the original learned seasonal decoder in all FM arms."""
import sys
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from data import ROOT,PERIOD_DECOMP_DIR,T,L
sys.path.insert(0,str(PERIOD_DECOMP_DIR))
import period_decomp as pdm
import baselines
sys.path.insert(0,str(ROOT))
from decovaefm.models.decovae_fm import DecoVAEFM, _build_mlp

@lru_cache(maxsize=8)
def context_decomposition(values):
    return pdm.decompose(np.frombuffer(values,dtype=np.float64))

def algorithm(x,method,oracle=None):
    if method=='original':
        d=baselines.original_decompose(x,L,k_top=16)
        return d,np.full(L,d['trend'][-1])+d['seasonal_horizon'],[]
    d=context_decomposition(np.asarray(x,dtype=np.float64).tobytes())
    periods=[s.period for s in d.seasons]
    if method.startswith('stl'):
        p=oracle if method=='stl_oracle' else periods
        result=baselines.stl_decompose(x,L,p)
        if result is None:
            trend=pdm.hp_filter(x,pdm.hp_lambda(T/8))
            result=dict(trend=trend,seasonal=np.zeros(T),noise=x-trend,seasonal_horizon=np.zeros(L))
        return result,np.full(L,result['trend'][-1])+result['seasonal_horizon'],p
    if method=='fourier_ar':
        time=np.arange(T+L,dtype=float)
        design=np.column_stack([np.ones(T+L),time]+[pdm.fourier_design(time,s.period,s.n_harm) for s in d.seasons])
        coeff=np.linalg.lstsq(design[:T],x,rcond=None)[0]
        trend=design[:,:2]@coeff[:2]; seasonal=design[:,2:]@coeff[2:]
        residual=x-trend[:T]-seasonal[:T]
        forecast=trend[T:]+seasonal[T:]+ar_forecast(residual,False)
        return dict(trend=trend[:T],seasonal=seasonal[:T],noise=residual),forecast,periods
    return dict(trend=d.trend,seasonal=d.seasonal,noise=d.noise),d.trend[-1]+d.seasonal_at(np.arange(T,T+L)),periods

def ar_forecast(x,linear=True):
    from statsmodels.tsa.ar_model import AutoReg
    fits=[AutoReg(x,lags=p,trend='ct' if linear else 'c',hold_back=20).fit() for p in (1,2,5,10,20)]
    fit=min(fits,key=lambda a:a.bic)
    return fit.predict(len(x),len(x)+L-1)

class Arm(DecoVAEFM):
    def __init__(self,arm):
        super().__init__(pred_len=L,d_model=64,patch_size=16,stride=16,n_heads=4,enc_layers=2,dec_layers=2,k_top=16)
        self.arm=arm
        if arm in 'CD': del self.decomposition
        if arm in 'BD':
            del self.noise_encoder
            self.logvar_decoder=_build_mlp(3*64,64,16,2)
            nn.init.zeros_(self.logvar_decoder[-1].bias)
            with torch.no_grad(): self.logvar_decoder[-1].weight.mul_(.01)
    def encode(self,x,n_target=0):
        if self.arm in 'AB':
            enc=super().encode(x,n_target) if self.arm=='A' else self._encode(x,n_target)
            return enc
        return self._encode(x,n_target)
    def _encode(self,x,n_target):
        if self.arm in 'CD':
            # Precomputed deterministic components, attached by the runner for this batch.
            trend,seasonal=self.components
        else: trend,seasonal=self.decomposition(x)
        tm,tl=self.trend_encoder(trend,n_target=n_target)
        sm,sl,rec,idx=self.seasonal_encoder(seasonal)
        # C/D use the complete period-aware seasonal estimate for the residual;
        # sparse rec is only the seasonal encoder representation.
        target=seasonal if self.arm in 'CD' else rec
        enc=dict(trend_mu=tm,trend_logvar=tl,seasonal_mu=sm,seasonal_logvar=sl,topk_idx=idx,target_trend=trend,target_seasonal=target)
        if self.arm in 'AC': enc['noise_mu'],enc['noise_logvar']=self.noise_encoder(x-trend-target,n_target=n_target)
        return enc
    def forward(self,x,pred_len=None,noise_permutation=None):
        length=x.shape[-1]; e=self.encode(x,n_target=L//16)
        zt=self.reparameterize(e['trend_mu'],e['trend_logvar']); zs=self.reparameterize(e['seasonal_mu'],e['seasonal_logvar'])
        trend=self._fold_trend_patches(self.trend_decoder(zt),length+L)
        seasonal=self._reconstruct_time_from_freq_tokens(self.seasonal_decoder(zs),e['topk_idx'],length,length+L)
        pooled=torch.cat([zs.real,zs.imag],-1).mean(1,keepdim=True).expand(-1,zt.size(1),-1)
        inputs=[pooled,zt]
        if self.arm in 'AC':
            zn=self.reparameterize(e['noise_mu'],e['noise_logvar'])
            if noise_permutation is not None: zn=zn[noise_permutation]
            inputs.insert(0,zn)
        lv=self._fold_trend_patches(self.logvar_decoder(torch.cat(inputs,-1)),length+L).clamp(-20,10)
        mu=trend+seasonal
        return dict(e,recon_mu=mu[:,:length],recon_logvar=lv[:,:length],forecast_mu=mu[:,length:],forecast_logvar=lv[:,length:],decoded_trend=trend,decoded_seasonal=seasonal)
