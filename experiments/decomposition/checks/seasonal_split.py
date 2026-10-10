import math, torch
T, L, K = 512, 96, 8
t = torch.arange(T + L, dtype=torch.float64)
def topk(x):
    fft = torch.fft.rfft(x); idx = torch.topk(fft.abs(), K).indices; return idx, fft[idx]
def as_code(idx, coef, scale=1.0):          # exactly as in the repo (+ optional rescale)
    n_freq=(T+L)//2+1; sidx=(idx.double()*(T+L)/T).round().long().clamp(0,n_freq-1)
    sp=torch.zeros(n_freq,dtype=coef.dtype); sp.scatter_(0,sidx,coef); return scale*torch.fft.irfft(sp,n=T+L)
def exact(idx, coef):                        # continuous evaluation at f=k/T, no rounding
    out=torch.zeros(T+L,dtype=torch.float64)
    for k,c in zip(idx.tolist(),coef):
        w = 1.0 if k in (0, T//2) else 2.0
        out += w*(c*torch.exp(2j*math.pi*k*t/T)).real/T
    return out
print("forecast RMSE on a unit sinusoid (zero forecast = 0.707)")
print("period | code as is | + amp fix T/(T+L) | no rounding (pure T-periodic DFT ext.) | rounding: k->k'")
for p in [7,12,24,32,64,96,168]:
    x=torch.sin(2*math.pi*t/p); idx,coef=topk(x[:T])
    r=lambda y:(y[T:]-x[T:]).pow(2).mean().sqrt().item()
    k0=idx[0].item(); k1=k0*(T+L)/T
    print(f"{p:6d} |   {r(as_code(idx,coef)):5.3f}    |      {r(as_code(idx,coef,(T+L)/T)):5.3f}        |            {r(exact(idx,coef)):5.3f}                       | {k0}->{k1:.3f}->{round(k1)}")
