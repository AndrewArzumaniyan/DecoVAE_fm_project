import sys, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decovaefm.models.decovae_fm import MultiScaleDecomposition, DecoVAEFM
torch.manual_seed(0)
T, L, K = 512, 96, 8   # base experiment config: context 512, pred 96, k_top 8
t = torch.arange(T + L, dtype=torch.float64)

print("1) MA gain on a period-p sinusoid (fraction of seasonal amplitude left in TREND)")
print("   period |  k=5   k=9  k=13  k=25")
for p in [7, 12, 24, 48, 96, 168]:
    g = [abs(math.sin(math.pi*k/p)/(k*math.sin(math.pi/p))) for k in (5,9,13,25)]
    print(f"   {p:5d}  | " + "  ".join(f"{v:4.2f}" for v in g))

print("\n2) softmax weights at init of the original module (randn), 5 seeds:")
for s in range(5):
    torch.manual_seed(s); m = MultiScaleDecomposition()
    print("   ", np.round(F.softmax(m.weights, 0).detach().numpy(), 2))

print("\n3) right-edge bias of the trend on a pure linear ramp x=t (reflect padding), in units of slope:")
m = MultiScaleDecomposition()
for k in (5, 9, 13, 25):
    m.kernels=[k]; m.weights = torch.nn.Parameter(torch.zeros(1))
    tr, _ = m(torch.arange(T, dtype=torch.float32)[None])
    print(f"   k={k:2d}: bias at last point = {tr[0,-1].item() - (T-1):+.2f}, mid = {tr[0,T//2].item()-T//2:+.2f}")

print("\n4) 'Oracle' seasonal forecast: decoder = identity on the exact top-K rFFT coeffs,")
print("   remapped bins as in _reconstruct_time_from_freq_tokens. Pure sinusoid, amp 1.")
print("   period | context recon RMSE | forecast RMSE | naive-seasonal RMSE")
def oracle(x_ctx, K):
    fft = torch.fft.rfft(x_ctx); amp = fft.abs()
    idx = torch.topk(amp, K).indices; coef = fft[idx]
    n_freq = (T+L)//2+1
    sidx = (idx.double()*(T+L)/T).round().long().clamp(0, n_freq-1)
    sp = torch.zeros(n_freq, dtype=fft.dtype); sp.scatter_(0, sidx, coef)
    return torch.fft.irfft(sp, n=T+L)
for p in [7, 12, 24, 32, 64, 96, 168]:
    x = torch.sin(2*math.pi*t/p)
    y = oracle(x[:T], K)
    rc = (y[:T]-x[:T]).pow(2).mean().sqrt().item()
    fc = (y[T:]-x[T:]).pow(2).mean().sqrt().item()
    naive = (x[T-p:T].repeat(math.ceil(L/p)+1)[:L] - x[T:]).pow(2).mean().sqrt().item()
    print(f"   {p:5d}  |      {rc:5.3f}         |    {fc:5.3f}      |   {naive:5.3f}")
print("   (RMSE of a forecast equal to zero = 0.707)")
