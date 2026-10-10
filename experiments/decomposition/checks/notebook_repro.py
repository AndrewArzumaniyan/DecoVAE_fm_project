"""Reproduce the original debug_training.ipynb (DummyRandomDataset) without Lightning and
measure what the notebook only plots: calibration vs the KNOWN true distribution,
the variance double-counting (A2), and the use of each stream.

Loss / normalisation are copied 1:1 from decovaefm/lightning_module.py.
Usage: python notebook_repro.py <kl_weight> <out_prefix>
"""
import sys, math, json, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from decovaefm.models.decovae_fm import DecoVAEFM

KL_W = float(sys.argv[1]); OUT = sys.argv[2]
T, L, EPOCHS, BS, LR = 512, 256, 30, 32, 1e-3          # notebook settings
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

def make(n, seed):   # same law as DummyRandomDataset, but also returns ground truth
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(20, 70, (n, 1), generator=g).float()
    xs = torch.linspace(0, 1, T + L)[None] * w + torch.arange(n)[:, None]
    s = torch.sin(xs)
    sc = torch.randint(1, 10000, (n, 1), generator=g).float()
    noisy = (s + torch.randn(n, T + L, generator=g) * (s + 1) * 0.15) * sc
    return noisy[:, :T], noisy[:, T:], (s * sc)[:, T:], (0.15 * (s + 1) * sc)[:, T:], sc

xtr, ytr, *_ = make(2048, 1); xva, yva, *_ = make(256, 2)
xte, yte, mte, ste, scte = make(512, 123)

# ---- copied from lightning_module.py ----
def stats(x): return x.mean(1, keepdim=True), x.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
def nrm(x, m, s): return torch.asinh((x - m) / s)
def dnrm(x, m, s): return torch.sinh(x) * s + m
def kl(mu, lv):
    lv = lv.clamp(-20, 10); mu2 = mu.real**2 + mu.imag**2 if mu.is_complex() else mu**2
    return -0.5 * torch.mean(1 + lv - mu2 - lv.exp())
def nll(t, mu, lv):
    lv = lv.clamp(-20, 10); return 0.5 * (math.log(2*math.pi) + lv + (t-mu).pow(2).clamp_max(1e4) * (-lv).exp()).mean()
def losses(model, x, y):
    m, s = stats(x); xn, yn = nrm(x, m, s), nrm(y, m, s); o = model(xn, pred_len=L)
    k = kl(o["trend_mu"], o["trend_logvar"]) + kl(o["seasonal_mu"], o["seasonal_logvar"]) + kl(o["noise_mu"], o["noise_logvar"])
    return nll(xn, o["recon_mu"], o["recon_logvar"]) + nll(yn, o["forecast_mu"], o["forecast_logvar"]) + KL_W * k
# -----------------------------------------

model = DecoVAEFM(pred_len=L, d_model=64, patch_size=16, stride=16, n_heads=4, enc_layers=2,
                  dec_layers=2, k_top=16, kernels=[5, 9, 13, 25]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
best, best_state = 1e9, None
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(xtr))
    for i in range(0, len(xtr), BS):
        b = perm[i:i+BS]; loss = losses(model, xtr[b].to(dev), ytr[b].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): vl = losses(model, xva.to(dev), yva.to(dev)).item()
    if vl < best: best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    print(f"ep {ep:2d} val {vl:.4f}", flush=True)
model.load_state_dict(best_state); model.eval()

# ---------------- evaluation against the true predictive law ----------------
NS = 200
def sample(mode):
    outs = []
    with torch.no_grad():
        for i in range(0, len(xte), 32):
            x = xte[i:i+32].to(dev); m, s = stats(x); xn = nrm(x, m, s); B = len(x)
            xr = xn.repeat_interleave(NS, 0)
            orig = model.reparameterize
            if mode == "mean_latent": model.reparameterize = lambda mu, lv: mu
            o = model(xr, pred_len=L); model.reparameterize = orig
            mu, lv = o["forecast_mu"], o["forecast_logvar"].clamp(-20, 10)
            yn = mu if mode == "latent_only" else mu + torch.randn_like(mu) * (0.5 * lv).exp()
            outs.append(dnrm(yn.view(B, NS, L), m[:, None], s[:, None]).cpu())
    return torch.cat(outs)

levels = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
res = {}
true_q = lambda q: mte + ste * torch.distributions.Normal(0, 1).icdf(torch.tensor(q))
for mode in ["full", "mean_latent", "latent_only"]:
    S = sample(mode)                                    # (N, NS, L) raw scale
    emp = np.array([(yte <= torch.quantile(S, q, dim=1)).float().mean().item() for q in levels])
    cov = {c: ((yte >= torch.quantile(S, (1-c)/2, dim=1)) & (yte <= torch.quantile(S, (1+c)/2, dim=1))).float().mean().item() for c in (.5, .8, .9)}
    a = (S - yte[:, None]).abs().mean(1); bterm = (S[:, :, None] - S[:, None, :]).abs().mean((1, 2)) if False else None
    Ssort = S.sort(1).values; k = torch.arange(1, NS+1).float()[None, :, None]
    crps = (a - ((2*k - NS - 1) * Ssort).sum(1) / NS**2)          # sample CRPS, O(n log n)
    width_ratio = ((torch.quantile(S, .95, 1) - torch.quantile(S, .05, 1)) / (true_q(.95) - true_q(.05)).clamp_min(1e-6))
    peak = ste > 0.15 * scte                                # points where true sigma is not ~0
    pstd = S.std(1)
    res[mode] = dict(PCE=float(np.abs(emp - levels).mean()), coverage=cov,
                     CRPS_over_scale=float((crps / scte).mean()),
                     width90_ratio_median_at_peaks=float(width_ratio[peak].median()),
                     corr_pred_std_true_std=float(np.corrcoef(pstd.flatten(), ste.flatten())[0, 1]))
# oracle CRPS of the true law (Gaussian closed form)
z = (yte - mte) / ste.clamp_min(1e-9); nd = torch.distributions.Normal(0, 1)
oc = ste * (z * (2 * nd.cdf(z) - 1) + 2 * nd.log_prob(z).exp() - 1 / math.sqrt(math.pi))
res["oracle_CRPS_over_scale"] = float((oc / scte).mean())

# stream usage (posterior-mean decode, normalized space)
with torch.no_grad():
    x = xte[:64].to(dev); m, s = stats(x); xn = nrm(x, m, s)
    orig = model.reparameterize; model.reparameterize = lambda mu, lv: mu
    o = model(xn, pred_len=L); model.reparameterize = orig
    res["rms_decoded_trend"] = o["decoded_trend"].pow(2).mean().sqrt().item()
    res["rms_decoded_seasonal"] = o["decoded_seasonal"].pow(2).mean().sqrt().item()
    res["KL_per_dim"] = {n: kl(o[f"{n}_mu"], o[f"{n}_logvar"]).item() for n in ("trend", "seasonal", "noise")}
    res["softmax_kernel_weights"] = torch.softmax(model.decomposition.weights, 0).cpu().numpy().round(3).tolist()
# which stream drives sigma? shuffle one latent across the batch, keep the others
def logvar_from(enc, zt, zs, zn):
    zs_ = torch.cat([zs.real, zs.imag], -1).mean(1, keepdim=True).expand(-1, zt.size(1), -1)
    return model._fold_trend_patches(model.logvar_decoder(torch.cat([zn, zs_, zt], -1)), T + L).clamp(-20, 10)
with torch.no_grad():
    x = xte[:256].to(dev); m, s_ = stats(x); xn = nrm(x, m, s_)
    n_t = max(1, math.ceil(L / model.trend_encoder.stride)); enc = model.encode(xn, n_target=n_t)
    zt, zs, zn = enc["trend_mu"], enc["seasonal_mu"], enc["noise_mu"]; perm = torch.randperm(len(x), device=dev)
    base = logvar_from(enc, zt, zs, zn)[:, T:]
    res["sigma_change_when_shuffled(mean|dlog sigma|)"] = {
        "noise": (0.5*(logvar_from(enc, zt, zs, zn[perm])[:, T:] - base)).abs().mean().item(),
        "trend": (0.5*(logvar_from(enc, zt[perm], zs, zn)[:, T:] - base)).abs().mean().item(),
        "seasonal": (0.5*(logvar_from(enc, zt, zs[perm], zn)[:, T:] - base)).abs().mean().item()}
print(json.dumps(res, indent=1)); json.dump(res, open(OUT + ".json", "w"), indent=1)
