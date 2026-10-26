"""Train one structural variant on synthetic data with ground truth and evaluate.

Usage: python train.py <variant> <seed> [gamma]
Results are appended to experiments/decomposition/runs/pinn/results.jsonl
Loss = the original loss (recon NLL + forecast NLL + kl_w * KL, copied from lightning_module.py)
       + optional structural terms:
  hp    : gamma * mean squared HIGH-PASS part of the decoded trend over [0, T+L), using the HP
          response with cutoff = 3 * p (T/8 if no season). Bounded, scale = variance units;
          zero iff the trend is already HP-smooth at that cutoff ("physics residual").
  per   : gamma * mean (mu_s[t] - mu_s[t-p])^2   (soft periodicity, for the remap decoder)
  teach : gamma * (MSE(trend, teacher trend) + MSE(seasonal, teacher seasonal)) on [0, T+L),
          teacher = period-aware decomposition of the FULL window (targets only, never inputs)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import T, L, build
from model import StructDecoVAE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RUNS_DIR  # noqa: E402
OUT = RUNS_DIR / "pinn"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

VARIANTS = {
    #            input      decoder    structural terms
    "V0_original":       ("original", "remap",   {}),
    "V1_period_in":     ("period",  "remap",   {}),
    "V2_hard_periodic": ("period",  "fourier", {}),
    "V2det_fixed_seas": ("period",  "det",     {}),
    "V3_hard+HP":       ("period",  "fourier", {"hp": 1.0}),
    "V4_soft_periodic": ("period",  "remap",   {"per": 1.0}),
    "V5_hard+teacher":  ("period",  "fourier", {"teach": 1.0}),
    "V0_original+HP":    ("original", "remap",   {"hp": 1.0}),
    # follow-ups (de-confounding V2 / V5)
    "V5b_period_in+teacher": ("period",  "remap",     {"teach": 1.0}),
    "V5c_original+teacher":   ("original", "remap",     {"teach": 1.0}),
    "V2r_det+learned_resid": ("period",  "det_resid", {}),
}

dev = "cuda" if torch.cuda.is_available() else "cpu"


def hp_gain(period: torch.Tensor, n_freq: int, n_ext: int) -> torch.Tensor:
    cutoff = torch.where(period > 0, 3.0 * period, torch.full_like(period, T / 8))
    wc = 2 * math.pi / cutoff
    lam = 1.0 / (4 * (1 - torch.cos(wc)) ** 2)
    w = 2 * math.pi * torch.arange(n_freq, device=period.device) / n_ext
    return 1.0 / (1.0 + 4 * lam[:, None] * (1 - torch.cos(w))[None] ** 2)


def hp_highpass_energy(trend: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    ext = torch.cat([trend, trend.flip(-1)], -1)            # even extension: no wrap jump
    F = torch.fft.rfft(ext)
    g = hp_gain(period, F.shape[-1], ext.shape[-1])
    hp = torch.fft.irfft(F * (1 - g), n=ext.shape[-1])[:, : trend.shape[-1]]
    return hp.pow(2).mean()


def kl(mu, lv):
    lv = lv.clamp(-20, 10)
    mu2 = mu.real ** 2 + mu.imag ** 2 if mu.is_complex() else mu ** 2
    return -0.5 * torch.mean(1 + lv - mu2 - lv.exp())


def nll(t, mu, lv):
    lv = lv.clamp(-20, 10)
    return 0.5 * (math.log(2 * math.pi) + lv + (t - mu).pow(2).clamp_max(1e4) * (-lv).exp()).mean()


def batch_to(d, idx):
    return {k: torch.as_tensor(v[idx], device=dev) for k, v in d.items()}


def run(variant, seed, gamma=None, epochs=60, bs=64, lr=1e-3, kl_w=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)
    inp, dec, terms = VARIANTS[variant]
    terms = dict(terms)
    if gamma is not None:
        terms = {k: gamma for k in terms}
    tr = build(4096, 10_000, CACHE / "pinn_train.npz")
    va = build(512, 50_000, CACHE / "pinn_val.npz")
    te = build(1024, 90_000, CACHE / "pinn_test.npz")
    model = StructDecoVAE(input_decomp=inp, seas_decoder=dec, pred_len=L, d_model=64,
                          patch_size=16, stride=16, n_heads=4, enc_layers=2, dec_layers=2,
                          k_top=16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def fwd(b):
        return model(b["x"], L, trend_in=b["trend_in"], seas_in=b["seas_in"],
                     period=b["period"], seas_det=b["seas_det"])

    def losses(b, o):
        data = (nll(b["x"], o["mu"][:, :T], o["logvar"][:, :T])
                + nll(b["y"], o["mu"][:, T:], o["logvar"][:, T:])
                + kl_w * sum(kl(m, v) for m, v in o["kl_terms"]))
        struct = torch.zeros((), device=dev)
        if "hp" in terms:
            struct = struct + terms["hp"] * hp_highpass_energy(o["trend"], b["period"])
        if "per" in terms:
            p = b["period"].round().long()
            s = o["seasonal"]
            idx = torch.arange(s.shape[1], device=dev)[None]
            src = (idx - p[:, None]).clamp_min(0)
            valid = ((idx >= p[:, None]) & (p[:, None] > 0)).float()
            diff = (s - s.gather(1, src)) * valid
            struct = struct + terms["per"] * diff.pow(2).sum() / valid.sum().clamp_min(1)
        if "teach" in terms:
            struct = struct + terms["teach"] * ((o["trend"] - b["teach_trend"]).pow(2).mean()
                                                + (o["seasonal"] - b["teach_seas"]).pow(2).mean())
        return data, struct

    best, best_state, log = 1e9, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr["x"]))
        for i in range(0, len(perm), bs):
            b = batch_to(tr, perm[i:i + bs])
            data, struct = losses(b, fwd(b))
            opt.zero_grad()
            (data + struct).backward()
            opt.step()
        # gradient balance on the trend decoder (the PINN "pathology" check)
        b = batch_to(tr, perm[:256])
        model.zero_grad()
        data, struct = losses(b, fwd(b))
        params = [p for p in model.trend_decoder.parameters()]
        gd = torch.autograd.grad(data, params, retain_graph=True, allow_unused=True)
        gs = torch.autograd.grad(struct, params, allow_unused=True) if struct.requires_grad else None
        gnorm = lambda g: float(torch.sqrt(sum(((x ** 2).sum() for x in g if x is not None),
                                               torch.zeros((), device=dev)))) if g else 0.0
        model.eval()
        with torch.no_grad():
            model.sample_latents = False
            bv = batch_to(va, np.arange(len(va["x"])))
            vm = float((fwd(bv)["mu"][:, T:] - bv["y"]).pow(2).mean())
            model.sample_latents = True
        log.append(dict(ep=ep, val_fc_mse=vm, grad_data=gnorm(gd), grad_struct=gnorm(gs)))
        if vm < best:
            best, best_state = vm, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    model.sample_latents = False
    with torch.no_grad():
        b = batch_to(te, np.arange(len(te["x"])))
        o = fwd(b)
    res = evaluate(b, o)
    res.update(variant=variant, seed=seed, gamma=gamma, terms=terms, train_min=(time.time() - t0) / 60,
               grad_ratio_last=log[-1]["grad_struct"] / max(log[-1]["grad_data"], 1e-12),
               best_val_fc_mse=best, log=log)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "log"}))


def evaluate(b, o):
    mu, lv = o["mu"], o["logvar"]
    tr, se = o["trend"], o["seasonal"]
    gt_t, gt_s = b["gt_trend"], b["gt_seas"]
    has = b["true_p"] > 0
    r = lambda a, c: float(torch.sqrt(((a - c) ** 2).mean()))
    ctx, hor = slice(0, T), slice(T, T + L)
    sd = (0.5 * lv[:, hor]).exp()
    z = (b["y"] - mu[:, hor]) / sd
    v_s = se[:, ctx].var(1)
    v_t = tr[:, ctx].var(1)
    gt_share = gt_s[:, ctx].var(1) / (gt_s[:, ctx].var(1) + gt_t[:, ctx].var(1))
    share = v_s / (v_s + v_t)
    hp_rough = lambda x: float((x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).abs().mean())
    return dict(
        fc_mse=float(((mu[:, hor] - b["y"]) ** 2).mean()),
        fc_mae=float((mu[:, hor] - b["y"]).abs().mean()),
        cov90=float((z.abs() <= 1.645).float().mean()),
        trend_ctx=r(tr[:, ctx], gt_t[:, ctx]), trend_hor=r(tr[:, hor], gt_t[:, hor]),
        seas_ctx=r(se[has][:, ctx], gt_s[has][:, ctx]), seas_hor=r(se[has][:, hor], gt_s[has][:, hor]),
        seas_rms=float(se.pow(2).mean().sqrt()), gt_seas_rms=float(gt_s.pow(2).mean().sqrt()),
        seas_share_median=float(share[has].median()), gt_share_median=float(gt_share[has].median()),
        trend_roughness=hp_rough(tr), gt_trend_roughness=hp_rough(gt_t),
        seas_on_noseason_rms=float(se[~has].pow(2).mean().sqrt()),
        period_hit=float(((b["period"][has] - b["true_p"][has]).abs() / b["true_p"][has] <= 0.02).float().mean()))


if __name__ == "__main__":
    v, s = sys.argv[1], int(sys.argv[2])
    g = float(sys.argv[3]) if len(sys.argv) > 3 else None
    run(v, s, g)
