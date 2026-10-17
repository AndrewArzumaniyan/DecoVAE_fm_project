"""Diagnostics of the CVAE arms: how much the posterior uses the latent and how far the prior is from it.

On a split: KL(q || p) per latent dim, posterior and prior sd, and the normalised CRPS of the forecast mixture when z is
drawn from the posterior (sees the future, training-time view) vs from the prior (test-time view), 32 draws each.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train import runner, T  # noqa: E402
from metrics import distribution  # noqa: E402


@torch.no_grad()
def diag(run, split, device, n=32, limit=512):
    cfg = json.loads((run/'provenance.json').read_text())['config']
    model = train.make_model(cfg['arm']).to(device)
    model.load_state_dict(torch.load(run/'best.pt', weights_only=False, map_location=device)['model'])
    raw = train.load_split(train.ROOT/cfg['cache'], split, cfg['arm'], cfg['clip']); idx = np.arange(min(limit, len(raw['z'])))
    y = raw['z'][idx, T:]; res = {}
    for mode in ('posterior', 'prior'):
        model.train(mode == 'posterior'); torch.manual_seed(1729); mus, vs, kls, zs = [], [], [], []
        for _ in range(n):
            x = runner.set_batch(model, raw, idx, device); o = model(x)
            mus.append(o['forecast_mu'].cpu().numpy()); vs.append(o['forecast_logvar'].exp().cpu().numpy()); zs.append(o['cvae_z'].cpu().numpy())
            if 'cvae_kl' in o: kls.append(o['cvae_kl'].cpu().numpy())
        mu, var = np.stack(mus), np.stack(vs); smp = mu+np.sqrt(var)*np.random.default_rng(0).standard_normal(mu.shape)
        res[mode] = dict(ncrps=float(distribution(y, smp)['crps'].mean()), latent_share=float((mu.var(0)/(mu.var(0)+var.mean(0))).mean()),
                         z_sd=np.stack(zs).std(0).mean(0).round(3).tolist(), kl=float(np.mean(kls)) if kls else None)
    print(run.name, split, json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--split', default='val'); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    for r in a.runs: diag(Path(r), a.split, a.device)


if __name__ == '__main__':
    main()
