"""Evaluate trained mini-FM runs in the original x-space, also for runs trained in asinh space.

For every run: rebuild the arm from provenance.json, load best.pt, draw 128 predictive samples per window exactly as
train.py does for test_ood (runner.evaluate, evaluation seed 1729). If the run's cache stores eps (build_asinh.py),
the samples are in u = asinh(x / eps) and are mapped back with x = eps * sinh(u) (u clipped to +-50 against overflow);
they are scored against x_lin, normalised by the raw context std. For an ordinary cache the result reproduces
ood_rows.csv. Also joint scores of the 96-step sample paths, normalised the same way: energy score
ES = mean_i ||x_i - y|| - sum_{i != j} ||x_i - x_j|| / (2 n (n - 1)) (fair) and variogram score (p = 0.5, all step pairs,
unit weights) VS = sum_{t < t'} (|y_t - y_t'|^.5 - mean_i |x_it - x_it'|^.5)^2 / (number of pairs); and the latent share
of the predictive variance (mixture over latent draws). ncrps_g: the same CRPS with the samples clipped to
[context min - 3 range, context max + 3 range] in x-space, one sanity guard for every model (an asinh-space Gaussian
has exponential tails in x, so a single window with a huge sd can dominate the mean). Output per run: space_eval_<split>.csv (window rows) and
space_eval_pit_<split>.npz (PIT per point).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train import runner, T, normalize  # noqa: E402
from metrics import distribution  # noqa: E402  (shared benchmark module, on sys.path via train)


def joint_scores(smp, y, s, chunk=16):
    """Fair energy score and variogram score (p = 0.5) per window; smp (S, N, L), y (N, L), s (N,) context std."""
    S, N, _ = smp.shape; es = np.empty(N); vs = np.empty(N); iu = np.triu_indices(smp.shape[-1], 1)
    for a in range(0, N, chunk):
        x = smp[:, a:a+chunk]/s[None, a:a+chunk, None]; yy = y[a:a+chunk]/s[a:a+chunk, None]
        d1 = np.linalg.norm(x-yy[None], axis=-1).mean(0)
        xt = np.transpose(x, (1, 0, 2)); g = np.einsum('nik,njk->nij', xt, xt); sq = np.einsum('nik,nik->ni', xt, xt)
        d2 = np.sqrt(np.maximum(sq[:, :, None]+sq[:, None, :]-2*g, 0)).sum((1, 2))/(2*S*(S-1))
        es[a:a+chunk] = d1-d2
        vx = (np.abs(x[..., :, None]-x[..., None, :])**.5).mean(0)[:, iu[0], iu[1]]
        vy = (np.abs(yy[:, :, None]-yy[:, None, :])**.5)[:, iu[0], iu[1]]
        vs[a:a+chunk] = ((vy-vx)**2).mean(-1)
    return es, vs


def evaluate(run, device, splits):
    cfg = json.loads((run/'provenance.json').read_text())['config']
    train.SIZE.clear(); train.SIZE.update(cfg.get('size') or {})
    model = train.make_model(cfg['arm']).to(device)
    model.load_state_dict(torch.load(run/'best.pt', weights_only=False, map_location=device)['model'])
    for split in splits:
        raw = train.load_split(train.ROOT/cfg['cache'], split, cfg['arm'], cfg['clip'])
        _, outputs = runner.evaluate(model, raw, device, 128)
        idx = np.concatenate([o['indices'] for o in outputs]); smp = np.concatenate([o['samples'] for o in outputs], axis=1)
        if 'eps' in raw:
            clipped = float((np.abs(smp) > 50).mean()); smp = raw['eps'][idx][None, :, None]*np.sinh(np.clip(smp, -50, 50))
            x = raw['x_lin'][idx].astype(np.float64)
        else:
            clipped = 0.; x = raw['x'][idx].astype(np.float64)
        y = x[:, T:]; s = normalize(x)[2][:, 0]
        m = distribution(y, smp); es, vs = joint_scores(smp, y, s)
        lo, hi = x[:, :T].min(1), x[:, :T].max(1); rg = hi-lo
        crps_g = distribution(y, np.clip(smp, (lo-3*rg)[None, :, None], (hi+3*rg)[None, :, None]))['crps']
        lat = np.concatenate([o['latent_mean_variance'] for o in outputs]); obs = np.concatenate([o['observation_variance'] for o in outputs])
        df = pd.DataFrame(dict(dataset=np.asarray(raw['dataset'])[idx], series_id=raw['series_id'][idx], ncrps=m['crps']/s, ncrps_g=crps_g/s,
                               nmae=m['mae']/s, coverage50=m['coverage50'], coverage80=m['coverage80'], coverage90=m['coverage90'],
                               w80=m['width80']/s, nes=es, nvs=vs, latent_share=(lat/(lat+obs)).mean(1),
                               **{f'ncrps_h{k}': m[f'crps_h{k}']/s for k in (1, 2, 3)}))
        df['arm'] = cfg['arm']; df['seed'] = cfg['seed']; df['run'] = run.name
        df.to_csv(run/f'space_eval_{split}.csv', index=False)
        np.savez_compressed(run/f'space_eval_pit_{split}.npz', pit=(smp < y[None]).mean(0).astype('float32'))
        print(run.name, split, len(idx), 'ncrps', round(float(df.ncrps.mean()), 5), 'clipped', clipped, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--device', default='cuda')
    p.add_argument('--splits', default='test_id,test_ood'); a = p.parse_args()
    for r in a.runs: evaluate(Path(r), a.device, a.splits.split(','))


if __name__ == '__main__':
    main()
