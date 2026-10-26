"""Build an asinh-space copy of a mini-FM cache.

Every window x (context + horizon) becomes u = asinh(x / eps), with eps from the raw context only (no leakage):
- std:   eps = std of the context (1 if the context is constant), as asinh_std in period_decomp/asinh_space_check.py;
- small: eps = max(0.1 * mean |x| of the context, 1e-8), as asinh_small there (log-like for positive series).
The new <cache>_asinh_<variant>/<split>.npz stores x = u plus x_lin (the original window) and eps, so the unchanged
training pipeline (per-window normalisation, period estimator, Fourier and S1/S2 caches) runs in u-space, and
eval_space.py maps the predictive samples back with x = eps * sinh(u) to score them against x_lin.
Afterwards build the Fourier and S1/S2 caches of the new directory (cache_fourier.py, seasonal_arms.py).
"""
import argparse
import json
from pathlib import Path
import numpy as np

T = 512
SPLITS = ('train', 'val', 'test_id', 'test_ood')


def eps_of(x, variant):
    c = x[:, :T].astype(np.float64)
    if variant == 'std':
        s = c.std(1); return np.where(s > 1e-8, s, 1.)
    return np.maximum(.1*np.abs(c).mean(1), 1e-8)


def main():
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--variant', choices=('std', 'small'), required=True)
    a = p.parse_args(); src = Path(a.cache); dst = src.with_name(f'{src.name}_asinh_{a.variant}'); dst.mkdir(exist_ok=True)
    for split in SPLITS:
        d = dict(np.load(src/f'{split}.npz', allow_pickle=True)); x = d['x'].astype(np.float64); eps = eps_of(x, a.variant)
        u = np.arcsinh(x/eps[:, None])
        assert np.allclose(eps[:, None]*np.sinh(u), x, rtol=1e-9, atol=1e-9*np.abs(x).max())
        d.update(x=u.astype(d['x'].dtype), x_lin=d['x'], eps=eps)
        np.savez_compressed(dst/f'{split}.npz', **d); print(split, len(x), 'eps median', float(np.median(eps)), flush=True)
    cfg = json.loads((src/'config.json').read_text()) if (src/'config.json').exists() else {}
    cfg.update(source_cache=src.name, transform=f'u = asinh(x / eps), eps variant {a.variant!r} (context only)')
    (dst/'config.json').write_text(json.dumps(cfg, indent=2)); print('saved', dst)


if __name__ == '__main__':
    main()
