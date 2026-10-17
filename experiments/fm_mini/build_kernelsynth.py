"""KernelSynth windows and mixed training caches for the diversity control.

Generator (Ansari et al. 2024, Chronos, Appendix A, Algorithm 2 and Table 2), reimplemented here (nothing downloaded):
j ~ U{1..5} kernels drawn with replacement from the bank, combined left to right with random + or x, one draw of a
zero-mean GP of length 1024 on t = linspace(0, 1, 1024) (periods are given in steps, i.e. p / 1024 on this axis).
Bank: Constant(1); WhiteNoise(0.1, 1); Linear sigma^2 + t t' with sigma in {0, 1, 10}; RBF l in {0.1, 1, 10};
RationalQuadratic alpha in {0.1, 1, 10} (length scale 1); Periodic exp(-2 sin^2(pi |t - t'| / p)) with
p in {24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10} steps.
A random window of T + L = 608 points is cut from each series (as training windows are cut from real series).

Steps: `generate <out_dir> --n N` writes <out_dir>/train.npz (dataset 'KernelSynth'); build its Fourier and S1/S2
caches with cache_fourier.py and seasonal_arms.build(splits=('train',)); then `mix <base_cache> <synth_dir> --n K`
writes <base_cache>_ks<pct>/ with train = real + the first K synthetic windows (arrays and caches concatenated) and
val/test files linked from the base cache, so only training data change.
"""
import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path
import numpy as np

LSYN, W = 1024, 608
PERIODS = (24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10)
BANK = ([('const', 1.)] + [('white', v) for v in (.1, 1.)] + [('linear', v) for v in (0., 1., 10.)] + [('rbf', v) for v in (.1, 1., 10.)]
        + [('rq', v) for v in (.1, 1., 10.)] + [('periodic', p/LSYN) for p in PERIODS])
TAG = '83a0a46efb39'


def kernel(name, v, t):
    d = t[:, None]-t[None, :]
    if name == 'const': return np.full_like(d, v)
    if name == 'white': return v*np.eye(len(t))
    if name == 'linear': return v**2+t[:, None]*t[None, :]
    if name == 'rbf': return np.exp(-d**2/(2*v**2))
    if name == 'rq': return (1+d**2/(2*v))**(-v)
    return np.exp(-2*np.sin(np.pi*np.abs(d)/v)**2)


def sample(seed):
    rng = np.random.default_rng(seed); t = np.linspace(0, 1, LSYN)
    j = rng.integers(1, 6); picks = rng.integers(0, len(BANK), j); K = kernel(*BANK[picks[0]], t); desc = [BANK[picks[0]]]
    for k in picks[1:]:
        op = rng.integers(0, 2); K = K+kernel(*BANK[k], t) if op == 0 else K*kernel(*BANK[k], t); desc.append(('+x'[op], BANK[k]))
    jitter = 1e-6*max(np.trace(K)/LSYN, 1e-12)
    for _ in range(8):
        try: c = np.linalg.cholesky(K+jitter*np.eye(LSYN)); break
        except np.linalg.LinAlgError: jitter *= 10
    else:
        w, V = np.linalg.eigh(K); c = V*np.sqrt(np.clip(w, 0, None))
    x = c@rng.standard_normal(LSYN); off = rng.integers(0, LSYN-W+1)
    return x[off:off+W].astype('float32'), json.dumps([str(q) for q in desc])


def generate(out, n, workers, seed):
    out.mkdir(parents=True, exist_ok=True)
    with get_context('spawn').Pool(workers) as pool: res = pool.map(sample, [seed*10**6+i for i in range(n)], chunksize=64)
    x = np.stack([r[0] for r in res])
    np.savez_compressed(out/'train.npz', x=x, dataset=np.array(['KernelSynth']*n), series_id=np.arange(n)+10**6,
                        start=np.zeros(n, int), split=np.array(['train']*n), kernels=np.array([r[1] for r in res]))
    print('saved', out/'train.npz', x.shape, 'std of window std', float(x.std(1).std()), flush=True)


def mix(base, synth, n):
    real = len(np.load(base/'train.npz')['x']); pct = round(100*n/real); dst = base.with_name(f'{base.name}_ks{pct}'); dst.mkdir(exist_ok=True)
    for name in ('train', f'train_fourier_{TAG}', f'train_s12_{TAG}'):
        a = dict(np.load(base/f'{name}.npz', allow_pickle=True)); b = dict(np.load(synth/f'{name}.npz', allow_pickle=True))
        keys = [k for k in a if k in b]; assert keys == list(a), (name, list(a), list(b))
        np.savez_compressed(dst/f'{name}.npz', **{k: np.concatenate([a[k], b[k][:n]]) for k in keys})
    for f in base.glob('*.npz'):
        if not f.name.startswith('train') and not (dst/f.name).exists(): os.symlink(f.resolve(), dst/f.name)
    cfg = json.loads((base/'config.json').read_text()); cfg.update(source_cache=base.name, kernelsynth_windows=n, kernelsynth_share_of_real=n/real)
    (dst/'config.json').write_text(json.dumps(cfg, indent=2)); print('saved', dst, real, '+', n, flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate'); g.add_argument('out'); g.add_argument('--n', type=int, required=True); g.add_argument('--workers', type=int, default=8); g.add_argument('--seed', type=int, default=20261015)
    m = sub.add_parser('mix'); m.add_argument('base'); m.add_argument('synth'); m.add_argument('--n', type=int, required=True)
    a = p.parse_args()
    if a.cmd == 'generate': generate(Path(a.out), a.n, a.workers, a.seed)
    else: mix(Path(a.base), Path(a.synth), a.n)


if __name__ == '__main__':
    main()
