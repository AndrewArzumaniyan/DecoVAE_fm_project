"""Parallel, resumable period-decomposition cache in the exact format fourier_arm.attach expects."""
import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from fm_paths import ROOT, VER0_DIR, RUNS, CACHE  # noqa: E402,F401  (also puts the shared benchmark modules on sys.path)
from data import normalize
from fourier_arm import cache_tag, fourier_components

KEYS = ('trend', 'seasonal', 'periods', 'coefs', 'seasonal_full')
CHUNK = 512


def z_of(cache, split, clip=0., rows=slice(None)):
    # normalize is per window, so slicing before it gives the same rows with less memory.
    raw = cache/f'{split}_x.npy'  # uncompressed copy for large caches: read only the rows of this chunk
    x = np.load(raw, mmap_mode='r')[rows] if raw.exists() else np.load(cache/f'{split}.npz')['x'][rows]
    z = normalize(np.asarray(x, dtype=np.float64))[0]
    if clip: z = np.clip(z, -clip, clip)
    return z.astype('float32')


def work(job):
    cache, split, k, clip = job
    z = z_of(cache, split, clip, slice(k*CHUNK, (k+1)*CHUNK))
    fourier_components(z, cache/'chunks'/f'{split}_{cache_tag()}_{k:04d}.npz')
    return split, k


def main():
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('--workers', type=int, default=3)
    p.add_argument('--clip', type=float, default=0.); p.add_argument('--splits', default='train,val,test_id,test_ood')
    a = p.parse_args(); cache = Path(a.cache); (cache/'chunks').mkdir(exist_ok=True)
    jobs = []
    for split in a.splits.split(','):
        if (cache/f'{split}_fourier_{cache_tag()}.npz').exists(): continue
        raw = cache/f'{split}_x.npy'
        n = len(np.load(raw, mmap_mode='r')) if raw.exists() else len(np.load(cache/f'{split}.npz')['x'])
        done = cache/'chunks'
        jobs += [(cache, split, k, a.clip) for k in range((n+CHUNK-1)//CHUNK)
                 if not (done/f'{split}_{cache_tag()}_{k:04d}.npz').exists()]
    with get_context('spawn').Pool(a.workers, maxtasksperchild=4) as pool:
        for split, k in pool.imap_unordered(work, jobs): print('chunk', split, k, flush=True)
    for split in a.splits.split(','):
        final = cache/f'{split}_fourier_{cache_tag()}.npz'
        if final.exists(): continue
        n = len(np.load(cache/f'{split}.npz')['x'])
        tag = cache_tag(); parts = {key: [] for key in KEYS}
        for k in range((n+CHUNK-1)//CHUNK):  # one chunk file open at a time (thousands of chunks exceed the fd limit)
            with np.load(cache/'chunks'/f'{split}_{tag}_{k:04d}.npz') as q:
                for key in KEYS: parts[key].append(q[key])
        out = {key: np.concatenate(v) for key, v in parts.items()}
        assert len(out['trend']) == n
        tmp = final.with_name(final.stem+'.tmp.npz'); np.savez_compressed(tmp, **out); os.replace(tmp, final)
        print('saved', final.name, flush=True)


if __name__ == '__main__':
    main()
