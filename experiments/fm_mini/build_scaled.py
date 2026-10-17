"""Nested, larger training corpus for the r2 data-scaling test.

The base (1x) cache train windows are kept first and in their original order; new unique windows from the same
nine sources and the ver0 'train' split are appended up to FACTOR x the base count per source.
Evaluation splits and their Fourier caches are copied byte for byte from the base cache, and the
full Fourier chunks covering base rows are written from the base cache, so cache_fourier.py only
decomposes the new windows (run it with --clip 20, the cache key does not include clipping).
`manifest.json` records per-source counts, shortfalls and SHA-256 of every file.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_data as bd
from cache_fourier import CHUNK, KEYS
from fourier_arm import cache_tag

EVAL_SPLITS = ('val', 'test_id', 'test_ood')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def write_manifest(cache, base, cfg):
    tag = cache_tag(); files = sorted(p for p in cache.glob('*.npz'))
    out = dict(cfg, base_cache=str(base), fourier_tag=tag, files={p.name: sha(p) for p in files})
    for split in ('train', *EVAL_SPLITS):
        out[f'{split}_counts'] = dict(Counter(np.load(cache/f'{split}.npz')['dataset'].tolist()))
    (cache/'manifest.json').write_text(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True); p.add_argument('--factor', type=int, default=4)
    p.add_argument('--max-per-series', type=int, default=64); p.add_argument('--seed', type=int, default=20261014)
    a = p.parse_args(); base = Path(a.base).resolve()
    old = dict(np.load(base/'train.npz')); n_old = len(old['x'])
    counts = Counter(old['dataset'].tolist())
    assert set(counts) == set(bd.TRAIN_SOURCES), counts
    target = {d: a.factor*counts[d] for d in bd.TRAIN_SOURCES}
    cfg = dict(factor=a.factor, max_per_series=a.max_per_series, seed=a.seed, target=target, n_old=n_old)
    digest = hashlib.sha256((json.dumps(cfg, sort_keys=True)+Path(__file__).read_text()
                             + Path(bd.__file__).read_text()).encode()).hexdigest()[:12]
    cache = base.parent/f'fm_scale{a.factor}_{digest}'; (cache/'chunks').mkdir(parents=True, exist_ok=True)
    (cache/'config.json').write_text(json.dumps(dict(cfg, base=str(base)), indent=2))

    if not (cache/'train.npz').exists():
        rng = np.random.default_rng(a.seed)
        seen = set(zip(old['dataset'].tolist(), old['series_id'].tolist(), old['start'].tolist()))
        rows = bd.collect(('train',), bd.TRAIN_SOURCES, max(target.values()), a.max_per_series, rng)
        new, shortfall = [], {}
        for d in bd.TRAIN_SOURCES:
            cand = [r for r in rows if r[1] == d and (d, str(r[2]), int(r[3])) not in seen]
            need = target[d]-counts[d]
            if len(cand) > need: cand = [cand[i] for i in sorted(rng.choice(len(cand), need, replace=False))]
            else: shortfall[d] = need-len(cand)
            new.extend(cand)
        rows = None
        tmp = cache/'train.tmp.npz'
        np.savez_compressed(tmp, x=np.concatenate([old['x'], np.stack([r[0] for r in new])]),
                            dataset=np.concatenate([old['dataset'], np.array([r[1] for r in new])]),
                            series_id=np.concatenate([old['series_id'], np.array([str(r[2]) for r in new])]),
                            start=np.concatenate([old['start'], np.array([r[3] for r in new])]),
                            split=np.concatenate([old['split'], np.array([r[4] for r in new])]))
        tmp.replace(cache/'train.npz')
        (cache/'shortfall.json').write_text(json.dumps(shortfall, indent=2))
        print('new windows', Counter(r[1] for r in new), 'shortfall', shortfall, flush=True)

    tag = cache_tag()
    for split in EVAL_SPLITS:
        for name in (f'{split}.npz', f'{split}_fourier_{tag}.npz'):
            if not (cache/name).exists(): shutil.copy2(base/name, cache/name)
    f = np.load(base/f'train_fourier_{tag}.npz')
    for k in range(n_old//CHUNK):
        path = cache/'chunks'/f'train_{tag}_{k:04d}.npz'
        if not path.exists(): np.savez_compressed(path, **{key: f[key][k*CHUNK:(k+1)*CHUNK] for key in KEYS})

    # Checks: base rows are an exact prefix, eval splits are byte-identical, no series leakage.
    train = np.load(cache/'train.npz')
    assert np.array_equal(train['x'][:n_old], old['x']) and np.array_equal(train['series_id'][:n_old], old['series_id'])
    for split in EVAL_SPLITS: assert sha(cache/f'{split}.npz') == sha(base/f'{split}.npz'), split
    assert set(train['dataset'].tolist()) == set(bd.TRAIN_SOURCES)
    keys = set(zip(train['dataset'].tolist(), train['series_id'].tolist()))
    for split in ('val', 'test_id'):
        ev = np.load(cache/f'{split}.npz'); assert not keys & set(zip(ev['dataset'].tolist(), ev['series_id'].tolist())), split
    assert len(set(zip(train['dataset'].tolist(), train['series_id'].tolist(), train['start'].tolist()))) == len(train['x'])
    write_manifest(cache, base, cfg)
    print(cache, flush=True)


if __name__ == '__main__':
    main()
