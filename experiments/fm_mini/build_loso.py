"""Leave-one-source-out folds over the 15 usable ver0 sources.

`union`: one cache with the windows of all 15 sources, drawn exactly as build_data.py draws them (same caps, seed and
sampler): train (ver0 train split, cap 2000 per source), val and test_id (ver0 val / test splits, cap 128), and the
held-out-style windows test_ood (all ver0 splits, cap 256) for every source. Build its Fourier and S1/S2 caches once
(cache_fourier.py, seasonal_arms.py) and the seasonal-naive rows (baselines_eval.py --splits test_id,test_ood).
`folds`: for every source k a cache loso_<k>/ with train / val / test_id = the union windows of the other 14 sources and
test_ood = the held-out-style windows of k, plus the matching rows of the Fourier and S1/S2 caches. In fold k source k
is never trained on, so test_ood of k is zero-shot; the other folds never see k's held-out-style windows.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import build_data as bd
from fourier_arm import cache_tag

SOURCES = bd.TRAIN_SOURCES+bd.HELDOUT_SOURCES
from fm_paths import CACHE  # noqa: E402


def union(seed=20261013):
    out = CACHE/'loso_union'; out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    plan = [('train', ('train',), 2000, 8), ('val', ('val',), 128, 2), ('test_id', ('test',), 128, 2), ('test_ood', ('train', 'val', 'test'), 256, 16)]
    for name, splits, cap, mps in plan:
        if (out/f'{name}.npz').exists(): continue
        rows = bd.collect(splits, SOURCES, cap, mps, rng); bd.save(out/f'{name}.npz', rows)
        print('saved', name, Counter(r[1] for r in rows), flush=True)
    ids = {k: set(zip(*[np.load(out/f'{k}.npz')[c] for c in ('dataset', 'series_id')])) for k in ('train', 'val', 'test_id')}
    assert not ids['train'] & ids['val'] and not ids['train'] & ids['test_id'], 'series overlap'
    (out/'config.json').write_text(json.dumps(dict(T=bd.T, L=bd.L, seed=seed, sources=SOURCES, note='LOSO union (build_loso.py)'), indent=2))


def folds():
    u = CACHE/'loso_union'; tag = cache_tag()
    for k in SOURCES:
        dst = CACHE/f'loso_{k}'; dst.mkdir(exist_ok=True)
        for split in ('train', 'val', 'test_id', 'test_ood'):
            d = dict(np.load(u/f'{split}.npz', allow_pickle=True)); ds = d['dataset'].astype(str)
            keep = ds == k if split == 'test_ood' else ds != k
            np.savez_compressed(dst/f'{split}.npz', **{c: v[keep] for c, v in d.items()})
            for kind in ('fourier', 's12'):
                f = dict(np.load(u/f'{split}_{kind}_{tag}.npz')); np.savez_compressed(dst/f'{split}_{kind}_{tag}.npz', **{c: v[keep] for c, v in f.items()})
        (dst/'config.json').write_text(json.dumps(dict(T=bd.T, L=bd.L, union='loso_union', heldout=k, train_sources=[s for s in SOURCES if s != k]), indent=2))
        print('fold', k, {s: int((np.load(dst/f'{s}.npz')['dataset'] != '').sum()) for s in ('train', 'val', 'test_id', 'test_ood')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('step', choices=('union', 'folds')); a = p.parse_args()
    union() if a.step == 'union' else folds()
