"""Nested 16x / 64x training corpora for the scaling experiment (one ver0 pass, one Fourier pass).

The 64x train split is ordered  base rows | tier-16 new rows | tier-64 new rows,  so the 16x train split is an exact
prefix of it and its Fourier components are a prefix of the 64x Fourier chunks (derive_prefix.py). Per source the
target is FACTOR x its base count; new windows are drawn at random (per-source permutation) from a ver0 'train' pool,
never repeating a base (series, start). Sources whose ver0 train split has fewer disjoint windows than the target are
capped (--cap SOURCE=N); caps and shortfalls go to the manifest. Evaluation splits and their Fourier caches are
copied byte for byte from the base cache; base Fourier chunks are reused. Run cache_fourier.py --clip 20 afterwards.
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
from build_scaled import EVAL_SPLITS, sha, write_manifest
from cache_fourier import CHUNK, KEYS
from fourier_arm import cache_tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True); p.add_argument('--factors', default='16,64')
    p.add_argument('--max-per-series', type=int, default=512); p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--cap', action='append', default=[], help='SOURCE=N upper bound on the largest tier')
    a = p.parse_args(); base = Path(a.base).resolve(); f_mid, f_top = map(int, a.factors.split(','))
    old = dict(np.load(base/'train.npz')); n_old = len(old['x']); counts = Counter(old['dataset'].tolist())
    assert set(counts) == set(bd.TRAIN_SOURCES), counts
    caps = {k: int(v) for k, v in (c.split('=') for c in a.cap)}
    top = {d: min(f_top*counts[d], caps.get(d, 10**12)) for d in bd.TRAIN_SOURCES}
    mid = {d: min(f_mid*counts[d], top[d]) for d in bd.TRAIN_SOURCES}
    cfg = dict(factors=[f_mid, f_top], max_per_series=a.max_per_series, seed=a.seed, caps=caps, target_mid=mid, target_top=top, n_old=n_old)
    digest = hashlib.sha256((json.dumps(cfg, sort_keys=True)+Path(__file__).read_text()+Path(bd.__file__).read_text()).encode()).hexdigest()[:12]
    caches = {f: base.parent/f'fm_nest{f}_{digest}' for f in (f_mid, f_top)}
    for c in caches.values(): (c/'chunks').mkdir(parents=True, exist_ok=True)
    (caches[f_top]/'config.json').write_text(json.dumps(dict(cfg, base=str(base)), indent=2))
    top_cache, mid_cache = caches[f_top], caches[f_mid]

    if not (top_cache/'train.npz').exists():
        rng = np.random.default_rng(a.seed)
        seen = set(zip(old['dataset'].tolist(), old['series_id'].tolist(), old['start'].tolist()))
        rows = bd.collect(('train',), bd.TRAIN_SOURCES, max(top.values()), a.max_per_series, rng)
        tier_mid, tier_top, shortfall = [], [], {}
        for d in bd.TRAIN_SOURCES:
            cand = [r for r in rows if r[1] == d and (d, str(r[2]), int(r[3])) not in seen]
            cand = [cand[i] for i in rng.permutation(len(cand))]
            need_top = top[d]-counts[d]; need_mid = mid[d]-counts[d]
            if len(cand) < need_top: shortfall[d] = need_top-len(cand)
            tier_mid.extend(cand[:need_mid]); tier_top.extend(cand[need_mid:need_top])
        rows = None
        new = tier_mid+tier_top
        tmp = top_cache/'train.tmp.npz'
        np.savez_compressed(tmp, x=np.concatenate([old['x'], np.stack([r[0] for r in new])]),
                            dataset=np.concatenate([old['dataset'], np.array([r[1] for r in new])]),
                            series_id=np.concatenate([old['series_id'], np.array([str(r[2]) for r in new])]),
                            start=np.concatenate([old['start'], np.array([r[3] for r in new])]),
                            split=np.concatenate([old['split'], np.array([r[4] for r in new])]))
        tmp.replace(top_cache/'train.npz')
        (top_cache/'tiers.json').write_text(json.dumps(dict(n_old=n_old, n_mid=n_old+len(tier_mid), n_top=n_old+len(new),
                                                            mid_counts=Counter(r[1] for r in tier_mid), top_counts=Counter(r[1] for r in tier_top),
                                                            shortfall=shortfall), indent=2))
        print('tiers', json.loads((top_cache/'tiers.json').read_text()), flush=True)
    tiers = json.loads((top_cache/'tiers.json').read_text())

    # 16x train = exact prefix of 64x train.
    train = dict(np.load(top_cache/'train.npz'))
    if not (top_cache/'train_x.npy').exists():  # uncompressed copy, read by cache_fourier per chunk via mmap
        np.save(top_cache/'train_x.tmp.npy', train['x']); (top_cache/'train_x.tmp.npy').replace(top_cache/'train_x.npy')
    if not (mid_cache/'train.npz').exists():
        tmp = mid_cache/'train.tmp.npz'; np.savez_compressed(tmp, **{k: v[:tiers['n_mid']] for k, v in train.items()}); tmp.replace(mid_cache/'train.npz')
    tag = cache_tag()
    for c in (top_cache, mid_cache):
        for split in EVAL_SPLITS:
            for name in (f'{split}.npz', f'{split}_fourier_{tag}.npz'):
                if not (c/name).exists(): shutil.copy2(base/name, c/name)
    f = np.load(base/f'train_fourier_{tag}.npz')
    for k in range(n_old//CHUNK):
        path = top_cache/'chunks'/f'train_{tag}_{k:04d}.npz'
        if not path.exists(): np.savez_compressed(path, **{key: f[key][k*CHUNK:(k+1)*CHUNK] for key in KEYS})

    # Checks: base rows are an exact prefix, eval splits byte-identical, no series leakage, unique windows.
    assert np.array_equal(train['x'][:n_old], old['x']) and np.array_equal(train['series_id'][:n_old], old['series_id'])
    mid_train = np.load(mid_cache/'train.npz')
    assert np.array_equal(mid_train['x'], train['x'][:tiers['n_mid']])
    for c in (top_cache, mid_cache):
        for split in EVAL_SPLITS: assert sha(c/f'{split}.npz') == sha(base/f'{split}.npz'), (c, split)
    assert set(train['dataset'].tolist()) == set(bd.TRAIN_SOURCES)
    keys = set(zip(train['dataset'].tolist(), train['series_id'].tolist()))
    for split in ('val', 'test_id'):
        ev = np.load(base/f'{split}.npz'); assert not keys & set(zip(ev['dataset'].tolist(), ev['series_id'].tolist())), split
    assert len(set(zip(train['dataset'].tolist(), train['series_id'].tolist(), train['start'].tolist()))) == len(train['x'])
    write_manifest(top_cache, base, dict(cfg, tiers=tiers))
    print('top', top_cache, '\nmid', mid_cache, flush=True)


if __name__ == '__main__':
    main()
