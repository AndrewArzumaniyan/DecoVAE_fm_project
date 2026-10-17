"""Mini-FM corpus: leave-sources-out windows from ver0.

Training sources contribute train/val/test-ID windows from their own ver0 splits; held-out sources
are never seen in training and contribute zero-shot windows from all three ver0 splits.
Windows are T+L long, sampled at random starts; a series may give several windows (non-identical
starts). Output: <cache>/{train,val,test_id,test_ood}.npz with x, dataset, series_id, start.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fm_paths import ROOT, VER0_DIR, RUNS, CACHE  # noqa: E402,F401  (also puts the shared benchmark modules on sys.path)
from data import T, L

TRAIN_SOURCES = ('London_Smart_Meters', 'Q-TRAFFIC', 'azure_vm_traces_2017', 'borg_cluster_data_2011',
                 'largest_2017', 'weatherbench_weekly', 'ushcn_daily', 'Wind_Farms', 'mexico_city_bikes')
HELDOUT_SOURCES = ('Electricity', 'Solar', 'KDD_Cup_2018', 'Pedestrian_Counts', 'M4_Daily', 'Uber_TLC')


def series_counts(splits):
    layout = json.loads((Path(__file__).parent/'ver0_layout.json').read_text())
    c = Counter()
    for k, v in layout.items():
        if k.split('/')[0] in splits: c.update(v)
    return c


def collect(splits, sources, cap, max_per_series, rng):
    """Random windows; each series is accepted with probability chosen so caps fill roughly uniformly."""
    counts = series_counts(splits)
    per_series = {d: int(min(max_per_series, max(1, np.ceil(2.*cap/max(1, counts[d]))))) for d in sources}
    accept = {d: min(1., 2.*cap/max(1, counts[d]*per_series[d])) for d in sources}
    pool = defaultdict(list)
    layout = json.loads((Path(__file__).parent/'ver0_layout.json').read_text())
    for split in splits:
        for path in sorted((VER0_DIR/split).glob('*.parquet')):
            if not set(layout[f'{split}/{path.name}']) & set(sources): continue
            pf = pq.ParquetFile(path, pre_buffer=False)
            for batch in pf.iter_batches(batch_size=16, columns=['dataset', 'series_id', 'values'], use_threads=False):
                names = batch.column(0).to_pylist(); ids = batch.column(1).to_pylist()
                for j, ds in enumerate(names):
                    if ds not in accept or rng.random() > accept[ds]: continue
                    x = batch.column(2)[j].values.to_numpy(zero_copy_only=False)
                    n = len(x) - T - L + 1
                    if n < 1: continue
                    starts = rng.choice(n, size=min(per_series[ds], n), replace=False)
                    for s in starts:
                        w = np.asarray(x[s:s+T+L], dtype=np.float32)
                        if not np.isfinite(w).all() or w[:T].std() < 1e-6: continue
                        # Skip windows whose horizon is almost entirely constant-zero padding.
                        if np.mean(w[T:] == 0) > .9 and np.mean(w[:T] == 0) < .5: continue
                        pool[ds].append((w.copy(), ds, ids[j], int(s), split))
                batch = None
            pf = None
            pa.default_memory_pool().release_unused()
            print(split, path.name, {d: len(v) for d, v in pool.items()}, flush=True)
    out = []
    for ds in sources:
        rows = pool.get(ds, [])
        if len(rows) > cap: rows = [rows[i] for i in sorted(rng.choice(len(rows), cap, replace=False))]
        out.extend(rows)
    return out


def save(path, rows):
    np.savez_compressed(path, x=np.stack([r[0] for r in rows]), dataset=np.array([r[1] for r in rows]),
                        series_id=np.array([str(r[2]) for r in rows]), start=np.array([r[3] for r in rows]),
                        split=np.array([r[4] for r in rows]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cap', type=int, default=2000); p.add_argument('--eval-cap', type=int, default=128)
    p.add_argument('--ood-cap', type=int, default=256); p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--train-sources', help='comma list; default TRAIN_SOURCES'); p.add_argument('--heldout-sources', help='comma list; default HELDOUT_SOURCES')
    p.add_argument('--name', default='fm_mini', help='cache directory prefix')
    a = p.parse_args()
    train_sources = tuple(a.train_sources.split(',')) if a.train_sources else TRAIN_SOURCES
    heldout_sources = tuple(a.heldout_sources.split(',')) if a.heldout_sources else HELDOUT_SOURCES
    assert not set(train_sources) & set(heldout_sources), 'a source is both trained on and held out'
    cfg = dict({k: v for k, v in vars(a).items() if k not in ('train_sources', 'heldout_sources', 'name')}, T=T, L=L, train_sources=train_sources, heldout_sources=heldout_sources)
    digest = hashlib.sha256((json.dumps(cfg, sort_keys=True)+Path(__file__).read_text()).encode()).hexdigest()[:12]
    cache = CACHE/f'{a.name}_{digest}'; cache.mkdir(parents=True, exist_ok=True)
    (cache/'config.json').write_text(json.dumps(cfg, indent=2))
    rng = np.random.default_rng(a.seed)
    plan = [('train', ('train',), train_sources, a.train_cap, 8), ('val', ('val',), train_sources, a.eval_cap, 2),
            ('test_id', ('test',), train_sources, a.eval_cap, 2),
            ('test_ood', ('train', 'val', 'test'), heldout_sources, a.ood_cap, 16)]
    for name, splits, sources, cap, mps in plan:
        if (cache/f'{name}.npz').exists(): continue
        rows = collect(splits, sources, cap, mps, rng)
        save(cache/f'{name}.npz', rows)
        print('saved', name, Counter(r[1] for r in rows), flush=True)
    # Series-level disjointness of training data and in-distribution evaluation.
    ids = {k: set(zip(*[np.load(cache/f'{k}.npz')[c] for c in ('dataset', 'series_id')])) for k in ('train', 'val', 'test_id')}
    assert not ids['train'] & ids['val'] and not ids['train'] & ids['test_id'], 'series overlap'
    print(cache)


if __name__ == '__main__':
    main()
