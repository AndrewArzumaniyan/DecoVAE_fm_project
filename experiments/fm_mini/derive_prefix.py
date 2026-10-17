"""Assemble the 16x train Fourier cache as a prefix of the 64x Fourier chunks (see build_nested.py).

Waits until the 64x chunks covering the first n_mid rows exist, concatenates them, truncates to n_mid, writes
<mid>/train_fourier_<tag>.npz and the 16x manifest."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_data  # noqa: F401  (puts the shared benchmark modules on sys.path)
from build_scaled import write_manifest
from cache_fourier import CHUNK, KEYS
from fourier_arm import cache_tag

p = argparse.ArgumentParser(); p.add_argument('top'); p.add_argument('mid'); p.add_argument('--base', required=True); a = p.parse_args()
top, mid = Path(a.top), Path(a.mid); tag = cache_tag(); tiers = json.loads((top/'tiers.json').read_text()); n = tiers['n_mid']
paths = [top/'chunks'/f'train_{tag}_{k:04d}.npz' for k in range((n+CHUNK-1)//CHUNK)]
while not all(q.exists() for q in paths):
    print(time.strftime('%T'), 'waiting', sum(q.exists() for q in paths), '/', len(paths), flush=True); time.sleep(60)
time.sleep(5)
parts = [np.load(q) for q in paths]
out = {key: np.concatenate([q[key] for q in parts])[:n] for key in KEYS}
assert len(out['trend']) == n == len(np.load(mid/'train.npz')['x'])
final = mid/f'train_fourier_{tag}.npz'; tmp = final.with_name(final.stem+'.tmp.npz'); np.savez_compressed(tmp, **out); os.replace(tmp, final)
cfg = json.loads((top/'config.json').read_text()); write_manifest(mid, Path(a.base).resolve(), dict(cfg, tiers=tiers, prefix_of=str(top)))
print('saved', final, flush=True)
