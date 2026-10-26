"""Synthetic windows with ground-truth components + cached period-aware decompositions.

Normalisation: LINEAR per-window standardisation with context stats, (x - m) / s, instead of
the model's asinh. asinh would break the additivity of the ground truth (trend + seasonal +
noise), and component recovery could not be measured exactly. (The asinh effect itself is
studied separately in the multiplicative-seasonality diagnostics.)
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "period_decomp"))
import period_decomp as pdm  # noqa: E402
from synthetic import make_series  # noqa: E402

T, L = 512, 96
PERIODS = (12, 24, 48, 96)
TRENDS = ("none", "linear", "kink", "rw")


def _one(seed):
    rng = np.random.default_rng(seed)
    has_season = rng.random() > 0.2
    per = (float(rng.choice(PERIODS)),) if has_season else ()
    s = make_series(rng, T, L, per, str(rng.choice(TRENDS)), float(rng.choice([1.0, 2.0])),
                    ar=float(rng.choice([0.0, 0.5])))
    m, sd = s.x[:T].mean(), s.x[:T].std()
    xn = (s.x - m) / sd
    gt_trend, gt_seas, gt_noise = (s.trend - m) / sd, s.seasonal / sd, s.noise / sd
    seasons = pdm.estimate_seasons(xn[:T])
    d = pdm.decompose(xn[:T], seasons)
    full = pdm.decompose(xn, seasons)                       # teacher: same periods, full window
    t_all = np.arange(T + L)
    return dict(
        x=xn[:T], y=xn[T:], gt_trend=gt_trend, gt_seas=gt_seas, gt_noise=gt_noise,
        sigma=s.sigma / sd, true_p=per[0] if per else 0.0,
        period=seasons[0].period if seasons else 0.0,
        trend_in=d.trend, seas_in=d.seasonal, seas_det=d.seasonal_at(t_all),
        teach_trend=full.trend, teach_seas=full.seasonal)


def build(n, seed0, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(16) as pool:
        rows = pool.map(_one, range(seed0, seed0 + n), chunksize=16)
    out = {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    np.savez_compressed(cache, **out)
    return out
