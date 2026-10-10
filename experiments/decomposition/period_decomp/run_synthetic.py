"""Synthetic benchmark: per-window period estimation + period-aware decomposition
vs the original fixed multi-scale MA + Top-K FFT, and STL/MSTL with the oracle period.

Writes tables and figures to experiments/decomposition/runs/period_decomp/.
Run:  python run_synthetic.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as bl
import period_decomp as pdm
from synthetic import make_series

T, L = 512, 96                      # base config of decovaefm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RUNS_DIR  # noqa: E402
OUT = RUNS_DIR / "period_decomp"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS_1 = (7, 12, 24, 48, 96, 144, 168)
TRENDS = ("none", "linear", "kink", "rw")


def rmse(a, b, scale):
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def detection(est: list[float], truth: tuple[float, ...]) -> dict:
    hits, harm = 0, 0
    used = set()
    for p in truth:
        j = next((i for i, q in enumerate(est) if abs(q - p) / p <= 0.02 and i not in used), None)
        if j is not None:
            hits += 1
            used.add(j)
            continue
        if any(abs(q - p * m) / (p * m) <= 0.03 or abs(q - p / m) / (p / m) <= 0.03
               for q in est for m in (2, 3)):
            harm += 1
    return dict(n_true=len(truth), hits=hits, harmonic=harm, misses=len(truth) - hits - harm,
                extra=max(0, len(est) - hits - harm), any_detected=int(len(est) > 0))


def methods(x_ctx, truth):
    """Return {name: dict(trend, seasonal, noise, seasonal_horizon, periods)}."""
    out = {}
    for name, logits in (("original_uniform_w", bl.W_UNIFORM), ("original_trained_w", bl.W_TRAINED)):
        out[name] = bl.original_decompose(x_ctx, L, k_top=8, logits=logits)
    for name, cfg in (("ours_est", pdm.EstimatorConfig()),
                      ("ours_est_no_holdout", pdm.EstimatorConfig(use_holdout=False)),
                      ("ours_est_calendar_snap", pdm.EstimatorConfig(snap_calendar=True))):
        seasons = pdm.estimate_seasons(x_ctx, cfg)
        d = pdm.decompose(x_ctx, seasons, cfg)
        out[name] = dict(trend=d.trend, seasonal=d.seasonal, noise=d.noise,
                         seasonal_horizon=d.seasonal_at(np.arange(T, T + L)),
                         periods=[s.period for s in seasons])
    if truth:
        orc = [pdm.SeasonSpec(p, min(4, pdm.max_harmonics(p))) for p in truth]
        d = pdm.decompose(x_ctx, orc)
        out["ours_oracle_p"] = dict(trend=d.trend, seasonal=d.seasonal, noise=d.noise,
                                    seasonal_horizon=d.seasonal_at(np.arange(T, T + L)))
        out["stl_oracle_p"] = bl.stl_decompose(x_ctx, L, truth)
    return out


def run_one(task):
    scenario, truth, trend, snr, ar, seed = task
    rng = np.random.default_rng(seed)
    s = make_series(rng, T, L, truth, trend, snr, ar=ar)
    x = s.x[:T]
    sc = x.std()
    rows = []
    for name, m in methods(x, truth).items():
        row = dict(scenario=scenario, truth=str(truth), trend=trend, snr=snr, ar=ar, seed=seed,
                   method=name,
                   trend_rmse=rmse(m["trend"], s.trend[:T], sc),
                   seas_rmse=rmse(m["seasonal"], s.seasonal[:T], sc),
                   noise_rmse=rmse(m["noise"], s.noise[:T], sc),
                   seas_horizon_rmse=rmse(m["seasonal_horizon"], s.seasonal[T:], sc))
        if "periods" in m:
            row.update(detection(m["periods"], truth))
            row["est_periods"] = json.dumps([round(p, 2) for p in m["periods"]])
        rows.append(row)
    return rows


def misspec_one(task):
    p, trend, snr, seed = task
    rng = np.random.default_rng(seed)
    s = make_series(rng, T, L, (p,), trend, snr)
    x = s.x[:T]
    sc = x.std()
    rows = []
    for label, q in (("p-1", p - 1), ("p-0.5", p - 0.5), ("p", p), ("p+0.5", p + 0.5),
                     ("p+1", p + 1), ("p/2", p / 2), ("2p", 2 * p)):
        d = pdm.decompose(x, [pdm.SeasonSpec(q, min(4, pdm.max_harmonics(q)))])
        rows.append(dict(p=p, used=label, seas_rmse=rmse(d.seasonal, s.seasonal[:T], sc),
                         seas_horizon_rmse=rmse(d.seasonal_at(np.arange(T, T + L)), s.seasonal[T:], sc),
                         trend_rmse=rmse(d.trend, s.trend[:T], sc)))
    return rows


def cutoff_one(task):
    p, trend, snr, ar, seed = task
    rng = np.random.default_rng(seed)
    s = make_series(rng, T, L, (p,), trend, snr, ar=ar)
    x = s.x[:T]
    sc = x.std()
    rows = []
    spec = [pdm.SeasonSpec(p, min(4, pdm.max_harmonics(p)))]
    for mult in (0.75, 1.5, 3.0, 6.0):
        for floor in (0.0, 0.125):
            cfg = pdm.EstimatorConfig(trend_cutoff_mult=mult, trend_cutoff_floor_frac=floor)
            d = pdm.decompose(x, spec, cfg)
            rows.append(dict(p=p, trend=trend, ar=ar, mult=mult, floor=floor,
                             trend_rmse=rmse(d.trend, s.trend[:T], sc),
                             seas_rmse=rmse(d.seasonal, s.seasonal[:T], sc)))
    return rows


def main():
    seed = itertools.count(1000)
    tasks = []
    for p, tr, snr, ar in itertools.product(PERIODS_1, TRENDS, (0.5, 1.0, 2.0), (0.0, 0.5)):
        tasks += [("S1_single", (p,), tr, snr, ar, next(seed)) for _ in range(6)]
    for pp, tr, snr, ar in itertools.product(((24, 168), (7, 30.4)), TRENDS, (1.0, 2.0), (0.0, 0.5)):
        tasks += [("S2_two", pp, tr, snr, ar, next(seed)) for _ in range(6)]
    for tr, ar in itertools.product(TRENDS, (0.0, 0.5, 0.8)):
        tasks += [("S3_none", (), tr, 0.0, ar, next(seed)) for _ in range(30)]

    with Pool(min(16, os.cpu_count() or 4)) as pool:
        rows = sum(pool.map(run_one, tasks, chunksize=4), [])
        mis = sum(pool.map(misspec_one, [(p, tr, snr, next(seed)) for p in (24, 96)
                                         for tr in ("linear", "kink") for snr in (1.0, 2.0)
                                         for _ in range(10)]), [])
        cut = sum(pool.map(cutoff_one, [(p, tr, 1.0, ar, next(seed)) for p in (12, 24, 96)
                                        for tr in TRENDS for ar in (0.0, 0.5)
                                        for _ in range(8)]), [])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_rows.csv", index=False)
    pd.DataFrame(mis).to_csv(OUT / "misspec_rows.csv", index=False)
    pd.DataFrame(cut).to_csv(OUT / "cutoff_rows.csv", index=False)
    print("rows", len(df), len(mis), len(cut))


if __name__ == "__main__":
    main()
