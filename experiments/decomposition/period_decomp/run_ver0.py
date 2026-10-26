"""Real-data sanity check on ver0 (no ground truth): which periods are detected per dataset,
and is the extracted seasonal component *predictive* on the horizon?

Windows: T=512 context + L=96 horizon, asinh-normalised with context stats (as in the model).
Horizon proxies (RMSE in normalised space, lower is better), all with a flat trend
(last trend value) so that only the seasonal extrapolation differs between decompositions:
  * ours        : last trend + Fourier seasonal at the estimated period(s)
  * original    : last trend + Top-K rFFT seasonal remapped to T+L (identity decoder)
  * snaive      : repeat the last cycle of the detected period (naive if none detected)
  * naive       : last value
Run:  OMP_NUM_THREADS=1 python run_ver0.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import baselines as bl
import period_decomp as pdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T, L = 512, 96
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RUNS_DIR, VER0_DIR  # noqa: E402
OUT = RUNS_DIR / "period_decomp"
OUT.mkdir(parents=True, exist_ok=True)
N_PER_DATASET = 150


def windows(seed=0):
    tab = pq.read_table(VER0_DIR / "val" / "part-00000.parquet").to_pandas()
    rng = np.random.default_rng(seed)
    out = []
    for ds, sub in tab.groupby("dataset"):
        got, tries = 0, 0
        while got < N_PER_DATASET and tries < 20 * N_PER_DATASET:
            tries += 1
            v = np.asarray(sub.iloc[rng.integers(len(sub))]["values"], float)
            if len(v) < T + L:
                continue
            st = rng.integers(0, len(v) - T - L + 1)
            w = v[st:st + T + L]
            if not np.isfinite(w).all() or w[:T].std() == 0:
                continue
            m, s = w[:T].mean(), w[:T].std()
            out.append((ds, np.arcsinh((w - m) / s)))
            got += 1
    return out


def one(item):
    ds, z = item
    x, y = z[:T], z[T:]
    seasons = pdm.estimate_seasons(x)
    d = pdm.decompose(x, seasons)
    ours = d.trend[-1] + d.seasonal_at(np.arange(T, T + L))
    mt = bl.original_decompose(x, L, k_top=8)
    original = mt["trend"][-1] + mt["seasonal_horizon"]
    if seasons:
        p = int(round(seasons[0].period if len(seasons) == 1 else seasons[-1].period))
        snaive = np.array([x[T - p + (i % p)] for i in range(L)])
    else:
        snaive = np.full(L, x[-1])
    naive = np.full(L, x[-1])
    r = lambda f: float(np.sqrt(np.mean((f - y) ** 2)))
    # how much of the detected-period power stays in each method's trend
    p_main = seasons[0].period if seasons else None

    def leak(trend):
        if p_main is None:
            return np.nan
        c = trend - trend.mean()
        P = np.abs(np.fft.rfft(c)) ** 2
        f = np.fft.rfftfreq(T)
        mask = np.zeros_like(P, bool)
        for h in (1, 2, 3):
            mask |= np.abs(f - h / p_main) <= 1.0 / T
        P0 = np.abs(np.fft.rfft(x - x.mean())) ** 2
        return float(P[mask].sum() / max(P0[mask].sum(), 1e-12))

    return dict(dataset=ds, periods=json.dumps([round(s.period, 1) for s in seasons]),
                n_seasons=len(seasons), rmse_ours=r(ours), rmse_original=r(original),
                rmse_snaive=r(snaive), rmse_naive=r(naive),
                leak_ours=leak(d.trend), leak_original=leak(mt["trend"]),
                noise_var_ours=float(d.noise.var() / x.var()),
                noise_var_original=float(mt["noise"].var() / x.var()))


def figure(items, path):
    pick = {}
    for ds, z in items:
        if ds in ("Electricity", "London_Smart_Meters", "Q-TRAFFIC", "Solar", "M4_Daily", "KDD_Cup_2018") \
                and ds not in pick:
            pick[ds] = z
    fig, axes = plt.subplots(len(pick), 2, figsize=(17, 2.9 * len(pick)))
    for row, (ds, z) in zip(axes, pick.items()):
        x = z[:T]
        seasons = pdm.estimate_seasons(x)
        d = pdm.decompose(x, seasons)
        mt = bl.original_decompose(x, L, k_top=8)
        t_all = np.arange(T + L)
        row[0].plot(t_all, z, color="0.65", lw=0.8, label="series (norm)")
        row[0].plot(np.arange(T), mt["trend"], lw=1.2, label="trend: original")
        row[0].plot(np.arange(T), d.trend, lw=1.6, label="trend: ours")
        row[0].axvline(T, color="k", ls="--", lw=0.8)
        row[0].set_title(f"{ds}: trend   (est. periods {[round(s.period, 1) for s in seasons]})", fontsize=9)
        row[0].legend(fontsize=7, loc="upper left")
        lo = max(0, T - 3 * int(max([s.period for s in seasons], default=48)))
        tt = np.arange(lo, T + L)
        row[1].plot(tt, z[lo:] - np.r_[d.trend, np.full(L, d.trend[-1])][lo:], color="0.65", lw=0.8,
                    label="series - our trend")
        row[1].plot(tt, np.r_[mt["seasonal"], mt["seasonal_horizon"]][lo:], lw=1.2, label="seasonal: original (Top-K)")
        row[1].plot(tt, d.seasonal_at(tt), lw=1.6, label="seasonal: ours (Fourier @ est. p)")
        row[1].axvline(T, color="k", ls="--", lw=0.8)
        row[1].set_title("seasonal: context + horizon extrapolation", fontsize=9)
        row[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=85)


def main():
    items = windows()
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        rows = pool.map(one, items, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ver0_rows.csv", index=False)
    figure(items, OUT / "ver0_examples.png")
    print(len(df))


if __name__ == "__main__":
    main()
