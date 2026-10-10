"""Reference decompositions to compare against.

* ``original_decompose``: numpy re-implementation of decovaefm's deterministic path
  (MultiScaleDecomposition with reflect padding + Top-K rFFT band-limited seasonal), and the
  horizon extrapolation of the seasonal bins exactly as in
  ``DecoVAEFM._reconstruct_time_from_freq_tokens`` with an identity decoder (best case).
* ``stl_decompose``: statsmodels STL / MSTL with a given (oracle) integer period.
"""

from __future__ import annotations

import numpy as np

KERNELS = (5, 9, 13, 25)
W_UNIFORM = np.zeros(4)
# softmax weights the model converged to in our reproduction of debug_training.ipynb
W_TRAINED = np.log(np.array([0.654, 0.105, 0.016, 0.225]))


def _softmax(w):
    e = np.exp(w - w.max())
    return e / e.sum()


def original_decompose(x: np.ndarray, L: int, k_top: int = 8, logits=W_UNIFORM):
    T = len(x)
    w = _softmax(np.asarray(logits, float))
    trend = np.zeros(T)
    for wk, k in zip(w, KERNELS):
        xp = np.pad(x, (k // 2, k // 2), mode="reflect")
        trend += wk * np.convolve(xp, np.ones(k) / k, mode="valid")
    s_raw = x - trend
    f = np.fft.rfft(s_raw)
    K = min(k_top, len(f))
    idx = np.argsort(-np.abs(f), kind="stable")[:K]
    sp = np.zeros_like(f)
    sp[idx] = f[idx]
    seasonal = np.fft.irfft(sp, n=T)
    # horizon: remap bins to length T+L as in the model (identity decoder = best case)
    n_freq = (T + L) // 2 + 1
    sidx = np.clip(np.round(idx * (T + L) / T).astype(int), 0, n_freq - 1)
    sp2 = np.zeros(n_freq, dtype=complex)
    sp2[sidx] = f[idx]
    seas_full = np.fft.irfft(sp2, n=T + L)
    return dict(trend=trend, seasonal=seasonal, noise=x - trend - seasonal,
                seasonal_horizon=seas_full[T:])


def stl_decompose(x: np.ndarray, L: int, periods):
    from statsmodels.tsa.seasonal import MSTL, STL

    periods = [int(round(p)) for p in periods]
    T = len(x)
    if len(periods) == 0:
        return None
    if len(periods) == 1:
        res = STL(x, period=periods[0], robust=False).fit()
        seas_list = [np.asarray(res.seasonal)]
    else:
        res = MSTL(x, periods=periods).fit()
        s = np.asarray(res.seasonal)
        seas_list = [s[:, j] for j in range(s.shape[1])]
    seasonal = sum(seas_list)
    horizon = np.zeros(L)
    for p, s in zip(periods, seas_list):   # tile the last full cycle of each component
        last = s[T - p:]
        horizon += np.array([last[(i) % p] for i in range(L)])
    trend = np.asarray(res.trend)
    return dict(trend=trend, seasonal=seasonal, noise=x - trend - seasonal,
                seasonal_horizon=horizon)
