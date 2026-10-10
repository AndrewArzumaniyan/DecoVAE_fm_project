"""Synthetic series with known trend / seasonal / noise components and known sigma_t.

Design choices (so that later experiments can test what the original dummy data cannot):
  * seasonal shapes are non-sinusoidal (1-4 harmonics), periods come from a calendar set and
    usually do NOT divide the context length;
  * noise volatility follows its own stochastic-volatility process, independent of the level
    (unlike DummyRandomDataset where sigma = f(signal));
  * optional coloured (AR(1)) noise and slowly varying seasonal amplitude as stress tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Series:
    x: np.ndarray            # full series, length T + L
    trend: np.ndarray
    seasonal: np.ndarray     # sum of seasonal components (zero mean over each period)
    noise: np.ndarray
    sigma: np.ndarray        # true conditional std of the innovation
    periods: tuple[float, ...]
    meta: dict


def _trend(rng, n, kind):
    t = np.arange(n) / n
    if kind == "none":
        return np.zeros(n)
    if kind == "linear":
        return rng.uniform(-2, 2) * t
    if kind == "kink":
        k = rng.uniform(0.2, 0.8)
        s1, s2 = rng.uniform(-2, 2, size=2)
        return np.where(t < k, s1 * t, s1 * k + s2 * (t - k))
    if kind == "rw":   # smooth random walk: integrated, then smoothed noise
        z = np.cumsum(rng.standard_normal(n))
        z = np.convolve(z, np.ones(31) / 31, mode="same")
        z -= z.mean()
        return 1.5 * z / (z.std() + 1e-12)
    raise ValueError(kind)


def _season(rng, n, p):
    t = np.arange(n)
    H = rng.integers(1, 5)
    amps = rng.uniform(0.3, 1.0, size=H) / np.arange(1, H + 1)
    phases = rng.uniform(0, 2 * np.pi, size=H)
    s = sum(a * np.cos(2 * np.pi * (h + 1) * t / p + ph) for h, (a, ph) in enumerate(zip(amps, phases)))
    return s / (s.std() + 1e-12)


def _sv_noise(rng, n, ar=0.0, phi=0.97, eta=0.25):
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = phi * h[i - 1] + eta * rng.standard_normal()
    sigma = np.exp(h / 2)
    sigma /= sigma.mean()
    eps = sigma * rng.standard_normal(n)
    if ar > 0:
        e = np.zeros(n)
        for i in range(n):
            e[i] = (ar * e[i - 1] if i else 0.0) + eps[i]
        eps = e * np.sqrt(1 - ar ** 2)      # keep marginal scale comparable
    return eps, sigma


def make_series(rng, T, L, periods=(), trend="linear", snr=2.0, ar=0.0,
                amp_mod=False, season_amps=(1.0, 0.6)) -> Series:
    """snr = std(total seasonal) / mean noise std. Trend amplitude is O(1) in the same units."""
    n = T + L
    tr = _trend(rng, n, trend)
    seas = np.zeros(n)
    for j, p in enumerate(periods):
        seas += season_amps[j] * _season(rng, n, p)
    if amp_mod and len(periods):
        seas *= 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / (n * rng.uniform(1, 3)) + rng.uniform(0, 6.3))
    noise, sigma = _sv_noise(rng, n, ar=ar)
    if len(periods):
        seas *= snr / (seas[:T].std() + 1e-12)
        tr *= snr                                   # keep trend comparable to the seasonal
    x = tr + seas + noise
    return Series(x, tr, seas, noise, sigma, tuple(float(p) for p in periods),
                  dict(trend=trend, snr=snr, ar=ar, amp_mod=amp_mod))
