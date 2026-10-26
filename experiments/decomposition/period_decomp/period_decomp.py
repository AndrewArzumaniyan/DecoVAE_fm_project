"""Per-window period estimation and period-aware trend/seasonal/noise decomposition.

Prototype for "strategy 2": the decomposition scale is estimated from each context
window by a fixed algorithm (not learned through the forecasting loss).

Pipeline for one context window x of length T:
  1. Candidate periods: peaks of the Hann-windowed periodogram of a stiffly detrended x,
     refined on the ACF (integer lag) and then on a fine continuous grid (harmonic fit RSS).
  2. Model selection: {no seasonality} U candidates, with the number of harmonics H,
     chosen by BIC on an effective sample size (corrects for coloured residuals).
     A second period is searched on the residual of the first (MSTL-like).
  3. Decomposition by backfitting: trend = HP filter whose cutoff is tied to the period
     (lambda ~ p^4, Ravn-Uhlig scaling); seasonal = Fourier series at the estimated
     (possibly non-integer) period, so its extrapolation to the horizon is exact.

Everything is plain numpy/scipy and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solveh_banded

CALENDAR_PERIODS = (7.0, 12.0, 24.0, 48.0, 52.0, 96.0, 144.0, 168.0, 288.0, 336.0, 365.0)
HARMONIC_GRID = (1, 2, 3, 4, 6, 8, 12)


# ----------------------------------------------------------------------------- HP filter
def hp_lambda(cutoff_period: float) -> float:
    """HP smoothing parameter whose gain is 0.5 at ``cutoff_period``.

    HP trend gain at angular frequency w is 1 / (1 + 4*lam*(1 - cos w)^2); solving for
    gain = 1/2 gives lam = 1 / (4 (1 - cos w_c)^2) ~ (p_c / 2pi)^4 for long periods,
    i.e. the Ravn-Uhlig fourth-power scaling.
    """
    w = 2.0 * np.pi / cutoff_period
    return 1.0 / (4.0 * (1.0 - np.cos(w)) ** 2)


def hp_filter(x: np.ndarray, lam: float) -> np.ndarray:
    """argmin_t ||x - t||^2 + lam ||D2 t||^2, solved as a banded SPD system."""
    n = len(x)
    if n < 3:
        return x.copy()
    main = np.full(n, 6.0)
    main[[0, -1]] = 1.0
    main[[1, -2]] = 5.0
    off1 = np.full(n - 1, -4.0)
    off1[[0, -1]] = -2.0
    off2 = np.ones(n - 2)
    ab = np.zeros((3, n))
    ab[2] = 1.0 + lam * main
    ab[1, 1:] = lam * off1
    ab[0, 2:] = lam * off2
    return solveh_banded(ab, x, lower=False)


# ----------------------------------------------------------------------- seasonal fits
def fourier_design(t: np.ndarray, period: float, n_harm: int) -> np.ndarray:
    cols = []
    for h in range(1, n_harm + 1):
        w = 2.0 * np.pi * h * t / period
        cols.append(np.cos(w))
        cols.append(np.sin(w))
    return np.stack(cols, axis=1)


def fit_seasonal(r: np.ndarray, period: float, n_harm: int):
    """Least-squares Fourier seasonal on residual r. Returns (coef, fitted, rss)."""
    t = np.arange(len(r), dtype=float)
    X = fourier_design(t, period, n_harm)
    coef, *_ = np.linalg.lstsq(X, r - r.mean(), rcond=None)
    fitted = X @ coef
    rss = float(((r - r.mean() - fitted) ** 2).sum())
    return coef, fitted, rss


def eval_seasonal(coef: np.ndarray, period: float, t: np.ndarray) -> np.ndarray:
    n_harm = len(coef) // 2
    return fourier_design(t.astype(float), period, n_harm) @ coef


def max_harmonics(period: float) -> int:
    return max(1, int(np.floor(period / 2.0 - 1e-9)))


# ------------------------------------------------------------------------- utilities
def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    a = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return a / a[0] if a[0] > 0 else np.zeros(max_lag + 1)


def effective_n(resid: np.ndarray) -> float:
    """n_eff = n (1 - rho) / (1 + rho) for AR(1)-like residual correlation (rho >= 0)."""
    rho = max(0.0, float(acf(resid, 1)[1]))
    return len(resid) * (1.0 - rho) / (1.0 + rho)


def bic(rss: float, n_eff: float, n_params: int) -> float:
    return n_eff * np.log(max(rss, 1e-12) / n_eff) + n_params * np.log(n_eff)


# ---------------------------------------------------------------- candidate periods
def candidate_periods(r: np.ndarray, p_min: float, p_max: float, n_peaks: int = 6,
                      peak_ratio: float = 4.0) -> list[float]:
    """Periodogram peaks (local background ratio) refined on ACF to an integer lag."""
    T = len(r)
    win = np.hanning(T)
    P = np.abs(np.fft.rfft((r - r.mean()) * win)) ** 2
    k_lo = max(1, int(np.ceil(T / p_max)))
    k_hi = min(len(P) - 2, int(np.floor(T / p_min)))
    if k_hi <= k_lo:
        return []
    peaks = []
    for k in range(k_lo, k_hi + 1):
        if P[k] >= P[k - 1] and P[k] >= P[k + 1]:
            lo, hi = max(1, k - 12), min(len(P), k + 13)
            bg = np.median(P[lo:hi])
            if bg > 0 and P[k] / bg >= peak_ratio:
                peaks.append((P[k] / bg * P[k], k))
    peaks.sort(reverse=True)
    a = acf(r, int(min(T // 2, np.ceil(p_max * 1.3))))
    cands: list[float] = []
    for _, k in peaks[:n_peaks]:
        lag_lo = max(2, int(np.floor(T / (k + 0.5))))
        lag_hi = min(len(a) - 1, int(np.ceil(T / (k - 0.5))) if k > 0.5 else len(a) - 1)
        if lag_hi < lag_lo:
            continue
        lags = np.arange(lag_lo, lag_hi + 1)
        lags = lags[lags <= p_max]
        if len(lags) == 0:
            continue
        best = float(lags[np.argmax(a[lags])])
        cands.append(best)
        for m in (2, 3):  # periodogram peak may be a harmonic of the true period
            if best * m <= p_max:
                cands.append(best * m)
    return sorted(set(cands))


def refine_period(r: np.ndarray, p0: float, n_harm: int, half_width: float = 1.0,
                  step: float = 0.05) -> float:
    grid = np.arange(max(2.0, p0 - half_width), p0 + half_width + 1e-9, step)
    rss = [fit_seasonal(r, p, min(n_harm, max_harmonics(p)))[2] for p in grid]
    return float(grid[int(np.argmin(rss))])


def holdout_skill(r: np.ndarray, period: float, n_harm: int) -> float:
    """Does the seasonal pattern EXTRAPOLATE? Fit on one half, predict the other (both ways).

    skill = 1 - RSS(seasonal) / RSS(mean only) on the held-out half, averaged over the two
    directions. A real periodic component predicts the other half; red-noise bumps or trend
    leakage that happen to fit one half do not.
    """
    T = len(r)
    t = np.arange(T, dtype=float)
    h = T // 2
    skills = []
    for tr_idx, te_idx in ((slice(0, h), slice(h, T)), (slice(h, T), slice(0, h))):
        Xtr = fourier_design(t[tr_idx], period, n_harm)
        mu = r[tr_idx].mean()
        coef, *_ = np.linalg.lstsq(Xtr, r[tr_idx] - mu, rcond=None)
        pred = fourier_design(t[te_idx], period, n_harm) @ coef
        rss0 = ((r[te_idx] - mu) ** 2).sum()
        rss1 = ((r[te_idx] - mu - pred) ** 2).sum()
        skills.append(1.0 - rss1 / rss0 if rss0 > 0 else 0.0)
    return float(np.mean(skills))


# --------------------------------------------------------------------- the estimator
@dataclass
class SeasonSpec:
    period: float
    n_harm: int


@dataclass
class Decomposition:
    trend: np.ndarray
    seasonal: np.ndarray            # sum of seasonal components on the context
    noise: np.ndarray
    seasons: list[SeasonSpec] = field(default_factory=list)
    coefs: list[np.ndarray] = field(default_factory=list)
    trend_cutoff: float = 0.0

    def seasonal_at(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros(len(t))
        for s, c in zip(self.seasons, self.coefs):
            out += eval_seasonal(c, s.period, t)
        return out


@dataclass
class EstimatorConfig:
    p_min: float = 3.0
    max_cycles_frac: float = 1.0 / 3.0      # need >= 3 cycles in the window
    max_seasons: int = 2
    trend_cutoff_mult: float = 1.5          # trend cutoff = mult * (largest period)
    no_season_cutoff_frac: float = 0.125    # trend cutoff when no seasonality: T/8
    trend_cutoff_floor_frac: float = 0.0    # optional floor: cutoff >= frac * T
    selection_cutoff_frac: float = 0.5      # stiff HP detrend used only for selection
    snap_calendar: bool = False
    snap_tol: float = 0.02
    min_explained: float = 0.05             # seasonal must explain >= 5% of residual var
    min_holdout_skill: float = 0.10         # ... and predict the other half of the window
    min_holdout_ratio: float = 0.5          # ... about as well as it fits in-sample
    use_holdout: bool = True

    def __post_init__(self):
        if not self.use_holdout:
            self.min_holdout_skill = -np.inf
            self.min_holdout_ratio = -np.inf


def _select_one(r: np.ndarray, cfg: EstimatorConfig, p_max: float,
                exclude: list[float]) -> SeasonSpec | None:
    rss0 = float(((r - r.mean()) ** 2).sum())
    fits = []                                   # (p, H, rss, fitted)
    for p0 in candidate_periods(r, cfg.p_min, p_max):
        if any(abs(p0 - e) / e < 0.05 for e in exclude):
            continue
        p = refine_period(r, p0, min(4, max_harmonics(p0)))
        if p > p_max:
            continue
        if cfg.snap_calendar:
            near = [c for c in CALENDAR_PERIODS if abs(p - c) / c <= cfg.snap_tol]
            if near:
                p = min(near, key=lambda c: abs(p - c))
        for H in HARMONIC_GRID:
            if H > max_harmonics(p):
                break
            _, fitted, rss = fit_seasonal(r, p, H)
            fits.append((p, H, rss, fitted))
    if not fits:
        return None
    # ONE effective sample size for every candidate and for "no season": taken from the
    # residual of the best-fitting candidate (coloured residual -> fewer effective obs).
    # Per-candidate n_eff would favour over-parameterised multiples (2p, 3p, ...), whose
    # whiter residual inflates n_eff and hence |delta BIC|.
    _, _, _, f_best = min(fits, key=lambda f: f[2])
    n_eff = effective_n(r - r.mean() - f_best)
    scored = sorted(((bic(rss, n_eff, 2 * H + 1) - bic(rss0, n_eff, 0), p, H, rss)
                     for p, H, rss, _ in fits))
    for delta, p, H, rss in scored:
        if delta >= 0:
            break
        explained = 1.0 - rss / rss0
        if explained < cfg.min_explained:
            continue
        ho = holdout_skill(r, p, H)
        # a real seasonal generalises about as well as it fits in-sample
        if ho >= cfg.min_holdout_skill and ho >= cfg.min_holdout_ratio * explained:
            return SeasonSpec(p, H)
    return None


def estimate_seasons(x: np.ndarray, cfg: EstimatorConfig | None = None) -> list[SeasonSpec]:
    cfg = cfg or EstimatorConfig()
    T = len(x)
    p_max = T * cfg.max_cycles_frac
    r = x - hp_filter(x, hp_lambda(T * cfg.selection_cutoff_frac))
    seasons: list[SeasonSpec] = []
    for _ in range(cfg.max_seasons):
        s = _select_one(r, cfg, p_max, [q.period for q in seasons])
        if s is None:
            break
        seasons.append(s)
        _, fitted, _ = fit_seasonal(r, s.period, s.n_harm)
        r = r - r.mean() - fitted
    return sorted(seasons, key=lambda s: s.period)


def decompose(x: np.ndarray, seasons: list[SeasonSpec] | None = None,
              cfg: EstimatorConfig | None = None, n_iter: int = 3,
              trend_cutoff: float | None = None) -> Decomposition:
    """Backfitting: trend (HP, cutoff tied to the period) <-> Fourier seasonals."""
    cfg = cfg or EstimatorConfig()
    T = len(x)
    if seasons is None:
        seasons = estimate_seasons(x, cfg)
    if trend_cutoff is None:
        trend_cutoff = (cfg.trend_cutoff_mult * max(s.period for s in seasons)
                        if seasons else T * cfg.no_season_cutoff_frac)
        trend_cutoff = max(trend_cutoff, cfg.trend_cutoff_floor_frac * T)
    lam = hp_lambda(trend_cutoff)
    comps = [np.zeros(T) for _ in seasons]
    coefs = [np.zeros(2 * s.n_harm) for s in seasons]
    trend = hp_filter(x, hp_lambda(max(trend_cutoff, T * cfg.selection_cutoff_frac)))
    for _ in range(n_iter):
        for j, s in enumerate(seasons):
            others = sum((c for i, c in enumerate(comps) if i != j), np.zeros(T))
            coefs[j], comps[j], _ = fit_seasonal(x - trend - others, s.period, s.n_harm)
        seas = sum(comps, np.zeros(T))
        trend = hp_filter(x - seas, lam)
    seas = sum(comps, np.zeros(T))
    return Decomposition(trend=trend, seasonal=seas, noise=x - trend - seas,
                         seasons=list(seasons), coefs=coefs, trend_cutoff=trend_cutoff)
