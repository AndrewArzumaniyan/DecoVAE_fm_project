"""Calibration metrics in the style of Adler et al. (ICLR'26, "Beyond Accuracy"), from the per-point PIT of a forecast.

For a window with PIT values u_t = fraction of predictive samples below y_t (t over the 96 horizon steps):
- PCE  = mean_{q in .1..0.9} | q - mean_t 1[u_t <= q] |              (per window; lower is better, max 0.5)
- CCE  = mean_{s in .2,.4,.6,.8} ( s - mean_t 1[|u_t - .5| <= s/2] )  (per window; > 0 over-confident, < 0 under-confident)
- per-third coverage c50/c80 over horizon steps 1-32, 33-64, 65-96 (calibration along the horizon).
Window values are averaged per dataset, then over datasets (equal weight), as our skill. SIW in the paper has an
ambiguous denominator; we report instead W80 = mean 80%-interval width / context std from the metric rows (width80),
marked as our variant.
"""
import numpy as np

Q = np.round(np.arange(.1, 1., .1), 1)
S = (.2, .4, .6, .8)
THIRDS = (slice(0, 32), slice(32, 64), slice(64, 96))


def window_metrics(pit):
    """pit (N, L) -> dict of (N,) arrays."""
    below = (pit[:, :, None] <= Q).mean(1)                       # (N, |Q|)
    pce = np.abs(Q-below).mean(1)
    cov = np.stack([(np.abs(pit-.5) <= s/2).mean(1) for s in S], 1)
    cce = (np.asarray(S)-cov).mean(1)
    out = dict(pce=pce, cce=cce)
    for k, sl in enumerate(THIRDS, 1):
        for lev in (.5, .8):
            out[f'c{int(lev*100)}_h{k}'] = (np.abs(pit[:, sl]-.5) <= lev/2).mean(1)
    return out
