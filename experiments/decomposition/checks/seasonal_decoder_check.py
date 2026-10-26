"""Identity-decoder check of the seasonal path in decovaefm (no training).

The seasonal encoder keeps the top-K rFFT bins of the length-T context. Here the "decoder" is the
identity: the true top-K coefficients are passed unchanged to
DecoVAEFM._reconstruct_time_from_freq_tokens, which scatters them onto the rFFT grid of length T+L
and inverts. Any error is therefore caused by the reconstruction and the frequency transfer alone.

Variants
  repo       the method as implemented (bin index rounded to the T+L grid, no rescaling)
  rescaled   same, with coefficients multiplied by (T+L)/T to undo the irfft length normalisation
  exact_k    rescaled, but each kept bin is evaluated at its exact frequency k/T (no rounding);
             the extension is then T-periodic, so it is still wrong when p does not divide T

The remap is exact only when k(T+L)/T is an integer for every kept bin k (here: p = 32).
Seasonal signal: 3 harmonics of period p (amplitudes 1, 0.5, 0.3, random phases), 64 draws per p.
Metric: RMSE / std(true seasonal) on the context [0, T) and on the horizon [T, T+L).
Usage: python seasonal_decoder_check.py [K]    (default K=32; the base config uses k_top=8)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decovaefm.models.decovae_fm import DecoVAEFM  # noqa: E402

T, L = 512, 96
K = int(sys.argv[1]) if len(sys.argv) > 1 else 32
PERIODS = (32, 12, 24, 48, 96, 17.3, 19.7, 30.4, 37.1, 64, 144, 168)
DRAWS = 64
t = torch.arange(T + L, dtype=torch.float64)


def seasonal(p, rng):
    phases = rng.uniform(-math.pi, math.pi, 3)
    return sum(a * torch.sin(2 * math.pi * h * t / p + ph) for h, a, ph in zip((1, 2, 3), (1., .5, .3), phases))


def exact_k(idx, coef):
    out = torch.zeros(T + L, dtype=torch.float64)
    for k, c in zip(idx.tolist(), coef):
        w = 1.0 if k in (0, T // 2) else 2.0
        out += w * (c * torch.exp(2j * math.pi * k * t / T)).real / T
    return out


def nrmse(est, truth, sl):
    return float(torch.sqrt(((est[sl] - truth[sl]) ** 2).mean()) / truth[sl].std())


def main():
    rng = np.random.default_rng(0)
    ctx, hor = slice(0, T), slice(T, T + L)
    print(f'K={K}, T={T}, L={L}; NRMSE context / horizon, mean over {DRAWS} draws')
    print(f'{"period":>7} | {"repo":>13} | {"rescaled":>13} | {"exact_k":>13} | fundamental bin T/p -> (T+L)/p')
    for p in PERIODS:
        res = {v: [] for v in ('repo', 'rescaled', 'exact_k')}
        for _ in range(DRAWS):
            s = seasonal(p, rng)
            fft = torch.fft.rfft(s[:T])
            idx = torch.topk(fft.abs(), K).indices
            coef = fft[idx]
            for name, scale in (('repo', 1.0), ('rescaled', (T + L) / T)):
                est = DecoVAEFM._reconstruct_time_from_freq_tokens(
                    None, (coef * scale)[None, :, None], idx[None], T, T + L)[0]
                res[name].append((nrmse(est, s, ctx), nrmse(est, s, hor)))
            est = exact_k(idx, coef)
            res['exact_k'].append((nrmse(est, s, ctx), nrmse(est, s, hor)))
        cells = [f'{np.mean([r[0] for r in res[v]]):5.2f} / {np.mean([r[1] for r in res[v]]):5.2f}' for v in res]
        print(f'{p:7.1f} | ' + ' | '.join(cells) + f' | {T / p:.3f} -> {(T + L) / p:.3f}')
    print('A zero forecast has NRMSE 1.0.')


if __name__ == '__main__':
    main()
