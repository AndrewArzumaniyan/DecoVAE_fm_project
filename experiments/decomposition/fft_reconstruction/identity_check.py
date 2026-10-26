"""Identity coefficients: isolate mathematical errors before any training."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import FFTReconstructionArm, VARIANTS


def identity_table(horizons=(96, 128, 256), draws=64, context=512, k_top=16):
    torch.set_num_threads(1)
    time = torch.arange(context + max(horizons), dtype=torch.float64)
    rng = np.random.default_rng(0)
    models = {variant: FFTReconstructionArm(variant) for variant in VARIANTS}
    rows = []
    for period in (12., 24., 32., 48., 64., 96., 128., 256., 19.7, 37.1, 168.):
        for draw in range(draws):
            truth = sum(amp * torch.sin(2 * torch.pi * harmonic * time / period + phase)
                        for harmonic, amp, phase in zip((1, 2, 3), (1., .5, .3),
                                                       rng.uniform(-np.pi, np.pi, 3)))
            fft = torch.fft.rfft(truth[:context])
            indices = torch.topk(fft.abs(), k_top).indices[None]
            coefficients = fft[indices][..., None]
            for horizon in horizons:
                for variant in VARIANTS:
                    predicted = models[variant]._reconstruct_time_from_freq_tokens(
                        coefficients, indices, context, context + horizon)[0]
                    errors = {}
                    for name, sl in (("context", slice(0, context)),
                                     ("horizon", slice(context, context + horizon))):
                        target = truth[sl]
                        errors[f"nrmse_{name}"] = float(((predicted[sl] - target) ** 2).mean().sqrt()
                                                       / target.std(unbiased=False))
                    rows.append(dict(period=period, horizon=horizon, variant=variant, draw=draw,
                                     **errors))
    return pd.DataFrame(rows).groupby(["period", "horizon", "variant"], as_index=False).mean().drop(columns="draw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = identity_table()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
