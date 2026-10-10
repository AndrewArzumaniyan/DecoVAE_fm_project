"""Use the existing benchmark generator, with shared paths for all horizons."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
spec = importlib.util.spec_from_file_location("fft_benchmark_data", EXPERIMENTS / "benchmark/data.py")
benchmark_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_data)
T = benchmark_data.T
DIAGNOSTICS = (
    ("period_32", "period", 32.), ("period_64", "period", 64.),
    ("period_128", "period", 128.), ("period_256", "period", 256.),
    ("period_19_7", "period", 19.7), ("period_37_1", "period", 37.1),
    ("period_168", "period", 168.),
    ("dual_24_168", "dual", (24., 168., 1.)),
    ("dual_17_3_96", "dual", (17.3, 96., 1.)),
    ("amplitude_growth", "amplitude", .5),
    ("negative_noise_only", "negative", "noise"),
    ("negative_trend_noise", "negative", "trend"),
)


@contextmanager
def generator_horizon(horizon):
    previous = benchmark_data.L
    benchmark_data.L = horizon
    try:
        yield
    finally:
        benchmark_data.L = previous


def build_split(n, seed, max_horizon, n_diagnostic=0):
    """45 stationary base scenarios; diagnostics are held out from training/validation.

    Generate T+max_horizon once, and slice the very same paths for each horizon.
    The amplitude-growth diagnostic's envelope is thus identical across horizons.
    """
    with generator_horizon(max_horizon):
        rows = []
        labels = []
        cases = []
        for i in range(n):
            label = i % len(benchmark_data.SCENARIOS)
            rows.append(benchmark_data.sample(seed + i, benchmark_data.SCENARIOS[label]))
            labels.append(label)
            cases.append("base")
        for ci, case in enumerate(DIAGNOSTICS):
            for i in range(n_diagnostic):
                rows.append(benchmark_data.diagnostic_sample(seed + 1_000_000 + ci * 10_000 + i, case))
                labels.append(-1)
                cases.append(case[0])
    arrays = {key: np.stack([row[key] for row in rows]).astype("float32")
              for key in ("x", "trend", "seasonal", "noise", "variance")}
    arrays["case"] = np.asarray(cases)
    arrays["scenario"] = np.asarray(labels)
    z, mean, scale = benchmark_data.normalize(arrays["x"])
    arrays.update(z=z.astype("float32"), mean=mean.astype("float32"), scale=scale.astype("float32"))
    return arrays
