"""Paired training of the three FFT-only ablations; run from any working directory.

python experiments/decomposition/fft_reconstruction/train.py --profile full
python experiments/decomposition/fft_reconstruction/train.py --profile smoke
The full profile runs 3 horizons x 3 seeds x 3 variants, each for 30 epochs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from model import FFTReconstructionArm, VARIANTS
from datasets import T, EXPERIMENTS, build_split

sys.path.insert(0, str(EXPERIMENTS / "benchmark"))
from runner import loss as benchmark_loss
from paths import RUNS_DIR, REPO_ROOT

MODEL_KWARGS = dict(d_model=64, patch_size=16, stride=16, n_heads=4,
                    enc_layers=2, dec_layers=2, k_top=16)


def initialize_worker(root, config, threads):
    global WORKER_RAW, WORKER_CONFIG, WORKER_ROOT
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    WORKER_ROOT, WORKER_CONFIG = Path(root), config
    WORKER_RAW = {split: dict(np.load(WORKER_ROOT / f"{split}.npz"))
                  for split in ("train", "val", "test")}


def worker_task(variant, seed, horizon):
    return train_one(variant, seed, horizon, WORKER_RAW, WORKER_CONFIG, WORKER_ROOT)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def provenance(path, config):
    sources = list((REPO_ROOT / "decovaefm/models").glob("*.py"))
    sources += list((EXPERIMENTS / "benchmark").glob("*.py"))
    sources += list(Path(__file__).parent.glob("*.py"))
    sources += [EXPERIMENTS / "paths.py", REPO_ROOT / "uv.lock"]
    hashes = {}
    for source in sources:
        name = str(source.relative_to(REPO_ROOT))
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        target = path / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    versions = {}
    for name in ("torch", "numpy", "pandas", "matplotlib", "scipy", "lightning"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(config=config, python=sys.version, platform=platform.platform(), versions=versions,
                  source_hashes=hashes, device="cpu", torch_threads=torch.get_num_threads(),
                  loss="benchmark.runner.loss, unchanged (Gaussian recon + forecast NLL + 0.01 KL)",
                  normalization="linear; per-series context statistics, unchanged benchmark convention",
                  environment_note="Available local Python/PyTorch; not the uv.lock-pinned environment",
                  model=MODEL_KWARGS, evaluation_seed=1729,
                  data_seeds=dict(train=719, val=100719, test=200719))
    write_json(path / "provenance.json", result)
    return result


@torch.no_grad()
def evaluate(model, raw, horizon, draws, save_path=None):
    """Common latent/observation draws; metrics in context-standardized units."""
    model.eval()
    rows = []
    saved = {key: [] for key in ("mean", "trend", "seasonal", "variance")}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        for start in range(0, len(raw["z"]), 16):
            sl = slice(start, start + 16)
            x = torch.from_numpy(raw["z"][sl, :T])
            y = raw["z"][sl, T:T + horizon]
            means = []
            samples = []
            trends = []
            seasons = []
            variances = []
            for _ in range(draws):
                out = model(x)
                means.append(out["forecast_mu"].numpy())
                var = out["forecast_logvar"].exp()
                samples.append((out["forecast_mu"] + var.sqrt() * torch.randn_like(var)).numpy())
                variances.append(var.numpy())
                trends.append(out["decoded_trend"].numpy())
                seasons.append(out["decoded_seasonal"].numpy())
            mean = np.mean(means, axis=0)
            seasonal = np.mean(seasons, axis=0)
            trend = np.mean(trends, axis=0)
            observation_variance = np.mean(variances, axis=0)
            observations = np.stack(samples)
            ordered = np.sort(observations, axis=0)
            weights = (2 * np.arange(1, draws + 1) - draws - 1)[:, None, None]
            crps = np.abs(observations - y).mean(0) - (weights * ordered).sum(0) / draws ** 2
            low, high = np.quantile(observations, [.05, .95], axis=0)
            center, scale = raw["mean"][sl], raw["scale"][sl]
            gt_t = (raw["trend"][sl, :T + horizon] - center) / scale
            gt_s = raw["seasonal"][sl, :T + horizon] / scale
            input_t = out["target_trend"].numpy()
            input_s = out["target_seasonal"].numpy()
            s_centered = gt_s[:, :T] - gt_s[:, :T].mean(1, keepdims=True)
            t_error = input_t - gt_t[:, :T]
            t_error -= t_error.mean(1, keepdims=True)
            denominator = (s_centered ** 2).sum(1)
            gain = (seasonal[:, :T] * s_centered).sum(1) / np.maximum(denominator, 1e-12)
            leak = (t_error * s_centered).sum(1) / np.maximum(denominator, 1e-12)
            gain[denominator < 1e-10] = np.nan
            leak[denominator < 1e-10] = np.nan
            metrics = dict(
                ncrps=crps.mean(1), nrmse=np.sqrt(((mean - y) ** 2).mean(1)),
                coverage90=((y >= low) & (y <= high)).mean(1), width90=(high - low).mean(1),
                recon_nrmse=np.sqrt(((trend[:, :T] + seasonal[:, :T] - x.numpy()) ** 2).mean(1)),
                seasonal_ctx_nrmse=np.sqrt(((seasonal[:, :T] - gt_s[:, :T]) ** 2).mean(1)),
                seasonal_hor_nrmse=np.sqrt(((seasonal[:, T:] - gt_s[:, T:]) ** 2).mean(1)),
                trend_ctx_nrmse=np.sqrt(((trend[:, :T] - gt_t[:, :T]) ** 2).mean(1)),
                trend_hor_nrmse=np.sqrt(((trend[:, T:] - gt_t[:, T:]) ** 2).mean(1)),
                input_trend_nrmse=np.sqrt(((input_t - gt_t[:, :T]) ** 2).mean(1)),
                input_seasonal_nrmse=np.sqrt(((input_s - gt_s[:, :T]) ** 2).mean(1)),
                input_leak_abs=np.abs(leak), seasonal_gain=gain,
                seasonal_rms=np.sqrt((seasonal[:, :T] ** 2).mean(1)),
                seasonal_share=seasonal[:, :T].var(1) / np.maximum(
                    seasonal[:, :T].var(1) + trend[:, :T].var(1), 1e-12),
                observation_variance=observation_variance.mean(1))
            for part in range(3):
                a, b = part * horizon // 3, (part + 1) * horizon // 3
                metrics[f"ncrps_h{part + 1}"] = crps[:, a:b].mean(1)
            for j in range(len(x)):
                i = start + j
                rows.append(dict(series_id=i, case=str(raw["case"][i]), scenario=int(raw["scenario"][i]),
                                 **{key: float(value[j]) for key, value in metrics.items()}))
            for key, value in (("mean", mean), ("trend", trend), ("seasonal", seasonal),
                               ("variance", observation_variance)):
                saved[key].append(value)
    if save_path is not None:
        np.savez_compressed(save_path, **{key: np.concatenate(value) for key, value in saved.items()})
    return pd.DataFrame(rows)


def train_one(variant, seed, horizon, raw, config, root):
    path = root / f"h{horizon}_s{seed}_{variant}"
    path.mkdir(exist_ok=True)
    if (path / "metrics.csv").exists():
        status = json.loads((path / "status.json").read_text())
        if status["status"] == "complete":
            return pd.read_csv(path / "metrics.csv"), status
    torch.manual_seed(seed)
    model = FFTReconstructionArm(variant=variant, pred_len=horizon, **MODEL_KWARGS)
    initial_hash = state_hash(model.state_dict())
    initial_decomp = model.decomposition.weights.detach().softmax(0).tolist()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Reinitialise dropout/latent RNG independently of model construction, identically in all arms.
    torch.manual_seed(40_000 + seed)
    random.seed(seed)
    order_rng = np.random.default_rng(seed)
    history = []
    best = float("inf")
    started = time.monotonic()
    order_hashes = []
    for epoch in range(config["epochs"]):
        model.train()
        order = order_rng.permutation(len(raw["train"]["z"]))
        order_hashes.append(hashlib.sha256(order.tobytes()).hexdigest())
        values = []
        for start in range(0, len(order), 32):
            ids = order[start:start + 32]
            x = torch.from_numpy(raw["train"]["z"][ids, :T])
            y = torch.from_numpy(raw["train"]["z"][ids, T:T + horizon])
            optimizer.zero_grad(set_to_none=True)
            value = benchmark_loss(model(x), x, y)
            if not torch.isfinite(value):
                raise RuntimeError(f"Nonfinite loss: {variant}, seed {seed}, horizon {horizon}")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            values.append(value.item())
        validation = evaluate(model, raw["val"], horizon, config["val_draws"])
        score = float(validation.ncrps.mean())
        if score < best:
            best = score
            best_epoch = epoch + 1
            torch.save(dict(model=model.state_dict(), variant=variant, horizon=horizon, seed=seed,
                            epoch=best_epoch, initial_state_sha256=initial_hash,
                            model_kwargs=MODEL_KWARGS), path / "best.pt")
        history.append(dict(epoch=epoch + 1, train_loss=np.mean(values), val_ncrps=score,
                            seconds=time.monotonic() - started))
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps(dict(event="epoch", horizon=horizon, seed=seed, variant=variant,
                                  **history[-1])), flush=True)
    model.load_state_dict(torch.load(path / "best.pt", weights_only=False)["model"])
    metrics = evaluate(model, raw["test"], horizon, config["test_draws"], path / "predictions.npz")
    metrics = metrics.assign(variant=variant, seed=seed, horizon=horizon)
    metrics.to_csv(path / "metrics.csv", index=False)
    status = dict(status="complete", seed=seed, variant=variant, horizon=horizon,
                  epochs=config["epochs"], best_epoch=best_epoch, best_val_ncrps=best,
                  initial_state_sha256=initial_hash, parameter_count=sum(p.numel() for p in model.parameters()),
                  initial_decomposition_weights=initial_decomp,
                  final_decomposition_weights=model.decomposition.weights.detach().softmax(0).tolist(),
                  batch_order_sha256=order_hashes, seconds=time.monotonic() - started)
    write_json(path / "status.json", status)
    print(json.dumps(dict(event="run_complete", **status)), flush=True)
    return metrics, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=3, help="Independent CPU processes, each with its own RNG")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if any(h < 16 or h % 16 for h in args.horizons):
        parser.error("Horizons must be positive multiples of the native model's patch/stride 16")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = (dict(train=2048, val=256, test=512, diagnostic_per_case=32, epochs=30,
                   val_draws=32, test_draws=128) if args.profile == "full" else
              dict(train=64, val=16, test=16, diagnostic_per_case=2, epochs=1,
                   val_draws=2, test_draws=4))
    config.update(profile=args.profile, horizons=args.horizons, seeds=args.seeds, variants=VARIANTS,
                  jobs=args.jobs, threads=args.threads,
                  optimizer="AdamW", lr=.001, batch_size=32, clip_grad=1.,
                  checkpoint_selection="minimum validation CRPS over fixed 30-epoch budget (full)")
    run_id = args.run_id or f"fft_reconstruction_{args.profile}"
    root = RUNS_DIR / run_id
    root.mkdir(parents=True, exist_ok=True)
    if (root / "provenance.json").exists():
        previous = json.loads((root / "provenance.json").read_text())
        if previous["config"] != json.loads(json.dumps(config)):
            raise ValueError("Run directory contains a different configuration; choose another --run-id")
        # Completed arms may only be reused with unchanged sources/runtime.
        for name, sha in previous["source_hashes"].items():
            if hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Source changed since this run: {name}; choose another --run-id")
        if previous["python"] != sys.version or previous["versions"]["torch"] != torch.__version__:
            raise ValueError("Runtime changed since this run; choose another --run-id")
    else:
        provenance(root, config)
    raw = {}
    for split, seed in (("train", 719), ("val", 100719), ("test", 200719)):
        target = root / f"{split}.npz"
        if not target.exists():
            arrays = build_split(config[split], seed, max(args.horizons),
                                 config["diagnostic_per_case"] if split == "test" else 0)
            np.savez_compressed(target, **arrays)
        raw[split] = dict(np.load(target))
    write_json(root / "data_hashes.json", {split: hashlib.sha256((root / f"{split}.npz").read_bytes()).hexdigest()
                                         for split in raw})
    results = []
    statuses = []
    print(json.dumps(dict(event="start", config=config, output=str(root))), flush=True)
    tasks = [(variant, seed, horizon) for horizon in args.horizons
             for seed in args.seeds for variant in VARIANTS]
    # Process isolation is required: threads would share PyTorch's dropout/latent RNG.
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(root), config, args.threads)) as pool:
        futures = [pool.submit(worker_task, *task) for task in tasks]
        for future in as_completed(futures):
            metrics, status = future.result()
            results.append(metrics)
            statuses.append(status)
            pd.concat(results).sort_values(["horizon", "seed", "variant", "series_id"]).to_csv(
                root / "all_metrics.csv", index=False)
            write_json(root / "progress.json", dict(completed=len(statuses), total=len(tasks), statuses=statuses))
    for horizon in args.horizons:
        for seed in args.seeds:
            paired = [s for s in statuses if s["horizon"] == horizon and s["seed"] == seed]
            assert len(paired) == 3
            assert len({s["initial_state_sha256"] for s in paired}) == 1
            assert len({tuple(s["batch_order_sha256"]) for s in paired}) == 1
            assert len({s["parameter_count"] for s in paired}) == 1
    write_json(root / "status.json", dict(status="complete", runs=len(statuses), config=config))
    print(json.dumps(dict(event="complete", runs=len(statuses), output=str(root))), flush=True)


if __name__ == "__main__":
    main()
