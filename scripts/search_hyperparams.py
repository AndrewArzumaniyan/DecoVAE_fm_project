"""Hyperparameter search for DecoVAE-FM, selecting the best run by validation WQL.

Search is staged to keep the grid tractable:

1. **dynamics** — sweep learning rate × KL weight on a fixed baseline architecture
2. **capacity** — sweep d_model × enc_layers (and compatible n_heads) with best dynamics
3. **patching** — sweep (patch_size, stride) × k_top with best dynamics + capacity

The best config overall (lowest validation WQL) is written to ``best_params.json``
and used to generate ``train_full.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decovaefm.cli import getparser as get_train_parser
from decovaefm.evaluation import DEFAULT_QUANTILE_LEVELS, evaluate_metrics_on_dataloader
from decovaefm.training import (
    create_dataloaders,
    create_model,
    parse_float_list,
    parse_int_list,
    parse_int_pair_list,
)

ARCH_KEYS = (
    "learning_rate",
    "kl_weight",
    "d_model",
    "n_heads",
    "enc_layers",
    "dec_layers",
    "patch_size",
    "stride",
    "k_top",
)


def getparser():
    parser = get_train_parser()
    parser.description = (
        "Staged DecoVAE-FM hyperparameter search; select by validation WQL "
        f"(quantiles {DEFAULT_QUANTILE_LEVELS[0]}..{DEFAULT_QUANTILE_LEVELS[-1]})"
    )
    parser.add_argument(
        "--search-dir",
        type=str,
        default="./outputs/search",
        help="Directory for search logs, checkpoints, and result files",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="dynamics,capacity,patching",
        help="Comma-separated stages to run: dynamics,capacity,patching",
    )
    parser.add_argument(
        "--learning-rates",
        type=parse_float_list,
        default=[3e-5, 1e-4, 3e-4],
        help="Stage dynamics: learning rates to sweep",
    )
    parser.add_argument(
        "--kl-weights",
        type=parse_float_list,
        default=[1e-4, 1e-3, 1e-2],
        help="Stage dynamics: KL weights to sweep",
    )
    parser.add_argument(
        "--d-models",
        type=parse_int_list,
        default=[64, 128],
        help="Stage capacity: model widths to sweep",
    )
    parser.add_argument(
        "--enc-layers-grid",
        type=parse_int_list,
        default=[2, 3],
        help="Stage capacity: encoder depths to sweep",
    )
    parser.add_argument(
        "--dec-layers-grid",
        type=parse_int_list,
        default=[2],
        help="Stage capacity: decoder MLP depths to sweep",
    )
    parser.add_argument(
        "--n-heads-grid",
        type=parse_int_list,
        default=[4, 8],
        help="Stage capacity: candidate head counts (kept only if d_model %% n_heads == 0)",
    )
    parser.add_argument(
        "--patch-stride-pairs",
        type=parse_int_pair_list,
        default=[(16, 8), (16, 16), (32, 16)],
        help="Stage patching: patch:stride pairs, e.g. 16:8,16:16,32:16",
    )
    parser.add_argument(
        "--k-tops",
        type=parse_int_list,
        default=[8, 16, 32],
        help="Stage patching: k_top values to sweep",
    )
    parser.add_argument(
        "--wql-num-samples",
        type=int,
        default=100,
        help="Monte Carlo samples used to form forecast quantiles for WQL",
    )
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=50,
        help="Epochs to put into the generated full-training script",
    )
    parser.add_argument(
        "--full-window-stride",
        type=int,
        default=96,
        help="Window stride to put into the generated full-training script",
    )
    parser.add_argument(
        "--skip-full-train-script",
        action="store_true",
        help="Do not write train_full.sh after the search finishes",
    )
    return parser


def _devices_cli(devices) -> str:
    if isinstance(devices, list):
        return '"' + str(devices).replace(" ", "") + '"'
    return str(devices)


def _fmt(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    return str(value)


def _trial_name(stage: str, overrides: dict[str, Any]) -> str:
    parts = [stage] + [f"{key}{_fmt(overrides[key])}" for key in sorted(overrides)]
    return "_".join(parts)


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _snapshot_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in ARCH_KEYS}


def _apply_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    out = deepcopy(args)
    for key, value in config.items():
        setattr(out, key, value)
    return out


def _compatible_n_heads(d_model: int, candidates: list[int]) -> list[int]:
    heads = [h for h in candidates if d_model % h == 0]
    if not heads:
        raise ValueError(
            f"No n_heads in {candidates} divides d_model={d_model}. "
            "Adjust --n-heads-grid or --d-models."
        )
    return heads


def _write_full_train_script(path: Path, args: argparse.Namespace, best: dict) -> None:
    datasets = ",".join(args.datasets) if args.datasets else ""
    datasets_flag = f" --datasets {datasets}" if datasets else ""
    python = getattr(args, "python", ".venv/bin/python")
    cmd = (
        f"{python} scripts/train.py"
        f" --data-dir {args.data_dir}"
        f"{datasets_flag}"
        f" --context-len {args.context_len}"
        f" --pred-len {args.pred_len}"
        f" --batch-size {args.batch_size}"
        f" --num-workers {args.num_workers}"
        f" --window-stride {args.full_window_stride}"
        f" --epochs {args.full_epochs}"
        f" --accelerator {args.accelerator}"
        f" --devices {_devices_cli(args.devices)}"
        f" --d-model {best['d_model']}"
        f" --patch-size {best['patch_size']}"
        f" --stride {best['stride']}"
        f" --n-heads {best['n_heads']}"
        f" --enc-layers {best['enc_layers']}"
        f" --dec-layers {best['dec_layers']}"
        f" --k-top {best['k_top']}"
        f" --rope-theta {args.rope_theta}"
        f" --learning-rate {best['learning_rate']}"
        f" --kl-weight {best['kl_weight']}"
        f" --load-fraction 1.0"
    )
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + cmd + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def _save_best(search_dir: Path, best: dict, args: argparse.Namespace) -> None:
    best_path = search_dir / "best_params.json"
    payload = dict(best)
    if best.get("checkpoint"):
        dest = search_dir / "best.ckpt"
        shutil.copy2(best["checkpoint"], dest)
        payload["best_checkpoint_copy"] = str(dest)
    best_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.skip_full_train_script:
        _write_full_train_script(search_dir / "train_full.sh", args, payload)


def _run_trial(
    *,
    stage: str,
    overrides: dict[str, Any],
    base_args: argparse.Namespace,
    train_loader,
    val_loader,
    search_dir: Path,
    trial_idx: int,
    n_trials: int,
) -> dict:
    trial_args = _apply_config(base_args, overrides)
    if trial_args.d_model % trial_args.n_heads != 0:
        raise ValueError(
            f"Invalid architecture: d_model={trial_args.d_model} "
            f"not divisible by n_heads={trial_args.n_heads}"
        )

    trial_name = _trial_name(stage, overrides)
    trial_dir = search_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"[{stage}] Trial {trial_idx}/{n_trials}: {trial_name}")
    for key, value in overrides.items():
        print(f"  {key}={value}")
    print("=" * 72)

    model = create_model(trial_args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(trial_dir / "checkpoints"),
        filename="epoch_{epoch}",
        auto_insert_metric_name=False,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=base_args.epochs,
        accelerator=base_args.accelerator,
        devices=base_args.devices,
        default_root_dir=str(trial_dir),
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )
    t0 = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    metrics = evaluate_metrics_on_dataloader(
        trainer,
        model,
        val_loader,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        num_samples=base_args.wql_num_samples,
    )
    wql = metrics["WQL"]
    wape = metrics["WAPE"]
    elapsed_sec = time.perf_counter() - t0
    print(
        f"Trial {trial_name} val_WQL={wql:.6f}  val_WAPE={wape:.6f}  "
        f"elapsed_sec={elapsed_sec:.1f}"
    )

    config = _snapshot_config(trial_args)
    return {
        "stage": stage,
        "trial": trial_name,
        "val_wql": wql,
        "val_wape": wape,
        "elapsed_sec": round(elapsed_sec, 1),
        "checkpoint": checkpoint_callback.last_model_path
        or checkpoint_callback.best_model_path,
        "epochs": base_args.epochs,
        "window_stride": base_args.window_stride,
        "batch_size": base_args.batch_size,
        **config,
    }


def _build_dynamics_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"learning_rate": lr, "kl_weight": kl}
        for lr, kl in product(args.learning_rates, args.kl_weights)
    ]


def _build_capacity_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Full grid over d_model × enc_layers × dec_layers × compatible n_heads."""
    grid: list[dict[str, Any]] = []
    for d_model, enc_layers, dec_layers in product(
        args.d_models, args.enc_layers_grid, args.dec_layers_grid
    ):
        for n_heads in _compatible_n_heads(d_model, args.n_heads_grid):
            grid.append(
                {
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "enc_layers": enc_layers,
                    "dec_layers": dec_layers,
                }
            )
    # Stable unique ordering.
    unique: dict[tuple, dict[str, Any]] = {}
    for item in grid:
        key = (item["d_model"], item["n_heads"], item["enc_layers"], item["dec_layers"])
        unique[key] = item
    return list(unique.values())


def _build_patching_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"patch_size": patch, "stride": stride, "k_top": k_top}
        for (patch, stride), k_top in product(args.patch_stride_pairs, args.k_tops)
    ]


def run_search(args: argparse.Namespace) -> dict:
    search_dir = Path(args.search_dir)
    search_dir.mkdir(parents=True, exist_ok=True)

    stages = [part.strip() for part in args.stages.split(",") if part.strip()]
    unknown = [s for s in stages if s not in {"dynamics", "capacity", "patching"}]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}. Use dynamics,capacity,patching.")

    train_loader, val_loader = create_dataloaders(args)
    print(f"Selection metric: WQL @ {DEFAULT_QUANTILE_LEVELS}")
    print(f"Stages: {stages}")

    rows: list[dict] = []
    best: dict | None = None
    # Working config starts from CLI architecture defaults, then absorbs stage winners.
    working = _snapshot_config(args)

    stage_builders = {
        "dynamics": _build_dynamics_grid,
        "capacity": _build_capacity_grid,
        "patching": _build_patching_grid,
    }

    for stage in stages:
        grid = stage_builders[stage](args)
        print("\n" + "#" * 72)
        print(f"Stage `{stage}`: {len(grid)} trials")
        print(f"Locked config: {working}")
        print("#" * 72)

        stage_best: dict | None = None
        for trial_idx, overrides in enumerate(grid, start=1):
            # Skip exact duplicate of the locked config to avoid wasting a slot,
            # except we still evaluate once if it hasn't been scored in this run.
            merged_overrides = dict(overrides)
            row = _run_trial(
                stage=stage,
                overrides=merged_overrides,
                base_args=_apply_config(args, working),
                train_loader=train_loader,
                val_loader=val_loader,
                search_dir=search_dir,
                trial_idx=trial_idx,
                n_trials=len(grid),
            )
            rows.append(row)
            _write_results_csv(search_dir / "results.csv", rows)

            if stage_best is None or row["val_wql"] < stage_best["val_wql"]:
                stage_best = row
            if best is None or row["val_wql"] < best["val_wql"]:
                best = row
                _save_best(search_dir, best, args)

        assert stage_best is not None
        for key in stage_builders[stage](args)[0].keys():
            working[key] = stage_best[key]
        print(
            f"Stage `{stage}` winner: {stage_best['trial']}  "
            f"val_WQL={stage_best['val_wql']:.6f}"
        )
        print(f"Updated working config: {working}")

    assert best is not None
    _save_best(search_dir, best, args)

    print("\n" + "=" * 72)
    print(
        f"Best overall: {best['trial']}  val_WQL={best['val_wql']:.6f}\n"
        + "\n".join(f"  {k}={best[k]}" for k in ARCH_KEYS)
    )
    print(f"Wrote {search_dir / 'best_params.json'}")
    print("=" * 72)
    return best


def main():
    parser = getparser()
    parser.set_defaults(
        datasets=["electricity", "pedestrian_counts", "solar"],
        window_stride=384,
        epochs=8,
        learning_rate=1e-4,
        kl_weight=1e-3,
        batch_size=256,
        num_workers=8,
        accelerator="gpu",
        d_model=64,
        n_heads=4,
        enc_layers=2,
        dec_layers=2,
        patch_size=16,
        stride=8,
        k_top=16,
    )
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
