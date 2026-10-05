"""Argparse CLI for scripts that still use flat flags (e.g. hyperparameter search)."""

from __future__ import annotations

import argparse
import ast

from decovaefm.training import parse_int_list


def parse_devices_arg(raw_value):
    """Parse Trainer devices arg from CLI, supporting int or list syntax."""
    if isinstance(raw_value, int):
        return raw_value

    value = str(raw_value).strip()
    if value.lower() == "auto":
        return "auto"

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(
            "Invalid --devices value. Use an integer (e.g. 1), "
            'a list (e.g. [1,3]), or "auto".'
        ) from exc

    if isinstance(parsed, int):
        return parsed

    if isinstance(parsed, list) and all(isinstance(device, int) for device in parsed):
        return parsed

    raise argparse.ArgumentTypeError(
        "Invalid --devices value. Use an integer (e.g. 1), "
        'a list (e.g. [1,3]), or "auto".'
    )


def parse_datasets_arg(raw_value: str) -> list[str]:
    """Parse a comma-separated dataset list from CLI."""
    names = [part.strip() for part in str(raw_value).split(",")]
    names = [name for name in names if name]
    if not names:
        raise argparse.ArgumentTypeError(
            "Invalid --datasets value. Provide one or more names, "
            "e.g. electricity,pedestrian_counts,solar"
        )
    return names


def getparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DecoVAE-FM model")

    parser.add_argument(
        "--context-len", type=int, default=512, help="Input context length"
    )
    parser.add_argument("--pred-len", type=int, default=96, help="Prediction length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing parquet data and manifest",
    )
    parser.add_argument(
        "--datasets",
        type=parse_datasets_arg,
        default=None,
        help=(
            "Comma-separated subset of source datasets to train on "
            "(case-insensitive), e.g. electricity,pedestrian_counts,solar. "
            "Default: all datasets in the manifest"
        ),
    )
    parser.add_argument(
        "--load-fraction",
        type=float,
        default=1.0,
        help="Fraction of parquet shards to load per split (0 < value <= 1), e.g. 0.05",
    )
    parser.add_argument(
        "--load-seed",
        type=int,
        default=42,
        help="Random seed used when load-fraction < 1.0",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=96,
        help="Stride for dynamic sliding-window extraction at training time",
    )
    parser.add_argument(
        "--stream-from-disk",
        action="store_true",
        help="Stream parquet data from disk instead of loading it all into RAM",
    )
    parser.add_argument(
        "--debug-on-dummy",
        action="store_true",
        help="Run training on a tiny deterministic random dataset for fast debugging",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes",
    )

    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--stride", type=int, default=16, help="Stride for patching")
    parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--enc-layers", type=int, default=2, help="Number of transformer encoder layers"
    )
    parser.add_argument(
        "--dec-layers",
        type=int,
        default=2,
        help="Number of linear layers in decoder MLPs",
    )
    parser.add_argument(
        "--k-top", type=int, default=16, help="Top-K seasonal frequency tokens"
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE base frequency (theta)",
    )
    parser.add_argument(
        "--kernels",
        type=parse_int_list,
        default=[5, 9, 13, 25],
        help=(
            "Comma-separated avg-pool kernel sizes for MultiScaleDecomposition, "
            "e.g. 5,9,13,25"
        ),
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="EarlyStopping patience in epochs (monitor=val_loss)",
    )
    parser.add_argument(
        "--viz-enabled",
        action="store_true",
        default=True,
        help="Save forecast visualization PNGs after each validation epoch",
    )
    parser.add_argument(
        "--viz-num-forecasts",
        type=int,
        default=8,
        help="Number of fixed val windows to plot each validation epoch",
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=42,
        help="RNG seed used to pick the fixed visualization windows",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices_arg,
        default="auto",
        help='Lightning Trainer devices arg: int (e.g. 1), list (e.g. [1,3]), or "auto"',
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Lightning Trainer accelerator: gpu or cpu",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--kl-weight", type=float, default=0.01, help="KL divergence weight"
    )
    parser.add_argument(
        "--recon-weight", type=float, default=1.0, help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--forecast-weight", type=float, default=1.0, help="Forecast loss weight"
    )

    return parser
