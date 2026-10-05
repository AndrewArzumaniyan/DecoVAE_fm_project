"""Shared helpers for training and hyperparameter search scripts."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from decovaefm.data import ParquetStreamingDataset, ParquetWindowDataset
from decovaefm.lightning_module import DecoVAEFMLightning


def create_dataloaders(args: argparse.Namespace, *, dummy_dataset_cls: Any | None = None):
    """Build train/val dataloaders from parsed CLI args.

    If ``args.debug_on_dummy`` is set, ``dummy_dataset_cls`` must be provided.
    """
    if args.debug_on_dummy:
        if dummy_dataset_cls is None:
            raise ValueError("dummy_dataset_cls is required when debug_on_dummy=True")
        train_dataset = dummy_dataset_cls(
            num_samples=2048,
            context_len=args.context_len,
            pred_len=args.pred_len,
            seed=args.load_seed,
        )
        val_dataset = dummy_dataset_cls(
            num_samples=256,
            context_len=args.context_len,
            pred_len=args.pred_len,
            seed=args.load_seed + 1,
        )
    else:
        selected = args.datasets or "all"
        print(f"Loading dataset from: {args.data_dir} (datasets={selected})")
        dataset_cls = (
            ParquetStreamingDataset if args.stream_from_disk else ParquetWindowDataset
        )
        train_dataset = dataset_cls(
            data_dir=args.data_dir,
            split="train",
            context_len=args.context_len,
            pred_len=args.pred_len,
            load_fraction=args.load_fraction,
            random_seed=args.load_seed,
            stride=args.window_stride,
            datasets=args.datasets,
        )
        val_dataset = dataset_cls(
            data_dir=args.data_dir,
            split="val",
            context_len=args.context_len,
            pred_len=args.pred_len,
            load_fraction=args.load_fraction,
            random_seed=args.load_seed + 1,
            stride=args.window_stride,
            datasets=args.datasets,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(train_dataset, IterableDataset),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation set is empty, so EarlyStopping/ModelCheckpoint cannot "
            "monitor val_loss. Check that the val split has series longer than "
            f"context_len+pred_len ({args.context_len}+{args.pred_len})."
        )
    return train_loader, val_loader


def create_model(args: argparse.Namespace) -> DecoVAEFMLightning:
    """Build the Lightning module from parsed CLI args."""
    print("Creating model...")
    return DecoVAEFMLightning(
        context_len=args.context_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        patch_size=args.patch_size,
        stride=args.stride,
        n_heads=args.n_heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        k_top=args.k_top,
        rope_theta=args.rope_theta,
        kernels=args.kernels,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        recon_weight=args.recon_weight,
        forecast_weight=args.forecast_weight,
    )


def parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated list of floats from CLI."""
    parts = [part.strip() for part in str(raw_value).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of floats, e.g. 1e-4,1e-3"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float list '{raw_value}': {exc}"
        ) from exc


def parse_int_list(raw_value: str) -> list[int]:
    """Parse a comma-separated list of ints from CLI."""
    parts = [part.strip() for part in str(raw_value).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of ints, e.g. 64,128"
        )
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid int list '{raw_value}': {exc}"
        ) from exc


def parse_int_pair_list(raw_value: str) -> list[tuple[int, int]]:
    """Parse comma-separated ``a:b`` pairs, e.g. ``16:8,16:16,32:16``."""
    parts = [part.strip() for part in str(raw_value).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated pairs like 16:8,16:16,32:16"
        )
    pairs: list[tuple[int, int]] = []
    for part in parts:
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid pair '{part}'. Use patch:stride, e.g. 16:8"
            )
        left, right = part.split(":", 1)
        try:
            pairs.append((int(left.strip()), int(right.strip())))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid pair '{part}': {exc}"
            ) from exc
    return pairs