"""Evaluate WQL and WAPE for a trained DecoVAE-FM checkpoint via Lightning test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from decovaefm.config import flatten_eval_config, prepare_hydra_argv
from decovaefm.data import ParquetStreamingDataset, ParquetWindowDataset
from decovaefm.debug_data import DummyRandomDataset
from decovaefm.evaluation import DEFAULT_QUANTILE_LEVELS, evaluate_metrics_on_dataloader
from decovaefm.lightning_module import DecoVAEFMLightning
from decovaefm.mlflow_utils import build_mlflow_logger, log_evaluation_checkpoint
from decovaefm.training import create_dataloaders


def _build_split_loader(args: argparse.Namespace, split: str):
    if split in ("train", "val"):
        train_loader, val_loader = create_dataloaders(
            args, dummy_dataset_cls=DummyRandomDataset
        )
        return train_loader if split == "train" else val_loader

    dataset_cls = (
        ParquetStreamingDataset if args.stream_from_disk else ParquetWindowDataset
    )
    dataset = dataset_cls(
        data_dir=args.data_dir,
        split="test",
        context_len=args.context_len,
        pred_len=args.pred_len,
        load_fraction=args.load_fraction,
        random_seed=args.load_seed + 2,
        stride=args.window_stride,
        datasets=args.datasets,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def evaluate(args: argparse.Namespace, cfg: DictConfig) -> tuple[dict, L.Trainer]:
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=build_mlflow_logger(cfg, run_name_prefix="eval_"),
        enable_checkpointing=False,
    )
    loader = _build_split_loader(args, args.split)
    model = DecoVAEFMLightning.load_from_checkpoint(args.checkpoint)
    metrics = evaluate_metrics_on_dataloader(
        trainer,
        model,
        loader,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        num_samples=args.num_samples,
    )
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "test_wql": metrics["WQL"],
        "test_wape": metrics["WAPE"],
        "quantile_levels": DEFAULT_QUANTILE_LEVELS,
        "num_samples": args.num_samples,
        "window_stride": args.window_stride,
        "datasets": args.datasets,
    }
    if trainer.is_global_zero:
        log_evaluation_checkpoint(trainer, args.checkpoint)
    return result, trainer


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    args = flatten_eval_config(cfg)
    result, trainer = evaluate(args, cfg)
    if not trainer.is_global_zero:
        return
    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    sys.argv = prepare_hydra_argv(sys.argv)
    main()
