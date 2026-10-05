"""Minimal training script for DecoVAE-FM using PyTorch Lightning."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from decovaefm.callbacks import ForecastVizCallback
from decovaefm.config import flatten_train_config
from decovaefm.debug_data import DummyRandomDataset
from decovaefm.lightning_module import DecoVAEFMLightning
from decovaefm.mlflow_utils import build_mlflow_logger
from decovaefm.training import create_dataloaders, create_model


def _runtime_output_dir() -> Path:
    try:
        output_dir = HydraConfig.get().runtime.output_dir
        if output_dir:
            return Path(str(output_dir))
    except ValueError:
        pass
    return Path("outputs") / "notebook"


def _checkpoint_dir() -> Path:
    return _runtime_output_dir() / "checkpoints"


def train(args, cfg: DictConfig | None = None) -> DecoVAEFMLightning:
    """Main training function.

    ``cfg`` is optional so the notebook can call ``train(args)`` without Hydra.
    """
    if cfg is None:
        cfg = OmegaConf.create({"mlflow": {"enabled": False}})

    train_loader, val_loader = create_dataloaders(
        args, dummy_dataset_cls=DummyRandomDataset
    )
    model = create_model(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(_checkpoint_dir()),
        filename="epoch_{epoch}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(getattr(args, "early_stopping_patience", 5)),
        # Lightning 2.6 defaults this to True, which checks before val_loss exists.
        check_on_train_epoch_end=False,
    )
    forecast_viz = ForecastVizCallback(
        val_dataset=val_loader.dataset,
        output_dir=_runtime_output_dir() / "forecasts",
        num_forecasts=int(getattr(args, "viz_num_forecasts", 8)),
        seed=int(getattr(args, "viz_seed", 42)),
        enabled=bool(getattr(args, "viz_enabled", True)),
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stopping, forecast_viz],
        logger=build_mlflow_logger(cfg),
    )

    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    args = flatten_train_config(cfg)
    train(args, cfg)


if __name__ == "__main__":
    main()
