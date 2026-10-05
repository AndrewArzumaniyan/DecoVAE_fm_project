"""Lightning callbacks for DecoVAE-FM training."""

from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import Dataset, IterableDataset

from decovaefm.plotting import save_forecast_samples


def _collect_fixed_windows(
    dataset: Dataset | IterableDataset,
    num_forecasts: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed (context, target) batch reused across epochs."""
    if isinstance(dataset, IterableDataset):
        contexts: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for context, target in dataset:
            contexts.append(context.detach().cpu())
            targets.append(target.detach().cpu())
            if len(contexts) >= num_forecasts:
                break
        if not contexts:
            raise RuntimeError("Val dataset produced no windows for forecast visualization")
        return torch.stack(contexts), torch.stack(targets)

    n = min(num_forecasts, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n]
    contexts = []
    targets = []
    for idx in indices.tolist():
        context, target = dataset[idx]
        contexts.append(context.detach().cpu())
        targets.append(target.detach().cpu())
    return torch.stack(contexts), torch.stack(targets)


class ForecastVizCallback(Callback):
    """Save notebook-style forecast PNGs for a fixed val subset each epoch."""

    def __init__(
        self,
        val_dataset: Dataset | IterableDataset,
        output_dir: str | Path,
        *,
        num_forecasts: int = 8,
        seed: int = 42,
        enabled: bool = True,
    ) -> None:
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.num_forecasts = num_forecasts
        self.seed = seed
        self.enabled = enabled and num_forecasts > 0
        self._contexts: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        self._contexts, self._targets = _collect_fixed_windows(
            self.val_dataset, self.num_forecasts, self.seed
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Forecast viz: cached {self._contexts.shape[0]} val windows "
            f"(seed={self.seed}) -> {self.output_dir}"
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled or self._contexts is None or self._targets is None:
            return
        if trainer.sanity_checking:
            return

        device = pl_module.device
        contexts = self._contexts.to(device)
        targets = self._targets.to(device)
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module._sample_forecasts(contexts, targets)
        if was_training:
            pl_module.train()

        epoch = int(trainer.current_epoch)
        epoch_dir = save_forecast_samples(
            self.output_dir,
            contexts.cpu(),
            targets.cpu(),
            outputs["recon_samples"].detach().cpu(),
            outputs["forecast_samples"].detach().cpu(),
            epoch=epoch,
        )

        logger = trainer.logger
        if isinstance(logger, MLFlowLogger) and logger.run_id:
            logger.experiment.log_artifacts(
                logger.run_id, str(epoch_dir), artifact_path=f"forecasts/epoch_{epoch}"
            )
