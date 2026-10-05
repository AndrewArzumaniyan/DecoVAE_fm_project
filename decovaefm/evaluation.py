"""Evaluation helpers for DecoVAE-FM forecasts."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import WAPE, WQL, _quantile_loss

DEFAULT_QUANTILE_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]


def forecast_samples_to_quantiles(
    forecast_samples: np.ndarray,
    quantile_levels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Monte Carlo forecast samples to point and quantile predictions.

    Parameters
    ----------
    forecast_samples:
        Array of shape ``[N, S, H]`` (univariate) or ``[N, S, H, D]``.
    quantile_levels:
        Quantile levels in ``(0, 1)``.

    Returns
    -------
    y_pred:
        Point forecast (median) with shape ``[N, H, D]``.
    q_pred:
        Quantile forecasts with shape ``[N, H, D, Q]``.
    """
    if forecast_samples.ndim == 3:
        forecast_samples = forecast_samples[..., None]
    if forecast_samples.ndim != 4:
        raise ValueError(
            f"forecast_samples must have shape [N, S, H] or [N, S, H, D], "
            f"got {forecast_samples.shape}"
        )

    q_pred = np.quantile(forecast_samples, q=quantile_levels, axis=1)
    # np.quantile with a list returns [Q, N, H, D] -> [N, H, D, Q]
    q_pred = np.moveaxis(q_pred, 0, -1)
    median_idx = int(np.argmin(np.abs(np.asarray(quantile_levels) - 0.5)))
    y_pred = q_pred[..., median_idx]
    return y_pred.astype(np.float64), q_pred.astype(np.float64)


def _prepare_metric_arrays(
    *,
    forecast_samples: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray | None,
    quantile_levels: list[float],
) -> dict:
    y_pred, q_pred = forecast_samples_to_quantiles(forecast_samples, quantile_levels)

    y_true = targets[..., None] if targets.ndim == 2 else targets
    if y_true.ndim != 3:
        raise ValueError(f"targets must have shape [N, H] or [N, H, D], got {targets.shape}")

    n = y_true.shape[0]
    if contexts is None:
        y_past = np.zeros((n, y_true.shape[-1]), dtype=np.float64)
        y_past_lengths = np.ones(n, dtype=np.int64)
    else:
        past = contexts[..., None] if contexts.ndim == 2 else contexts
        y_past = past.reshape(-1, past.shape[-1]).astype(np.float64)
        y_past_lengths = np.full(n, past.shape[1], dtype=np.int64)

    return {
        "y_true": y_true.astype(np.float64),
        "y_pred": y_pred,
        "y_past": y_past,
        "y_past_lengths": y_past_lengths,
        "q_pred": q_pred,
        "quantile_levels": quantile_levels,
    }


def compute_wql(
    *,
    forecast_samples: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    epsilon: float = 1e-8,
) -> float:
    """Compute WQL from forecast samples and targets."""
    levels = list(DEFAULT_QUANTILE_LEVELS if quantile_levels is None else quantile_levels)
    arrays = _prepare_metric_arrays(
        forecast_samples=forecast_samples,
        targets=targets,
        contexts=contexts,
        quantile_levels=levels,
    )
    return WQL(epsilon=epsilon).compute(
        **arrays,
        seasonality=1,
    )


def compute_wape(
    *,
    forecast_samples: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    epsilon: float = 1e-8,
) -> float:
    """Compute WAPE from forecast samples (uses median point forecast)."""
    levels = list(DEFAULT_QUANTILE_LEVELS if quantile_levels is None else quantile_levels)
    arrays = _prepare_metric_arrays(
        forecast_samples=forecast_samples,
        targets=targets,
        contexts=contexts,
        quantile_levels=levels,
    )
    return WAPE(epsilon=epsilon).compute(
        **arrays,
        seasonality=1,
    )


def compute_forecast_metrics(
    *,
    forecast_samples: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Compute WQL and WAPE from Monte Carlo forecast samples."""
    levels = list(DEFAULT_QUANTILE_LEVELS if quantile_levels is None else quantile_levels)
    arrays = _prepare_metric_arrays(
        forecast_samples=forecast_samples,
        targets=targets,
        contexts=contexts,
        quantile_levels=levels,
    )
    return {
        "WQL": WQL(epsilon=epsilon).compute(**arrays, seasonality=1),
        "WAPE": WAPE(epsilon=epsilon).compute(**arrays, seasonality=1),
    }


def new_forecast_metric_stats(
    num_quantiles: int, num_dims: int = 1
) -> dict[str, np.ndarray]:
    """Allocate running sums for dataset-level WQL / WAPE."""
    return {
        "abs_err_sum": np.zeros(num_dims, dtype=np.float64),
        "abs_err_count": np.zeros(num_dims, dtype=np.float64),
        "abs_true_sum": np.zeros(num_dims, dtype=np.float64),
        "abs_true_count": np.zeros(num_dims, dtype=np.float64),
        "ql_sum": np.zeros((num_dims, num_quantiles), dtype=np.float64),
        "ql_count": np.zeros((num_dims, num_quantiles), dtype=np.float64),
    }


def _nansum_and_count(arr: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    summed = np.where(finite, arr, 0).sum(axis=axis).astype(np.float64)
    count = finite.sum(axis=axis, dtype=np.float64)
    return summed, count


def update_forecast_metric_stats(
    stats: dict[str, np.ndarray],
    *,
    forecast_samples: np.ndarray,
    targets: np.ndarray,
    quantile_levels: list[float],
) -> None:
    """Add one batch of forecasts into running WQL / WAPE sums."""
    arrays = _prepare_metric_arrays(
        forecast_samples=forecast_samples,
        targets=targets,
        contexts=None,
        quantile_levels=quantile_levels,
    )
    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]
    q_pred = arrays["q_pred"]

    abs_err_sum, abs_err_count = _nansum_and_count(np.abs(y_true - y_pred), axis=(0, 1))
    abs_true_sum, abs_true_count = _nansum_and_count(np.abs(y_true), axis=(0, 1))
    ql = _quantile_loss(
        y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels
    )
    ql_sum, ql_count = _nansum_and_count(ql, axis=(0, 1))

    stats["abs_err_sum"] += abs_err_sum
    stats["abs_err_count"] += abs_err_count
    stats["abs_true_sum"] += abs_true_sum
    stats["abs_true_count"] += abs_true_count
    stats["ql_sum"] += ql_sum
    stats["ql_count"] += ql_count


def reduce_forecast_metric_stats(stats: dict[str, np.ndarray]) -> None:
    """Sum running metric stats across DDP ranks in-place."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if torch.distributed.get_world_size() < 2:
        return

    keys = (
        "abs_err_sum",
        "abs_err_count",
        "abs_true_sum",
        "abs_true_count",
        "ql_sum",
        "ql_count",
    )
    pieces = [np.ascontiguousarray(stats[key], dtype=np.float64).reshape(-1) for key in keys]
    packed = np.concatenate(pieces)
    tensor = torch.from_numpy(packed)
    if torch.distributed.get_backend() == "nccl":
        tensor = tensor.to(device=torch.device("cuda", torch.cuda.current_device()))
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    packed = tensor.detach().cpu().numpy()

    offset = 0
    for key in keys:
        size = stats[key].size
        stats[key] = packed[offset : offset + size].reshape(stats[key].shape)
        offset += size


def finalize_forecast_metric_stats(
    stats: dict[str, np.ndarray],
    *,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Turn running sums into dataset-level WQL / WAPE."""

    def _nanmean(summed: np.ndarray, count: np.ndarray) -> np.ndarray:
        out = np.full_like(summed, np.nan, dtype=np.float64)
        np.divide(summed, count, out=out, where=count > 0)
        return out

    abs_err_mean = _nanmean(stats["abs_err_sum"], stats["abs_err_count"])
    abs_true_mean = _nanmean(stats["abs_true_sum"], stats["abs_true_count"])
    ql_mean = _nanmean(stats["ql_sum"], stats["ql_count"])
    wape = float(np.mean(abs_err_mean / np.maximum(abs_true_mean, epsilon)))
    per_dim = ql_mean / np.maximum(abs_true_mean, epsilon)[:, None]
    wql = float(np.mean(np.mean(per_dim, axis=0)))
    return {"WQL": wql, "WAPE": wape}


def collect_predict_outputs(
    predictions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Stack Lightning predict/test batch dicts into arrays."""
    forecast_chunks = []
    target_chunks = []
    context_chunks = []

    for batch in predictions:
        forecast_chunks.append(batch["forecast_samples"].detach().cpu().numpy())
        target_chunks.append(batch["target"].detach().cpu().numpy())
        if "context" in batch:
            context_chunks.append(batch["context"].detach().cpu().numpy())

    forecasts = np.concatenate(forecast_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    contexts = np.concatenate(context_chunks, axis=0) if context_chunks else None
    return forecasts, targets, contexts


@torch.no_grad()
def evaluate_metrics_on_dataloader(
    trainer,
    model,
    dataloader: DataLoader,
    *,
    quantile_levels: list[float] | None = None,
    num_samples: int | None = None,
) -> dict[str, float]:
    """Run ``trainer.test`` and return logged WQL/WAPE metrics."""
    if num_samples is not None:
        model.num_samples = num_samples
    if quantile_levels is not None:
        model.quantile_levels = list(quantile_levels)

    trainer.test(model, dataloaders=dataloader)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    return {
        "WQL": metrics.get("test_wql", metrics.get("WQL", float("nan"))),
        "WAPE": metrics.get("test_wape", metrics.get("WAPE", float("nan"))),
    }


@torch.no_grad()
def evaluate_wql_on_dataloader(
    trainer,
    model,
    dataloader: DataLoader,
    *,
    quantile_levels: list[float] | None = None,
    num_samples: int | None = None,
) -> float:
    """Backward-compatible wrapper returning only WQL."""
    return evaluate_metrics_on_dataloader(
        trainer,
        model,
        dataloader,
        quantile_levels=quantile_levels,
        num_samples=num_samples,
    )["WQL"]
