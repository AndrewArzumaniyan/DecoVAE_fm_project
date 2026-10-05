"""Forecast visualization helpers matching notebooks/plot_forecasts.ipynb."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _to_1d(array: np.ndarray | torch.Tensor) -> np.ndarray:
    values = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else np.asarray(array)
    return np.squeeze(values)


def _quantiles(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.quantile(samples, 0.5, dim=1),
        torch.quantile(samples, 0.05, dim=1),
        torch.quantile(samples, 0.95, dim=1),
    )


def plot_forecast_panel(
    ax: Axes,
    context: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    recon_samples: torch.Tensor,
    forecast_samples: torch.Tensor,
    *,
    title: str,
    show_legend: bool = False,
) -> None:
    """Plot context, reconstruction, target, and forecast with 90% CI."""
    context = _to_1d(context)
    target = _to_1d(target)
    recon_median, recon_q05, recon_q95 = _quantiles(recon_samples.unsqueeze(0))
    forecast_median, forecast_q05, forecast_q95 = _quantiles(forecast_samples.unsqueeze(0))

    context_len = context.shape[0]
    pred_len = target.shape[0]
    x_context = np.arange(context_len)
    x_forecast = np.arange(context_len, context_len + pred_len)

    ax.plot(x_context, context, label="Context", color="tab:blue")
    ax.plot(x_context, _to_1d(recon_median), color="tab:orange")
    ax.fill_between(
        x_context,
        _to_1d(recon_q05),
        _to_1d(recon_q95),
        color="tab:orange",
        alpha=0.5,
        linewidth=0,
    )
    ax.plot(x_forecast, target, label="Target", color="tab:blue")
    ax.plot(x_forecast, _to_1d(forecast_median), label="Median forecast", color="tab:orange")
    ax.fill_between(
        x_forecast,
        _to_1d(forecast_q05),
        _to_1d(forecast_q95),
        color="tab:orange",
        alpha=0.5,
        label="90% Forecast CI",
        linewidth=0,
    )
    ax.axvline(context_len - 1, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(loc="upper center", ncol=4, fontsize=8)


def save_forecast_samples(
    output_dir: str | Path,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    recon_samples: torch.Tensor,
    forecast_samples: torch.Tensor,
    *,
    epoch: int,
) -> Path:
    """Save one 3x1 horizontal PNG per sample under ``output_dir/epoch_{epoch}/``."""
    epoch_dir = Path(output_dir) / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for i in range(contexts.shape[0]):
        fig = Figure(figsize=(12, 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        plot_forecast_panel(
            ax,
            contexts[i],
            targets[i],
            recon_samples[i],
            forecast_samples[i],
            title=f"Epoch {epoch} — Sample {i}",
            show_legend=True,
        )
        fig.tight_layout()
        fig.savefig(epoch_dir / f"sample_{i}.png", dpi=120)

    return epoch_dir
