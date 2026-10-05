"""PyTorch Lightning module for training DecoVAE-FM."""

import math

import torch
import lightning as L

from .evaluation import (
    DEFAULT_QUANTILE_LEVELS,
    finalize_forecast_metric_stats,
    new_forecast_metric_stats,
    reduce_forecast_metric_stats,
    update_forecast_metric_stats,
)
from .models import DecoVAEFM


class DecoVAEFMLightning(L.LightningModule):
    """Lightning module for DecoVAE-FM training."""

    def __init__(
        self,
        context_len: int = 96,
        pred_len: int = 24,
        num_samples: int = 100,
        d_model: int = 64,
        patch_size: int = 16,
        stride: int = 8,
        n_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        k_top: int = 32,
        rope_theta: float = 10000.0,
        kernels: list[int] | None = None,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.001,
        recon_weight: float = 1.0,
        forecast_weight: float = 1.0,
        quantile_levels: list[float] | None = None,
    ):
        super().__init__()
        self.context_len = context_len
        self.save_hyperparameters()

        self.model = DecoVAEFM(
            pred_len=pred_len,
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            n_heads=n_heads,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            k_top=k_top,
            rope_theta=rope_theta,
            kernels=kernels,
        )

        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.forecast_weight = forecast_weight
        self.num_samples = num_samples
        self.quantile_levels = list(
            DEFAULT_QUANTILE_LEVELS if quantile_levels is None else quantile_levels
        )
        self._test_stats: dict | None = None

    def _get_normalization_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-window mean and standard deviation from context windows."""
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return x_mean, x_std

    def _normalize(self, x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        """Apply asinh normalization using context statistics."""
        return torch.asinh((x - x_mean) / x_std)

    def _denormalize(
        self,
        x: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the asinh normalization."""
        return torch.sinh(x) * x_std + x_mean

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for VAE."""
        logvar = logvar.clamp(-20.0, 10.0)
        if mu.is_complex():
            mu_sq = mu.real.pow(2) + mu.imag.pow(2)
        else:
            mu_sq = mu.pow(2)

        return -0.5 * torch.mean(1 + logvar - mu_sq - logvar.exp())

    def _gaussian_nll(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood under N(mu, sigma^2), mean-reduced."""
        logvar = logvar.clamp(-20.0, 10.0)
        inv_var = (-logvar).exp()
        # Cap residual contribution to avoid overflow when variance is small.
        sq_error = (target - mu).pow(2).clamp_max(1e4)
        return 0.5 * (math.log(2 * math.pi) + logvar + sq_error * inv_var).mean()

    def _sample_observation(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the observation model N(mu, sigma^2)."""
        logvar = logvar.clamp(-20.0, 10.0)
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def _sample_forecasts(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        include_recon: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Draw observation-model samples for reconstruction and forecast."""
        pred_len = y.shape[1]
        batch_size = x.shape[0]
        num_samples = self.num_samples
        x_mean, x_std = self._get_normalization_stats(x)
        x_norm = self._normalize(x, x_mean, x_std)

        # One forward over a repeated batch covers all MC draws (latent + obs noise).
        x_norm_rep = (
            x_norm.unsqueeze(1)
            .expand(batch_size, num_samples, -1)
            .reshape(batch_size * num_samples, -1)
        )
        output = self.model(x_norm_rep, pred_len=pred_len)

        forecast_norm = self._sample_observation(
            output["forecast_mu"], output["forecast_logvar"]
        ).view(batch_size, num_samples, -1)

        x_mean = x_mean.unsqueeze(1)
        x_std = x_std.unsqueeze(1)
        result = {
            "forecast_samples": self._denormalize(forecast_norm, x_mean, x_std),
            "target": y,
            "context": x,
        }
        if include_recon:
            recon_norm = self._sample_observation(
                output["recon_mu"], output["recon_logvar"]
            ).view(batch_size, num_samples, -1)
            result["recon_samples"] = self._denormalize(recon_norm, x_mean, x_std)
        return result

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)

    def _elbo_losses(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
        output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction, forecast, KL, and component alignment losses."""
        recon_loss = self._gaussian_nll(
            x_norm, output["recon_mu"], output["recon_logvar"]
        )
        forecast_loss = self._gaussian_nll(
            y_norm, output["forecast_mu"], output["forecast_logvar"]
        )
        kl_loss = (
            self._kl_divergence(output["trend_mu"], output["trend_logvar"])
            + self._kl_divergence(output["seasonal_mu"], output["seasonal_logvar"])
            + self._kl_divergence(output["noise_mu"], output["noise_logvar"])
        )
        total_loss = (
            self.recon_weight * recon_loss
            + self.forecast_weight * forecast_loss
            + self.kl_weight * kl_loss
        )
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "forecast_loss": forecast_loss,
            "kl_loss": kl_loss,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        x_mean, x_std = self._get_normalization_stats(x)
        x_norm = self._normalize(x, x_mean, x_std)
        y_norm = self._normalize(y, x_mean, x_std)

        output = self.model(x_norm, pred_len=y.shape[1])
        losses = self._elbo_losses(x_norm, y_norm, output)

        self.log("train_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", losses["recon_loss"], sync_dist=True)
        self.log("train_forecast_loss", losses["forecast_loss"], sync_dist=True)
        self.log("train_kl_loss", losses["kl_loss"], sync_dist=True)

        return losses["total_loss"]

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        x_mean, x_std = self._get_normalization_stats(x)
        x_norm = self._normalize(x, x_mean, x_std)
        y_norm = self._normalize(y, x_mean, x_std)

        output = self.model(x_norm, pred_len=y.shape[1])
        losses = self._elbo_losses(x_norm, y_norm, output)

        self.log("val_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log("val_recon_loss", losses["recon_loss"], sync_dist=True)
        self.log("val_forecast_loss", losses["forecast_loss"], sync_dist=True)
        self.log("val_kl_loss", losses["kl_loss"], sync_dist=True)

        return losses["total_loss"]

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Test step: sample forecasts and update running WQL/WAPE sums."""
        x, y = batch
        outputs = self._sample_forecasts(x, y, include_recon=False)
        if self._test_stats is None:
            self._test_stats = new_forecast_metric_stats(len(self.quantile_levels))
        update_forecast_metric_stats(
            self._test_stats,
            forecast_samples=outputs["forecast_samples"].detach().cpu().numpy(),
            targets=outputs["target"].detach().cpu().numpy(),
            quantile_levels=self.quantile_levels,
        )

    def on_test_epoch_start(self) -> None:
        self._test_stats = new_forecast_metric_stats(len(self.quantile_levels))

    def on_test_epoch_end(self) -> None:
        """Reduce running sums across ranks and log dataset-level WQL / WAPE."""
        if self._test_stats is None:
            return

        reduce_forecast_metric_stats(self._test_stats)
        metrics = finalize_forecast_metric_stats(self._test_stats)
        self.log("test_wql", metrics["WQL"], prog_bar=True, sync_dist=True)
        self.log("test_wape", metrics["WAPE"], prog_bar=True, sync_dist=True)
        self._test_stats = None

    def predict_step(
        self, batch: tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Prediction step that returns samples from the observation model."""
        x, y = batch
        return self._sample_forecasts(x, y)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        #return [optimizer], [scheduler]
        return optimizer
