"""Small synthetic datasets for fast debugging."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyRandomDataset(Dataset):
    """Small deterministic random dataset for fast debugging."""

    def __init__(
        self,
        *,
        num_samples: int,
        context_len: int,
        pred_len: int,
        seed: int,
        device: str = "cpu",
    ) -> None:
        self.num_samples = num_samples
        self.context_len = context_len
        self.pred_len = pred_len
        xspace = (
            torch.linspace(0, 1, context_len + pred_len)[None, :]
            * torch.randint(20, 70, (num_samples, 1))
            + torch.arange(num_samples)[:, None]
        )

        sin_data = torch.sin(xspace)

        generator = torch.Generator().manual_seed(seed)
        sin_data = (
            sin_data
            + torch.randn(num_samples, context_len + pred_len, generator=generator)
            * (sin_data + 1)
            * 0.15
        )
        sin_data = sin_data * torch.randint(1, 10000, (num_samples, 1))
        context = sin_data[:, :context_len]
        target = sin_data[:, context_len : context_len + pred_len]

        self.target = target.to(device)
        self.context = context.to(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context[idx], self.target[idx]
