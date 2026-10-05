"""PyTorch dataset for unified parquet window shards."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm.auto import tqdm


def _load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {data_dir}. Run conversion first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_names(
    requested: Sequence[str] | None,
    available: Iterable[str],
) -> set[str] | None:
    """Resolve requested dataset names against available names (case-insensitive).

    Returns ``None`` when ``requested`` is empty/None (use all datasets).
    """
    if not requested:
        return None

    available_list = list(available)
    by_lower = {name.lower(): name for name in available_list}
    resolved: set[str] = set()
    unknown: list[str] = []

    for name in requested:
        key = name.strip()
        if not key:
            continue
        match = by_lower.get(key.lower())
        if match is None:
            unknown.append(name)
        else:
            resolved.add(match)

    if unknown:
        available_str = ", ".join(sorted(available_list)) or "(none)"
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {available_str}"
        )
    if not resolved:
        raise ValueError("No datasets selected after parsing --datasets")

    return resolved


class ParquetWindowDataset(Dataset):
    """Loads unified parquet shards and returns (context, target) tensors."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        pred_len: int,
        context_len: int,
        load_fraction: float = 1.0,
        random_seed: int = 42,
        stride: int = 1,
        datasets: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.context_len = context_len
        self.pred_len = pred_len

        manifest = _load_manifest(self.data_dir)
        available = manifest.get("stats", {}).get("series_by_dataset", {}).keys()
        self.datasets = resolve_dataset_names(datasets, available)

        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        split_dir = self.data_dir / split
        all_shard_files = sorted(split_dir.glob("*.parquet"))
        if not all_shard_files:
            raise FileNotFoundError(f"No parquet shards found for split '{split}' in {split_dir}")

        if not (0.0 < load_fraction <= 1.0):
            raise ValueError(f"load_fraction must be in (0, 1], got {load_fraction}")

        shard_files = all_shard_files
        if load_fraction < 1.0:
            rng = np.random.default_rng(random_seed)
            num_to_keep = max(1, math.ceil(len(all_shard_files) * load_fraction))
            chosen_idx = np.sort(rng.choice(len(all_shard_files), size=num_to_keep, replace=False))
            shard_files = [all_shard_files[i] for i in chosen_idx]

        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        shard_iter = tqdm(
            shard_files,
            desc=f"Loading {split} shards ({len(shard_files)}/{len(all_shard_files)})",
            unit="shard",
            leave=False,
        )
        for shard in shard_iter:
            df = pd.read_parquet(shard)
            if self.datasets is not None:
                if "dataset" not in df.columns:
                    raise ValueError(
                        f"Cannot filter by dataset: column 'dataset' missing in {shard}"
                    )
                df = df[df["dataset"].isin(self.datasets)]
            total_len = self.context_len + self.pred_len

            if "values" in df.columns:
                # New format: one row per full series; derive windows dynamically.
                for _, row in df.iterrows():
                    values = np.asarray(row["values"], dtype=np.float32)
                    if values.shape[0] < total_len:
                        continue

                    max_start = values.shape[0] - total_len
                    for start in range(0, max_start + 1, stride):
                        context = values[start : start + self.context_len]
                        target = values[start + self.context_len : start + total_len]
                        contexts.append(context)
                        targets.append(target)
            else:
                raise ValueError(
                    "Unified parquet shard is in unsupported format. "
                    "Expected a 'values' column (series-only shards). "
                    "Re-run convert_pretrain_data.py to regenerate series-only shards."
                )

            shard_iter.set_postfix(loaded_samples=len(contexts), refresh=False)

        if not contexts:
            selected = (
                ", ".join(sorted(self.datasets)) if self.datasets is not None else "all"
            )
            raise RuntimeError(
                f"No valid samples loaded for split '{split}' from {split_dir} "
                f"(datasets={selected})"
            )

        self.context = torch.from_numpy(np.stack(contexts, axis=0)).float()
        self.target = torch.from_numpy(np.stack(targets, axis=0)).float()

    def __len__(self) -> int:
        return self.context.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context[idx], self.target[idx]


class ParquetStreamingDataset(IterableDataset):
    """Streams unified parquet windows from disk without materializing them in RAM."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        pred_len: int,
        context_len: int,
        load_fraction: float = 1.0,
        random_seed: int = 42,
        row_batch_size: int = 1024,
        stride: int = 1,
        datasets: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.context_len = context_len
        self.pred_len = pred_len
        self.load_fraction = load_fraction
        self.random_seed = random_seed
        self.row_batch_size = row_batch_size
        self.stride = stride

        manifest = _load_manifest(self.data_dir)
        available = manifest.get("stats", {}).get("series_by_dataset", {}).keys()
        self.datasets = resolve_dataset_names(datasets, available)

        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        if not (0.0 < load_fraction <= 1.0):
            raise ValueError(f"load_fraction must be in (0, 1], got {load_fraction}")

        split_dir = self.data_dir / split
        shard_files = sorted(split_dir.glob("*.parquet"))
        if not shard_files:
            raise FileNotFoundError(f"No parquet shards found for split '{split}' in {split_dir}")

        if load_fraction < 1.0:
            rng = np.random.default_rng(random_seed)
            num_to_keep = max(1, math.ceil(len(shard_files) * load_fraction))
            chosen_idx = np.sort(rng.choice(len(shard_files), size=num_to_keep, replace=False))
            shard_files = [shard_files[i] for i in chosen_idx]

        self.shard_files = shard_files
        self.num_samples = self._estimate_num_windows()

    def _row_columns(self, schema_names: set[str]) -> list[str]:
        columns = ["values"]
        if self.datasets is not None:
            if "dataset" not in schema_names:
                raise ValueError(
                    "Cannot filter by dataset: column 'dataset' missing in parquet schema"
                )
            columns.append("dataset")
        return columns

    def _keep_row(self, row: dict) -> bool:
        if self.datasets is None:
            return True
        return row.get("dataset") in self.datasets

    def _estimate_num_windows(self) -> int:
        """Count windows using the ``length`` column when present (avoids decoding values)."""
        total = 0
        total_len = self.context_len + self.pred_len
        for shard in tqdm(self.shard_files, desc="Estimating number of windows"):
            parquet_file = pq.ParquetFile(shard)
            schema_names = set(parquet_file.schema_arrow.names)

            if "values" not in schema_names:
                raise ValueError(
                    "Unified parquet shard is in unsupported format. "
                    "Expected a 'values' column (series-only shards). "
                    "Re-run convert_pretrain_data.py to regenerate series-only shards."
                )

            if "length" in schema_names:
                columns = ["length"]
                if self.datasets is not None:
                    if "dataset" not in schema_names:
                        raise ValueError(
                            "Cannot filter by dataset: column 'dataset' missing "
                            f"in {shard}"
                        )
                    columns.append("dataset")

                for batch in parquet_file.iter_batches(
                    batch_size=self.row_batch_size,
                    columns=columns,
                ):
                    lengths = batch.column("length").to_numpy(zero_copy_only=False)
                    if self.datasets is not None:
                        datasets = batch.column("dataset").to_pylist()
                        keep = np.fromiter(
                            (name in self.datasets for name in datasets),
                            dtype=bool,
                            count=len(datasets),
                        )
                        lengths = lengths[keep]
                    eligible = lengths[lengths >= total_len]
                    if eligible.size:
                        total += int(
                            ((eligible - total_len) // self.stride + 1).sum()
                        )
                continue

            # Fallback for older shards without a length column.
            columns = self._row_columns(schema_names)
            for batch in parquet_file.iter_batches(
                batch_size=self.row_batch_size,
                columns=columns,
            ):
                for row in batch.to_pylist():
                    if not self._keep_row(row):
                        continue
                    values = row["values"]
                    if values is None:
                        continue
                    n = len(values)
                    if n >= total_len:
                        total += (n - total_len) // self.stride + 1

        return total

    @staticmethod
    def _distributed_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _shard_across_ranks(self) -> bool:
        """Split parquet files across DDP ranks only for training.

        Val/test keep a full replica on every rank. The val split is small and
        often a single shard, so rank-striping would leave most GPUs with an
        empty loader and Lightning would never log ``val_loss``.
        """
        world_size, _ = self._distributed_info()
        return self.split == "train" and world_size > 1

    def _iter_shard_files(self) -> list[Path]:
        shard_files = self.shard_files
        if self._shard_across_ranks():
            world_size, rank = self._distributed_info()
            shard_files = shard_files[rank::world_size]
        worker_info = get_worker_info()
        if worker_info is None:
            return shard_files
        return shard_files[worker_info.id :: worker_info.num_workers]

    def __len__(self) -> int:
        if self._shard_across_ranks():
            world_size, _ = self._distributed_info()
            return math.ceil(self.num_samples / world_size)
        return self.num_samples

    def _iter_rows(self, shard_files: list[Path]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        total_len = self.context_len + self.pred_len
        shard_iter = tqdm(
            shard_files,
            desc=f"Streaming {self.split} shards ({len(shard_files)})",
            unit="shard",
            leave=False,
        )

        for shard in shard_iter:
            parquet_file = pq.ParquetFile(shard)
            schema_names = set(parquet_file.schema_arrow.names)

            if "values" in schema_names:
                columns = self._row_columns(schema_names)
                for batch in parquet_file.iter_batches(
                    batch_size=self.row_batch_size,
                    columns=columns,
                ):
                    for row in batch.to_pylist():
                        if not self._keep_row(row):
                            continue
                        values = np.asarray(row["values"], dtype=np.float32)
                        if values.shape[0] < total_len:
                            continue

                        max_start = values.shape[0] - total_len
                        for start in range(0, max_start + 1, self.stride):
                            context = values[start : start + self.context_len]
                            target = values[start + self.context_len : start + total_len]
                            yield torch.from_numpy(context), torch.from_numpy(target)
            else:
                raise ValueError(
                    "Unified parquet shard is in unsupported format. "
                    "Expected a 'values' column (series-only shards). "
                    "Re-run convert_pretrain_data.py to regenerate series-only shards."
                )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self._iter_rows(self._iter_shard_files())
