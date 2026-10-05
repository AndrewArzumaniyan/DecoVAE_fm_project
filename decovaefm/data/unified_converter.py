"""Convert heterogeneous pretraining datasets into a unified parquet-window format."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm.auto import tqdm

# Folders currently present under source_data that the converter understands.
KNOWN_SOURCE_DATASETS: tuple[str, ...] = (
    "Electricity",
    "KDD_Cup_2018",
    "London_Smart_Meters",
    "M4_Daily",
    "M4_Hourly",
    "M4_Monthly",
    "M4_Weekly",
    "Pedestrian_Counts",
    "Solar",
    "Taxi",
    "Temperature_Rain",
    "Uber_TLC",
    "Wiki",
    "Wind_Farms",
    "mexico_city_bikes",
    "ushcn_daily",
    "weatherbench_weekly",
    "alibaba_cluster_trace_2018",
    "azure_vm_traces_2017",
    "borg_cluster_data_2011",
    "largest_2017",
    "Q-TRAFFIC",
)

_ID_COLUMNS = ("id", "item_id", "series_id")
_TARGET_COLUMNS = ("target", "values", "y")
_SKIP_COLUMNS = frozenset(
    {
        "id",
        "item_id",
        "series_id",
        "timestamp",
        "start",
        "freq",
        "latitude",
        "longitude",
        "level",
        "subset",
        "category",
        "past_feat_dynamic_real",
        "feat_dynamic_real",
        "feat_static_real",
        "feat_static_cat",
    }
)


@dataclass
class SplitConfig:
    """Dataset-level split ratios used to avoid leakage between windows."""

    train: float = 0.9
    val: float = 0.05
    test: float = 0.05

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


class ParquetShardWriter:
    """Buffered writer that materializes split parquet shards."""

    def __init__(self, base_dir: Path, split: str, shard_size: int) -> None:
        self.base_dir = base_dir
        self.split = split
        self.shard_size = shard_size
        self.rows: list[dict[str, Any]] = []
        self.part_idx = 0
        self.total_rows = 0

        self.output_dir = self.base_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_row(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        output_path = self.output_dir / f"part-{self.part_idx:05d}.parquet"
        df.to_parquet(output_path, index=False)
        self.total_rows += len(self.rows)
        self.rows = []
        self.part_idx += 1

    def close(self) -> int:
        self.flush()
        return self.total_rows


def _safe_float(value: str) -> float:
    if value == "?" or value == "" or value is None:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _clean_series(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return np.array([], dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    series = pd.Series(arr)
    series = series.interpolate(limit_direction="both").fillna(0.0)
    return series.to_numpy(dtype=np.float32)


def _iter_tsf_rows(tsf_text: io.TextIOBase) -> Iterator[tuple[list[str], list[float]]]:
    in_data = False
    attribute_count = 0

    for raw_line in tsf_text:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        lower = line.lower()
        if lower.startswith("@attribute"):
            attribute_count += 1
            continue
        if lower == "@data":
            in_data = True
            continue
        if not in_data:
            continue

        if attribute_count == 0:
            continue

        parts = line.split(":", maxsplit=attribute_count)
        if len(parts) != attribute_count + 1:
            continue

        attributes = [p.strip() for p in parts[:-1]]
        values = [_safe_float(v.strip()) for v in parts[-1].split(",")]
        yield attributes, values


def _iter_series_from_tsf_zip(zip_path: Path) -> Iterator[tuple[str, str, list[float]]]:
    dataset = zip_path.parent.name
    with zipfile.ZipFile(zip_path, "r") as zf:
        tsf_members = [n for n in zf.namelist() if n.lower().endswith(".tsf")]
        for member in tsf_members:
            with zf.open(member, "r") as f:
                wrapped = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
                for row_idx, (attrs, values) in enumerate(_iter_tsf_rows(wrapped)):
                    series_id = attrs[0] if attrs else f"{member}-row-{row_idx}"
                    yield dataset, str(series_id), values


def _iter_series_from_uber_zip(zip_path: Path) -> Iterator[tuple[str, str, list[float]]]:
    dataset = zip_path.parent.name
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for member in csv_members:
            with zf.open(member, "r") as f:
                raw = f.read()

            df = None
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    text = raw.decode(encoding, errors="replace")
                    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
                    break
                except Exception:
                    continue

            if df is None:
                continue
            for col in df.columns:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                values = numeric_col.dropna().astype(np.float32).tolist()
                if len(values) > 0:
                    series_id = f"{Path(member).name}:{col}"
                    yield dataset, series_id, values


def _parse_m4_row(row: list[str]) -> tuple[str, list[float]]:
    series_id = row[0]
    values = [_safe_float(v) for v in row[1:] if v is not None and v != ""]
    return series_id, values


def _iter_series_from_m4_dir(dataset_dir: Path) -> Iterator[tuple[str, str, list[float]]]:
    dataset = dataset_dir.name
    pattern = dataset.replace("M4_", "")
    train_file = dataset_dir / f"{pattern}-train.csv"
    test_file = dataset_dir / f"{pattern}-test.csv"
    if not train_file.exists() or not test_file.exists():
        return

    train_rows: dict[str, list[float]] = {}
    with train_file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            sid, vals = _parse_m4_row(row)
            train_rows[sid] = vals

    with test_file.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            sid, test_vals = _parse_m4_row(row)
            train_vals = train_rows.get(sid, [])
            yield dataset, sid, train_vals + test_vals


def _pick_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _expand_target(target: Any) -> list[list[float]]:
    """Normalize a target field into one or more univariate float series."""
    if target is None:
        return []

    # Multivariate: list/array of channels.
    if isinstance(target, (list, tuple)) and target and isinstance(target[0], (list, tuple, np.ndarray)):
        series_list: list[list[float]] = []
        for channel in target:
            arr = np.asarray(channel, dtype=np.float32).ravel()
            if arr.size > 0:
                series_list.append(arr.tolist())
        return series_list

    if isinstance(target, np.ndarray) and target.ndim >= 2:
        series_list = []
        flat = target.reshape(target.shape[0], -1) if target.ndim > 2 else target
        # Prefer (channels, time) when first dim is small.
        if flat.shape[0] < flat.shape[1]:
            channels = flat
        else:
            channels = flat.T
        for channel in channels:
            arr = np.asarray(channel, dtype=np.float32).ravel()
            if arr.size > 0:
                series_list.append(arr.tolist())
        return series_list

    try:
        arr = np.asarray(target, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return []
    if arr.size == 0:
        return []
    return [arr.tolist()]


def _series_id_from_row(row_id: Any, parquet_stem: str, row_idx: int, channel_idx: int, n_channels: int) -> str:
    base = str(row_id) if row_id is not None and str(row_id) != "" else f"{parquet_stem}:{row_idx}"
    if n_channels > 1:
        return f"{base}:ch{channel_idx}"
    return base


def _iter_series_from_parquet_file(parquet_path: Path) -> Iterator[tuple[str, str, list[float]]]:
    dataset = parquet_path.parent.name
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return

    id_col = _pick_column(list(df.columns), _ID_COLUMNS)
    target_col = _pick_column(list(df.columns), _TARGET_COLUMNS)

    # Chronos / HF style: one series (or multivariate channels) per row.
    if target_col is not None:
        for row_idx, row in df.iterrows():
            row_id = row[id_col] if id_col is not None else None
            channels = _expand_target(row[target_col])
            for ch_idx, values in enumerate(channels):
                series_id = _series_id_from_row(row_id, parquet_path.stem, int(row_idx), ch_idx, len(channels))
                yield dataset, series_id, values
        return

    # Legacy fallback for wide numeric tables.
    for col in df.columns:
        if col.lower() in _SKIP_COLUMNS:
            continue
        column = df[col]
        if len(column) == 0:
            continue

        if column.map(lambda x: isinstance(x, (list, tuple, np.ndarray))).all():
            for idx, item in enumerate(column):
                channels = _expand_target(item)
                for ch_idx, values in enumerate(channels):
                    suffix = f":ch{ch_idx}" if len(channels) > 1 else ""
                    yield dataset, f"{parquet_path.stem}:{col}:{idx}{suffix}", values
            continue

        numeric_col = pd.to_numeric(column, errors="coerce")
        vals = numeric_col.dropna().astype(np.float32).tolist()
        if len(vals) > 0:
            yield dataset, f"{parquet_path.stem}:{col}", vals


def _read_arrow_table(arrow_path: Path) -> pa.Table:
    with pa.memory_map(str(arrow_path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except (pa.ArrowInvalid, OSError):
            source.seek(0)
            return ipc.open_stream(source).read_all()


def _iter_series_from_arrow_file(arrow_path: Path) -> Iterator[tuple[str, str, list[float]]]:
    """Read LOTSA / HuggingFace Arrow shards (item_id + target)."""
    dataset = arrow_path.parent.name
    table = _read_arrow_table(arrow_path)
    if table.num_rows == 0:
        return

    names = table.column_names
    id_col = _pick_column(names, _ID_COLUMNS)
    target_col = _pick_column(names, _TARGET_COLUMNS)
    if target_col is None:
        return

    target_column = table.column(target_col)
    id_column = table.column(id_col) if id_col is not None else None

    for row_idx in range(table.num_rows):
        row_id = id_column[row_idx].as_py() if id_column is not None else None
        channels = _expand_target(target_column[row_idx].as_py())
        for ch_idx, values in enumerate(channels):
            series_id = _series_id_from_row(row_id, arrow_path.stem, row_idx, ch_idx, len(channels))
            yield dataset, series_id, values


def _dataset_source_files(dataset_dir: Path) -> list[Path]:
    name = dataset_dir.name
    if name.startswith("M4_"):
        pattern = name.replace("M4_", "")
        train_file = dataset_dir / f"{pattern}-train.csv"
        test_file = dataset_dir / f"{pattern}-test.csv"
        files = [p for p in (train_file, test_file) if p.exists()]
        return files

    files: list[Path] = []
    files.extend(sorted(dataset_dir.glob("*.zip")))
    files.extend(sorted(dataset_dir.glob("*.parquet")))
    files.extend(sorted(dataset_dir.glob("*.arrow")))
    return files


def _iter_series_from_dataset_dir(dataset_dir: Path) -> Iterator[tuple[str, str, list[float]]]:
    name = dataset_dir.name
    if name.startswith("M4_"):
        yield from _iter_series_from_m4_dir(dataset_dir)
        return

    for zip_path in sorted(dataset_dir.glob("*.zip")):
        if name == "Uber_TLC":
            yield from _iter_series_from_uber_zip(zip_path)
        else:
            yield from _iter_series_from_tsf_zip(zip_path)

    for parquet_path in sorted(dataset_dir.glob("*.parquet")):
        yield from _iter_series_from_parquet_file(parquet_path)

    for arrow_path in sorted(dataset_dir.glob("*.arrow")):
        yield from _iter_series_from_arrow_file(arrow_path)


def discover_source_datasets(
    input_dir: str | Path,
    datasets: Sequence[str] | None = None,
) -> list[Path]:
    """Return dataset directories under input_dir, optionally filtered by name."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    available = sorted(p for p in input_path.iterdir() if p.is_dir())
    if datasets is None:
        return available

    wanted = {name.strip() for name in datasets if name.strip()}
    selected = [p for p in available if p.name in wanted]
    missing = sorted(wanted - {p.name for p in selected})
    if missing:
        raise FileNotFoundError(
            f"Requested datasets not found under {input_path}: {', '.join(missing)}"
        )
    return selected


def _assign_split(dataset: str, series_id: str, split_config: SplitConfig) -> str:
    key = f"{dataset}::{series_id}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0
    if r < split_config.train:
        return "train"
    if r < split_config.train + split_config.val:
        return "val"
    return "test"


def convert_pretrain_data(
    input_dir: str | Path,
    output_dir: str | Path,
    shard_size: int = 5_000,
    split_config: SplitConfig | None = None,
    min_series_length: int = 2,
    datasets: Sequence[str] | None = None,
    clear_output: bool = True,
) -> dict[str, Any]:
    """Convert raw datasets into unified parquet series shards and write a manifest."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_cfg = split_config or SplitConfig()
    split_cfg.validate()

    if clear_output:
        for split in ("train", "val", "test"):
            split_dir = output_path / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    writers = {
        "train": ParquetShardWriter(output_path, "train", shard_size),
        "val": ParquetShardWriter(output_path, "val", shard_size),
        "test": ParquetShardWriter(output_path, "test", shard_size),
    }

    per_dataset_series: dict[str, int] = {}
    per_split_series: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    per_dataset_points: dict[str, int] = {}
    points_by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    skipped_short = 0
    skipped_all_zero = 0
    total_series = 0

    dataset_dirs = discover_source_datasets(input_path, datasets=datasets)
    # Never treat the output directory as an input dataset if nested under input_dir.
    excluded_resolved = output_path.resolve()
    dataset_dirs = [p for p in dataset_dirs if p.resolve() != excluded_resolved]

    total_files = sum(len(_dataset_source_files(dataset_dir)) for dataset_dir in dataset_dirs)
    progress = tqdm(desc="Converting files", unit="file", total=total_files)

    def _handle_series(dataset: str, series_id: str, raw_values: list[float]) -> None:
        nonlocal skipped_short, skipped_all_zero, total_series
        cleaned = _clean_series(raw_values)
        if cleaned.size < min_series_length:
            skipped_short += 1
            return
        if np.all(cleaned == 0.0):
            skipped_all_zero += 1
            return

        total_series += 1
        per_dataset_series[dataset] = per_dataset_series.get(dataset, 0) + 1

        split = _assign_split(dataset, series_id, split_cfg)
        values = cleaned.astype(np.float32)
        writers[split].add_row(
            {
                "dataset": dataset,
                "series_id": series_id,
                "values": values.tolist(),
                "length": int(values.shape[0]),
            }
        )
        per_split_series[split] += 1
        points_by_split[split] += int(values.shape[0])
        per_dataset_points[dataset] = per_dataset_points.get(dataset, 0) + int(values.shape[0])

    for dataset_dir in dataset_dirs:
        source_files = _dataset_source_files(dataset_dir)
        if not source_files:
            continue

        if dataset_dir.name.startswith("M4_"):
            for dataset, series_id, raw_values in _iter_series_from_m4_dir(dataset_dir):
                _handle_series(dataset, series_id, raw_values)
            progress.update(len(source_files))
            continue

        for zip_path in sorted(dataset_dir.glob("*.zip")):
            iterator = (
                _iter_series_from_uber_zip(zip_path)
                if dataset_dir.name == "Uber_TLC"
                else _iter_series_from_tsf_zip(zip_path)
            )
            for dataset, series_id, raw_values in iterator:
                _handle_series(dataset, series_id, raw_values)
            progress.update(1)

        for parquet_path in sorted(dataset_dir.glob("*.parquet")):
            for dataset, series_id, raw_values in _iter_series_from_parquet_file(parquet_path):
                _handle_series(dataset, series_id, raw_values)
            progress.update(1)

        for arrow_path in sorted(dataset_dir.glob("*.arrow")):
            for dataset, series_id, raw_values in _iter_series_from_arrow_file(arrow_path):
                _handle_series(dataset, series_id, raw_values)
            progress.update(1)

    progress.close()

    series_written = {split: writers[split].close() for split in writers}

    manifest = {
        "format_version": 1,
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "datasets": [p.name for p in dataset_dirs],
        "schema": {
            "dataset": "string",
            "series_id": "string",
            "values": "list[float32]",
            "length": "int",
        },
        "splits": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
        },
        "conversion": {
            "min_series_length": min_series_length,
            "drop_all_zero_series": True,
            "windowing_applied": False,
            "clear_output": clear_output,
        },
        "stats": {
            "total_series": total_series,
            "skipped_short_series": skipped_short,
            "skipped_all_zero_series": skipped_all_zero,
            "series_by_split": per_split_series,
            "series_written": series_written,
            "points_by_split": points_by_split,
            "series_by_dataset": dict(sorted(per_dataset_series.items())),
            "points_by_dataset": dict(sorted(per_dataset_points.items())),
        },
    }

    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
