"""CLI for converting raw pretraining datasets into unified parquet windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decovaefm.data.unified_converter import (
    KNOWN_SOURCE_DATASETS,
    convert_pretrain_data,
    discover_source_datasets,
)


def _parse_datasets(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded source_data datasets to unified parquet format"
    )
    parser.add_argument("--input-dir", type=str, default="./source_data", help="Raw dataset root")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for parquet shards + manifest",
    )
    parser.add_argument("--shard-size", type=int, default=5000, help="Series rows per parquet shard")
    parser.add_argument(
        "--min-series-length",
        type=int,
        default=2048,
        help="Drop raw series shorter than this length",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated dataset folder names under --input-dir, or 'all'. "
            f"Known folders: {', '.join(KNOWN_SOURCE_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available dataset folders under --input-dir and exit",
    )
    parser.add_argument(
        "--no-clear-output",
        action="store_true",
        help="Do not delete existing train/val/test shards before writing",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    selected = _parse_datasets(args.datasets)

    if args.list_datasets:
        available = discover_source_datasets(input_dir)
        print(f"Available datasets in {input_dir.resolve()}:")
        for path in available:
            marker = "known" if path.name in KNOWN_SOURCE_DATASETS else "extra"
            print(f"  - {path.name} ({marker})")
        return

    available = discover_source_datasets(input_dir, datasets=selected)
    print(f"Converting {len(available)} dataset(s) from {input_dir.resolve()}")
    for path in available:
        print(f"  - {path.name}")

    manifest = convert_pretrain_data(
        input_dir=input_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        min_series_length=args.min_series_length,
        datasets=selected,
        clear_output=not args.no_clear_output,
    )

    print("Conversion complete")
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()
