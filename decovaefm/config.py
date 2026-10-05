"""Hydra config adapters for training scripts."""

from __future__ import annotations

import argparse
from typing import Any

from omegaconf import DictConfig, OmegaConf


def prepare_hydra_argv(argv: list[str]) -> list[str]:
    """Quote Hydra override values that contain extra ``=``.

    Lightning/MLflow checkpoints are often named ``epoch=40.ckpt``. Hydra's
    override grammar treats every ``=`` as a separator, and the shell strips
    quotes, so ``eval.checkpoint=.../epoch=40.ckpt`` fails to parse. Quote the
    value in Hydra's grammar so the path is taken literally.
    """
    prepared = [argv[0]]
    for arg in argv[1:]:
        if arg.startswith("-") or "=" not in arg:
            prepared.append(arg)
            continue
        key, value = arg.split("=", 1)
        if "=" not in value:
            prepared.append(arg)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            prepared.append(arg)
            continue
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        prepared.append(f"{key}='{escaped}'")
    return prepared


def _normalize_devices(value: Any) -> int | list[int] | str:
    if isinstance(value, int):
        return value
    if isinstance(value, list) and all(isinstance(device, int) for device in value):
        return value
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    raise ValueError(
        f"Invalid devices value {value!r}. Use an int, a list of ints, or 'auto'."
    )


def _normalize_datasets(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        names = [str(name).strip() for name in value if str(name).strip()]
        return names or None
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
        return names or None
    raise ValueError(
        f"Invalid datasets value {value!r}. Use null, a YAML list, or comma-separated names."
    )


def _section_to_dict(cfg: DictConfig, key: str) -> dict[str, Any]:
    if key not in cfg:
        return {}
    section = OmegaConf.to_container(cfg[key], resolve=True)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section


def flatten_train_config(cfg: DictConfig) -> argparse.Namespace:
    """Merge grouped Hydra config into a flat namespace for training helpers."""
    flat: dict[str, Any] = {}
    for section in ("data", "model", "trainer"):
        flat.update(_section_to_dict(cfg, section))

    if "devices" in flat:
        flat["devices"] = _normalize_devices(flat["devices"])
    if "datasets" in flat:
        flat["datasets"] = _normalize_datasets(flat["datasets"])

    return argparse.Namespace(**flat)


def flatten_eval_config(cfg: DictConfig) -> argparse.Namespace:
    """Merge train groups plus eval settings into a flat namespace."""
    args = flatten_train_config(cfg)
    eval_cfg = _section_to_dict(cfg, "eval")
    for key, value in eval_cfg.items():
        setattr(args, key, value)
    return args
