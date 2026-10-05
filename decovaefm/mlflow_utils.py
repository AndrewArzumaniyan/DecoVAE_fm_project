"""MLflow helpers for Hydra-driven train and evaluate entrypoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

# Shared across Lightning DDP subprocesses so Hydra re-entry does not
# create a second MLflow run. The parent sets this before spawning workers.
_MLFLOW_RUN_ID_ENV = "DECOVAEFM_MLFLOW_RUN_ID"


def get_git_info() -> dict[str, str]:
    """Return git dirty flag for reproducibility logging.

    Branch and commit SHA are already recorded by MLflow's source tags.
    """
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git.dirty": "unknown"}

    return {"git.dirty": "true" if status.strip() else "false"}


def _ensure_tracking_uri(tracking_uri: str) -> str:
    """Normalize tracking URI for MLflow clients and Lightning's MLFlowLogger."""
    uri = str(tracking_uri).strip()
    if uri.startswith(("sqlite:", "http://", "https://", "postgresql", "mysql")):
        return uri
    # Legacy filesystem store (maintenance mode in MLflow 3.x).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if not uri.startswith("file:"):
        # Lightning MLFlowLogger.save_dir only resolves for file: URIs; without
        # this prefix checkpoints land at {experiment_id}/{run_id}/ in cwd.
        uri = f"file:{uri}"
    return uri


def _stringify_param_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def flatten_hydra_params(cfg: DictConfig) -> dict[str, Any]:
    """Dot-flatten Hydra training groups for MLflow param logging."""
    flat: dict[str, Any] = {}
    for section in ("data", "model", "trainer", "eval"):
        if section not in cfg:
            continue
        section_cfg = OmegaConf.to_container(cfg[section], resolve=True)
        if not isinstance(section_cfg, dict):
            continue
        for key, value in section_cfg.items():
            flat[f"{section}.{key}"] = _stringify_param_value(value)

    choices = OmegaConf.select(cfg, "hydra.runtime.choices", default={})
    if isinstance(choices, dict):
        for key, value in choices.items():
            if value is not None:
                flat[f"hydra.choice.{key}"] = _stringify_param_value(value)

    return flat


def _default_run_name(cfg: DictConfig) -> str:
    try:
        output_dir = HydraConfig.get().runtime.output_dir
        if output_dir:
            return Path(str(output_dir)).name
    except ValueError:
        pass

    job_name = OmegaConf.select(cfg, "hydra.job.name", default="train")
    return str(job_name)


def _resolve_run_name(cfg: DictConfig, *, prefix: str = "") -> str:
    mlflow_cfg = cfg.mlflow
    run_name = mlflow_cfg.get("run_name")
    if run_name:
        return str(run_name)
    base = _default_run_name(cfg)
    return f"{prefix}{base}" if prefix else base


def build_mlflow_logger(
    cfg: DictConfig, *, run_name_prefix: str = ""
) -> MLFlowLogger | bool:
    """Build a Lightning MLFlow logger or fall back to the default CSV logger."""
    mlflow_cfg = cfg.get("mlflow")
    if mlflow_cfg is None or not mlflow_cfg.get("enabled", False):
        return True

    tags = OmegaConf.to_container(mlflow_cfg.get("tags"), resolve=True)
    if not isinstance(tags, dict):
        tags = {}
    tags = {str(key): str(value) for key, value in tags.items()}

    tracking_uri = _ensure_tracking_uri(mlflow_cfg.tracking_uri)
    existing_run_id = os.environ.get(_MLFLOW_RUN_ID_ENV)
    logger = MLFlowLogger(
        experiment_name=str(mlflow_cfg.experiment_name),
        tracking_uri=tracking_uri,
        run_name=_resolve_run_name(cfg, prefix=run_name_prefix),
        tags=tags,
        log_model=bool(mlflow_cfg.get("log_model", False)),
        run_id=existing_run_id,
    )

    # Lightning DDP re-executes this script in child processes. Create the run
    # once in the pre-launch process and reuse its id so start time is correct
    # and workers do not open a second run.
    if existing_run_id is None and os.environ.get("LOCAL_RANK", "0") == "0":
        if mlflow_cfg.get("log_params", True):
            params = flatten_hydra_params(cfg)
            params.update(get_git_info())
            logger.log_hyperparams(params)
        run_id = logger.run_id
        if run_id:
            os.environ[_MLFLOW_RUN_ID_ENV] = str(run_id)

    return logger


def log_evaluation_checkpoint(trainer: Any, checkpoint: str | Path) -> None:
    """Attach the evaluated checkpoint to the active Lightning MLflow run."""
    logger = getattr(trainer, "logger", None)
    if not isinstance(logger, MLFlowLogger):
        return
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.is_file():
        return
    logger.experiment.log_artifact(logger.run_id, str(checkpoint_path))
