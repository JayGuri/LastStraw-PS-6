from __future__ import annotations

import os
from typing import Any

import mlflow


def init_mlflow(cfg: dict[str, Any]) -> None:
    ml_cfg = cfg["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required in environment (.env) when mlflow is enabled.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(ml_cfg["experiment_name"])


def log_config_params(cfg: dict[str, Any]) -> None:
    flat = _flatten(cfg)
    for key, value in flat.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
