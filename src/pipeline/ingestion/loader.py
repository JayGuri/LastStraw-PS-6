"""Ingestion — load and validate flood.csv, log metadata to MLflow."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


# The 20 environmental feature columns from the GFD dataset.
FEATURE_COLUMNS: list[str] = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]


def load_csv(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load flood.csv, validate schema, copy to processed dir.

    Logs ingestion metadata as MLflow params and tags.
    Returns the validated DataFrame.
    """
    raw_path = Path(cfg["paths"]["csv_raw"])
    processed_path = Path(cfg["paths"]["csv_processed"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"flood.csv not found at '{raw_path}'. "
            "Place the dataset there before running the pipeline."
        )

    df = pd.read_csv(raw_path)

    # ── Schema validation ────────────────────────────────────────────────────
    label_col = cfg["data"]["label_column"]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"CSV missing expected feature columns: {missing_features}")
    if label_col not in df.columns:
        raise ValueError(f"CSV missing label column '{label_col}'.")

    # ── Copy to processed ────────────────────────────────────────────────────
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, processed_path)

    # ── MLflow logging ───────────────────────────────────────────────────────
    try:
        mlflow.log_params({
            "ingestion.rows": len(df),
            "ingestion.columns": len(df.columns),
            "ingestion.source": str(raw_path),
            "ingestion.label_column": label_col,
            "ingestion.missing_values": int(df.isnull().sum().sum()),
        })
        mlflow.set_tags({
            "ingestion.status": "ok",
            "ingestion.feature_count": str(len(FEATURE_COLUMNS)),
        })
    except Exception:
        pass  # MLflow not active — don't crash ingestion

    print(f"[ingestion] Loaded {len(df):,} rows × {len(df.columns)} cols from '{raw_path}'")
    print(f"[ingestion] Copied to '{processed_path}'")

    return df
