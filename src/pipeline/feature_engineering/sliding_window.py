"""Feature Engineering — Sliding Window Dataset.

Converts a flat (N, 20) feature array into (N-W+1) overlapping sequences of
shape (W, 20), each labelled with the FloodProbability at the last time step.

The sliding window treats consecutive CSV rows as "time steps," capturing how
correlated environmental conditions build up over W observations before a
flood event.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset[dict[str, Tensor]]):
    """Sliding-window view over a scaled feature array.

    Args:
        features:    float32 array of shape [N, num_features].
        labels:      float32 array of shape [N] — FloodProbability per row.
        window_size: Number of consecutive rows in each input sequence (W).

    Each sample ``i`` yields:
        features: Tensor[W, num_features]  — rows [i, i+W)
        label:    Tensor scalar            — labels[i + W - 1]
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int,
    ) -> None:
        if len(features) < window_size:
            raise ValueError(
                f"Dataset has {len(features)} rows but window_size={window_size}. "
                "Need at least window_size rows."
            )
        self.features = torch.from_numpy(features)  # [N, F]
        self.labels = torch.from_numpy(labels)      # [N]
        self.window_size = window_size
        self._len = len(features) - window_size + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x = self.features[idx : idx + self.window_size]       # [W, F]
        y = self.labels[idx + self.window_size - 1]           # scalar
        return {"features": x, "label": y}


def build_datasets(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: dict[str, Any],
) -> tuple["SlidingWindowDataset", "SlidingWindowDataset", "SlidingWindowDataset"]:
    """Split features/labels into train/val/test and wrap in SlidingWindowDataset.

    Returns (train_ds, val_ds, test_ds).
    """
    window_size = int(cfg["data"]["window_size"])
    split = cfg["data"]["split"]
    n = len(features)
    n_train = int(n * float(split["train"]))
    n_val = int(n * float(split["val"]))

    train_ds = SlidingWindowDataset(features[:n_train], labels[:n_train], window_size)
    val_ds = SlidingWindowDataset(
        features[n_train : n_train + n_val],
        labels[n_train : n_train + n_val],
        window_size,
    )
    test_ds = SlidingWindowDataset(
        features[n_train + n_val :],
        labels[n_train + n_val :],
        window_size,
    )

    print(
        f"[feature_engineering] Sliding window W={window_size} → "
        f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} samples"
    )
    return train_ds, val_ds, test_ds
