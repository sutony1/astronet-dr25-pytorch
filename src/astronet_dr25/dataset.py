"""Datasets and leakage-safe KIC-grouped splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset


class ViewDataset(Dataset):
    def __init__(self, index: pd.DataFrame, root: Path) -> None:
        self.index = index.reset_index(drop=True)
        self.root = root

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        row = self.index.iloc[item]
        with np.load(self.root / row["view_path"]) as data:
            local = torch.from_numpy(data["local_view"].astype(np.float32, copy=False))
            global_view = torch.from_numpy(
                data["global_view"].astype(np.float32, copy=False)
            )
        return local, global_view, torch.tensor(float(row["label"])), int(row["row_id"])


def grouped_five_fold_split(index: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign fold 0=test, fold 1=validation, folds 2-4=training by KIC."""
    required = {"kepid", "label"}
    if not required.issubset(index.columns):
        raise KeyError(f"Index missing columns: {required - set(index.columns)}")
    output = index.copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    output["fold"] = -1
    dummy = np.zeros(len(output))
    for fold, (_, held_out) in enumerate(
        splitter.split(dummy, output["label"], groups=output["kepid"])
    ):
        output.loc[held_out, "fold"] = fold
    if (output["fold"] < 0).any():
        raise RuntimeError("Some examples were not assigned a fold")
    output["split"] = np.where(
        output["fold"] == 0,
        "test",
        np.where(output["fold"] == 1, "validation", "train"),
    )
    leakage = output.groupby("kepid")["split"].nunique().max()
    if leakage != 1:
        raise RuntimeError("KIC leakage detected across splits")
    return output
