"""Canonical m3 input container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Dataset:
    """Multimodal, multi-batch raw-count container.

    modalities: {name -> csr_matrix [n_cells, n_features]} raw counts
    obs:        DataFrame [n_cells, ...]; must contain a 'batch' column
    var:        {name -> pd.Index of feature names}, aligned to each modality's columns
    present:    {name -> bool ndarray [n_cells]} per-cell modality presence
    """

    modalities: dict[str, sp.csr_matrix]
    obs: pd.DataFrame
    var: dict[str, pd.Index]
    present: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("Dataset requires at least one modality.")
        n = self.obs.shape[0]
        if "batch" not in self.obs.columns:
            raise ValueError("obs must contain a 'batch' column.")
        for name, mat in self.modalities.items():
            if mat.shape[0] != n:
                raise ValueError(
                    f"modality '{name}' has {mat.shape[0]} rows but obs has {n} (n_cells)."
                )
            if name not in self.var or len(self.var[name]) != mat.shape[1]:
                raise ValueError(
                    f"var['{name}'] length must equal modality '{name}' column count."
                )
            if pd.Index(self.var[name]).has_duplicates:
                raise ValueError(
                    f"var['{name}'] has duplicate feature names; names must be unique."
                )
            if name not in self.present or self.present[name].shape[0] != n:
                raise ValueError(f"present['{name}'] must be a length-{n} boolean array.")

    @property
    def n_cells(self) -> int:
        return self.obs.shape[0]

    @property
    def modality_names(self) -> list[str]:
        return list(self.modalities.keys())

    @property
    def batches(self) -> list[str]:
        return list(pd.unique(self.obs["batch"]))

    def __repr__(self) -> str:
        mods = ", ".join(f"{k}:{v.shape[1]}" for k, v in self.modalities.items())
        return f"Dataset(n_cells={self.n_cells}, batches={self.batches}, modalities=[{mods}])"
