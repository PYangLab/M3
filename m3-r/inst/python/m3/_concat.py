"""Multi-batch assembly: stack single-batch Datasets into one."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from m3._dataset import Dataset


def _intersect_ordered(indices: list[pd.Index]) -> pd.Index:
    common = set(indices[0])
    for idx in indices[1:]:
        common &= set(idx)
    # stable order: follow the first batch's ordering
    return pd.Index([g for g in indices[0] if g in common])


def concat(datasets: list[Dataset]) -> Dataset:
    """Combine single-batch Datasets into one multi-batch Dataset.

    - features per modality are harmonized by NAME intersection (first batch's order)
    - a batch missing a whole modality is zero-filled with present[modality]=False
    - batch labels must be unique across inputs
    """
    if not datasets:
        raise ValueError("concat requires at least one Dataset.")
    if len(datasets) == 1:
        return datasets[0]

    labels = [c for d in datasets for c in d.batches]
    if len(labels) != len(set(labels)):
        raise ValueError(f"batch labels must be unique across inputs; got {labels}.")

    all_mods = sorted({m for d in datasets for m in d.modality_names})
    n_total = sum(d.n_cells for d in datasets)

    out_mods: dict[str, sp.csr_matrix] = {}
    out_var: dict[str, pd.Index] = {}
    out_present: dict[str, np.ndarray] = {}

    for mod in all_mods:
        carriers = [d for d in datasets if mod in d.modalities]
        common = _intersect_ordered([d.var[mod] for d in carriers])
        if len(common) == 0:
            raise ValueError(f"modality '{mod}': empty feature intersection across batches.")
        if any(len(common) < len(d.var[mod]) for d in carriers):
            warnings.warn(
                f"modality '{mod}': intersection kept {len(common)} features "
                f"(batches had {[len(d.var[mod]) for d in carriers]}).",
                stacklevel=2,
            )
        blocks = []
        present = []
        for d in datasets:
            if mod in d.modalities:
                col_pos = [d.var[mod].get_loc(g) for g in common]
                blocks.append(d.modalities[mod][:, col_pos])
                present.append(np.ones(d.n_cells, dtype=bool))
            else:
                blocks.append(sp.csr_matrix((d.n_cells, len(common)), dtype=np.float32))
                present.append(np.zeros(d.n_cells, dtype=bool))
        out_mods[mod] = sp.vstack(blocks).tocsr()
        out_var[mod] = common
        out_present[mod] = np.concatenate(present)

    obs = pd.concat([d.obs for d in datasets], ignore_index=True)
    assert obs.shape[0] == n_total
    return Dataset(modalities=out_mods, obs=obs, var=out_var, present=out_present)
