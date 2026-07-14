"""Loaders for built-in demo datasets shipped inside the wheel."""

from __future__ import annotations

from importlib import resources

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from m3._dataset import Dataset

# Maps AnnData var['feature_types'] values onto m3 modality names.
_FT_TO_MODALITY = {
    "Gene Expression": "rna",
    "Antibody Capture": "adt",
    "Peaks": "atac",
}


def liu_demo() -> Dataset:
    """Load the Liu et al. CITE-seq demo subsample (~30k cells, 3 batches).

    A ~30k-cell (30,534) stratified subsample of the full dataset (~135k cells)
    used in the publication, balanced by batch x condition x cell type so every
    annotated cell type survives. QC labels (Unk / dblt / dim) are pre-filtered.

    Returns
    -------
    :class:`m3.Dataset`
        Multi-batch, multimodal (RNA 1000 HVG + ADT 192) with obs columns
        ``batch`` (``B1``/``B2``/``B3``), ``sample_id``, ``Donor``,
        ``cond_group``, ``Age_interval``, ``mergedcelltype`` -- ready for
        ``m3.M3(...)``.

    Example
    -------
    >>> import m3
    >>> data = m3.datasets.liu_demo()
    >>> print(data)
    Dataset(n_cells=30534, batches=['B1', 'B2', 'B3'], modalities=[rna:1000, adt:192])
    >>> model = m3.M3(data,
    ...               condition_keys=["cond_group", "Age_interval"],
    ...               celltype_key="mergedcelltype",
    ...               donor_key="sample_id",
    ...               embedding_dim=30)          # batch_key defaults to "batch"
    >>> model.train(max_epochs=80)
    """
    with resources.as_file(
        resources.files("m3.datasets").joinpath("_data", "liu_demo.h5ad")
    ) as p:
        ad = anndata.read_h5ad(str(p))

    if "feature_types" not in ad.var.columns:
        raise ValueError(
            "liu_demo.h5ad is missing var['feature_types']; package data is corrupt."
        )

    modalities: dict = {}
    var: dict = {}
    present: dict = {}
    n = ad.shape[0]
    ft = ad.var["feature_types"].astype(str)
    for raw_type, mod_name in _FT_TO_MODALITY.items():
        mask = (ft == raw_type).to_numpy()
        if not mask.any():
            continue
        block = ad.X[:, mask]
        modalities[mod_name] = (block if sp.issparse(block)
                                else sp.csr_matrix(block, dtype=np.float32))
        var[mod_name] = pd.Index(list(ad.var_names[mask]))
        present[mod_name] = np.ones(n, dtype=bool)

    if not modalities:
        raise ValueError("liu_demo.h5ad has no recognised modality in var['feature_types'].")

    obs = ad.obs.reset_index(drop=True).copy()
    # The shipped demo carries two columns for the SAME 3-way split -- a stamped
    # 'cohort' (batch1/2/3) and the real 'Batch' (B1/B2/B3). m3 now uses a single
    # 'batch' key, so collapse to one 'batch' column with the real labels.
    if "Batch" in obs.columns:
        obs["batch"] = obs["Batch"].astype(str)
        obs = obs.drop(columns=[c for c in ("cohort", "Batch") if c in obs.columns])
    elif "cohort" in obs.columns:
        obs = obs.rename(columns={"cohort": "batch"})
    if "batch" not in obs.columns:
        raise ValueError("liu_demo.h5ad obs has no batch/Batch column.")
    return Dataset(modalities=modalities, obs=obs, var=var, present=present)
