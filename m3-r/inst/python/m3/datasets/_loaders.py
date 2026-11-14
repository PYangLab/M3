"""Loaders for built-in demo datasets shipped inside the wheel."""

from __future__ import annotations

import operator
import os
import warnings
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


# Fewest cells liu_demo(n_cells=...) keeps per (batch, cell type), so that a rare
# type is still a visible group in a small subsample rather than a few points.
_MIN_CELLS_PER_TYPE = 50


def _subsample_rows(obs: pd.DataFrame, n_cells) -> np.ndarray:
    """Row positions of a deterministic ``n_cells`` subsample of the demo.

    Every batch contributes the same number of cells: m3 trains on a
    batch-balanced subset by default, so cells beyond the smallest batch's share
    would be loaded and then left out of training. Within a batch each cell type
    first gets ``_MIN_CELLS_PER_TYPE`` cells (or all it has), and the rest of the
    batch's share is split in proportion to what each type has left.

    Cells are taken at even spacing through each (batch, cell type) group instead
    of with a random generator. The file's order within a group is already mixed
    across samples, and even spacing gives the same cells on every numpy version,
    which keeps the R and Python demos identical.
    """
    if isinstance(n_cells, bool):
        raise TypeError(f"n_cells must be an integer, got {n_cells!r}.")
    try:
        n = operator.index(n_cells)
    except TypeError:
        raise TypeError(f"n_cells must be an integer, got {n_cells!r}.") from None

    batch = obs["batch"].astype(str).to_numpy()
    celltype = obs["mergedcelltype"].astype(str).to_numpy()
    groups = {
        b: [np.flatnonzero((batch == b) & (celltype == t))
            for t in sorted(set(celltype[batch == b]))]
        for b in sorted(set(batch))
    }
    minimum = {b: sum(min(len(g), _MIN_CELLS_PER_TYPE) for g in gs) for b, gs in groups.items()}
    available = {b: sum(len(g) for g in gs) for b, gs in groups.items()}
    lowest = len(groups) * max(minimum.values())
    highest = len(groups) * min(available.values())
    if not lowest <= n <= highest:
        raise ValueError(
            f"n_cells must be between {lowest} and {highest} for this demo, got {n}. "
            f"It takes the same number of cells from each of the {len(groups)} batches, "
            f"and the smallest batch has {min(available.values())}. Omit n_cells to "
            f"load all {len(obs)} cells."
        )

    share, extra = divmod(n, len(groups))
    rows = []
    for i, gs in enumerate(groups.values()):
        size = np.array([len(g) for g in gs])
        take = np.minimum(size, _MIN_CELLS_PER_TYPE)
        spare = size - take
        rest = share + (i < extra) - take.sum()
        if rest:
            # Largest-remainder split of `rest` in proportion to `spare`, in whole
            # cells, so the batch lands on exactly its share.
            take = take + rest * spare // spare.sum()
            remainder = rest * spare % spare.sum()
            short = share + (i < extra) - take.sum()
            take[np.argsort(-remainder, kind="stable")[:short]] += 1
        for g, k in zip(gs, take):
            rows.append(g[(2 * np.arange(k) + 1) * len(g) // (2 * k)])
    return np.sort(np.concatenate(rows))


def liu_demo(n_cells: int | None = None) -> Dataset:
    """Load the Liu et al. CITE-seq demo subsample (~30k cells, 3 batches).

    A ~30k-cell (30,534) stratified subsample of the full dataset (~135k cells)
    used in the publication, balanced by batch x condition x cell type so every
    annotated cell type survives. QC labels (Unk / dblt / dim) are pre-filtered.

    Parameters
    ----------
    n_cells
        Load a smaller, fixed subsample of this many cells instead of all of
        them, for a quicker run: training time is proportional to the number of
        cells. The same cells are returned on every call. Each batch contributes
        equally, and every cell type of every batch keeps at least 50 cells (or
        all it has), so no cell type is lost. Must be between 2,250 and 21,489.
        ``None`` (the default) loads all 30,534 cells.

        This is meant for cell-level integration. Donor-level prediction and
        attribution need enough cells per donor, so use the full demo there.

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
        # 23 names appear twice across the whole var index -- each is a marker
        # measured both ways (gene CD14 and antibody CD14), which is normal for
        # CITE-seq. anndata warns about the combined index, but we split var per
        # modality just below and each block is unique on its own (asserted there),
        # so the warning is noise here. Do NOT "fix" it with var_names_make_unique():
        # that would rename the ADT CD14 to CD14-1 and surface a wrong protein name
        # in every downstream readout, attribution included.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Variable names are not unique", category=UserWarning
            )
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
        names = pd.Index(list(ad.var_names[mask]))
        if not names.is_unique:
            dups = sorted(set(names[names.duplicated()]))
            raise ValueError(
                f"liu_demo.h5ad has duplicate {mod_name} feature names "
                f"({', '.join(dups[:5])}); package data is corrupt."
            )
        var[mod_name] = names
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
    if n_cells is not None:
        rows = _subsample_rows(obs, n_cells)
        modalities = {m: x[rows] for m, x in modalities.items()}
        present = {m: p[rows] for m, p in present.items()}
        obs = obs.iloc[rows].reset_index(drop=True)
    return Dataset(modalities=modalities, obs=obs, var=var, present=present)


# Models trained in advance on the full demo, one per tutorial that trains on it.
_DEMO_MODELS = {
    "attribution": "liu_demo_attribution.m3",
    "patient_prediction": "liu_demo_patient_prediction.m3",
}


def liu_demo_model(name: str) -> str:
    """Path of a model trained in advance on the full :func:`liu_demo` data.

    Training on the full demo takes a while; these files let you go straight to
    the read-outs. Each was written by ``model.save(...)`` at the end of the
    tutorial of the same name and is read back with :meth:`m3.M3.load`, together
    with the full demo data.

    Parameters
    ----------
    name
        ``"patient_prediction"`` (batch ``B3`` held out, as in the donor-level
        prediction tutorial) or ``"attribution"`` (trained on all three batches,
        as in the feature attribution tutorial).

    Returns
    -------
    str
        The path of the model file inside the installed package.

    Example
    -------
    >>> import m3
    >>> data = m3.datasets.liu_demo()
    >>> model = m3.M3.load(m3.datasets.liu_demo_model("patient_prediction"), data)
    >>> preds = model.predict_donors()
    """
    if name not in _DEMO_MODELS:
        raise ValueError(
            f"unknown demo model {name!r}; choose one of {sorted(_DEMO_MODELS)}.")
    path = str(resources.files("m3.datasets").joinpath("_data", _DEMO_MODELS[name]))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing; the installed m3 package is incomplete (reinstall m3-sc).")
    return path
