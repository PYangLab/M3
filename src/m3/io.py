"""Readers: common single-cell formats -> one canonical m3 Dataset."""

from __future__ import annotations

import warnings

import anndata as _ad
import h5py as _h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from m3._dataset import Dataset


def _validate_raw(counts: dict, obs: pd.DataFrame) -> None:
    # Raw-count sanity check only. NaN in obs is checked later, at M3 construction,
    # and only on the columns the user actually selects as roles (condition / batch /
    # donor / celltype / batch) -- checking every obs column here just floods warnings.
    for name, mat in counts.items():
        arr = sp.csr_matrix(mat, dtype=np.float32)
        if arr.nnz:
            data = arr.data
            if not np.allclose(data, np.round(data)):
                warnings.warn(
                    f"modality '{name}' has non-integer values; m3 expects RAW counts "
                    "(it normalizes internally).",
                    stacklevel=3,
                )


def read_matrix(
    *,
    counts: dict,
    obs: pd.DataFrame,
    var: dict,
    batch: str = "batch0",
) -> Dataset:
    """Build a single-batch Dataset from in-memory per-modality count arrays.

    counts: {modality -> array/sparse [n_cells, n_features]} RAW counts
    obs:    DataFrame [n_cells, ...]
    var:    {modality -> list/Index of feature names}
    batch: label written to obs['batch'] for a single-batch input; ignored if
           obs already carries a 'batch' column (multi-batch data brings its own)
    """
    if not counts:
        raise ValueError("counts must contain at least one modality.")
    _validate_raw(counts, obs)
    n = obs.shape[0]
    modalities: dict[str, sp.csr_matrix] = {}
    var_out: dict[str, pd.Index] = {}
    present: dict[str, np.ndarray] = {}
    for name, mat in counts.items():
        m = sp.csr_matrix(mat, dtype=np.float32)
        if m.shape[0] != n:
            raise ValueError(f"modality '{name}' has {m.shape[0]} rows but obs has {n}.")
        names = pd.Index(list(var[name]))
        if len(names) != m.shape[1]:
            raise ValueError(f"var['{name}'] has {len(names)} names for {m.shape[1]} columns.")
        if names.has_duplicates:
            dups = names[names.duplicated()].unique().tolist()
            raise ValueError(
                f"modality '{name}': feature names must be unique, but {len(dups)} are "
                f"duplicated (e.g. {dups[:5]}). De-duplicate first -- e.g. AnnData's "
                "adata.var_names_make_unique() -- then read.")
        modalities[name] = m
        var_out[name] = names
        present[name] = np.ones(n, dtype=bool)
    obs_out = obs.reset_index(drop=True).copy()
    if "batch" not in obs_out.columns:
        obs_out["batch"] = batch
    return Dataset(modalities=modalities, obs=obs_out, var=var_out, present=present)


_FEATURE_TYPE_TO_MODALITY = {
    "Gene Expression": "rna",
    "Antibody Capture": "adt",
    "Peaks": "atac",
}


def from_anndata(adata, *, batch: str = "batch0", modality: str = "rna") -> Dataset:
    """Build a single-batch Dataset from an in-memory AnnData.

    If adata.var has a 'feature_types' column, the matrix is split into modalities;
    otherwise the whole matrix becomes the single modality `modality` (default 'rna').

    Note: only a single AnnData is supported. A dict/MuData (one AnnData per
    modality) input is deferred to a later plan along with the MuData reader.
    """
    obs = adata.obs.reset_index(drop=True).copy()
    if "feature_types" in adata.var.columns:
        counts: dict = {}
        var: dict = {}
        ft = adata.var["feature_types"].astype(str)
        for raw_type, mod in _FEATURE_TYPE_TO_MODALITY.items():
            mask = (ft == raw_type).to_numpy()
            if mask.any():
                counts[mod] = adata.X[:, mask]
                var[mod] = list(adata.var_names[mask])
        if not counts:
            raise ValueError("feature_types present but no recognized modality found.")
    else:
        counts = {modality: adata.X}
        var = {modality: list(adata.var_names)}
    return read_matrix(counts=counts, obs=obs, var=var, batch=batch)


def read_h5ad(path: str, *, batch: str = "batch0", modality: str = "rna") -> Dataset:
    """Read one batch from an AnnData .h5ad file (thin shim over from_anndata)."""
    return from_anndata(_ad.read_h5ad(path), batch=batch, modality=modality)


def _read_paper_h5(path: str):
    """Return (counts cells x features, feature_names or None) from a paper-format h5.

    Orientation is resolved against the feature count: ``matrix/data`` may be
    stored cells x features (e.g. the Liu/Stephenson files) or features x cells;
    either is normalized to cells x features.
    """
    with _h5py.File(path, "r") as f:
        if "matrix/data" in f:
            data = np.asarray(f["matrix/data"])
            feats = None
            if "matrix/features" in f:
                feats = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in np.asarray(f["matrix/features"])]
        elif "data" in f:
            data = np.asarray(f["data"])
            feats = None
        else:
            raise KeyError(f"{path}: neither 'matrix/data' nor 'data' found.")
    if feats is not None:
        nf = len(feats)
        if data.shape[1] == nf:
            counts = data                # already cells x features
        elif data.shape[0] == nf:
            counts = data.T              # features x cells -> cells x features
        else:
            raise ValueError(
                f"{path}: data shape {data.shape} matches neither orientation of {nf} features."
            )
    else:
        counts = data                    # no feature names: assume cells x features
    return counts, feats


def read_h5(
    *,
    rna: str | None = None,
    adt: str | None = None,
    atac: str | None = None,
    metadata: str,
    batch: str = "batch0",
) -> Dataset:
    """Read one batch from paper-format HDF5 count matrices + a metadata CSV."""
    obs = pd.read_csv(metadata)
    counts: dict = {}
    var: dict = {}
    for name, path in (("rna", rna), ("adt", adt), ("atac", atac)):
        if path is None:
            continue
        mat, feats = _read_paper_h5(path)
        if mat.shape[0] != obs.shape[0]:
            raise ValueError(
                f"{name}: {mat.shape[0]} cells in h5 but {obs.shape[0]} rows in metadata."
            )
        counts[name] = mat
        var[name] = feats if feats is not None else [f"{name}_{i}" for i in range(mat.shape[1])]
    if not counts:
        raise ValueError("read_h5 requires at least one of rna/adt/atac.")
    return read_matrix(counts=counts, obs=obs, var=var, batch=batch)
