"""Readers: common single-cell formats -> one canonical m3 Dataset."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

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
    batch: str | None = None,
) -> Dataset:
    """Build a Dataset from in-memory per-modality count arrays.

    counts: {modality -> array/sparse [n_cells, n_features]} RAW counts
    obs:    DataFrame [n_cells, ...]
    var:    {modality -> list/Index of feature names}
    batch:  when all cells are one batch, its label; written to obs['batch'].
            Leave it out when obs has a column that identifies the batches, and
            name that column as batch_key when building the model.
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
    # No label is invented here: which cells form a batch is the user's to say,
    # either with this label or with batch_key when the model is built.
    if batch is not None:
        if "batch" in obs_out.columns and not (obs_out["batch"].astype(str) == str(batch)).all():
            raise ValueError(
                f"obs already has a 'batch' column, and its values differ from batch={batch!r}. "
                "Leave out batch= to use the column, or drop the column to use the label.")
        obs_out["batch"] = batch
    return Dataset(modalities=modalities, obs=obs_out, var=var_out, present=present)


_FEATURE_TYPE_TO_MODALITY = {
    "Gene Expression": "rna",
    "Antibody Capture": "adt",
    "Peaks": "atac",
}


def _layer_name(layer, mod: str):
    """The layer that holds the raw counts of `mod`, or None for .X."""
    if layer is None or isinstance(layer, str):
        return layer
    return layer.get(mod)


def _counts_of(adata, layer, mod: str):
    name = _layer_name(layer, mod)
    if name is None:
        return adata.X
    if name not in adata.layers:
        raise ValueError(
            f"{mod}: the AnnData has no layer {name!r} (layers: "
            f"{', '.join(str(k) for k in adata.layers.keys() if k is not None) or 'none'}). "
            "Leave layer= out to use .X.")
    return adata.layers[name]


def _examples(names, k: int = 3) -> str:
    return ", ".join(repr(str(x)) for x in list(names)[:k])


def _with_batch_label(obs: pd.DataFrame, batch, mod: str) -> pd.DataFrame:
    if batch is None:
        return obs
    if "batch" in obs.columns and not (obs["batch"].astype(str) == str(batch)).all():
        raise ValueError(
            f"{mod}: obs already has a 'batch' column, and its values differ from "
            f"batch={batch!r}. Leave out batch= to use the column, or drop the column "
            "to use the label.")
    obs = obs.copy()
    obs["batch"] = batch
    return obs


def _plan_cells(obs_by_mod: dict, *, batch, batch_key: str, join: str):
    """Decide, from the cell names and batch columns alone, which cells to keep.

    Within a batch every modality must have the same cells, or none of them (a
    modality missing from a whole batch is a mosaic). This is checked here,
    before any count matrix is built, so a mismatch fails with the batch, the
    counts and example names rather than after the matrices are assembled.

    Returns (obs_by_mod with any batch label applied, kept names per batch,
    cells dropped per batch and modality).
    """
    if join not in ("exact", "inner"):
        raise ValueError(f"join must be 'exact' or 'inner', not {join!r}.")
    if batch is not None and batch_key != "batch":
        raise ValueError(
            f"batch={batch!r} writes the column 'batch', but batch_key={batch_key!r}. "
            "Use one or the other.")
    labelled: dict[str, pd.DataFrame] = {}
    cell_batch: dict[str, pd.Series] = {}
    for mod, obs in obs_by_mod.items():
        obs = _with_batch_label(obs, batch, mod)
        names = pd.Index(obs.index.astype(str))
        if names.has_duplicates:
            dups = names[names.duplicated()].unique()
            raise ValueError(
                f"{mod}: {len(dups)} cell names occur more than once (e.g. {_examples(dups)}). "
                "Cell names must be unique within each modality. Barcodes that repeat across "
                "batches can be made unique with the batch, e.g. when combining per-batch files: "
                "ad.concat(pieces, label='batch', keys=batch_names, index_unique='-'), or "
                "adata.obs_names = adata.obs['batch'].astype(str) + '_' + adata.obs_names. "
                "Do not use obs_names_make_unique(): it numbers repeats in file order, which "
                "can pair cells of different batches.")
        if batch_key not in obs.columns:
            raise ValueError(
                f"{mod}: obs has no {batch_key!r} column (columns: "
                f"{', '.join(map(str, obs.columns)) or 'none'}). Every modality needs the "
                "column that identifies the batches, so that m3 can check which cells each "
                "batch has. Name it with batch_key=, pass batch='<label>' if all cells are "
                "one batch, or add it when combining per-batch files: "
                "ad.concat(pieces, label='batch', keys=batch_names, index_unique='-').")
        labels = obs[batch_key]
        if labels.isna().any():
            raise ValueError(
                f"{mod}: {int(labels.isna().sum())} cells have no value in obs[{batch_key!r}].")
        labelled[mod] = obs
        cell_batch[mod] = pd.Series(labels.astype(str).to_numpy(), index=names)

    # one cell, one batch: the modalities must not disagree on it
    first = next(iter(cell_batch))
    for mod, s in cell_batch.items():
        if mod == first:
            continue
        ref = pd.concat([cell_batch[m] for m in cell_batch if m != mod])
        ref = ref[~ref.index.duplicated()]
        shared = s.index.intersection(ref.index)
        differ = shared[(s[shared] != ref[shared]).to_numpy()]
        if len(differ):
            c = differ[0]
            raise ValueError(
                f"obs[{batch_key!r}] gives {len(differ)} cells a different batch in {mod} than "
                f"in the other modalities (e.g. cell {c!r}: {s[c]!r} in {mod}, {ref[c]!r} "
                "elsewhere). Make the batch column agree across the modalities.")

    order = list(pd.unique(pd.concat(list(cell_batch.values())).to_numpy()))
    kept: dict[str, set] = {}
    dropped: dict[str, dict[str, int]] = {}
    for b in order:
        sets = {m: set(s.index[(s == b).to_numpy()]) for m, s in cell_batch.items()}
        sets = {m: v for m, v in sets.items() if v}
        common = set.intersection(*sets.values())
        if all(len(v) == len(common) for v in sets.values()):
            kept[b] = common
            continue
        if not common:
            (m1, v1), (m2, v2) = list(sets.items())[:2]
            raise ValueError(
                f"batch {b!r}: {m1} and {m2} share none of their cell names (e.g. "
                f"{_examples(sorted(v1))} in {m1}; {_examples(sorted(v2))} in {m2}). Cell names "
                "must match across modalities: check for a suffix or prefix changed in one "
                "file only (e.g. '-1' vs '.1').")
        if join == "exact":
            counts = ", ".join(f"{m} {len(v)}" for m, v in sets.items())
            extra = "; ".join(f"only in {m}: {_examples(sorted(v - common))}"
                              for m, v in sets.items() if len(v) > len(common))
            raise ValueError(
                f"batch {b!r}: the modalities do not have the same cells ({counts}; "
                f"{len(common)} in all). {extra}. Within a batch every modality must have the "
                "same cells (a modality may be missing from a whole batch). If cells were "
                "removed from one modality on purpose, e.g. by QC, pass join='inner' to keep "
                "only the cells every modality has; otherwise check that the cell names match.")
        kept[b] = common
        dropped[b] = {m: len(v) - len(common) for m, v in sets.items() if len(v) > len(common)}
    return labelled, kept, dropped


def _merge_obs(obs_by_mod: dict, cells: pd.Index) -> pd.DataFrame:
    """One obs over `cells`: the columns of every modality; where two modalities
    have the same column, its values must agree for the cells they share."""
    merged = None
    first = None
    for mod, obs in obs_by_mod.items():
        o = obs.copy()
        o.index = o.index.astype(str)
        o = o.reindex(cells)
        if merged is None:
            merged, first = o, mod
            continue
        for col in o.columns:
            if col not in merged.columns:
                merged[col] = o[col]
                continue
            a, b = merged[col], o[col]
            both = a.notna().to_numpy() & b.notna().to_numpy()
            differ = both & (a.astype(str).to_numpy() != b.astype(str).to_numpy())
            if differ.any():
                c = cells[differ][0]
                raise ValueError(
                    f"obs column {col!r} differs between {first} and {mod} for "
                    f"{int(differ.sum())} cells (e.g. cell {c!r}: {a[c]!r} vs {b[c]!r}). "
                    "Make it agree, or drop it from one of the modalities.")
            if isinstance(a.dtype, pd.CategoricalDtype) or isinstance(b.dtype, pd.CategoricalDtype):
                a, b = a.astype(object), b.astype(object)
            merged[col] = a.where(a.notna(), b)
    return merged


def _from_modality_dict(adatas, *, batch, batch_key: str, layer, join: str) -> Dataset:
    if not adatas:
        raise ValueError("pass at least one modality, e.g. {'rna': rna_adata}.")
    for mod, a in adatas.items():
        if not isinstance(mod, str):
            raise TypeError(f"modality names must be strings, not {mod!r}.")
        if not isinstance(a, _ad.AnnData):
            raise TypeError(f"{mod}: expected an AnnData, got {type(a).__name__}.")
    if isinstance(layer, Mapping):
        unknown = set(layer) - set(adatas)
        if unknown:
            raise ValueError(f"layer= names modalities that were not passed: {sorted(unknown)}.")
    obs_by_mod = {m: a.obs.set_axis(a.obs_names) for m, a in adatas.items()}
    obs_by_mod, kept, dropped = _plan_cells(obs_by_mod, batch=batch, batch_key=batch_key, join=join)

    keep_all = set().union(*kept.values())
    cells: list = []
    seen: set = set()
    for a in adatas.values():
        for c in a.obs_names.astype(str):
            if c in keep_all and c not in seen:
                seen.add(c)
                cells.append(c)
    cells = pd.Index(cells)

    modalities: dict[str, sp.csr_matrix] = {}
    var: dict[str, pd.Index] = {}
    present: dict[str, np.ndarray] = {}
    for mod, a in adatas.items():
        X = sp.csr_matrix(_counts_of(a, layer, mod), dtype=np.float32)
        if X.nnz and not np.allclose(X.data, np.round(X.data)):
            where = f"layer {_layer_name(layer, mod)!r}" if _layer_name(layer, mod) else ".X"
            raise ValueError(
                f"{mod}: {where} has non-integer values, but m3 needs raw counts (it "
                "normalizes internally). If the counts are in a layer, name it, e.g. "
                "layer='counts' or layer={'rna': 'counts', 'adt': 'counts'}.")
        names = pd.Index(a.var_names.astype(str))
        if names.has_duplicates:
            dups = names[names.duplicated()].unique().tolist()
            raise ValueError(
                f"{mod}: feature names must be unique, but {len(dups)} are duplicated "
                f"(e.g. {dups[:5]}). De-duplicate first, e.g. adata.var_names_make_unique().")
        src = np.flatnonzero(a.obs_names.astype(str).isin(keep_all))
        dst = cells.get_indexer(a.obs_names.astype(str)[src])
        place = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (dst, np.arange(len(src)))),
                              shape=(len(cells), len(src)))
        modalities[mod] = (place @ X[src]).tocsr()
        var[mod] = names
        present[mod] = np.zeros(len(cells), dtype=bool)
        present[mod][dst] = True

    obs = _merge_obs({m: o.loc[o.index.astype(str).isin(keep_all)] for m, o in obs_by_mod.items()},
                     cells)
    labels = obs[batch_key].astype(str).to_numpy()
    for mod, mat in modalities.items():
        totals = np.asarray(mat.sum(axis=1)).ravel()
        for b in kept:
            rows = (labels == b) & present[mod]
            if rows.any() and totals[rows].sum() == 0:
                raise ValueError(
                    f"{mod} is all zero for the {int(rows.sum())} cells of batch {b!r}: they "
                    f"were probably not measured for {mod} and filled with zeros (merging "
                    f"Seurat objects does this). Remove them from the {mod} AnnData, e.g. "
                    f"{mod} = {mod}[{mod}.obs[{batch_key!r}] != {b!r}], so that m3 marks "
                    f"{mod} as missing for that batch.")

    if "obs_names" in obs.columns and not (obs["obs_names"].astype(str).to_numpy()
                                           == cells.to_numpy()).all():
        raise ValueError(
            "obs already has an 'obs_names' column that differs from the cell names; "
            "rename it before reading.")
    obs["obs_names"] = cells.to_numpy()
    obs = obs.reset_index(drop=True)

    table = pd.DataFrame({m: [int(present[m][labels == b].sum()) for b in kept] for m in modalities},
                         index=list(kept))
    print(f"cells per batch ({batch_key!r}) and modality:\n{table.to_string()}")
    for b, per_mod in dropped.items():
        print(f"join='inner' dropped in batch {b!r}: "
              + ", ".join(f"{n} cells only in {m}" for m, n in per_mod.items()))
    return Dataset(modalities=modalities, obs=obs, var=var, present=present)


def from_anndata(adata, *, batch: str | None = None, modality: str = "rna",
                 batch_key: str = "batch", layer=None, join: str = "exact") -> Dataset:
    """Build a Dataset from in-memory AnnData.

    One AnnData: if adata.var has a 'feature_types' column, the matrix is split
    into modalities; otherwise the whole matrix becomes the single modality
    `modality` (default 'rna').

    One AnnData per modality, as a dict such as {"rna": rna, "adt": adt}: the
    cells are matched by name, and a cell missing from a modality is marked as
    not measured for it. Each modality's obs needs the batch column (batch_key),
    and within a batch every modality must have the same cells, or none of them;
    join="inner" instead keeps the cells all modalities of the batch have.
    Shared obs columns must agree. Cell names are kept as they are, in
    obs['obs_names']. Prints the number of cells per batch and modality.

    batch: when all cells are one batch, its label, written to obs['batch'].
    layer: where the raw counts are, if not in .X: a layer name, or (with a
           dict of AnnData) {modality: layer name}.
    """
    if isinstance(adata, Mapping):
        if modality != "rna":
            raise ValueError("modality= names the matrix of a single AnnData; with a dict, "
                             "the dict keys name the modalities.")
        return _from_modality_dict(adata, batch=batch, batch_key=batch_key, layer=layer, join=join)
    if batch_key != "batch" or join != "exact":
        raise ValueError("batch_key= and join= apply when passing one AnnData per modality, "
                         "e.g. from_anndata({'rna': rna, 'adt': adt}); with a single AnnData, "
                         "name the batch column with batch_key when building the model.")
    if layer is not None and not isinstance(layer, str):
        raise ValueError("with a single AnnData, layer= is one layer name.")
    # Keep the cell barcodes as a column: readouts are not guaranteed to come back in
    # input order (see M3.cell_metadata), so this is what aligns them to `adata`.
    obs = adata.obs.copy()
    names = adata.obs_names.to_numpy()
    if "obs_names" in obs.columns and not (obs["obs_names"].to_numpy() == names).all():
        raise ValueError(
            "adata.obs already has an 'obs_names' column that differs from "
            "adata.obs_names; rename it before reading.")
    obs["obs_names"] = names
    obs = obs.reset_index(drop=True)
    X = _counts_of(adata, layer, modality)
    if "feature_types" in adata.var.columns:
        counts: dict = {}
        var: dict = {}
        ft = adata.var["feature_types"].astype(str)
        for raw_type, mod in _FEATURE_TYPE_TO_MODALITY.items():
            mask = (ft == raw_type).to_numpy()
            if mask.any():
                counts[mod] = X[:, mask]
                var[mod] = list(adata.var_names[mask])
        if not counts:
            raise ValueError("feature_types present but no recognized modality found.")
    else:
        counts = {modality: X}
        var = {modality: list(adata.var_names)}
    return read_matrix(counts=counts, obs=obs, var=var, batch=batch)


def read_h5ad(path, *, batch: str | None = None, modality: str = "rna",
              batch_key: str = "batch", layer=None, join: str = "exact") -> Dataset:
    """Read AnnData .h5ad file(s); the arguments are those of from_anndata.

    path: one file, or one file per modality as a dict, e.g.
          {"rna": "rna.h5ad", "adt": "adt.h5ad"}. With a dict, the cells of the
          files are checked against each other from obs alone before any count
          matrix is loaded.
    """
    if isinstance(path, Mapping):
        obs_by_mod = {}
        for mod, p in path.items():
            backed = _ad.read_h5ad(str(p), backed="r")     # obs only; X stays on disk
            try:
                obs_by_mod[mod] = backed.obs.set_axis(backed.obs_names)
            finally:
                backed.file.close()
        _plan_cells(obs_by_mod, batch=batch, batch_key=batch_key, join=join)
        adatas = {mod: _ad.read_h5ad(str(p)) for mod, p in path.items()}
        return from_anndata(adatas, batch=batch, modality=modality, batch_key=batch_key,
                            layer=layer, join=join)
    return from_anndata(_ad.read_h5ad(path), batch=batch, modality=modality,
                        batch_key=batch_key, layer=layer, join=join)


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
    batch: str | None = None,
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
