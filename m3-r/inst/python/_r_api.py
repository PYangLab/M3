"""R bridge for the m3 package.

The R package ``m3`` drives the bundled Python package ``m3`` (engine + public
object API, under ``inst/python/m3``) via :pkg:`reticulate` inside a
:pkg:`basilisk` process.

A trained :class:`m3.M3` holds live torch state that cannot be serialised back
into R, so we keep the live objects in a **process-local registry** keyed by
small integer handles. R holds the handle plus a little converted metadata, and
every readout is a flat function here that takes the handle and returns
R-friendly values (numpy arrays -> R matrices, pandas DataFrames -> data.frame,
dicts -> named lists).

NOTHING in this file changes the model numerics. Determinism is added only via
:func:`m3._engine.util.setup_seed`, which the upstream engine leaves uncalled
for the Stage-1 integration VAE — seeding it makes a run reproducible (and so
makes Python and R bit-identical in the same env + device).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import m3
from m3._engine.util import setup_seed as _setup_seed


@contextlib.contextmanager
def _in_tmpdir():
    """Run inside a throwaway dir so the engine's cwd-relative artifacts (the
    EarlyStopping ``best_model.pth`` in train.py) never land in the user's cwd.
    The trained weights are already in memory on the M3 object afterwards, so the
    dir is disposable."""
    prev = os.getcwd()
    d = tempfile.mkdtemp(prefix="m3_run_")
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(prev)
        shutil.rmtree(d, ignore_errors=True)

# handle -> object (Dataset / M3 / Attribution)
_REG: dict[int, object] = {}
_NEXT = [0]


def _put(obj) -> int:
    h = _NEXT[0]
    _NEXT[0] += 1
    _REG[h] = obj
    return h


def _get(h):
    return _REG[int(h)]


def drop(h) -> None:
    _REG.pop(int(h), None)


def registry_size() -> int:
    return len(_REG)


def version() -> str:
    return m3.__version__


def seed(s) -> None:
    """Seed torch / numpy / random globally (covers the Stage-1 VAE init+shuffle)."""
    _setup_seed(int(s))


# --------------------------------------------------------------- datasets
def dataset_from_parts(counts: dict, obs, var: dict, batch: str) -> int:
    """Build a single-batch Dataset.

    counts: {mod -> 2D float array [cells, feats]} (RAW counts, cells x features)
    obs:    a pandas DataFrame (reticulate converts the R data.frame) or a dict
            of columns; m3 checks NaN only on the role columns later.
    var:    {mod -> [feature names]}
    """
    counts = {k: np.asarray(v, dtype=np.float32) for k, v in counts.items()}
    var = {k: [str(x) for x in v] for k, v in var.items()}
    obs = pd.DataFrame(obs).reset_index(drop=True)
    return _put(m3.read_matrix(counts=counts, obs=obs, var=var, batch=str(batch)))


def dataset_read_h5(rna=None, adt=None, atac=None, metadata=None, batch="batch0") -> int:
    def n(x):
        return None if (x is None or x == "") else str(x)
    ds = m3.read_h5(rna=n(rna), adt=n(adt), atac=n(atac),
                    metadata=str(metadata), batch=str(batch))
    return _put(ds)


def dataset_read_h5ad(path, batch="batch0", modality="rna") -> int:
    return _put(m3.read_h5ad(str(path), batch=str(batch), modality=str(modality)))


def dataset_concat(handles) -> int:
    ds = m3.concat([_get(h) for h in handles])
    return _put(ds)


def dataset_demo() -> int:
    return _put(m3.datasets.liu_demo())


def dataset_info(h) -> dict:
    ds = _get(h)
    return {
        "n_cells": int(ds.n_cells),
        "batches": [str(c) for c in ds.batches],
        "modalities": [str(m) for m in ds.modality_names],
        "n_features": {str(m): int(ds.modalities[m].shape[1]) for m in ds.modality_names},
        "obs_columns": [str(c) for c in ds.obs.columns],
    }


def dataset_obs(h):
    """Return obs as a pandas DataFrame (reticulate -> R data.frame)."""
    return _get(h).obs.reset_index(drop=True)


def dataset_var(h, modality):
    return [str(x) for x in list(_get(h).var[str(modality)])]


def dataset_unique(h, column):
    return [str(x) for x in pd.unique(_get(h).obs[str(column)].astype(str))]


def dataset_obs_matrix(h, modality):
    """Dense [cells, feats] of a modality (for SCE round-trips / sample-level viz)."""
    m = _get(h).modalities[str(modality)]
    return np.asarray(m.todense() if hasattr(m, "todense") else m, dtype=np.float32)


# ------------------------------------------------------------------ model
def model_train(dataset_handle, *, condition_keys, target_condition=None,
                donor_key=None, celltype_key=None,
                batch_key="batch", held_out=None, held_out_samples=None,
                hvg=None, embedding_dim=30,
                max_epochs=300, lr=1e-5, batch_size=256, early_stop_patience=300,
                min_delta=0.0, val_percentage=0.1, weight_batch_ae=1.0,
                weight_modality=None,
                balance_batches=True, donor_prediction=None, donor_predictor=None,
                seed=None) -> dict:
    """Construct m3.M3(...) and train; store the live model, return handle+meta."""
    if seed is not None:
        _setup_seed(int(seed))
    ds = _get(dataset_handle)

    def _clean(x):
        return None if x is None or (isinstance(x, str) and x == "") else x

    held = list(held_out) if held_out else None
    model = m3.M3(
        ds,
        condition_keys=list(condition_keys),
        target_condition=_clean(target_condition),
        donor_key=_clean(donor_key),
        celltype_key=_clean(celltype_key),
        batch_key=str(batch_key),
        held_out=held,
        held_out_samples=(list(held_out_samples) if held_out_samples else None),
        hvg=(dict(hvg) if hvg else None),
        embedding_dim=int(embedding_dim),
    )
    # re-seed AFTER construction so the train RNG state is identical regardless of
    # any RNG draws m3.M3.__init__ might make (it currently makes none, but this
    # keeps parity robust to future changes).
    if seed is not None:
        _setup_seed(int(seed))
    with _in_tmpdir():
        model.train(
            max_epochs=int(max_epochs), lr=float(lr), batch_size=int(batch_size),
            early_stop_patience=int(early_stop_patience), min_delta=float(min_delta),
            val_percentage=float(val_percentage), weight_batch_ae=float(weight_batch_ae),
            weight_modality=(dict(weight_modality)
                             if isinstance(weight_modality, dict) else weight_modality),
            balance_batches=bool(balance_batches),
            donor_prediction=(None if donor_prediction is None else bool(donor_prediction)),
            donor_predictor=(dict(donor_predictor) if donor_predictor else None),
        )
    h = _put(model)
    return {
        "handle": int(h),
        "capabilities": {k: bool(v) for k, v in model.capabilities.items()},
        "modalities": [str(m) for m in ds.modality_names],
        "embedding_dim": int(embedding_dim),
        "condition_keys": [str(c) for c in model.condition_keys],
        "target_condition": str(model.target_condition),
        "celltype_key": (None if model.celltype_key is None else str(model.celltype_key)),
        "donor_key": (None if model.donor_key is None else str(model.donor_key)),
        "batch_key": (None if model.batch_key is None else str(model.batch_key)),
        "held_out": [str(c) for c in model.held_out],
        "reference_vocab": {k: [str(x) for x in v]
                            for k, v in model.contract["reference_vocab"].items()},
    }


def model_capabilities(h) -> dict:
    return {k: bool(v) for k, v in _get(h).capabilities.items()}


def embedding(h, part="bio"):
    return np.asarray(_get(h).embedding(part=str(part)), dtype=np.float64)


def reconstruct(h, remove_batch=True) -> dict:
    out = _get(h).reconstruct(remove_batch=bool(remove_batch))
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in out.items()}


def cell_metadata(h):
    return _get(h).cell_metadata.reset_index(drop=True)


def predict_donors(h, include_reference=False):
    return _get(h).predict_donors(include_reference=bool(include_reference)).reset_index(drop=True)


def donor_embedding(h):
    df = _get(h).donor_embedding()
    return df.reset_index()  # bring the donor index out as a column


# ------------------------------------------------------------ attribution
def attribute(h, *, reference_labels, target_class=None, n_steps=50) -> int:
    attr = _get(h).attribute(
        reference_labels=list(reference_labels),
        target_class=(None if target_class is None else int(target_class)),
        n_steps=int(n_steps),
    )
    return _put(attr)


def attr_tables(h) -> dict:
    a = _get(h)
    out = {
        "target_label": (None if a.target_label is None else str(a.target_label)),
        "genes": a.genes.reset_index(drop=True),
        "celltypes": a.celltypes.reset_index(drop=True),
        "feature_names": [str(x) for x in a._feature_names],
        "celltype_names": [str(x) for x in list(a._res["celltype_names"])],
    }
    out["donors"] = (None if a.donors is None else a.donors.reset_index(drop=True))
    return out


def attr_matrix(h):
    return np.asarray(_get(h).attribution, dtype=np.float64)


def attr_gene_celltype_matrix(h):
    return np.asarray(_get(h).gene_celltype_matrix, dtype=np.float64)


def attr_top_genes(h, n=100, min_cells_per_condition=200,
                   exclude_regex="__default__", modality=None):
    from m3._model import _HOUSEKEEPING_RE
    rx = _HOUSEKEEPING_RE if exclude_regex == "__default__" else (
        None if exclude_regex in (None, "") else str(exclude_regex))
    df = _get(h).top_genes(
        n=int(n), min_cells_per_condition=int(min_cells_per_condition),
        exclude_regex=rx,
        modality=(None if modality in (None, "") else str(modality)),
    )
    return df.reset_index(drop=True)


def attr_top_celltypes(h, min_cells_per_condition=200):
    return _get(h).top_celltypes(
        min_cells_per_condition=int(min_cells_per_condition)).reset_index(drop=True)


# ------------------------------------------------------------- generation
def generate(h, tau=0.8, seed=42) -> dict:
    out = _get(h).generate(tau=float(tau), seed=int(seed))
    return {str(k): np.asarray(v, dtype=np.float64) for k, v in out.items()}


def augment(h, *, conditions, n_donors, tau=0.8, batch=None, seed=42) -> dict:
    res = _get(h).augment(
        conditions=list(conditions), n_donors=[int(x) for x in n_donors],
        tau=float(tau), batch=(None if batch in (None, "") else str(batch)),
        seed=int(seed),
    )
    expr = {str(k): np.asarray(v, dtype=np.float64) for k, v in res["expression"].items()}
    return {"expression": expr, "obs": res["obs"].reset_index(drop=True)}


# ----------------------------------------------------- tutorial projections
# These compute the SAME projections the Python tutorials use (scanpy / umap-learn
# / sklearn-PCA), so an R tutorial that plots their output gets byte-identical
# coordinates -- no uwot-vs-umap-learn rotation. They touch no model state.
def umap_scanpy(X, n_neighbors=15):
    """sc.pp.neighbors(use_rep='X') + sc.tl.umap, as in tutorial 1."""
    import anndata as ad
    import scanpy as sc
    X = np.asarray(X, dtype=np.float32)
    a = ad.AnnData(X=X)
    sc.pp.neighbors(a, use_rep="X", n_neighbors=int(n_neighbors))
    sc.tl.umap(a)
    return np.asarray(a.obsm["X_umap"], dtype=np.float64)


def umap_learn(X, n_neighbors=15, min_dist=0.1, spread=1.0,
               metric="euclidean", random_state=0):
    """umap.UMAP(...).fit_transform, as in tutorials 2 and 4."""
    import umap
    X = np.asarray(X, dtype=np.float32)
    nn = int(min(int(n_neighbors), X.shape[0] - 1))
    reducer = umap.UMAP(n_neighbors=max(2, nn), min_dist=float(min_dist),
                        spread=float(spread), metric=str(metric),
                        random_state=int(random_state))
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)


def pca2(X):
    """sklearn PCA to 2 components (tutorial 4's small-sample fallback)."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=np.float32)
    return np.asarray(PCA(n_components=2).fit_transform(X), dtype=np.float64)
