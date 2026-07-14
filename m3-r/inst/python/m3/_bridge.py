"""Bridge: marshal a :class:`m3.Dataset` to the engine's file-based inputs.

The engine (``m3._engine.run_M3``) consumes per-modality lists of
paper-format HDF5 files (one element per batch, ``None`` where a batch lacks a
modality) plus a per-batch metadata CSV list. This module writes those temp
files from an in-memory ``Dataset`` so the proven engine path runs unchanged.

Paper HDF5 layout (what ``m3._engine.util.read_h5_data`` expects):
``matrix/data`` is genes x cells (transpose of our cells x genes), and
``matrix/features`` holds the feature names.
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np

_MODALITY_ORDER = ("rna", "adt", "atac")  # maps to engine modality1/2/3_path


def _write_paper_h5(path: str, counts_cells_by_features, feature_names) -> None:
    dense = np.asarray(counts_cells_by_features.todense()
                       if hasattr(counts_cells_by_features, "todense")
                       else counts_cells_by_features, dtype=np.float32)
    # Store cells x features (the Liu/Stephenson layout). The engine's
    # read_h5_data + load_data_from_list double-transpose nets back to cells x features.
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=dense)
        g.create_dataset("features", data=np.array([str(x) for x in feature_names], dtype="S"))


def marshal(dataset, tmpdir: str | None = None) -> dict:
    """Write a Dataset to temp paper-format h5 + metadata CSVs.

    Returns a dict with:
      modality_paths: {"rna": [path|None per batch], "adt": [...], "atac": [...]}
      metadata_paths: [csv path per batch]
      batches: ordered batch labels (row order of the engine's concatenation)
      tmpdir: the directory holding the temp files (caller may clean up)
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="m3_marshal_")
    os.makedirs(tmpdir, exist_ok=True)

    obs = dataset.obs
    batches = list(dataset.batches)
    batch_rows = {c: (obs["batch"].to_numpy() == c) for c in batches}

    modality_paths: dict[str, list] = {m: [] for m in _MODALITY_ORDER}
    metadata_paths: list[str] = []

    for ci, c in enumerate(batches):
        rows = batch_rows[c]
        # metadata CSV for this batch (engine reads with pandas; needs the role columns)
        meta_path = os.path.join(tmpdir, f"metadata_{ci}.csv")
        obs.loc[rows].reset_index(drop=True).to_csv(meta_path, index=False)
        metadata_paths.append(meta_path)

        for m in _MODALITY_ORDER:
            if m in dataset.modalities and bool(dataset.present[m][rows].any()):
                mat = dataset.modalities[m][rows]
                path = os.path.join(tmpdir, f"{m}_{ci}.h5")
                _write_paper_h5(path, mat, dataset.var[m])
                modality_paths[m].append(path)
            else:
                modality_paths[m].append(None)

    return {
        "modality_paths": modality_paths,
        "metadata_paths": metadata_paths,
        "batches": batches,
        "tmpdir": tmpdir,
    }
