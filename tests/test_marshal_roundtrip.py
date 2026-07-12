"""_bridge.marshal: per-batch split, missing-modality handling, and the
cells x features orientation surviving the h5 round-trip."""
import shutil

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("torch")  # importing the m3 package pulls in the torch-backed model
from m3 import _bridge
from m3._dataset import Dataset
from m3.io import _read_paper_h5


def _two_batch_dataset():
    rng = np.random.default_rng(0)
    # batch A carries rna + adt; batch B carries rna only (adt absent).
    obs = pd.DataFrame({"batch": ["A", "A", "A", "B", "B", "B"],
                        "cond": ["hc", "dis", "hc", "dis", "hc", "dis"]})
    rna = sp.csr_matrix(rng.integers(1, 6, (6, 4)).astype(np.float32))
    adt = sp.csr_matrix(rng.integers(1, 6, (6, 2)).astype(np.float32)).tolil()
    adt[3:] = 0                                   # zero + present=False -> B lacks adt
    present = {"rna": np.ones(6, dtype=bool),
              "adt": np.array([True, True, True, False, False, False])}
    return Dataset(
        modalities={"rna": rna, "adt": adt.tocsr()}, obs=obs,
        var={"rna": pd.Index([f"g{i}" for i in range(4)]),
             "adt": pd.Index([f"p{i}" for i in range(2)])},
        present=present,
    )


def test_marshal_structure_and_orientation():
    ds = _two_batch_dataset()
    out = _bridge.marshal(ds)
    try:
        assert sorted(out["batches"]) == ["A", "B"]
        assert len(out["metadata_paths"]) == 2

        # rna present in both batches; adt only in A.
        assert all(p is not None for p in out["modality_paths"]["rna"])
        adt_paths = out["modality_paths"]["adt"]
        assert adt_paths[out["batches"].index("A")] is not None
        assert adt_paths[out["batches"].index("B")] is None

        # round-trip: each written h5 recovers the original per-batch cells x features.
        for bi, b in enumerate(out["batches"]):
            rows = ds.obs["batch"].to_numpy() == b
            mat, feats = _read_paper_h5(out["modality_paths"]["rna"][bi])
            expected = np.asarray(ds.modalities["rna"][rows].todense(), dtype=np.float32)
            assert mat.shape == expected.shape       # cells x features, not transposed
            assert np.allclose(mat, expected)
            assert list(feats) == [f"g{i}" for i in range(4)]

            meta = pd.read_csv(out["metadata_paths"][bi])
            assert len(meta) == int(rows.sum())      # one CSV row per cell in the batch
            assert "cond" in meta.columns            # role columns carried through
    finally:
        shutil.rmtree(out["tmpdir"], ignore_errors=True)
