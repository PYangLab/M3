import numpy as np
import pandas as pd
import scipy.sparse as sp

from m3._dataset import Dataset


def _tiny():
    rna = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "batch": ["A", "A"]})
    var = {"rna": pd.Index(["g1", "g2", "g3"])}
    present = {"rna": np.array([True, True])}
    return Dataset(modalities={"rna": rna}, obs=obs, var=var, present=present)


def test_basic_attributes():
    d = _tiny()
    assert d.n_cells == 2
    assert d.modality_names == ["rna"]
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]
    assert d.batches == ["A"]


def test_rejects_modality_var_length_mismatch():
    rna = sp.csr_matrix(np.zeros((2, 3), dtype=np.float32))
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "batch": ["A", "A"]})
    var = {"rna": pd.Index(["g1", "g2"])}  # only 2 names for 3 columns
    present = {"rna": np.array([True, True])}
    try:
        Dataset(modalities={"rna": rna}, obs=obs, var=var, present=present)
    except ValueError as e:
        assert "var" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on var/column mismatch")


def test_rejects_row_count_mismatch():
    rna = sp.csr_matrix(np.zeros((2, 3), dtype=np.float32))
    obs = pd.DataFrame({"barcode": ["c1", "c2", "c3"], "batch": ["A", "A", "A"]})  # 3 rows
    var = {"rna": pd.Index(["g1", "g2", "g3"])}
    present = {"rna": np.array([True, True])}
    try:
        Dataset(modalities={"rna": rna}, obs=obs, var=var, present=present)
    except ValueError as e:
        assert "rows" in str(e).lower() or "n_cells" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on row mismatch")


def test_rejects_duplicate_feature_names():
    rna = sp.csr_matrix(np.zeros((2, 3), dtype=np.float32))
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "batch": ["A", "A"]})
    var = {"rna": pd.Index(["g1", "g2", "g1"])}  # duplicate g1
    present = {"rna": np.array([True, True])}
    try:
        Dataset(modalities={"rna": rna}, obs=obs, var=var, present=present)
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on duplicate feature names")
