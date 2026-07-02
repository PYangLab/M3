import numpy as np
import pandas as pd

import m3


def _d(batch, rna_cols, rna_vals, adt=None, adt_cols=None):
    counts = {"rna": np.array(rna_vals, dtype=np.float32)}
    var = {"rna": rna_cols}
    if adt is not None:
        counts["adt"] = np.array(adt, dtype=np.float32)
        var["adt"] = adt_cols
    obs = pd.DataFrame({"barcode": [f"{batch}{i}" for i in range(len(rna_vals))]})
    return m3.read_matrix(counts=counts, obs=obs, var=var, batch=batch)


def test_concat_intersects_features():
    a = _d("A", ["g1", "g2", "g3"], [[1, 2, 3]])
    b = _d("B", ["g2", "g3", "g4"], [[4, 5, 6]])
    d = m3.concat([a, b])
    assert list(d.var["rna"]) == ["g2", "g3"]      # intersection, stable order
    assert d.n_cells == 2
    assert d.batches == ["A", "B"]
    # row A keeps g2,g3 = 2,3 ; row B keeps g2,g3 = 4,5
    dense = d.modalities["rna"].toarray()
    assert dense.tolist() == [[2.0, 3.0], [4.0, 5.0]]


def test_concat_missing_modality_zero_filled_and_masked():
    a = _d("A", ["g1", "g2"], [[1, 2]], adt=[[7]], adt_cols=["p1"])
    b = _d("B", ["g1", "g2"], [[3, 4]])  # no adt
    d = m3.concat([a, b])
    assert set(d.modality_names) == {"rna", "adt"}
    assert d.modalities["adt"].shape == (2, 1)
    assert d.modalities["adt"].toarray().tolist() == [[7.0], [0.0]]
    assert d.present["adt"].tolist() == [True, False]
    assert d.present["rna"].tolist() == [True, True]


def test_concat_rejects_duplicate_batch_labels():
    a = _d("A", ["g1"], [[1]])
    b = _d("A", ["g1"], [[2]])
    try:
        m3.concat([a, b])
    except ValueError as e:
        assert "batch" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on duplicate batch labels")


def test_concat_rejects_empty_feature_intersection():
    a = _d("A", ["g1", "g2"], [[1, 2]])
    b = _d("B", ["g3", "g4"], [[3, 4]])
    try:
        m3.concat([a, b])
    except ValueError as e:
        assert "intersection" in str(e).lower() or "no shared" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on empty feature intersection")
