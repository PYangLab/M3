import numpy as np
import pandas as pd

import m3


def test_read_matrix_single_modality():
    counts = {"rna": np.array([[1, 0, 2], [0, 3, 1]], dtype=np.float32)}
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "disease": ["hc", "covid"]})
    var = {"rna": ["g1", "g2", "g3"]}
    d = m3.read_matrix(counts=counts, obs=obs, var=var, batch="A")
    assert d.n_cells == 2
    assert d.batches == ["A"]
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]
    assert d.present["rna"].all()
    assert "disease" in d.obs.columns


def test_read_matrix_two_modalities():
    counts = {
        "rna": np.array([[1, 0], [0, 2]], dtype=np.float32),
        "adt": np.array([[5], [6]], dtype=np.float32),
    }
    obs = pd.DataFrame({"barcode": ["c1", "c2"]})
    var = {"rna": ["g1", "g2"], "adt": ["p1"]}
    d = m3.read_matrix(counts=counts, obs=obs, var=var, batch="A")
    assert set(d.modality_names) == {"rna", "adt"}
    assert d.modalities["adt"].shape == (2, 1)
