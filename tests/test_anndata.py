import anndata as ad
import numpy as np
import pandas as pd

import m3


def test_from_anndata_single_modality():
    X = np.array([[1, 0, 2], [0, 3, 1]], dtype=np.float32)
    a = ad.AnnData(X=X, obs=pd.DataFrame({"disease": ["hc", "covid"]}, index=["c1", "c2"]),
                   var=pd.DataFrame(index=["g1", "g2", "g3"]))
    d = m3.from_anndata(a, batch="A")
    assert d.n_cells == 2
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]
    assert "disease" in d.obs.columns


def test_from_anndata_split_by_feature_types():
    X = np.array([[1, 0, 2, 5], [0, 3, 1, 6]], dtype=np.float32)
    var = pd.DataFrame(
        {"feature_types": ["Gene Expression", "Gene Expression", "Gene Expression", "Antibody Capture"]},
        index=["g1", "g2", "g3", "p1"],
    )
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=["c1", "c2"]), var=var)
    d = m3.from_anndata(a, batch="A")
    assert set(d.modality_names) == {"rna", "adt"}
    assert d.modalities["adt"].shape == (2, 1)
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]


def test_read_h5ad_roundtrip(tmp_path):
    # Each cell is tagged (row i of X starts at 3*i; obs carries per-cell labels)
    # so that dropping obs, reordering it, or desyncing it from the counts is caught.
    n = 4
    X = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    obs = pd.DataFrame(
        {"disease": ["hc", "covid", "hc", "covid"],
         "donor": ["d0", "d1", "d2", "d3"],
         "n_genes": [10, 20, 30, 40]},
        index=[f"cell{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g1", "g2", "g3"]))
    p = tmp_path / "c.h5ad"
    a.write_h5ad(p)

    d = m3.read_h5ad(str(p), batch="A")

    assert d.n_cells == n
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]
    # every user obs column survives the file round-trip, values and row order intact
    assert list(d.obs["disease"]) == ["hc", "covid", "hc", "covid"]
    assert list(d.obs["donor"]) == ["d0", "d1", "d2", "d3"]
    assert d.obs["n_genes"].tolist() == [10, 20, 30, 40]
    # obs rows still align with their count rows (row i of X starts at 3*i)
    counts = d.modalities["rna"].toarray()
    assert np.allclose(counts[:, 0], np.arange(n, dtype=np.float32) * 3)
