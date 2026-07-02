import h5py
import numpy as np
import pandas as pd

import m3


def _write_paper_h5(path, mat_genes_by_cells, features):
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=mat_genes_by_cells.astype(np.float32))
        g.create_dataset("features", data=np.array(features, dtype="S"))


def test_read_h5_rna_plus_metadata(tmp_path):
    # 3 genes x 2 cells (paper layout: genes x cells)
    mat = np.array([[1, 0], [0, 3], [2, 1]], dtype=np.float32)
    rna = tmp_path / "rna.h5"
    _write_paper_h5(rna, mat, ["g1", "g2", "g3"])
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"barcode": ["c1", "c2"], "disease": ["hc", "covid"]}).to_csv(meta, index=False)

    d = m3.read_h5(rna=str(rna), metadata=str(meta), batch="A")
    assert d.n_cells == 2
    assert d.modalities["rna"].shape == (2, 3)  # transposed to cells x genes
    assert list(d.var["rna"]) == ["g1", "g2", "g3"]
    assert list(d.obs["disease"]) == ["hc", "covid"]
