"""Lock in the per-celltype-balanced gene ranking recipe in Attribution.top_genes."""
import numpy as np
import pandas as pd

from m3._model import Attribution


def _make():
    # 4 cell types (A,B,C,D), 6 genes (4 RNA + 2 ADT). C and D have <200 cells in
    # one condition and should be excluded. Gene 'MT-foo' is housekeeping.
    n_g = 6
    feature_names = ["gA", "gB", "MT-foo", "RPL3", "ADT1", "ADT2"]
    modality_of = ["rna"] * 4 + ["adt"] * 2
    ct_names = ["A", "B", "C", "D"]
    gcm = np.array([
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],  # A: high all over
        [9.0, 7.0, 0.0, 0.0, 0.0, 0.0],  # B: high on first two
        [0.0, 0.0, 99.0, 99.0, 99.0, 99.0],  # C: should be EXCLUDED
        [0.0, 0.0, 99.0, 99.0, 99.0, 99.0],  # D: should be EXCLUDED
    ], dtype=np.float32)

    # Synthesize a cell-metadata frame: 300 HC + 300 Severe per A,B; 50 each per C,D
    rows = []
    for ct, hc, sev in [("A", 300, 300), ("B", 300, 300), ("C", 50, 50), ("D", 50, 50)]:
        rows += [(ct, "HC")] * hc + [(ct, "Severe")] * sev
    meta = pd.DataFrame(rows, columns=["mergedcelltype", "cond_group"])

    res = {
        "gene_importance": np.zeros(n_g, dtype=np.float32),       # unused by top_genes
        "celltype_importance": np.zeros(len(ct_names), dtype=np.float32),
        "celltype_names": ct_names,
        "cell_importance": np.zeros(len(meta), dtype=np.float32),
        "attribution": np.zeros((len(meta), n_g), dtype=np.float32),
        "gene_celltype_matrix": gcm,
    }
    return Attribution(res, feature_names=feature_names, target_label="Severe",
                       modality_of=modality_of, cell_metadata=meta,
                       celltype_key="mergedcelltype", target_condition="cond_group",
                       reference_labels=["HC"])


def test_top_genes_filters_balances_excludes():
    attr = _make()
    out = attr.top_genes(n=4, min_cells_per_condition=200,
                         exclude_regex=r"^MT-|^RPL", modality="rna")
    # Cell types kept: A and B (C, D have only 50 cells per condition)
    assert int(out["n_celltypes_used"].iloc[0]) == 2
    # Ranked features: only gA and gB survive (MT-foo, RPL3 excluded; ADT masked)
    assert list(out["feature"]) == ["gA", "gB"]
    # gA score = mean(|9,9|) = 9 ; gB = mean(|8,7|) = 7.5
    assert out.loc[0, "score"] == 9.0
    assert abs(out.loc[1, "score"] - 7.5) < 1e-6


def test_top_genes_no_filter_keeps_all_celltypes():
    attr = _make()
    out = attr.top_genes(n=10, min_cells_per_condition=0,
                         exclude_regex=None, modality=None)
    assert int(out["n_celltypes_used"].iloc[0]) == 4
    assert set(out["feature"]) == {"gA", "gB", "MT-foo", "RPL3", "ADT1", "ADT2"}


def test_top_celltypes_keeps_only_eligible():
    attr = _make()
    out = attr.top_celltypes(min_cells_per_condition=200)
    # A and B (300/300 each) survive; C and D (50/50) are dropped
    assert list(out["celltype"]) == ["A", "B"] or list(out["celltype"]) == ["B", "A"]
    assert len(out) == 2


def test_top_celltypes_no_filter_returns_all():
    attr = _make()
    out = attr.top_celltypes(min_cells_per_condition=0)
    assert len(out) == 4   # all four cell types kept
