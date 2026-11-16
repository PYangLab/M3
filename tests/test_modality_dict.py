"""One AnnData per modality: cells are matched by name, and within a batch every
modality must have the same cells or none of them. Mismatches fail when reading,
with the batch, the counts and the fix."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import m3
import m3.io as m3io


def _modality(batches, n_feat, prefix, drop=(), value=None):
    """Cells 'c{i}-{batch}'; row of cell i starts at 10*i so pairing is checkable."""
    obs, rows = [], []
    for b, n in batches.items():
        for i in range(n):
            if (b, i) in drop:
                continue
            obs.append({"name": f"c{i}-{b}", "batch": b, "cond": "hc" if i % 2 else "dis"})
            rows.append(10 * i + np.arange(n_feat) if value is None else np.full(n_feat, value))
    obs = pd.DataFrame(obs).set_index("name")
    obs.index.name = None
    return ad.AnnData(np.asarray(rows, dtype=np.float32), obs=obs,
                      var=pd.DataFrame(index=[f"{prefix}{j}" for j in range(n_feat)]))


def _pair(drop=()):
    rna = _modality({"B1": 6, "B2": 6, "B3": 6}, 4, "g")
    adt = _modality({"B1": 6, "B2": 6}, 2, "p", drop=drop)
    return rna, adt


def test_mosaic_from_one_file_per_modality(capsys):
    rna, adt = _pair()
    d = m3.from_anndata({"rna": rna, "adt": adt})
    assert d.n_cells == 18 and d.batches == ["B1", "B2", "B3"]
    assert d.present["rna"].all()
    assert d.present["adt"].tolist() == [True] * 12 + [False] * 6
    # cell names are kept verbatim, and each row holds that cell's counts
    assert d.obs["obs_names"].tolist() == list(rna.obs_names)
    i = d.obs["obs_names"].tolist().index("c4-B2")
    assert d.modalities["adt"][i].toarray().tolist() == [[40, 41]]
    assert d.modalities["rna"][i].toarray().tolist() == [[40, 41, 42, 43]]
    assert d.modalities["adt"][d.n_cells - 1].nnz == 0
    out = capsys.readouterr().out
    assert "cells per batch ('batch') and modality" in out and "B3" in out
    model = m3.M3(d, condition_keys="cond")
    assert model.contract["batches"] == ["B1", "B2", "B3"]


def test_cells_are_matched_by_name_not_position():
    rna, adt = _pair()
    shuffled = adt[np.random.default_rng(0).permutation(adt.n_obs)].copy()
    a = m3.from_anndata({"rna": rna, "adt": adt})
    b = m3.from_anndata({"rna": rna, "adt": shuffled})
    assert (a.modalities["adt"] != b.modalities["adt"]).nnz == 0


def test_read_h5ad_takes_one_file_per_modality(tmp_path):
    rna, adt = _pair()
    rna.write_h5ad(tmp_path / "rna.h5ad")
    adt.write_h5ad(tmp_path / "adt.h5ad")
    d = m3.read_h5ad({"rna": str(tmp_path / "rna.h5ad"), "adt": tmp_path / "adt.h5ad"})
    ref = m3.from_anndata({"rna": rna, "adt": adt})
    assert (d.modalities["adt"] != ref.modalities["adt"]).nnz == 0
    assert d.obs["obs_names"].tolist() == ref.obs["obs_names"].tolist()


def test_read_h5ad_checks_the_cells_before_loading_counts(tmp_path, monkeypatch):
    rna, adt = _pair(drop={("B1", 0)})
    rna.write_h5ad(tmp_path / "rna.h5ad")
    adt.write_h5ad(tmp_path / "adt.h5ad")
    full_reads = []
    real = ad.read_h5ad
    def spy(path, *a, **kw):
        if kw.get("backed") is None:
            full_reads.append(path)
        return real(path, *a, **kw)
    monkeypatch.setattr(m3io._ad, "read_h5ad", spy)
    with pytest.raises(ValueError, match="do not have the same cells"):
        m3.read_h5ad({"rna": str(tmp_path / "rna.h5ad"), "adt": str(tmp_path / "adt.h5ad")})
    assert full_reads == []


def test_cells_missing_from_part_of_a_batch_are_an_error():
    rna, adt = _pair(drop={("B1", 0), ("B1", 3)})
    with pytest.raises(ValueError, match=r"batch 'B1'.*rna 6, adt 4; 4 in all.*join='inner'"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_join_inner_keeps_the_cells_every_modality_has(capsys):
    rna, adt = _pair(drop={("B1", 0), ("B1", 3)})
    d = m3.from_anndata({"rna": rna, "adt": adt}, join="inner")
    assert d.n_cells == 16
    assert "c0-B1" not in d.obs["obs_names"].tolist()
    assert "dropped in batch 'B1': 2 cells only in rna" in capsys.readouterr().out
    m3.M3(d, condition_keys="cond")          # the within-batch invariant holds


def test_names_that_do_not_match_are_reported_with_examples():
    rna, adt = _pair()
    adt.obs_names = [n.replace("-", ".") for n in adt.obs_names]
    with pytest.raises(ValueError, match=r"share none of their cell names.*'c0\.B1'"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_modalities_must_agree_on_the_batch_of_a_cell():
    rna, adt = _pair()
    adt.obs["batch"] = adt.obs["batch"].replace({"B2": "B1"})
    with pytest.raises(ValueError, match="different batch in adt"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_every_modality_needs_the_batch_column():
    rna, adt = _pair()
    adt.obs = adt.obs.drop(columns="batch")
    with pytest.raises(ValueError, match=r"adt: obs has no 'batch' column.*index_unique"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_batch_key_names_another_column():
    rna, adt = _pair()
    for a in (rna, adt):
        a.obs = a.obs.rename(columns={"batch": "run"})
    d = m3.from_anndata({"rna": rna, "adt": adt}, batch_key="run")
    assert "batch" not in d.obs.columns
    assert m3.M3(d, condition_keys="cond", batch_key="run").contract["batches"] == ["B1", "B2", "B3"]


def test_batch_label_for_a_single_batch():
    rna = _modality({"B1": 4}, 3, "g")
    adt = _modality({"B1": 4}, 2, "p")
    for a in (rna, adt):
        a.obs = a.obs.drop(columns="batch")
    assert m3.from_anndata({"rna": rna, "adt": adt}, batch="A").batches == ["A"]
    with pytest.raises(ValueError, match="Use one or the other"):
        m3.from_anndata({"rna": rna, "adt": adt}, batch="A", batch_key="run")


def test_duplicate_cell_names_name_the_fix():
    rna, adt = _pair()
    rna.obs_names = [n.split("-")[0] for n in rna.obs_names]     # barcodes repeat across batches
    with pytest.raises(ValueError, match=r"rna: 6 cell names occur more than once.*index_unique"
                                         r".*obs_names_make_unique"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_a_modality_that_is_all_zero_for_a_batch_is_refused():
    rna = _modality({"B1": 6, "B2": 6}, 4, "g")
    adt = ad.concat([_modality({"B1": 6}, 2, "p"), _modality({"B2": 6}, 2, "p", value=0)])
    with pytest.raises(ValueError, match="adt is all zero for the 6 cells of batch 'B2'"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_normalized_counts_are_refused_and_layer_reads_the_counts():
    rna, adt = _pair()
    rna.layers["counts"] = rna.X.copy()
    rna.X = np.log1p(rna.X)
    with pytest.raises(ValueError, match=r"rna: \.X has non-integer values.*layer='counts'"):
        m3.from_anndata({"rna": rna, "adt": adt})
    d = m3.from_anndata({"rna": rna, "adt": adt}, layer={"rna": "counts"})
    assert d.modalities["rna"].max() == rna.layers["counts"].max()
    with pytest.raises(ValueError, match="no layer 'counts'"):
        m3.from_anndata({"rna": rna, "adt": adt}, layer="counts")      # adt has none


def test_shared_obs_columns_must_agree():
    rna, adt = _pair()
    adt.obs["cond"] = "other"
    with pytest.raises(ValueError, match=r"obs column 'cond' differs between rna and adt.*'c0-B1'"):
        m3.from_anndata({"rna": rna, "adt": adt})


def test_obs_columns_of_every_modality_are_kept():
    rna, adt = _pair()
    adt.obs["n_adt"] = 7
    rna.obs["cond"] = rna.obs["cond"].astype("category")         # dtypes may differ
    d = m3.from_anndata({"rna": rna, "adt": adt})
    assert d.obs["n_adt"].tolist()[:12] == [7] * 12
    assert d.obs["n_adt"].iloc[12:].isna().all()


def test_arguments_that_do_not_fit_the_input_are_refused():
    rna, adt = _pair()
    with pytest.raises(ValueError, match="dict keys name the modalities"):
        m3.from_anndata({"rna": rna, "adt": adt}, modality="adt")
    with pytest.raises(ValueError, match="one AnnData per modality"):
        m3.from_anndata(rna, join="inner")
    with pytest.raises(ValueError, match="join must be"):
        m3.from_anndata({"rna": rna, "adt": adt}, join="outer")
    with pytest.raises(ValueError, match="not passed"):
        m3.from_anndata({"rna": rna, "adt": adt}, layer={"atac": "counts"})


def test_single_anndata_reads_a_layer():
    rna = _modality({"B1": 4}, 3, "g")
    rna.layers["counts"] = rna.X.copy()
    rna.X = rna.X / 2.5
    d = m3.from_anndata(rna, layer="counts")
    assert d.modalities["rna"].toarray().tolist() == rna.layers["counts"].tolist()
