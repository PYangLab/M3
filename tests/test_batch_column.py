"""Where the batch comes from: readers never invent one; the model reads it from
the obs column named by batch_key and refuses to train without it."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # the M3 constructor is torch-backed
import m3


def _read(obs, batch=None, adt=True, **kw):
    n = len(obs)
    counts = {"rna": np.arange(n * 3, dtype=np.float32).reshape(n, 3)}
    var = {"rna": ["g1", "g2", "g3"]}
    if adt:
        counts["adt"] = np.ones((n, 2), dtype=np.float32)
        var["adt"] = ["p1", "p2"]
    if batch is not None:
        kw["batch"] = batch
    return m3.read_matrix(counts=counts, obs=obs, var=var, **kw)


def _obs(n=6, **cols):
    base = {"cond": (["hc", "dis"] * n)[:n], "cty": (["T", "B", "B"] * n)[:n]}
    return pd.DataFrame({**base, **cols})


# ------------------------------------------------------------------ readers
def test_reader_does_not_invent_a_batch():
    d = _read(_obs(run=["r1", "r1", "r1", "r2", "r2", "r2"]))
    assert "batch" not in d.obs.columns
    assert d.batches == []
    assert "batches=not set" in repr(d)


def test_from_anndata_does_not_invent_a_batch():
    import anndata as ad
    obs = _obs(4).set_axis([f"c{i}" for i in range(4)])
    a = ad.AnnData(X=np.ones((4, 3), dtype=np.float32), obs=obs,
                   var=pd.DataFrame(index=["g1", "g2", "g3"]))
    assert "batch" not in m3.from_anndata(a).obs.columns


def test_reader_writes_the_label_it_is_given():
    d = _read(_obs(), batch="A")
    assert d.batches == ["A"]


def test_reader_keeps_an_existing_batch_column():
    d = _read(_obs(batch=["A", "A", "A", "B", "B", "B"]))
    assert d.batches == ["A", "B"]


def test_reader_rejects_a_label_that_contradicts_the_batch_column():
    with pytest.raises(ValueError, match="already has a 'batch' column"):
        _read(_obs(batch=["A", "A", "A", "B", "B", "B"]), batch="C")


def test_reader_accepts_a_label_equal_to_the_batch_column():
    d = _read(_obs(batch=["A"] * 6), batch="A")
    assert d.batches == ["A"]


# ------------------------------------------------------------------- concat
def test_concat_rejects_a_piece_without_a_batch():
    with pytest.raises(ValueError, match="batch="):
        m3.concat([_read(_obs(), batch="A"), _read(_obs())])


# -------------------------------------------------------------------- model
def test_model_refuses_to_train_without_a_batch_column():
    d = _read(_obs(run=["r1", "r1", "r1", "r2", "r2", "r2"]))
    with pytest.raises(ValueError, match="batch_key="):
        m3.M3(d, condition_keys="cond", celltype_key="cty")


def test_model_takes_the_batches_from_batch_key():
    d = _read(_obs(run=["r1", "r1", "r1", "r2", "r2", "r2"]))
    model = m3.M3(d, condition_keys="cond", celltype_key="cty", batch_key="run")
    assert model.contract["batches"] == ["r1", "r2"]
    assert "batch" not in d.obs.columns                    # caller's Dataset untouched


def test_model_rejects_a_batch_whose_cells_differ_in_modalities():
    # A has RNA + ADT, B has RNA only; 'site' puts cells of both into each batch.
    mosaic = m3.concat([_read(_obs(site=["s1", "s2"] * 3), batch="A"),
                        _read(_obs(site=["s1", "s2"] * 3), batch="B", adt=False)])
    with pytest.raises(ValueError, match="adt for 3 of 6 cells"):
        m3.M3(mosaic, condition_keys="cond", celltype_key="cty", batch_key="site")


def test_model_accepts_a_mosaic_whose_batches_follow_the_pieces():
    mosaic = m3.concat([_read(_obs(), batch="A"), _read(_obs(), batch="B", adt=False)])
    model = m3.M3(mosaic, condition_keys="cond", celltype_key="cty")
    assert model.contract["batches"] == ["A", "B"]
