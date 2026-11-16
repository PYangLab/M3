"""The cell-level model needs a condition and a batch column only: cell-type
labels do not enter its training, so they are neither required nor able to
change (or break) it."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")
import m3
from m3._model import M3CapabilityError


def _cohort(celltypes=None, extra=("donor",)):
    """3 batches x 48 cells, 4 donors per batch (2 hc, 2 dis)."""
    rng = np.random.default_rng(0)
    n_genes, per_batch = 24, 48
    records, counts = [], []
    for batch in ["B1", "B2", "B3"]:
        for j in range(per_batch):
            d = j % 4
            cond = "hc" if d < 2 else "dis"
            rec = {"batch": batch, "cond": cond}
            if "donor" in extra:
                rec["donor"] = f"{batch}_d{d}"
            records.append(rec)
            counts.append(rng.poisson(3 + (2 if cond == "dis" else 0), n_genes))
    obs = pd.DataFrame(records)
    if celltypes is not None:
        obs["cty"] = celltypes(len(obs))
    return m3.read_matrix(counts={"rna": np.asarray(counts, dtype=np.float32)}, obs=obs,
                          var={"rna": [f"g{i}" for i in range(n_genes)]})


def _two_types(n):
    return ["T" if i % 2 == 0 else "Bcell" for i in range(n)]


def _embedding(ds, **init):
    model = m3.M3(ds, condition_keys="cond", **init)
    model.train(max_epochs=2, seed=0, donor_prediction=False)
    return model, model.embedding("bio")


def test_trains_with_a_condition_and_a_batch_column_only():
    ds = _cohort(extra=())
    assert list(ds.obs.columns) == ["batch", "cond"]
    model, emb = _embedding(ds)
    assert emb.shape[0] == ds.n_cells and np.isfinite(emb).all()
    assert model.capabilities == {"embedding": True, "reconstruct": True,
                                  "predict_donors": False}
    assert set(model.reconstruct()) == {"rna"}
    assert model.generate()["rna"].shape == (ds.n_cells, 24)


def test_cell_type_labels_do_not_change_the_embedding():
    _, without = _embedding(_cohort())
    _, with_labels = _embedding(_cohort(_two_types), celltype_key="cty")
    assert np.array_equal(without, with_labels)


def test_held_out_batch_trains_without_cell_types():
    _, without = _embedding(_cohort(), held_out=["B3"])
    _, with_labels = _embedding(_cohort(_two_types), celltype_key="cty", held_out=["B3"])
    assert np.isfinite(without).all()
    assert np.array_equal(without, with_labels)


def test_more_than_fifty_cell_types_train():
    ds = _cohort(lambda n: [f"type{i % 60:02d}" for i in range(n)])
    _, emb = _embedding(ds, celltype_key="cty")
    assert np.isfinite(emb).all()


def test_the_internal_column_stays_internal():
    ds = _cohort()
    before = list(ds.obs.columns)
    model, _ = _embedding(ds)
    assert list(ds.obs.columns) == before
    assert not [c for c in model.cell_metadata.columns if c.startswith("__m3")]


def test_donor_prediction_still_needs_cell_types():
    model = m3.M3(_cohort(), condition_keys="cond", donor_key="donor")
    with pytest.raises(ValueError, match="celltype_key"):
        model.train(max_epochs=1, donor_prediction=True)


def test_donor_prediction_is_off_by_default_without_cell_types():
    model = m3.M3(_cohort(), condition_keys="cond", donor_key="donor")
    model.train(max_epochs=1, seed=0)
    assert model.capabilities["predict_donors"] is False


def test_augment_still_needs_cell_types():
    model = m3.M3(_cohort(), condition_keys="cond", donor_key="donor")
    model.train(max_epochs=1, seed=0)
    with pytest.raises(M3CapabilityError, match="celltype_key"):
        model.augment(conditions=["hc", "dis"], n_donors=[1, 1])


def test_save_and_load_without_cell_types(tmp_path):
    ds = _cohort()
    model, emb = _embedding(ds)
    path = model.save(tmp_path / "model.m3")
    loaded = m3.M3.load(path, ds)
    assert loaded.celltype_key is None
    assert np.array_equal(loaded.embedding("bio"), emb)


def test_reserved_column_name_is_refused():
    ds = _cohort(lambda n: ["x"] * n)
    ds.obs.rename(columns={"cty": "__m3_celltype__"}, inplace=True)
    with pytest.raises(ValueError, match="reserved by m3"):
        m3.M3(ds, condition_keys="cond").train(max_epochs=1)
