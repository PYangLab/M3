"""M3.save / M3.load: a loaded model gives the same read-outs as the model that
was saved, and a file or dataset that does not belong is refused."""
import contextlib
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")
import m3


def _cohort(n_rna=24, n_adt=6, sizes=(60, 48, 52), adt_missing_in=None, seed=0):
    """Three batches of unequal size; 4 donors per batch (2 hc, 2 dis), 2 cell types."""
    rng = np.random.default_rng(seed)
    records, rna, adt = [], [], []
    for batch, n in zip(["B1", "B2", "B3"], sizes):
        for j in range(n):
            donor = j % 4
            cond = "hc" if donor < 2 else "dis"
            records.append({"batch": batch, "donor": f"{batch}_d{donor}", "cond": cond,
                            "cty": "T" if (j // 4) % 2 == 0 else "Bcell",
                            "note": f"cell{len(records)}"})
            rna.append(rng.poisson(3 + (2 if cond == "dis" else 0), n_rna))
            adt.append(np.zeros(n_adt) if batch == adt_missing_in
                       else rng.poisson(20 + (10 if cond == "dis" else 0), n_adt))
    obs = pd.DataFrame(records)
    counts = {"rna": np.asarray(rna, dtype=np.float32)}
    var = {"rna": [f"g{i}" for i in range(n_rna)]}
    if n_adt:
        counts["adt"] = np.asarray(adt, dtype=np.float32)
        var["adt"] = [f"p{i}" for i in range(n_adt)]
    return m3.read_matrix(counts=counts, obs=obs, var=var)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _train(ds, *, train_kw=None, **init_kw):
    kw = dict(condition_keys=["cond"], target_condition="cond", celltype_key="cty",
              donor_key="donor", batch_key="batch")
    kw.update(init_kw)
    model = m3.M3(ds, **kw)
    tkw = dict(max_epochs=2, donor_predictor={"n_epochs": 3, "min_cells": 2})
    tkw.update(train_kw or {})
    if kw.get("donor_key") is None:
        tkw.pop("donor_predictor")
    _quiet(model.train, **tkw)
    return model


def _readouts(model):
    """Every read-out the model supports, as a flat {name: array-or-frame} dict."""
    out = {"cell_metadata": model.cell_metadata}
    for part in ["bio", "intrinsic", "batch", *model.condition_keys]:
        out[f"embedding:{part}"] = model.embedding(part)
    for rb in (True, False):
        for mod, arr in model.reconstruct(remove_batch=rb).items():
            out[f"reconstruct:{rb}:{mod}"] = arr
    for mod, arr in _quiet(model.generate).items():
        out[f"generate:{mod}"] = arr
    if model.capabilities["predict_donors"]:
        out["predict_donors"] = model.predict_donors(include_reference=True)
        out["donor_embedding"] = model.donor_embedding()
        attr = _quiet(model.attribute, reference_labels=["hc"], n_steps=4)
        out["attr:attribution"] = attr.attribution
        out["attr:gene_celltype_matrix"] = attr.gene_celltype_matrix
        out["attr:cells"] = attr.cells
        out["attr:genes"] = attr.genes
        out["attr:celltypes"] = attr.celltypes
        out["attr:donors"] = attr.donors
        out["attr:top_genes"] = attr.top_genes(n=10, min_cells_per_condition=0)
        out["attr:top_celltypes"] = attr.top_celltypes(min_cells_per_condition=0)
    if model.donor_key is not None:
        aug = _quiet(model.augment, conditions=["hc", "dis"], n_donors=[1, 1])
        out["augment:obs"] = aug["obs"]
        for mod, arr in aug["expression"].items():
            out[f"augment:{mod}"] = arr
    return out


def _assert_same(a: dict, b: dict):
    """Bit-for-bit: saving and loading must not change a single number."""
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], pd.DataFrame):
            pd.testing.assert_frame_equal(a[k], b[k], check_exact=True, obj=k)
        else:
            assert a[k].dtype == b[k].dtype, k
            assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k


def _roundtrip(model, ds, tmp_path):
    path = str(tmp_path / "model.m3")
    model.save(path)
    return m3.M3.load(path, ds)


# ------------------------------------------------------------ same read-outs
def test_loaded_model_reproduces_every_readout_with_held_out_batch(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    before = _readouts(model)
    loaded = _roundtrip(model, ds, tmp_path)
    assert loaded.capabilities == model.capabilities
    assert loaded.contract == model.contract
    assert loaded.history == model.history
    _assert_same(before, _readouts(loaded))
    _assert_same(before, _readouts(model))      # saving leaves the model as it was


def test_loaded_model_reproduces_every_readout_on_the_full_reference(tmp_path):
    ds = _cohort()
    model = _train(ds)                           # no held-out set: the attribution setup
    assert model.cell_metadata.shape[0] == ds.n_cells
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_model_reproduces_readouts_with_held_out_samples(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out_samples=["B1_d0", "B2_d3"])
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_integration_only_model(tmp_path):
    ds = _cohort()
    model = _train(ds, donor_key=None, target_condition=None)
    loaded = _roundtrip(model, ds, tmp_path)
    assert loaded.capabilities == {"embedding": True, "reconstruct": True,
                                   "predict_donors": False}
    _assert_same(_readouts(model), _readouts(loaded))
    with pytest.raises(m3.M3CapabilityError):
        loaded.predict_donors()


def test_loaded_model_without_batch_balancing(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"], train_kw={"balance_batches": False})
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_model_trained_without_a_seed(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"], train_kw={"seed": None})
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_model_with_in_model_hvg(tmp_path):
    ds = _cohort(n_rna=60)
    model = _train(ds, hvg={"rna": 20})
    loaded = _roundtrip(model, ds, tmp_path)
    assert loaded.reconstruct()["rna"].shape[1] == 20
    _assert_same(_readouts(model), _readouts(loaded))


def test_loaded_model_on_mosaic_data(tmp_path):
    ds = _cohort(adt_missing_in="B2")
    model = _train(ds, held_out=["B3"])
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_model_with_a_named_batch_column(tmp_path):
    ds = _cohort()
    obs = ds.obs.rename(columns={"batch": "site"})
    ds2 = m3.read_matrix(
        counts={k: np.asarray(v.todense()) for k, v in ds.modalities.items()},
        obs=obs, var={k: list(v) for k, v in ds.var.items()})
    model = _train(ds2, batch_key="site", held_out=["B3"])
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds2, tmp_path)))


def test_cell_metadata_keeps_only_the_users_columns():
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    assert list(model.cell_metadata.columns) == list(ds.obs.columns)


def test_load_in_a_fresh_process(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    path = str(tmp_path / "model.m3")
    model.save(path)
    np.save(tmp_path / "emb.npy", model.embedding("bio"))
    model.predict_donors(include_reference=True).to_pickle(tmp_path / "preds.pkl")
    code = (
        "import sys, numpy as np, pandas as pd\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import m3, test_save_load as t\n"
        "d = sys.argv[2]\n"
        "model = m3.M3.load(d + '/model.m3', t._cohort())\n"
        "assert np.array_equal(model.embedding('bio'), np.load(d + '/emb.npy'))\n"
        "pd.testing.assert_frame_equal(model.predict_donors(include_reference=True),\n"
        "                              pd.read_pickle(d + '/preds.pkl'), check_exact=True)\n"
        "print('fresh-process-ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code, os.path.dirname(__file__), str(tmp_path)],
                         capture_output=True, text=True)
    assert "fresh-process-ok" in res.stdout, res.stderr[-2000:]


# -------------------------------------------------------------- refusals
def test_save_needs_a_trained_model(tmp_path):
    ds = _cohort()
    model = m3.M3(ds, condition_keys=["cond"], celltype_key="cty")
    with pytest.raises(m3.M3CapabilityError, match="train"):
        model.save(str(tmp_path / "model.m3"))
    assert not (tmp_path / "model.m3").exists()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    path = str(tmp_path_factory.mktemp("saved") / "model.m3")
    model.save(path)
    return ds, path


def _edited(ds, *, counts=None, obs=None, var=None):
    return m3.read_matrix(
        counts=counts or {k: np.asarray(v.todense()) for k, v in ds.modalities.items()},
        obs=ds.obs if obs is None else obs,
        var=var or {k: list(v) for k, v in ds.var.items()})


def test_load_refuses_changed_counts(saved):
    ds, path = saved
    counts = {k: np.asarray(v.todense()) for k, v in ds.modalities.items()}
    counts["rna"][5, 3] += 1
    with pytest.raises(ValueError, match="counts of 'rna'"):
        m3.M3.load(path, _edited(ds, counts=counts))


def test_load_refuses_reordered_cells(saved):
    ds, path = saved
    order = np.random.default_rng(1).permutation(ds.n_cells)
    counts = {k: np.asarray(v.todense())[order] for k, v in ds.modalities.items()}
    obs = ds.obs.iloc[order].reset_index(drop=True)
    with pytest.raises(ValueError, match="not the dataset this model was trained on"):
        m3.M3.load(path, _edited(ds, counts=counts, obs=obs))


def test_load_refuses_a_subset_of_the_cells(saved):
    ds, path = saved
    keep = np.arange(ds.n_cells - 4)
    counts = {k: np.asarray(v.todense())[keep] for k, v in ds.modalities.items()}
    with pytest.raises(ValueError, match="not the dataset this model was trained on"):
        m3.M3.load(path, _edited(ds, counts=counts, obs=ds.obs.iloc[keep]))


def test_load_refuses_renamed_features(saved):
    ds, path = saved
    var = {k: list(v) for k, v in ds.var.items()}
    var["rna"][0], var["rna"][1] = var["rna"][1], var["rna"][0]
    with pytest.raises(ValueError, match="feature names of 'rna'"):
        m3.M3.load(path, _edited(ds, var=var))


def test_load_refuses_changed_role_column(saved):
    ds, path = saved
    obs = ds.obs.copy()
    obs.loc[0, "cty"] = "Bcell" if obs.loc[0, "cty"] == "T" else "T"
    with pytest.raises(ValueError, match="obs column 'cty'"):
        m3.M3.load(path, _edited(ds, obs=obs))


def test_load_accepts_a_changed_column_the_model_does_not_use(saved):
    ds, path = saved
    obs = ds.obs.assign(note="edited")
    loaded = m3.M3.load(path, _edited(ds, obs=obs))
    assert set(loaded.cell_metadata["note"]) == {"edited"}


def test_load_refuses_a_missing_role_column(saved):
    ds, path = saved
    with pytest.raises(ValueError, match="donor"):
        m3.M3.load(path, _edited(ds, obs=ds.obs.drop(columns=["donor"])))


def test_load_refuses_a_truncated_file(saved, tmp_path):
    _, path = saved
    raw = open(path, "rb").read()
    for cut in (0, 10, len(raw) // 2, len(raw) - 7):
        bad = tmp_path / f"cut{cut}.m3"
        bad.write_bytes(raw[:cut])
        with pytest.raises(ValueError, match="not a readable m3 model file"):
            m3.M3.load(str(bad), saved[0])


def test_a_corrupted_file_is_refused_or_still_exact(saved, tmp_path):
    """Flip one byte anywhere in the file, 40 times over. A flip that reaches the
    content must be refused; one that lands in zip bookkeeping the reader ignores
    (a timestamp, say) may load, and then the model must be the saved one."""
    ds, path = saved
    raw = open(path, "rb").read()
    expected = m3.M3.load(path, ds).embedding("bio")
    rng = np.random.default_rng(0)
    refused = 0
    for trial in range(40):
        bad = bytearray(raw)
        bad[int(rng.integers(0, len(bad)))] ^= 0xFF
        p = tmp_path / f"flip{trial}.m3"
        p.write_bytes(bad)
        try:
            loaded = m3.M3.load(str(p), ds)
        except ValueError:
            refused += 1
        else:
            assert np.array_equal(loaded.embedding("bio"), expected)
    assert refused >= 30


def test_load_refuses_other_files(tmp_path):
    ds = _cohort()
    text = tmp_path / "notes.m3"
    text.write_text("not a model")
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(str(text), ds)
    npz = tmp_path / "other.m3"
    with open(npz, "wb") as f:
        np.savez(f, a=np.arange(3))
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(str(npz), ds)
    with pytest.raises(FileNotFoundError):
        m3.M3.load(str(tmp_path / "absent.m3"), ds)


def _rewrite(path, out, *, header=None, arrays=None):
    """Copy a model file, editing its header and/or some of its arrays."""
    with np.load(path, allow_pickle=False) as z:
        members = {k: z[k] for k in z.files}
    if header is not None:
        h = json.loads(str(members["header"]))
        header(h)
        members["header"] = np.array(json.dumps(h))
    if arrays is not None:
        arrays(members)
    with open(out, "wb") as f:
        np.savez_compressed(f, **members)


def test_load_refuses_a_newer_file_format(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "future.m3")
    _rewrite(path, out, header=lambda h: h.update(format_version=h["format_version"] + 1))
    with pytest.raises(ValueError, match="format version"):
        m3.M3.load(out, ds)


def test_load_refuses_weights_that_do_not_reproduce_the_saved_outputs(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "tampered.m3")

    def scale(members):
        key = next(k for k in members if k.startswith("generator/") and k.endswith("weight"))
        members[key] = members[key] * 1.5

    _rewrite(path, out, arrays=scale)
    with pytest.raises(ValueError, match="does not reproduce"):
        m3.M3.load(out, ds)


def test_load_refuses_weights_of_another_shape(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "shape.m3")

    def crop(members):
        key = next(k for k in members if k.startswith("generator/") and k.endswith("weight"))
        members[key] = members[key][:-1]

    _rewrite(path, out, arrays=crop)
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(out, ds)


def test_model_file_holds_no_pickled_objects(saved):
    _, path = saved
    with zipfile.ZipFile(path) as z, np.load(path, allow_pickle=False) as npz:
        assert len(z.namelist()) == len(npz.files)
        for k in npz.files:
            npz[k]                                # raises if any member needs pickle


def test_save_replaces_an_existing_file_only_once_complete(saved, tmp_path, monkeypatch):
    ds, path = saved
    target = tmp_path / "model.m3"
    target.write_bytes(open(path, "rb").read())
    model = m3.M3.load(str(target), ds)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError, match="disk full"):
        model.save(str(target))
    monkeypatch.undo()
    assert target.read_bytes() == open(path, "rb").read()
    assert os.listdir(tmp_path) == ["model.m3"]   # no stray temporary file


def test_load_refuses_a_file_whose_cell_rows_were_changed(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "rows.m3")

    def swap(members):
        rows = members["rows"].copy()
        rows[[0, 1]] = rows[[1, 0]]
        members["rows"] = rows

    _rewrite(path, out, arrays=swap)
    with pytest.raises(ValueError, match="does not reproduce the model inputs"):
        m3.M3.load(out, ds)


def test_save_writes_nothing_if_the_model_cannot_be_rebuilt(saved, tmp_path, monkeypatch):
    ds, path = saved
    model = m3.M3.load(path, ds)
    real = m3.M3._assemble

    def off_by_a_bit(self, rows, n_ref):
        data, b, mask, meta = real(self, rows, n_ref)
        return data * (1 + 1e-6), b, mask, meta

    monkeypatch.setattr(m3.M3, "_assemble", off_by_a_bit)
    with pytest.raises(RuntimeError, match="nothing was written"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


# ------------------------------------------------------------- R bridge
def _r_api():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parents[1] / "m3-r" / "inst" / "python" / "_r_api.py"
    spec = importlib.util.spec_from_file_location("_r_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r_api_saves_and_loads_a_model(tmp_path):
    r_api = _r_api()
    ds = _cohort()
    h_ds = r_api._put(ds)
    trained = _quiet(
        r_api.model_train, h_ds, condition_keys=["cond"], target_condition="cond",
        celltype_key="cty", donor_key="donor", held_out=["B3"], max_epochs=2,
        donor_predictor={"n_epochs": 3, "min_cells": 2}, seed=0)
    path = r_api.model_save(trained["handle"], str(tmp_path / "model.m3"))
    assert path == str(tmp_path / "model.m3") and os.path.isfile(path)

    loaded = r_api.model_load(path, h_ds)
    assert loaded["handle"] != trained["handle"]
    assert {k: v for k, v in loaded.items() if k != "handle"} == \
           {k: v for k, v in trained.items() if k != "handle"}
    assert np.array_equal(r_api.embedding(loaded["handle"]), r_api.embedding(trained["handle"]))
    pd.testing.assert_frame_equal(r_api.predict_donors(loaded["handle"], True),
                                  r_api.predict_donors(trained["handle"], True),
                                  check_exact=True)


def test_r_api_gives_the_shipped_model_paths():
    r_api = _r_api()
    assert r_api.demo_model("attribution") == m3.datasets.liu_demo_model("attribution")


# ------------------------------------------------- less common situations
def test_loaded_model_with_unlabelled_held_out_cells(tmp_path):
    ds = _cohort()
    obs = ds.obs.copy()
    obs.loc[obs["batch"] == "B3", "cond"] = np.nan      # the query's labels are unknown
    ds = _edited(ds, obs=obs)
    model = _train(ds, held_out=["B3"])
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_loaded_model_on_a_single_non_rna_modality(tmp_path):
    full = _cohort()
    ds = m3.read_matrix(counts={"adt": np.asarray(full.modalities["adt"].todense())},
                        obs=full.obs, var={"adt": list(full.var["adt"])})
    model = _train(ds, held_out=["B3"])
    _assert_same(_readouts(model), _readouts(_roundtrip(model, ds, tmp_path)))


def test_load_needs_an_m3_dataset(saved):
    import anndata as ad
    ds, path = saved
    adata = ad.AnnData(X=np.asarray(ds.modalities["rna"].todense()), obs=ds.obs.copy())
    with pytest.raises(TypeError, match="m3 Dataset"):
        m3.M3.load(path, adata)


def test_save_after_annotating_the_dataset(tmp_path):
    """Adding an obs column after training does not touch what the model depends on."""
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    before = _readouts(model)
    ds.obs["cluster"] = np.arange(ds.n_cells) % 3
    loaded = _roundtrip(model, ds, tmp_path)
    assert "cluster" in loaded.cell_metadata.columns
    after = _readouts(loaded)
    after["cell_metadata"] = after["cell_metadata"].drop(columns="cluster")
    after["augment:obs"] = after["augment:obs"][before["augment:obs"].columns]
    _assert_same(before, after)


def test_save_refuses_a_dataset_changed_since_training(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    ds.modalities["rna"].data[:] += 1                     # edited in place after train()
    with pytest.raises(ValueError, match="changed since this model was trained.*counts of 'rna'"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("part", [
    "generator/decoder.decoders.0.net.0.weight",        # reconstruct / generate / augment
    "generator/batch_encoder.encoder1.net.0.weight",    # the batch term of every embedding
    "corrector/transform.0.0.weight",                   # donor embedding and prediction
    "corrector/status_head.1.weight",                   # donor prediction
])
def test_load_checks_every_part_of_the_model_a_readout_uses(saved, tmp_path, part):
    ds, path = saved
    out = str(tmp_path / "tampered.m3")

    def scale(members):
        assert part in members, sorted(members)
        members[part] = members[part] * 1.5

    _rewrite(path, out, arrays=scale)
    with pytest.raises(ValueError, match="does not reproduce"):
        m3.M3.load(out, ds)


def test_load_refuses_settings_that_disagree_with_the_recorded_cells(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "held.m3")
    _rewrite(path, out, header=lambda h: h["init"].update(held_out=["B1"]))
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(out, ds)
    _rewrite(path, out, header=lambda h: h.update(n_ref=h["n_ref"] - 1))
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(out, ds)


# ------------------------------------------- found by independent testers
def test_retraining_without_the_donor_predictor_drops_the_old_one(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    _quiet(model.train, max_epochs=2, donor_prediction=False, seed=1)
    assert model.capabilities == {"embedding": True, "reconstruct": True,
                                  "predict_donors": False}
    with pytest.raises(m3.M3CapabilityError):
        model.predict_donors()
    loaded = _roundtrip(model, ds, tmp_path)
    assert loaded.capabilities == model.capabilities
    _assert_same(_readouts(model), _readouts(loaded))


def test_loaded_model_whose_donor_predictions_are_nan(tmp_path):
    ds = _cohort()
    obs = ds.obs.copy()
    obs.loc[(obs["donor"] == "B1_d0") & (obs["cty"] == "Bcell"), "cty"] = "T"
    ds = _edited(ds, obs=obs)                 # one donor now lacks a cell type
    model = _train(ds, held_out=["B3"],
                   train_kw={"donor_predictor": {"n_epochs": 3, "min_cells": 0}})
    assert model.predict_donors(include_reference=True).filter(like="prob_").isna().any().any()
    loaded = _roundtrip(model, ds, tmp_path)
    pd.testing.assert_frame_equal(model.predict_donors(include_reference=True),
                                  loaded.predict_donors(include_reference=True),
                                  check_exact=True)
    pd.testing.assert_frame_equal(model.donor_embedding(), loaded.donor_embedding(),
                                  check_exact=True)
    assert np.array_equal(model.embedding("bio"), loaded.embedding("bio"))


def test_save_names_a_setting_it_cannot_store(tmp_path):
    ds = _cohort()
    days = {"B1": pd.Timestamp("2021-01-05"), "B2": pd.Timestamp("2021-02-09"),
            "B3": pd.Timestamp("2021-03-16")}
    ds = _edited(ds, obs=ds.obs.assign(batch=ds.obs["batch"].map(days)))
    model = _train(ds, held_out=[days["B3"]])
    with pytest.raises(TypeError, match="held_out.*Timestamp"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


def test_save_and_load_accept_a_bytes_path(saved, tmp_path):
    ds, path = saved
    model = m3.M3.load(os.fsencode(path), ds)
    out = os.fsencode(str(tmp_path / "again.m3"))
    assert model.save(out) == str(tmp_path / "again.m3")
    assert os.listdir(tmp_path) == ["again.m3"]


def test_save_says_so_when_cell_metadata_was_edited(tmp_path):
    ds = _cohort()
    model = _train(ds, held_out=["B3"])
    meta = model.cell_metadata
    meta["cty"] = meta["cty"].str.lower()
    with pytest.raises(ValueError, match="cell_metadata"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


def test_dataset_digest_tells_apart_labels_that_only_differ_in_where_they_split():
    ds = _cohort()
    a, b = ds.obs.copy(), ds.obs.copy()
    a.loc[[0, 1], "cty"] = ["T\x1fT", "B"]
    b.loc[[0, 1], "cty"] = ["T", "T\x1fB"]
    kw = dict(condition_keys=["cond"], celltype_key="cty")
    fa = m3.M3(_edited(ds, obs=a), **kw)._fingerprint()
    fb = m3.M3(_edited(ds, obs=b), **kw)._fingerprint()
    assert fa["obs column 'cty'"] != fb["obs column 'cty'"]


def test_load_reports_a_wrong_dataset_even_when_the_settings_do_not_fit_it(saved):
    ds, path = saved                          # saved with held_out=["B3"]
    renamed = ds.obs.assign(batch=ds.obs["batch"].map({"B1": "x", "B2": "y", "B3": "z"}))
    with pytest.raises(ValueError, match="not the dataset this model was trained on"):
        m3.M3.load(path, _edited(ds, obs=renamed))


def test_loaded_capabilities_follow_what_the_file_holds(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "caps.m3")
    _rewrite(path, out, header=lambda h: h.update(capabilities={"predict_donors": False}))
    assert m3.M3.load(out, ds).capabilities["predict_donors"] is True


def _unlabelled(ds, missing):
    obs = ds.obs.copy()
    col = obs["cond"].astype(object)
    col[(obs["batch"] == "B3").to_numpy()] = missing
    obs["cond"] = col
    return _edited(ds, obs=obs)


def test_dataset_digest_tells_a_missing_label_from_the_text_nan(tmp_path):
    base = _cohort()
    ds = _unlabelled(base, np.nan)
    model = _train(ds, held_out=["B3"])
    path = model.save(str(tmp_path / "model.m3"))
    with pytest.raises(ValueError, match="obs column 'cond'"):
        m3.M3.load(path, _unlabelled(base, "nan"))
    # None and NaN are the same missing label
    _assert_same(_readouts(model), _readouts(m3.M3.load(path, _unlabelled(base, None))))


def test_concurrent_saves_to_one_path_leave_a_whole_file(saved, tmp_path):
    import threading
    ds, path = saved
    models = [m3.M3.load(path, ds), m3.M3.load(path, ds)]
    target = str(tmp_path / "shared.m3")
    errors = []

    def work(model):
        try:
            for _ in range(6):
                model.save(target)
        except Exception as e:                    # pragma: no cover - the failure itself
            errors.append(e)

    threads = [threading.Thread(target=work, args=(m,)) for m in models]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert os.listdir(tmp_path) == ["shared.m3"]
    assert np.array_equal(m3.M3.load(target, ds).embedding("bio"), models[0].embedding("bio"))


def test_loading_from_several_threads_leaves_stdout_alone(saved):
    import threading
    ds, path = saved
    before = sys.stdout
    threads = [threading.Thread(target=lambda: [m3.M3.load(path, ds) for _ in range(4)])
               for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sys.stdout is before


def test_load_does_not_suggest_unpickling_a_file_that_is_not_a_model(tmp_path):
    ds = _cohort()
    text = tmp_path / "notes.m3"
    text.write_text("this is not a model file\n")
    with pytest.raises(ValueError, match="not a readable m3 model file") as err:
        m3.M3.load(str(text), ds)
    assert "pickle" not in str(err.value)


@pytest.mark.parametrize("edit", [
    lambda h: h["init"].update(batch_key=None),
    lambda h: h["init"].update(condition_keys=["cond", "cty"]),
    lambda h: h.update(format_version="1"),
    lambda h: h.update(format_version=True),
    lambda h: h.update(n_ref=0),
    lambda h: h["generator"].update(nfeatures=[200000, 6]),
    lambda h: h["generator"].update(batch_classify_dim=7),
    lambda h: h["generator"].update(condition_dim=[3000000]),
    lambda h: h["donor_predictor"].update(min_cells="2x"),
    lambda h: h.update(donor_predictor=None),
    lambda h: h.update(hvg_cols={"rna": [0, 1, 2]}),
], ids=["batch_key", "condition_keys", "version-str", "version-bool", "n_ref", "nfeatures",
        "batches", "condition-classes", "min_cells", "no-predictor", "hvg_cols"])
def test_load_refuses_a_header_that_disagrees_with_the_file_or_the_data(saved, tmp_path, edit):
    ds, path = saved
    out = str(tmp_path / "header.m3")
    _rewrite(path, out, header=edit)
    with pytest.raises(ValueError, match="not a readable m3 model file"):
        m3.M3.load(out, ds)


def test_load_refuses_entries_save_never_writes(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "extra.m3")
    _rewrite(path, out, arrays=lambda members: members.update(padding=np.zeros(8)))
    with pytest.raises(ValueError, match="not a readable m3 model file.*padding"):
        m3.M3.load(out, ds)


def test_load_refuses_a_donor_embedding_rescaled_behind_unchanged_predictions(saved, tmp_path):
    ds, path = saved
    out = str(tmp_path / "rescaled.m3")

    def rescale(members):
        members["corrector/transform.1.0.weight"] = members["corrector/transform.1.0.weight"] * 100
        members["corrector/transform.1.0.bias"] = members["corrector/transform.1.0.bias"] * 100
        members["corrector/status_head.0.0.weight"] = members["corrector/status_head.0.0.weight"] / 100

    _rewrite(path, out, arrays=rescale)
    with pytest.raises(ValueError, match="does not reproduce"):
        m3.M3.load(out, ds)


def test_recorded_outputs_cover_every_cell():
    ds = _cohort(sizes=(140, 120, 130))
    model = _train(ds)
    n = model.cell_metadata.shape[0]
    assert n > 256
    recorded = model._probes()
    assert recorded["probe_embedding_cells"].shape == (n,)
    assert recorded["probe_reconstruction_cells"].shape == (n,)


def test_load_says_which_argument_comes_first(saved):
    ds, path = saved
    with pytest.raises(TypeError, match="file path comes first"):
        m3.M3.load(ds, path)


# --------------------------------- found by the second round of testers
def test_other_obs_columns_do_not_change_how_role_columns_are_read(tmp_path):
    """Cell types that mix number-like and text labels ('1', '2', 'unassigned'), in
    batches long enough for a wide obs table to be parsed in several pieces."""
    rng = np.random.default_rng(0)
    recs = []
    for batch, n in zip(["B1", "B2", "B3"], (3000, 2600, 2800)):
        for j in range(n):
            d = j % 4
            numbered = "1" if (j // 4) % 2 == 0 else "2"
            cty = numbered if j < n - 400 or numbered == "1" else "unassigned"
            recs.append({"batch": batch, "donor": f"{batch}_d{d}",
                         "cond": "hc" if d < 2 else "dis", "cty": cty})
    obs = pd.DataFrame(recs)
    counts = {"rna": rng.poisson(np.where((obs["cond"] == "dis").to_numpy()[:, None], 5, 3),
                                 (len(obs), 24)).astype(np.float32),
              "adt": rng.poisson(20, (len(obs), 6)).astype(np.float32)}
    var = {"rna": [f"g{i}" for i in range(24)], "adt": [f"p{i}" for i in range(6)]}
    wide = pd.concat([obs, pd.DataFrame(0, index=obs.index,
                                        columns=[f"qc{i}" for i in range(300)])], axis=1)
    model = _train(m3.read_matrix(counts=counts, obs=wide, var=var), held_out=["B3"],
                   train_kw={"max_epochs": 1})
    assert {type(v) for v in model.cell_metadata["cty"]} == {str}
    path = model.save(str(tmp_path / "model.m3"))
    loaded = m3.M3.load(path, m3.read_matrix(counts=counts, obs=obs, var=var))
    a = _quiet(model.augment, conditions=["hc"], n_donors=[2])
    b = _quiet(loaded.augment, conditions=["hc"], n_donors=[2])
    assert np.array_equal(a["expression"]["rna"], b["expression"]["rna"])
    pd.testing.assert_frame_equal(a["obs"], b["obs"], check_exact=True)
    assert np.array_equal(model.embedding("bio"), loaded.embedding("bio"))


def test_save_refuses_a_model_whose_held_out_set_is_not_the_one_requested(tmp_path):
    """Donor ids that read as numbers ('001') lose their text form inside the engine,
    so held_out_samples matches nothing there. Such a model must not be written."""
    base = _cohort()
    ids = {d: f"{i + 1:03d}" for i, d in enumerate(sorted(base.obs["donor"].unique()))}
    ds = _edited(base, obs=base.obs.assign(donor=base.obs["donor"].map(ids)))
    model = _train(ds, held_out_samples=["001", "012"])
    with pytest.raises(ValueError, match="held-out"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


def test_load_lets_a_warning_raised_as_an_error_through(tmp_path):
    import warnings
    base = _cohort()
    ds = _edited(base, obs=base.obs.assign(batch="B1"))
    with pytest.warns(UserWarning, match="only one batch"):
        model = _train(ds)
    path = model.save(str(tmp_path / "model.m3"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="only one batch"):
            m3.M3.load(path, ds)


def test_a_dataset_whose_parts_no_longer_fit_together_is_reported_as_such(saved, tmp_path):
    ds, path = saved
    broken = _edited(ds)
    broken.obs = broken.obs[broken.obs["cty"] == "T"]          # rows filtered in place
    with pytest.raises(ValueError, match="rows but obs has"):
        m3.M3.load(path, broken)
    model = m3.M3.load(path, ds2 := _edited(ds))
    ds2.obs.drop(index=ds2.obs.index[-5:], inplace=True)
    with pytest.raises(ValueError, match="changed since this model was trained"):
        model.save(str(tmp_path / "model.m3"))
    assert os.listdir(tmp_path) == []


def test_loading_does_not_swallow_what_other_threads_print(saved):
    import threading
    import time
    ds, path = saved

    class Count(io.TextIOBase):
        lines = 0

        def write(self, s):
            Count.lines += s.count("tick")
            return len(s)

    real, sent, stop = sys.stdout, [0], []

    def chatter():
        while not stop:
            print("tick")
            sent[0] += 1
            time.sleep(0.0005)

    sys.stdout = Count()
    try:
        t = threading.Thread(target=chatter)
        t.start()
        for _ in range(8):
            m3.M3.load(path, ds)
        stop.append(1)
        t.join()
    finally:
        sys.stdout = real
    assert sent[0] > 0 and Count.lines == sent[0]


def test_dataset_digest_tells_a_number_label_from_its_text():
    ds = _cohort()
    numbered = ds.obs.assign(cond=(ds.obs["cond"] == "dis").astype(int))
    kw = dict(condition_keys=["cond"], celltype_key="cty")
    as_int = m3.M3(_edited(ds, obs=numbered), **kw)._fingerprint()
    as_text = m3.M3(_edited(ds, obs=numbered.assign(cond=numbered["cond"].astype(str))),
                    **kw)._fingerprint()
    assert as_int["obs column 'cond'"] != as_text["obs column 'cond'"]


def test_load_names_the_real_problem_for_a_file_with_leading_junk(saved, tmp_path):
    ds, path = saved
    bad = tmp_path / "junk.m3"
    bad.write_bytes(b"xyz" + open(path, "rb").read())
    with pytest.raises(ValueError, match="not a readable m3 model file") as err:
        m3.M3.load(str(bad), ds)
    assert "pickle" not in str(err.value)
