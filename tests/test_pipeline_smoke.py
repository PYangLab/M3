"""End-to-end smoke test: train() then the embedding / reconstruct /
predict_donors readouts on a tiny synthetic cohort."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")
import m3


def _synthetic_cohort():
    """3 batches x 48 cells; 4 donors/batch (2 HC, 2 disease), 2 cell types."""
    rng = np.random.default_rng(0)
    n_genes, per_batch = 24, 48
    records, counts = [], []
    for batch in ["B1", "B2", "B3"]:
        for j in range(per_batch):
            donor_idx = j % 4
            cond = "hc" if donor_idx < 2 else "dis"
            records.append({"batch": batch, "donor": f"{batch}_d{donor_idx}",
                            "cond": cond, "cty": "T" if j % 2 == 0 else "Bcell"})
            counts.append(rng.poisson(3 + (2 if cond == "dis" else 0), n_genes))
    obs = pd.DataFrame(records)
    return m3.read_matrix(counts={"rna": np.asarray(counts, dtype=np.float32)}, obs=obs,
                          var={"rna": [f"g{i}" for i in range(n_genes)]})


def test_train_predict_smoke():
    ds = _synthetic_cohort()
    model = m3.M3(ds, condition_keys=["cond"], target_condition="cond",
                  celltype_key="cty", donor_key="donor", batch_key="batch",
                  held_out=["B3"])
    model.train(max_epochs=2, seed=0, donor_predictor={"n_epochs": 2, "min_cells": 2})
    assert model.capabilities == {"embedding": True, "reconstruct": True,
                                  "predict_donors": True}

    emb = model.embedding("bio")
    assert emb.shape[0] == ds.n_cells
    assert np.isfinite(emb).all()

    rec = model.reconstruct()
    assert set(rec) == {"rna"}
    assert rec["rna"].shape == (ds.n_cells, 24)
    assert np.isfinite(rec["rna"]).all()

    donors = model.predict_donors()          # held-out B3 donors by default
    prob_cols = [c for c in donors.columns if c.startswith("prob_")]
    assert len(donors) > 0
    assert len(prob_cols) == 2
    assert np.allclose(donors[prob_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-4)
    assert set(donors["predicted_label"]).issubset({"hc", "dis"})
    assert not donors["is_reference"].any()  # only query donors reported


def _interleaved_adata(n=180, n_genes=30):
    """One AnnData, 3 sites with cells INTERLEAVED and no 'batch' column."""
    import anndata as ad
    rng = np.random.default_rng(0)
    recs, X = [], []
    for i in range(n):
        site, d = f"S{i % 3}", (i // 3) % 4
        cond = "hc" if d < 2 else "dis"
        recs.append({"site": site, "donor": f"{site}_d{d}", "cond": cond,
                     "cty": "T" if (i // 12) % 2 == 0 else "Bcell"})
        X.append(rng.poisson(3 + 4 * (i % 3) + (2 if cond == "dis" else 0), n_genes))
    obs = pd.DataFrame(recs, index=[f"cell{i:04d}" for i in range(n)])
    return ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs,
                      var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))


def test_batch_key_selects_the_integration_batches():
    adata = _interleaved_adata()
    ds = m3.from_anndata(adata)
    model = m3.M3(ds, condition_keys=["cond"], celltype_key="cty", donor_key="donor",
                  batch_key="site")
    model.train(max_epochs=1, donor_prediction=False)
    assert model._generator.batch_classify_dim == adata.obs["site"].nunique() == 3
    assert list(ds.obs["batch"].unique()) == ["batch0"]   # caller's Dataset untouched


def test_single_batch_warns():
    ds = m3.from_anndata(_interleaved_adata())            # only the 'batch0' placeholder
    with pytest.warns(UserWarning, match="only one batch"):
        m3.M3(ds, condition_keys=["cond"], celltype_key="cty")


def test_batch_key_refuses_to_overwrite_a_role_column():
    adata = _interleaved_adata()
    adata.obs["batch"] = adata.obs["cond"]
    ds = m3.from_anndata(adata)
    with pytest.raises(ValueError, match="would be overwritten"):
        m3.M3(ds, condition_keys=["batch"], celltype_key="cty", batch_key="site")


def test_readouts_align_to_input_by_obs_names():
    adata = _interleaved_adata()
    model = m3.M3(m3.from_anndata(adata), condition_keys=["cond"], celltype_key="cty",
                  donor_key="donor", batch_key="site")
    model.train(max_epochs=1, donor_prediction=False)
    names = model.cell_metadata["obs_names"]
    assert set(names) == set(adata.obs_names) and names.is_unique
    assert list(names) != list(adata.obs_names)           # rows really are regrouped
    emb = pd.DataFrame(model.embedding("intrinsic"), index=names).reindex(adata.obs_names)
    assert np.isfinite(emb.to_numpy()).all()
    # the metadata carried alongside is the right cell's metadata
    back = model.cell_metadata.set_index("obs_names").reindex(adata.obs_names)
    assert (back["donor"].to_numpy() == adata.obs["donor"].to_numpy()).all()


def test_r_api_seed_reaches_training():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).parents[1] / "m3-r" / "inst" / "python" / "_r_api.py"
    spec = importlib.util.spec_from_file_location("_r_api", path)
    r_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r_api)

    ds = _synthetic_cohort()

    def emb(seed):
        out = r_api.model_train(r_api._put(ds), condition_keys=["cond"], celltype_key="cty",
                                donor_key="donor", max_epochs=1, donor_prediction=False,
                                seed=seed)
        return r_api._get(out["handle"]).embedding("bio")

    assert np.array_equal(emb(1), emb(1))
    assert not np.array_equal(emb(1), emb(2))
