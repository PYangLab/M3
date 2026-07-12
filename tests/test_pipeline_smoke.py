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
