"""M3 construction validation: each guard raises a clear ValueError."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("torch")  # importing m3 pulls in the torch-backed model
import m3
from m3._dataset import Dataset


def _ds(n=8):
    rng = np.random.default_rng(0)
    rna = sp.csr_matrix(rng.integers(0, 5, (n, 4)).astype(np.float32))
    half = n // 2
    obs = pd.DataFrame({
        "batch": ["A"] * half + ["B"] * (n - half),
        "cond": (["hc", "dis"] * n)[:n],
        "cty": (["T", "B"] * n)[:n],
        "donor": [f"d{i % 3}" for i in range(n)],
    })
    var = {"rna": pd.Index([f"g{i}" for i in range(4)])}
    present = {"rna": np.ones(n, dtype=bool)}
    return Dataset(modalities={"rna": rna}, obs=obs, var=var, present=present)


def test_valid_construction_ok():
    m = m3.M3(_ds(), condition_keys=["cond"], celltype_key="cty",
              donor_key="donor", held_out=["B"])
    assert m.target_condition == "cond"


def test_rejects_empty_condition_keys():
    with pytest.raises(ValueError, match="condition_keys is required"):
        m3.M3(_ds(), condition_keys=[])


def test_rejects_more_than_two_condition_keys():
    with pytest.raises(ValueError, match="at most 2 condition"):
        m3.M3(_ds(), condition_keys=["cond", "cty", "donor"])


def test_rejects_unknown_condition_key():
    with pytest.raises(ValueError, match="condition key 'nope' not in dataset.obs"):
        m3.M3(_ds(), condition_keys=["nope"])


def test_rejects_unknown_celltype_key():
    with pytest.raises(ValueError, match="celltype_key 'nope' not in dataset.obs"):
        m3.M3(_ds(), condition_keys=["cond"], celltype_key="nope")


def test_rejects_target_condition_not_in_keys():
    with pytest.raises(ValueError, match="target_condition must be one of"):
        m3.M3(_ds(), condition_keys=["cond"], target_condition="cty")


def test_rejects_held_out_and_held_out_samples_together():
    with pytest.raises(ValueError, match="not both"):
        m3.M3(_ds(), condition_keys=["cond"], donor_key="donor",
              held_out=["A"], held_out_samples=["d0"])


def test_held_out_samples_requires_donor_key():
    with pytest.raises(ValueError, match="held_out_samples requires donor_key"):
        m3.M3(_ds(), condition_keys=["cond"], held_out_samples=["d0"])


def test_rejects_unknown_held_out_batch():
    with pytest.raises(ValueError, match="held_out batch 'ZZ'"):
        m3.M3(_ds(), condition_keys=["cond"], held_out=["ZZ"])


def test_rejects_unknown_held_out_sample():
    with pytest.raises(ValueError, match="held_out_samples not in donor_key"):
        m3.M3(_ds(), condition_keys=["cond"], donor_key="donor",
              held_out_samples=["nobody"])
