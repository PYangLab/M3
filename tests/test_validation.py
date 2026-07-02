import warnings

import numpy as np
import pandas as pd

import m3


def test_warns_on_non_integer_counts():
    counts = {"rna": np.array([[0.5, 1.2], [3.3, 0.0]], dtype=np.float32)}
    obs = pd.DataFrame({"barcode": ["c1", "c2"]})
    var = {"rna": ["g1", "g2"]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m3.read_matrix(counts=counts, obs=obs, var=var, batch="A")
    assert any("count" in str(x.message).lower() for x in w)


def test_read_matrix_does_not_warn_on_nan_obs():
    # NaN in obs is no longer flagged at read time (it would flood on unused columns);
    # it is checked at M3 construction, only on the selected role columns.
    counts = {"rna": np.array([[1, 2], [3, 4]], dtype=np.float32)}
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "junk": ["x", np.nan]})
    var = {"rna": ["g1", "g2"]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m3.read_matrix(counts=counts, obs=obs, var=var, batch="A")
    assert not any("nan" in str(x.message).lower() for x in w)


def test_m3_warns_on_nan_only_in_role_columns():
    counts = {"rna": np.array([[1, 2], [3, 4]], dtype=np.float32)}
    obs = pd.DataFrame({"barcode": ["c1", "c2"], "disease": ["hc", np.nan],
                        "junk": [np.nan, np.nan]})
    d = m3.read_matrix(counts=counts, obs=obs, var={"rna": ["g1", "g2"]}, batch="A")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m3.M3(d, condition_keys=["disease"])   # 'disease' is a role -> warns; 'junk' is not
    msgs = [str(x.message).lower() for x in w]
    assert any("disease" in m and "nan" in m for m in msgs)
    assert not any("junk" in m for m in msgs)


def test_integer_counts_no_count_warning():
    counts = {"rna": np.array([[1, 2], [3, 4]], dtype=np.float32)}
    obs = pd.DataFrame({"barcode": ["c1", "c2"]})
    var = {"rna": ["g1", "g2"]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m3.read_matrix(counts=counts, obs=obs, var=var, batch="A")
    assert not any("count" in str(x.message).lower() for x in w)
