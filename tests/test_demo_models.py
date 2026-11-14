"""`m3.datasets.liu_demo_model(...)`: models trained in advance on the full demo,
shipped with the package so a tutorial can load one instead of training."""
import os

import pytest

import m3


def test_liu_demo_model_gives_the_path_of_a_shipped_file():
    for name in ("patient_prediction", "attribution"):
        path = m3.datasets.liu_demo_model(name)
        assert os.path.isfile(path), path


def test_liu_demo_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="attribution.*patient_prediction"):
        m3.datasets.liu_demo_model("augmentation")


def test_shipped_models_load_on_the_full_demo():
    """load() refuses a file whose weights do not give the outputs recorded when it
    was saved, or whose dataset digest differs, so loading is the check that the
    shipped models and the shipped demo data still belong together."""
    pytest.importorskip("torch")
    data = m3.datasets.liu_demo()

    prediction = m3.M3.load(m3.datasets.liu_demo_model("patient_prediction"), data)
    assert prediction.held_out == ["B3"]
    preds = prediction.predict_donors()
    assert len(preds) == 23
    assert set(preds["predicted_label"]) <= {"HC", "Severe"}

    attribution = m3.M3.load(m3.datasets.liu_demo_model("attribution"), data)
    assert attribution.held_out == []
    assert attribution.cell_metadata.shape[0] == data.n_cells
    assert attribution.capabilities["predict_donors"]


def test_shipped_models_do_not_fit_the_subsampled_demo():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="not the dataset this model was trained on"):
        m3.M3.load(m3.datasets.liu_demo_model("attribution"), m3.datasets.liu_demo(n_cells=3000))
