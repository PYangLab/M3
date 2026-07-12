import m3


def test_version_exposed():
    assert isinstance(m3.__version__, str)
    assert m3.__version__.count(".") >= 2


def test_liu_demo_matches_documented_contract():
    # The quickstart's first line. Loading it here proves the bundled .h5ad ships
    # in the package and that the loader's obs remapping produces the shape the
    # docstring promises -- checked structurally, not against volatile cell counts.
    data = m3.datasets.liu_demo()

    assert data.n_cells > 0

    # Three batches, canonicalized to the real B1/B2/B3 labels: the cohort/Batch
    # collapse ran, and both source columns are gone.
    assert set(data.batches) == {"B1", "B2", "B3"}
    assert "cohort" not in data.obs.columns
    assert "Batch" not in data.obs.columns

    # RNA + ADT, each row-aligned to every cell and to its own var; ADT is smaller.
    assert set(data.modality_names) == {"rna", "adt"}
    for mod in ("rna", "adt"):
        assert data.modalities[mod].shape[0] == data.n_cells
        assert data.modalities[mod].shape[1] == len(data.var[mod])
        assert data.present[mod].shape[0] == data.n_cells
    assert data.modalities["adt"].shape[1] < data.modalities["rna"].shape[1]

    # obs carries the columns the quickstart's M3(...) call relies on.
    for col in ("batch", "sample_id", "Donor", "cond_group", "Age_interval", "mergedcelltype"):
        assert col in data.obs.columns
