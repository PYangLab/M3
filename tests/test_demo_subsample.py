"""`m3.datasets.liu_demo(n_cells=...)`: a smaller demo for quick, CPU-only runs."""

import numpy as np
import pandas as pd
import pytest

import m3


@pytest.fixture(scope="module")
def full():
    return m3.datasets.liu_demo()


def _row_keys(data):
    """One hashable key per cell, drawn from obs AND both count matrices."""
    obs = data.obs
    return list(zip(
        obs["batch"].astype(str),
        obs["sample_id"].astype(str),
        obs["mergedcelltype"].astype(str),
        np.asarray(data.modalities["rna"].sum(axis=1)).ravel().tolist(),
        np.asarray(data.modalities["adt"].sum(axis=1)).ravel().tolist(),
    ))


@pytest.mark.parametrize("n_cells", [3000, 6000])
def test_n_cells_returns_exactly_that_many_aligned_cells(n_cells):
    data = m3.datasets.liu_demo(n_cells=n_cells)

    assert data.n_cells == n_cells
    assert data.obs.shape[0] == n_cells
    assert list(data.obs.index) == list(range(n_cells))
    for mod, n_features in (("rna", 1000), ("adt", 192)):
        assert data.modalities[mod].shape == (n_cells, n_features)
        assert data.present[mod].shape == (n_cells,)
        assert len(data.var[mod]) == n_features


@pytest.mark.parametrize("n_cells", [3000, 6000])
def test_n_cells_draws_the_same_number_of_cells_from_every_batch(n_cells):
    # m3 trains on a batch-balanced subset by default, so an unbalanced subsample
    # would have part of its cells discarded before training.
    data = m3.datasets.liu_demo(n_cells=n_cells)

    assert data.obs["batch"].value_counts().to_dict() == {
        "B1": n_cells // 3, "B2": n_cells // 3, "B3": n_cells // 3}


@pytest.mark.parametrize("n_cells", [3000, 6000])
def test_n_cells_keeps_every_cell_type_of_every_batch(full, n_cells):
    # Proportional sampling alone would cut B2's 35 pDCs to a handful at 3,000 cells.
    data = m3.datasets.liu_demo(n_cells=n_cells)

    before = full.obs.groupby(["batch", "mergedcelltype"], observed=True).size()
    after = data.obs.groupby(["batch", "mergedcelltype"], observed=True).size()
    assert set(after.index) == set(before.index)
    short = {k: (int(after[k]), int(before[k])) for k in before.index
             if after[k] < min(before[k], 50)}
    assert not short, f"(batch, cell type) -> (kept, available): {short}"


def test_n_cells_rows_are_rows_of_the_full_demo_in_the_original_order(full):
    # Catches obs and counts being subset with different indices: every kept cell
    # must match a full-demo cell on its labels and on both modalities' totals.
    data = m3.datasets.liu_demo(n_cells=3000)

    remaining = iter(_row_keys(full))
    missing = [key for key in _row_keys(data) if key not in remaining]
    assert not missing, f"{len(missing)} cells are not an in-order subset, e.g. {missing[0]}"


def test_n_cells_gives_the_same_cells_on_every_call():
    first = m3.datasets.liu_demo(n_cells=3000)
    second = m3.datasets.liu_demo(n_cells=3000)

    pd.testing.assert_frame_equal(first.obs, second.obs)
    for mod in ("rna", "adt"):
        assert (first.modalities[mod] != second.modalities[mod]).nnz == 0


def test_omitting_n_cells_still_returns_the_whole_demo(full):
    assert m3.datasets.liu_demo(n_cells=None).n_cells == full.n_cells == 30534


@pytest.mark.parametrize("n_cells", [0, -5, 300, 25000, 30534, 10**6])
def test_n_cells_outside_the_supported_range_is_refused(n_cells):
    # 300 cannot hold the per-cell-type minimum; 25,000 needs 8,334 cells from B1,
    # which has 7,163.
    with pytest.raises(ValueError, match="n_cells must be between"):
        m3.datasets.liu_demo(n_cells=n_cells)


@pytest.mark.parametrize("n_cells", [3000.5, "3000", True])
def test_n_cells_that_is_not_a_whole_number_is_refused(n_cells):
    with pytest.raises(TypeError, match="n_cells must be an integer"):
        m3.datasets.liu_demo(n_cells=n_cells)
