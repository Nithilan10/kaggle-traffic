import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.low_rank.matrix_completion import matrix_complete


def test_matrix_completion_recovers_a_low_rank_matrix():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(40, 5)) @ rng.normal(size=(5, 60))
    hidden = rng.random(truth.shape) < 0.4
    matrix = pd.DataFrame(truth, index=[f"s{i}" for i in range(40)])
    matrix = matrix.mask(hidden)
    before = matrix.copy()

    filled = matrix_complete(matrix, shrinkage=0.01)

    assert filled.index.equals(matrix.index)
    assert filled.columns.equals(matrix.columns)
    assert filled.attrs["shrinkage"] == 0.01
    assert filled.attrs["rank"] > 0
    np.testing.assert_allclose(filled.to_numpy()[~hidden], truth[~hidden])
    assert np.mean(np.abs(filled.to_numpy()[hidden] - truth[hidden])) < 0.5
    pd.testing.assert_frame_equal(matrix, before)


def test_matrix_completion_with_no_shrinkage_fills_missing_cells_by_column_mean():
    matrix = pd.DataFrame([[1.0, np.nan], [3.0, 5.0]], index=["a", "b"], columns=["t0", "t1"])
    filled = matrix_complete(matrix, shrinkage=0.0)

    assert filled.loc["a", "t0"] == 1.0
    assert filled.loc["b", "t1"] == 5.0
    assert filled.loc["a", "t1"] == pytest.approx(5.0)
    assert filled.attrs["rank"] == 1


def test_stronger_shrinkage_keeps_fewer_components():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(25, 8)) @ rng.normal(size=(8, 30))
    hidden = rng.random(truth.shape) < 0.2
    matrix = pd.DataFrame(truth).mask(hidden)

    loose = matrix_complete(matrix, shrinkage=0.01)
    tight = matrix_complete(matrix, shrinkage=0.3)

    assert tight.attrs["rank"] < loose.attrs["rank"]


def test_matrix_completion_keeps_a_fully_observed_matrix():
    values = np.arange(12, dtype=float).reshape(3, 4)
    matrix = pd.DataFrame(values)
    filled = matrix_complete(matrix, shrinkage=0.2)
    np.testing.assert_allclose(filled.to_numpy(), values)


def test_matrix_completion_rejects_bad_shrinkage_and_an_empty_observation_set():
    matrix = pd.DataFrame(np.eye(4))
    with pytest.raises(ValueError, match="shrinkage"):
        matrix_complete(matrix, shrinkage=1)
    with pytest.raises(ValueError, match="shrinkage"):
        matrix_complete(matrix, shrinkage=-0.1)

    empty = pd.DataFrame(np.full((2, 3), np.nan))
    with pytest.raises(ValueError, match="no observed entries"):
        matrix_complete(empty, shrinkage=0.1)
