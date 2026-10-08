import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.low_rank.svd.matrix import build_matrix
from kaggle_traffic.models.low_rank.svd.svd import svd_impute


def _times():
    return pd.date_range("2030-06-01 00:10", periods=3, freq="5min", tz="UTC")


def test_build_matrix_pivots_stations_by_sorted_time():
    times = _times()
    frame = pd.DataFrame(
        {
            "station_id": ["B", "A", "A", "B", "A", "B"],
            "timestamp": [times[2], times[0], times[2], times[0], times[1], times[1]],
            "speed_kmh": [6.0, 1.0, np.nan, 4.0, 2.0, 5.0],
        }
    )
    matrix = build_matrix(frame, "speed_kmh")

    assert set(matrix.index) == {"A", "B"}
    assert matrix.columns.is_monotonic_increasing
    assert list(matrix.columns) == list(times)
    assert matrix.loc["A", times[0]] == 1.0
    assert matrix.loc["B", times[2]] == 6.0
    assert np.isnan(matrix.loc["A", times[2]])


def test_svd_recovers_a_low_rank_matrix_and_keeps_observed_entries():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(30, 3)) @ rng.normal(size=(3, 40))
    hidden = rng.random(truth.shape) < 0.3
    matrix = pd.DataFrame(truth, index=[f"s{i}" for i in range(30)])
    matrix = matrix.mask(hidden)

    filled = svd_impute(matrix, rank=3)

    assert filled.index.equals(matrix.index)
    assert filled.columns.equals(matrix.columns)
    np.testing.assert_allclose(filled.to_numpy()[~hidden], truth[~hidden])
    assert np.mean(np.abs(filled.to_numpy()[hidden] - truth[hidden])) < 0.5
    assert matrix.isna().sum().sum() == hidden.sum()


def test_svd_returns_a_complete_matrix_unchanged():
    values = np.arange(12, dtype=float).reshape(3, 4)
    matrix = pd.DataFrame(values, index=list("abc"))
    filled = svd_impute(matrix, rank=2)
    np.testing.assert_allclose(filled.to_numpy(), values)


def test_svd_clamps_a_rank_above_the_matrix_shape():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 10))
    hidden = np.zeros(truth.shape, dtype=bool)
    hidden[::2, ::3] = True
    matrix = pd.DataFrame(truth).mask(hidden)

    filled = svd_impute(matrix, rank=100)

    assert np.mean(np.abs(filled.to_numpy()[hidden] - truth[hidden])) < 0.5


def test_svd_rank_zero_matches_rank_one():
    matrix = pd.DataFrame(np.arange(20, dtype=float).reshape(4, 5))
    matrix.iloc[0, 1] = np.nan
    np.testing.assert_allclose(
        svd_impute(matrix, rank=0).to_numpy(),
        svd_impute(matrix, rank=1).to_numpy(),
    )


def test_svd_rejects_a_matrix_with_no_observations():
    matrix = pd.DataFrame(np.full((3, 4), np.nan))
    with pytest.raises(ValueError, match="no observed entries"):
        svd_impute(matrix)
