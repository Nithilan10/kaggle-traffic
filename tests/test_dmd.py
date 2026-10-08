import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.low_rank.dmd import dmd_impute, dmd_reconstruct


def _linear_system(n_stations=12, n_times=40):
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(n_stations, 2))
    eigenvalues = np.array([0.95, 0.85])
    amplitudes = np.array([1.0, 0.7])
    steps = np.arange(n_times)
    snapshots = np.real(modes @ (amplitudes[:, None] * eigenvalues[:, None] ** steps))
    index = [f"s{i}" for i in range(n_stations)]
    columns = pd.date_range("2030-06-01", periods=n_times, freq="5min", tz="UTC")
    return snapshots, pd.DataFrame(snapshots, index=index, columns=columns)


def test_dmd_reconstructs_a_linear_system():
    snapshots, matrix = _linear_system()
    reconstructed = dmd_reconstruct(matrix, rank=2)

    assert reconstructed.index.equals(matrix.index)
    assert reconstructed.columns.equals(matrix.columns)
    np.testing.assert_allclose(reconstructed.to_numpy(), snapshots, atol=1e-8)


def test_dmd_reconstruct_returns_a_zero_matrix_unchanged():
    matrix = pd.DataFrame(np.zeros((4, 6)))
    reconstructed = dmd_reconstruct(matrix, rank=3)
    np.testing.assert_allclose(reconstructed.to_numpy(), 0.0)


def test_dmd_reconstruct_rejects_missing_values_and_a_single_snapshot():
    snapshots, matrix = _linear_system()
    broken = matrix.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete matrix"):
        dmd_reconstruct(broken)

    with pytest.raises(ValueError, match="two timestamps"):
        dmd_reconstruct(matrix.iloc[:, :1])


def test_dmd_impute_fills_only_the_holes_of_a_linear_system():
    snapshots, matrix = _linear_system()
    hidden = np.zeros(snapshots.shape, dtype=bool)
    hidden[::4, 1::3] = True
    masked = matrix.mask(hidden)
    before = masked.copy()

    filled = dmd_impute(masked, rank=2)

    np.testing.assert_allclose(filled.to_numpy()[~hidden], snapshots[~hidden])
    assert np.mean(np.abs(filled.to_numpy()[hidden] - snapshots[hidden])) < 0.25
    assert np.isfinite(filled.to_numpy()).all()
    pd.testing.assert_frame_equal(masked, before)


def test_dmd_impute_rejects_a_matrix_with_no_observations():
    matrix = pd.DataFrame(np.full((3, 4), np.nan))
    with pytest.raises(ValueError, match="no observed entries"):
        dmd_impute(matrix)
