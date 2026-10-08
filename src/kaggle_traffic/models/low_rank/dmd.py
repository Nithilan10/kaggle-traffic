from typing import Optional

import numpy as np
import pandas as pd

from kaggle_traffic.models.low_rank.svd.svd import svd_impute


def dmd_reconstruct(matrix: pd.DataFrame, rank: int = 10) -> pd.DataFrame:
    """Reconstruct a complete station-by-time matrix with exact dynamic mode decomposition.

    Columns must be consecutive timestamps. Each column is one spatial snapshot,
    and the next column is the snapshot one step later.
    """
    snapshots = np.asarray(matrix.to_numpy(), dtype=np.float64)
    if not np.isfinite(snapshots).all():
        raise ValueError("DMD reconstruct expects a complete matrix")

    n_stations, n_times = snapshots.shape
    if n_times < 2:
        raise ValueError("DMD needs at least two timestamps")

    earlier = snapshots[:, :-1]
    later = snapshots[:, 1:]
    left, singular_values, right = np.linalg.svd(earlier, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return pd.DataFrame(snapshots, index=matrix.index, columns=matrix.columns)

    keep = singular_values > singular_values[0] * 1e-10
    max_rank = int(keep.sum())
    rank = int(max(1, min(rank, max_rank)))
    left_r = left[:, :rank]
    singular_r = singular_values[:rank]
    right_r = right[:rank, :].conj().T
    reduced_step = right_r / singular_r

    reduced_operator = left_r.conj().T @ later @ reduced_step
    eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
    modes = later @ reduced_step @ eigenvectors

    powers = _eigenvalue_powers(eigenvalues, n_times)
    design = np.einsum("ni,it->nti", modes, powers).reshape(n_stations * n_times, rank)
    amplitudes, *_ = np.linalg.lstsq(design, snapshots.reshape(-1), rcond=None)
    reconstructed = np.real(design @ amplitudes).reshape(n_stations, n_times)
    return pd.DataFrame(reconstructed, index=matrix.index, columns=matrix.columns)


def dmd_impute(
    matrix: pd.DataFrame,
    rank: int = 10,
    n_iter: int = 5,
    initial: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Fill missing entries from a DMD reconstruction. Observed entries stay fixed.

    The time-of-day level is taken from observed values in each timestamp.
    DMD is fit to the deviation from that level, starting from an SVD completion.
    """
    original = matrix.to_numpy(dtype=np.float64)
    observed = np.isfinite(original)
    column_mean = _observed_column_mean(original)
    if initial is None:
        current = svd_impute(matrix, rank=rank)
    else:
        current = initial.copy()

    for _ in range(n_iter):
        residual = current.to_numpy(dtype=np.float64) - column_mean
        residual_frame = pd.DataFrame(residual, index=current.index, columns=current.columns)
        reconstructed = dmd_reconstruct(residual_frame, rank=rank).to_numpy(dtype=np.float64) + column_mean
        updated = np.where(observed, original, reconstructed)
        current = pd.DataFrame(updated, index=matrix.index, columns=matrix.columns)
    return current


def _observed_column_mean(values: np.ndarray) -> np.ndarray:
    observed = np.isfinite(values)
    col_count = observed.sum(axis=0)
    col_sum = np.nansum(values, axis=0)
    global_mean = float(np.nansum(values) / observed.sum())
    column_mean = np.full(values.shape[1], global_mean, dtype=np.float64)
    np.divide(col_sum, col_count, out=column_mean, where=col_count > 0)
    return column_mean


def _eigenvalue_powers(eigenvalues: np.ndarray, n_times: int) -> np.ndarray:
    powers = np.empty((eigenvalues.size, n_times), dtype=np.complex128)
    powers[:, 0] = 1
    if n_times == 1:
        return powers
    steps = np.arange(1, n_times, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        powers[:, 1:] = np.exp(np.log(eigenvalues.astype(np.complex128))[:, None] * steps[None, :])
    powers[~np.isfinite(powers)] = 0
    return powers
