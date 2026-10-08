import numpy as np
import pandas as pd


def matrix_complete(
    matrix: pd.DataFrame,
    shrinkage: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Fill missing entries by soft-impute, the nuclear-norm matrix completion method.

    ``shrinkage`` is the fraction of the leading singular value removed from every
    singular value. Components that fall to zero are dropped, so the rank is chosen
    by the data. Observed entries stay fixed.
    """
    if not 0 <= shrinkage < 1:
        raise ValueError("shrinkage must be in [0, 1)")

    values = matrix.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError("matrix has no observed entries")

    column_mean = _observed_column_mean(values, observed)
    residual = values - column_mean
    estimate = np.zeros(values.shape, dtype=np.float64)
    threshold = _shrinkage_threshold(np.where(observed, residual, 0.0), shrinkage)
    rank = 0

    for _ in range(max_iter):
        filled = np.where(observed, residual, estimate)
        low_rank, rank = _soft_threshold(filled, threshold)
        change = np.linalg.norm(low_rank - estimate)
        scale = np.linalg.norm(estimate) + 1e-12
        estimate = low_rank
        if change / scale < tol:
            break

    completed = np.where(observed, values, estimate + column_mean)
    result = pd.DataFrame(completed, index=matrix.index, columns=matrix.columns)
    result.attrs["rank"] = rank
    result.attrs["shrinkage"] = shrinkage
    return result


def _shrinkage_threshold(filled: np.ndarray, shrinkage: float) -> float:
    _, singular_values, _ = np.linalg.svd(filled, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0.0
    return float(shrinkage) * float(singular_values[0])


def _soft_threshold(filled: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    left, singular_values, right = np.linalg.svd(filled, full_matrices=False)
    kept = np.maximum(singular_values - threshold, 0.0)
    active = kept > 0
    rank = int(active.sum())
    if rank == 0:
        return np.zeros_like(filled), 0
    low_rank = (left[:, active] * kept[active]) @ right[active, :]
    return low_rank, rank


def _observed_column_mean(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    col_count = observed.sum(axis=0)
    col_sum = np.nansum(values, axis=0)
    global_mean = float(np.nansum(values) / observed.sum())
    column_mean = np.full(values.shape[1], global_mean, dtype=np.float64)
    np.divide(col_sum, col_count, out=column_mean, where=col_count > 0)
    return column_mean
