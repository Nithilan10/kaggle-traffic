import numpy as np
import pandas as pd


def svd_impute(matrix: pd.DataFrame, rank: int = 10, max_iter: int = 25, tol: float = 1e-4) -> pd.DataFrame:
    """Fill missing entries with a rank-truncated SVD, keeping observed entries fixed."""
    values = matrix.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError("matrix has no observed entries")

    col_count = observed.sum(axis=0)
    col_sum = np.nansum(values, axis=0)
    global_mean = float(np.nansum(values) / observed.sum())
    col_mean = np.full(values.shape[1], global_mean, dtype=np.float64)
    np.divide(col_sum, col_count, out=col_mean, where=col_count > 0)

    residual = values - col_mean
    filled = np.where(observed, residual, 0.0)
    rank = int(max(1, min(rank, *filled.shape)))
    previous_missing = None

    for _ in range(max_iter):
        left, singular_values, right = np.linalg.svd(filled, full_matrices=False)
        low_rank = (left[:, :rank] * singular_values[:rank]) @ right[:rank, :]
        filled = np.where(observed, residual, low_rank)
        if not (~observed).any():
            break
        missing = filled[~observed]
        if previous_missing is not None:
            scale = np.linalg.norm(previous_missing) + 1e-12
            if np.linalg.norm(missing - previous_missing) / scale < tol:
                break
        previous_missing = missing

    completed = np.where(observed, values, filled + col_mean)
    return pd.DataFrame(completed, index=matrix.index, columns=matrix.columns)
