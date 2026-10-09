import numpy as np
import pandas as pd

from kaggle_traffic.models.low_rank.matrix_completion import matrix_complete
from kaggle_traffic.models.low_rank.svd.matrix import build_matrix
from kaggle_traffic.models.low_rank.svd.svd import svd_impute

COLUMNS = ("speed_kmh", "flow_vph")


def blend_frames(frames, weights, columns=COLUMNS) -> pd.DataFrame:
    """Weighted average of imputed frames. Weights are rescaled so they sum to 1."""
    if not frames:
        raise ValueError("ensemble needs at least one frame")
    if len(frames) != len(weights):
        raise ValueError("one weight is required per frame")
    scale = float(np.sum(weights))
    if scale <= 0:
        raise ValueError("weights must sum to a positive number")
    normalized = np.asarray(weights, dtype=float) / scale
    result = frames[0].copy()
    for column in columns:
        total = np.zeros(len(result), dtype=float)
        for frame, weight in zip(frames, normalized):
            aligned = frame[column].reindex(result.index).to_numpy(dtype=float)
            total += weight * aligned
        result[column] = total
    return result


def apply_matrix_imputer(df: pd.DataFrame, imputer) -> pd.DataFrame:
    """Run a station-by-time imputer on speed and flow, keeping observed values."""
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in COLUMNS:
        matrix = build_matrix(result, column)
        completed = imputer(matrix)
        filled = _from_matrix(result, completed)
        observed = df[column].notna().to_numpy()
        original = df[column].to_numpy(dtype=float)
        result[column] = np.where(observed, original, filled)
    return result


def component_fills(df: pd.DataFrame):
    """Rank-5 SVD and soft-impute at shrinkage 0.05, in that weight order."""
    return [
        apply_matrix_imputer(df, lambda matrix: svd_impute(matrix, rank=5)),
        apply_matrix_imputer(df, lambda matrix: matrix_complete(matrix, shrinkage=0.05)),
    ]


def ensemble_impute(df: pd.DataFrame, weights=(0.5, 0.5)) -> pd.DataFrame:
    """Blend truncated SVD and soft-impute.

    Weights are SVD first, then matrix completion. On a scored training day,
    an even mix beat either method by itself.
    """
    parts = component_fills(df)
    blended = blend_frames(parts, weights)
    for column in COLUMNS:
        observed = df[column].notna().to_numpy()
        blended[column] = np.where(observed, df[column].to_numpy(dtype=float), blended[column].to_numpy(dtype=float))
    blended["timestamp"] = pd.to_datetime(blended["timestamp"], utc=True)
    return blended


def _from_matrix(frame: pd.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
    lookup = matrix.stack()
    lookup.index = lookup.index.set_names(["station_id", "timestamp"])
    keys = pd.MultiIndex.from_arrays([frame["station_id"].to_numpy(), frame["timestamp"].to_numpy()])
    return lookup.reindex(keys).to_numpy(dtype=float)
