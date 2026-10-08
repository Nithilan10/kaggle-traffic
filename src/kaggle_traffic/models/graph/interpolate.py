import numpy as np
import pandas as pd

TRAFFIC_COLUMNS = ["speed_kmh", "flow_vph", "occupancy"]


def graph_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps along the mainline path, using distance between stations.

    Stations ordered by milepost are the nodes. Each hop to the next station is
    an edge. On that path the harmonic fill is linear in milepost, so a long
    gap counts for more than a short hop. Observed values stay fixed.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    result = df.copy()

    for labels in df.groupby("timestamp", sort=False).groups.values():
        labels = pd.Index(labels)
        milepost = result.loc[labels, "milepost"].to_numpy(dtype=float)
        order = np.argsort(milepost, kind="mergesort")
        ordered_milepost = milepost[order]
        for column in TRAFFIC_COLUMNS:
            values = result.loc[labels, column].to_numpy(dtype=float)
            filled = _interpolate_along_path(ordered_milepost, values[order])
            restored = np.empty_like(filled)
            restored[order] = filled
            result.loc[labels, column] = pd.Series(restored, index=labels)
    return result


def _interpolate_along_path(milepost: np.ndarray, values: np.ndarray) -> np.ndarray:
    observed = np.isfinite(values)
    if observed.sum() == 0:
        return values.copy()

    positions = milepost[observed]
    readings = values[observed]
    if len(positions) >= 2 and np.any(np.diff(positions) <= 0):
        _, unique_index = np.unique(positions, return_index=True)
        unique_index.sort()
        positions = positions[unique_index]
        readings = readings[unique_index]
    if len(positions) == 1:
        return np.full(values.shape, readings[0], dtype=float)

    filled = np.interp(milepost, positions, readings)
    filled[observed] = values[observed]
    return filled
