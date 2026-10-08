import numpy as np
import pandas as pd

from kaggle_traffic.models.linear_interpolation.spatial import spatial_interpolation
from kaggle_traffic.models.linear_interpolation.spatiotemporal import spatiotemporal_interpolation
from kaggle_traffic.models.linear_interpolation.temporal import temporal_interpolation


def _times(n):
    return pd.date_range("2030-06-01", periods=n, freq="5min", tz="UTC")


def _frame(rows):
    return pd.DataFrame(rows)


def test_temporal_interpolation_fills_an_interior_gap_per_station():
    times = _times(3)
    original = _frame(
        {
            "station_id": ["A", "A", "A", "B", "B", "B"],
            "timestamp": list(times) + list(times),
            "milepost": [0, 0, 0, 1, 1, 1],
            "speed_kmh": [10.0, np.nan, 40.0, np.nan, 5.0, 9.0],
            "flow_vph": [0.0, np.nan, 30.0, np.nan, 2.0, 4.0],
            "occupancy": [0.1, np.nan, 0.4, np.nan, 0.2, 0.4],
        }
    )
    filled = temporal_interpolation(original)

    speeds = filled.set_index(["station_id", "timestamp"])["speed_kmh"]
    assert speeds.loc[("A", times[1])] == 25.0
    assert speeds.loc[("B", times[0])] == 5.0
    assert speeds.loc[("A", times[0])] == 10.0
    assert speeds.loc[("B", times[2])] == 9.0
    assert original["speed_kmh"].isna().sum() == 2


def test_temporal_interpolation_does_not_change_the_input():
    times = _times(3)
    original = _frame(
        {
            "station_id": ["A", "A", "A"],
            "timestamp": list(times),
            "milepost": [0, 0, 0],
            "speed_kmh": [10.0, np.nan, 40.0],
            "flow_vph": [0.0, np.nan, 30.0],
            "occupancy": [0.1, np.nan, 0.4],
        }
    )
    before = original.copy()
    temporal_interpolation(original)
    pd.testing.assert_frame_equal(original, before)


def test_spatial_interpolation_follows_milepost_within_one_timestamp():
    times = _times(2)
    original = _frame(
        {
            "station_id": ["C", "A", "B", "C", "A", "B"],
            "timestamp": [times[0], times[0], times[0], times[1], times[1], times[1]],
            "milepost": [20.0, 0.0, 10.0, 20.0, 0.0, 10.0],
            "speed_kmh": [30.0, 10.0, np.nan, 8.0, np.nan, np.nan],
            "flow_vph": [3.0, 1.0, np.nan, 4.0, np.nan, np.nan],
            "occupancy": [0.3, 0.1, np.nan, 0.2, np.nan, np.nan],
        }
    )
    filled = spatial_interpolation(original).set_index(["station_id", "timestamp"])

    assert filled.loc[("B", times[0]), "speed_kmh"] == 20.0
    assert filled.loc[("A", times[1]), "speed_kmh"] == 8.0
    assert filled.loc[("B", times[1]), "speed_kmh"] == 8.0
    assert filled.loc[("C", times[0]), "flow_vph"] == 3.0


def test_spatiotemporal_blend_uses_the_weight_and_falls_back_when_one_side_is_missing():
    times = _times(3)
    frame = _frame(
        {
            "station_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "timestamp": list(times) * 3,
            "milepost": [0, 0, 0, 10, 10, 10, 20, 20, 20],
            "speed_kmh": [0.0, 0.0, 0.0, 0.0, np.nan, 40.0, 0.0, 20.0, 0.0],
            "flow_vph": [0.0, 0.0, 0.0, 0.0, np.nan, 40.0, 0.0, 20.0, 0.0],
            "occupancy": [0.0, 0.0, 0.0, 0.0, np.nan, 0.4, 0.0, 0.2, 0.0],
        }
    )

    half = spatiotemporal_interpolation(
        temporal_interpolation, spatial_interpolation, frame, temporal_weight=0.5
    )
    full_temporal = spatiotemporal_interpolation(
        temporal_interpolation, spatial_interpolation, frame, temporal_weight=1
    )
    full_spatial = spatiotemporal_interpolation(
        temporal_interpolation, spatial_interpolation, frame, temporal_weight=0
    )
    temporal = temporal_interpolation(frame)
    spatial = spatial_interpolation(frame)

    blended = half.set_index(["station_id", "timestamp"])
    # B at the middle time is 20 along its own series and 10 between A and C.
    assert blended.loc[("B", times[1]), "speed_kmh"] == 15.0

    pd.testing.assert_series_equal(
        full_temporal.sort_index()["speed_kmh"].reset_index(drop=True),
        temporal.sort_values(["station_id", "timestamp"])["speed_kmh"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        full_spatial.sort_values(["station_id", "timestamp"])["speed_kmh"].reset_index(drop=True),
        spatial.sort_values(["station_id", "timestamp"])["speed_kmh"].reset_index(drop=True),
    )
