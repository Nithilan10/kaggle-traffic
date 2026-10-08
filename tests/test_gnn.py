import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.graph.gnn import gnn_impute


def _line(times=24, stations=7):
    timestamps = pd.date_range("2030-06-01", periods=times, freq="5min", tz="UTC")
    rows = []
    for timestamp in timestamps:
        for index in range(stations):
            speed = index * 10.0
            rows.append(
                {
                    "station_id": f"s{index}",
                    "timestamp": timestamp,
                    "milepost": float(index),
                    "speed_kmh": speed,
                    "flow_vph": speed * 5,
                    "occupancy": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_gnn_fills_a_hidden_station_from_its_neighbors():
    frame = _line()
    hidden = frame["station_id"] == "s3"
    frame.loc[hidden, ["speed_kmh", "flow_vph"]] = np.nan
    before = frame.copy()

    filled = gnn_impute(frame, epochs=120, hidden=16, seed=0)

    middle = filled.loc[hidden, "speed_kmh"]
    assert np.isfinite(middle).all()
    assert middle.mean() == pytest.approx(30.0, abs=3.0)
    assert filled.loc[~hidden, "speed_kmh"].equals(before.loc[~hidden, "speed_kmh"])
    pd.testing.assert_frame_equal(frame, before)


def test_gnn_rejects_a_frame_with_no_observed_speed_and_flow():
    frame = _line(times=2)
    frame[["speed_kmh", "flow_vph"]] = np.nan
    with pytest.raises(ValueError, match="no observed entries"):
        gnn_impute(frame, epochs=1)
