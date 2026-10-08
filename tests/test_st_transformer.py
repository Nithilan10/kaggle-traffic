import numpy as np
import pandas as pd
import pytest
import torch

from kaggle_traffic.models.transformer.spatial_attention import SpatialAttention
from kaggle_traffic.models.transformer.st_transformer import st_impute
from kaggle_traffic.models.transformer.temporal_attention import TemporalAttention


def test_attention_blocks_keep_the_station_time_shape():
    spatial = SpatialAttention(channels=8, heads=2)
    temporal = TemporalAttention(channels=8, heads=2)
    hidden = torch.zeros(5, 4, 8)

    assert spatial(hidden).shape == hidden.shape
    assert temporal(hidden).shape == hidden.shape


def _panel(times=12, stations=7):
    timestamps = pd.date_range("2030-06-01", periods=times, freq="5min", tz="UTC")
    rows = []
    for time_index, timestamp in enumerate(timestamps):
        for station_index in range(stations):
            speed = station_index * 10.0 + time_index
            rows.append(
                {
                    "station_id": f"s{station_index}",
                    "timestamp": timestamp,
                    "milepost": float(station_index),
                    "speed_kmh": speed,
                    "flow_vph": speed * 4,
                    "occupancy": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_transformer_fills_a_hidden_station_using_space_and_time():
    frame = _panel()
    hidden = frame["station_id"] == "s3"
    truth = frame.loc[hidden, "speed_kmh"].to_numpy()
    frame.loc[hidden, ["speed_kmh", "flow_vph"]] = np.nan
    before = frame.copy()

    filled = st_impute(frame, epochs=80, channels=16, heads=2, seed=0)

    prediction = filled.loc[hidden, "speed_kmh"].to_numpy()
    assert np.isfinite(prediction).all()
    assert np.abs(prediction - truth).mean() < 4.0
    pd.testing.assert_series_equal(filled.loc[~hidden, "speed_kmh"], before.loc[~hidden, "speed_kmh"])
    pd.testing.assert_frame_equal(frame, before)


def test_transformer_rejects_a_frame_with_no_observed_speed_and_flow():
    frame = _panel(times=2, stations=3)
    frame[["speed_kmh", "flow_vph"]] = np.nan
    with pytest.raises(ValueError, match="no observed entries"):
        st_impute(frame, epochs=1)
