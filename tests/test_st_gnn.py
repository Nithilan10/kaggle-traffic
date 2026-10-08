import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.graph.st_gnn import pretrain_stgnn, stgnn_impute


def _line(times=16, stations=7, noise=0.0, seed=0):
    timestamps = pd.date_range("2030-06-01", periods=times, freq="5min", tz="UTC")
    generator = np.random.default_rng(seed)
    rows = []
    for timestamp in timestamps:
        for index in range(stations):
            speed = index * 10.0 + float(generator.normal(0.0, noise))
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


def test_pretrained_stgnn_fills_a_hidden_station_without_seeing_that_day():
    train_days = [_line(seed=day) for day in range(4)]
    target = _line(seed=9)
    hidden = target["station_id"] == "s3"
    truth = target.loc[hidden, "speed_kmh"].mean()
    target.loc[hidden, ["speed_kmh", "flow_vph"]] = np.nan
    before = target.copy()

    weights = pretrain_stgnn(train_days, epochs=8, hidden=16, seed=0)
    filled = stgnn_impute(target, pretrained=weights, finetune_epochs=0, hidden=16, seed=0)

    prediction = filled.loc[hidden, "speed_kmh"]
    assert np.isfinite(prediction).all()
    assert prediction.mean() == pytest.approx(truth, abs=5.0)
    pd.testing.assert_series_equal(filled.loc[~hidden, "speed_kmh"], before.loc[~hidden, "speed_kmh"])
    pd.testing.assert_frame_equal(target, before)


def test_stgnn_rejects_a_frame_with_no_observed_speed_and_flow():
    frame = _line(times=2, stations=3)
    frame[["speed_kmh", "flow_vph"]] = np.nan
    with pytest.raises(ValueError, match="no observed entries"):
        stgnn_impute(frame, finetune_epochs=1)
