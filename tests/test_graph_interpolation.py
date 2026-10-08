import numpy as np
import pandas as pd

from kaggle_traffic.models.graph.interpolate import graph_interpolation


def _frame(mileposts, speeds, flows=None, occupancies=None, timestamp="2030-06-01 08:00"):
    n = len(mileposts)
    if flows is None:
        flows = speeds
    if occupancies is None:
        occupancies = [0.1] * n
    return pd.DataFrame(
        {
            "station_id": [f"s{i}" for i in range(n)],
            "timestamp": [timestamp] * n,
            "milepost": mileposts,
            "speed_kmh": speeds,
            "flow_vph": flows,
            "occupancy": occupancies,
        }
    )


def test_graph_interpolation_weights_a_gap_by_milepost_not_by_hop_count():
    original = _frame([0.0, 1.0, 10.0], [0.0, np.nan, 100.0])
    before = original.copy()

    filled = graph_interpolation(original)

    assert filled.loc[1, "speed_kmh"] == 10.0
    assert filled.loc[0, "speed_kmh"] == 0.0
    assert filled.loc[2, "flow_vph"] == 100.0
    pd.testing.assert_frame_equal(original, before)


def test_graph_interpolation_fills_a_run_of_missing_stations():
    filled = graph_interpolation(_frame([0.0, 2.0, 3.0, 10.0], [0.0, np.nan, np.nan, 100.0]))

    assert filled.loc[1, "speed_kmh"] == 20.0
    assert filled.loc[2, "speed_kmh"] == 30.0


def test_graph_interpolation_copies_the_nearest_reading_past_the_ends():
    filled = graph_interpolation(_frame([0.0, 5.0, 9.0], [np.nan, 8.0, 12.0]))

    assert filled.loc[0, "speed_kmh"] == 8.0


def test_graph_interpolation_keeps_timestamps_separate():
    first = _frame([0.0, 10.0], [0.0, np.nan], timestamp="2030-06-01 08:00")
    second = _frame([0.0, 10.0], [50.0, np.nan], timestamp="2030-06-01 08:05")
    second.index = [2, 3]
    filled = graph_interpolation(pd.concat([first, second]))

    assert filled.loc[1, "speed_kmh"] == 0.0
    assert filled.loc[3, "speed_kmh"] == 50.0


def test_graph_interpolation_leaves_an_empty_timestamp_missing():
    filled = graph_interpolation(_frame([0.0, 4.0], [np.nan, np.nan]))

    assert filled["speed_kmh"].isna().all()
