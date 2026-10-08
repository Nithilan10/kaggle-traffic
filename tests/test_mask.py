import numpy as np
import pandas as pd

from kaggle_traffic.evaluation.synthetic_mask import mask


def _observations(n_complete, n_incomplete=0):
    n = n_complete + n_incomplete
    frame = pd.DataFrame(
        {
            "station_id": [f"s{i}" for i in range(n)],
            "speed_kmh": np.arange(n, dtype=float) + 10,
            "flow_vph": np.arange(n, dtype=float) * 100 + 5,
            "occupancy": np.linspace(0.1, 0.5, n),
        }
    )
    if n_incomplete:
        frame.loc[frame.index[-n_incomplete:], "speed_kmh"] = np.nan
    return frame


def test_mask_hides_twenty_percent_of_complete_rows_and_keeps_the_answers():
    original = _observations(10, n_incomplete=3)
    before = original.copy()

    masked, truth, indices = mask(original)

    assert len(indices) == 2
    assert list(truth.index) == list(indices)
    assert masked.loc[indices, ["speed_kmh", "flow_vph"]].isna().all().all()
    pd.testing.assert_frame_equal(truth, before.loc[indices, ["speed_kmh", "flow_vph"]])
    pd.testing.assert_series_equal(masked["occupancy"], before["occupancy"])
    pd.testing.assert_frame_equal(original, before)


def test_mask_leaves_incomplete_rows_out_of_the_draw():
    original = _observations(10, n_incomplete=3)
    incomplete = original.index[original["speed_kmh"].isna()]

    _, _, indices = mask(original)

    assert incomplete.isin(indices).sum() == 0


def test_mask_is_reproducible():
    original = _observations(15)
    first, _, first_index = mask(original)
    second, _, second_index = mask(original)
    assert list(first_index) == list(second_index)
    pd.testing.assert_frame_equal(first, second)


def test_mask_draws_nothing_when_twenty_percent_rounds_to_zero():
    original = _observations(4)
    masked, truth, indices = mask(original)
    assert len(indices) == 0
    assert truth.empty
    pd.testing.assert_series_equal(masked["speed_kmh"], original["speed_kmh"])
