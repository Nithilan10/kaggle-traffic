import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.models.ensemble import blend_frames


def _frame(speed, flow):
    return pd.DataFrame(
        {
            "station_id": ["a", "b"],
            "speed_kmh": speed,
            "flow_vph": flow,
        }
    )


def test_blend_is_the_weighted_average():
    first = _frame([0.0, 10.0], [0.0, 100.0])
    second = _frame([10.0, 30.0], [40.0, 200.0])
    blended = blend_frames([first, second], weights=(1, 3))

    assert blended.loc[0, "speed_kmh"] == pytest.approx(7.5)
    assert blended.loc[1, "flow_vph"] == pytest.approx(175.0)


def test_blend_matches_frames_by_index_rather_than_row_order():
    first = pd.DataFrame({"speed_kmh": [0.0, 10.0], "flow_vph": [0.0, 0.0]}, index=["b", "a"])
    second = pd.DataFrame({"speed_kmh": [10.0, 30.0], "flow_vph": [0.0, 0.0]}, index=["a", "b"])
    blended = blend_frames([first, second], weights=(0.5, 0.5))

    assert blended.loc["b", "speed_kmh"] == pytest.approx(15.0)
    assert blended.loc["a", "speed_kmh"] == pytest.approx(10.0)


def test_blend_rejects_a_weight_vector_of_the_wrong_length():
    with pytest.raises(ValueError, match="one weight"):
        blend_frames([_frame([1.0, 2.0], [3.0, 4.0])], weights=(0.2, 0.8))
