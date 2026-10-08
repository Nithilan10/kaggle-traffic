import numpy as np
import pandas as pd
import pytest

from kaggle_traffic.evaluation.metrics import rmse


def test_rmse_of_a_known_error():
    error = rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert error == pytest.approx(np.sqrt(12.5))


def test_rmse_is_zero_when_prediction_matches():
    values = np.array([10.0, 20.0, 30.0])
    assert rmse(values, values.copy()) == pytest.approx(0.0)


def test_rmse_uses_index_alignment_for_series():
    truth = pd.Series([0.0, 0.0], index=["b", "a"])
    prediction = pd.Series([3.0, 4.0], index=["a", "b"])
    assert rmse(truth, prediction) == pytest.approx(np.sqrt((16.0 + 9.0) / 2))
