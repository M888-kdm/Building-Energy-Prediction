import numpy as np
import pandas as pd
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

from src.metrics import eval_metrics

# Sample data
y_actual = np.array([3, -0, 2, 7])
y_pred = np.array([2.5, 0, 2, 8])


def test_eval_metrics_rmse():
    expected_rmse = mean_squared_error(y_actual, y_pred) ** 0.5
    metrics = eval_metrics(y_actual, y_pred)
    assert np.isclose(
        metrics["rmse"], expected_rmse
    ), f"Expected RMSE: {expected_rmse}, but got {metrics['rmse']}"


def test_eval_metrics_mae():
    expected_mae = mean_absolute_error(y_actual, y_pred)
    metrics = eval_metrics(y_actual, y_pred)
    assert np.isclose(
        metrics["mae"], expected_mae
    ), f"Expected MAE: {expected_mae}, but got {metrics['mae']}"


def test_eval_metrics_r2():
    expected_r2 = r2_score(y_actual, y_pred)
    metrics = eval_metrics(y_actual, y_pred)
    assert np.isclose(
        metrics["r2"], expected_r2
    ), f"Expected R2: {expected_r2}, but got {metrics['r2']}"


def test_eval_metrics_max_error():
    expected_max_error = max_error(y_actual, y_pred)
    metrics = eval_metrics(y_actual, y_pred)
    assert np.isclose(
        metrics["maxerror"], expected_max_error
    ), f"Expected max error: {expected_max_error}, but got {metrics['maxerror']}"
