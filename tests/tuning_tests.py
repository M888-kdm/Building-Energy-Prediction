from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.tuning import fine_tune_models, rmse


def test_rmse():
    """Test the RMSE calculation function."""
    y_actual = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    result = rmse(y_actual, y_pred)
    expected = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    assert np.isclose(result, expected), "RMSE calculation is incorrect."
