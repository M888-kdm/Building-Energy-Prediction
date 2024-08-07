import numpy as np
from unittest.mock import  patch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.tuning import fine_tune_models, rmse

# @patch('src.tuning.mlflow_log_search')
# def test_fine_tune_models(mock_mlflow_log_search):
#     # Setup mock return values
#     mock_mlflow_log_search.return_value = {"param1": 0.1, "param2": 0.01}

#     # Define the estimator_params with estimators and parameter grids
#     estimator_params = {
#         "LinearRegression": {
#             "estimator": LinearRegression(),
#             "params": {
#                 "estimator__regressor__fit_intercept": [True, False]
#             }
#         },
#         "RandomForest": {
#             "estimator": RandomForestRegressor(),
#             "params": {
#                 "estimator__regressor__n_estimators": [10, 50, 100],
#                 "estimator__regressor__max_depth": [None, 10, 20]
#             }
#         }
#     }

#     # Create synthetic data for testing
#     x_train = pd.DataFrame(np.random.randn(100, 2), columns=["feature1", "feature2"])
#     y_train = pd.Series(np.random.randn(100))

#     # Call the function to be tested
#     result = fine_tune_models(estimator_params, x_train, y_train)

#     # Assertions and other test logic here
#     assert "LinearRegression" in result, "LinearRegression model should be present in the result."
#     assert "RandomForest" in result, "RandomForest model should be present in the result."
    
def test_rmse():
    """Test the RMSE calculation function."""
    y_actual = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    result = rmse(y_actual, y_pred)
    expected = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    assert np.isclose(result, expected), "RMSE calculation is incorrect."
