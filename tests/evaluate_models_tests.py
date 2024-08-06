# import pytest
# from unittest.mock import MagicMock, patch
# import pandas as pd
# from sklearn.linear_model import LinearRegression

# from src.evaluate import evaluate_models

# # Mock the dependencies
# mock_eval_metrics = MagicMock(return_value={'r2': 0.85, 'mse': 0.1})
# mock_get_experiment_id = MagicMock(return_value=1)
# mock_get_current_date = MagicMock(return_value='2024-08-04')
# mock_add_prefix_to_keys = MagicMock()

# # Sample data for testing
# x_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
# y_train = pd.Series([1, 2, 3])
# x_test = pd.DataFrame({'feature1': [2, 3, 4], 'feature2': [5, 6, 7]})
# y_test = pd.Series([2, 3, 4])

# # Sample estimators
# estimators = {
#     'Linear Regression': MagicMock(spec=LinearRegression, predict=MagicMock(return_value=[1, 2, 3]))
# }

# @pytest.fixture
# def mock_mlflow():
#     with patch('mlflow.start_run'), \
#          patch('mlflow.log_params'), \
#          patch('mlflow.log_metrics'), \
#          patch('mlflow.sklearn.log_model'), \
#          patch('mlflow.end_run'):
#         yield

# @pytest.fixture
# def patch_dependencies():
#     with patch('src.evaluate.eval_metrics', mock_eval_metrics), \
#          patch('src.evaluate.get_experiment_id', mock_get_experiment_id), \
#          patch('src.evaluate.get_current_date', mock_get_current_date), \
#          patch('src.evaluate.add_prefix_to_keys', mock_add_prefix_to_keys):
#         yield

# def test_evaluate_models(mock_mlflow, patch_dependencies):
#     mock_estimator = MagicMock()
#     mock_regressor = MagicMock()
#     mock_regressor.get_params.return_value = {"fit_intercept": True}
#     mock_estimator.regressor = mock_regressor

#     estimators = {"MockModel": mock_estimator}
#     x_train, x_test, y_train, y_test = MagicMock(), MagicMock(), MagicMock(), MagicMock()

#     best_model_name, best_r2_score = evaluate_models(estimators, x_train, x_test, y_train, y_test)

#     # Check if the function returned the correct best model
#     assert best_model_name == 'Linear Regression'
#     assert best_r2_score == 0.85

#     # Check if eval_metrics was called with correct arguments
#     mock_eval_metrics.assert_any_call(y_train, [1, 2, 3])
#     mock_eval_metrics.assert_any_call(y_test, [1, 2, 3])

#     # Check if add_prefix_to_keys was called with the correct metrics
#     mock_add_prefix_to_keys.assert_any_call({'r2': 0.85, 'mse': 0.1}, 'train')
#     mock_add_prefix_to_keys.assert_any_call({'r2': 0.85, 'mse': 0.1}, 'test')

