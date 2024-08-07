import pandas as pd
from unittest.mock import MagicMock, patch
from src.tracking import mlflow_log_search, find_best_run_id_by_name, get_experiment_id

# def test_mlflow_log_search():
#     """Test logging of GridSearchCV results to MLflow."""
#     search = MagicMock()
#     search.best_params_ = {'param1': 1, 'param2': 2}
#     search.best_score_ = 0.9
#     search.best_estimator_ = MagicMock()

#     with patch("mlflow.log_params") as mock_log_params, \
#          patch("mlflow.log_metric") as mock_log_metric, \
#          patch("mlflow.sklearn.log_model") as mock_log_model:
#         mlflow_log_search(search)

#         mock_log_params.assert_called_once_with({'param1': 1, 'param2': 2})
#         mock_log_metric.assert_called_once_with("best_score", 0.9)
#         mock_log_model.assert_called_once_with(search.best_estimator_, "model")

# def test_find_best_run_id_by_name():
#     """Test finding the run ID with the best R2 score."""
#     experiment_name = "building-energy-prediction-evaluation"
#     mock_experiment = MagicMock()
#     mock_experiment.experiment_id = "1234"

#     mock_runs = pd.DataFrame({
#         'run_id': ['run1', 'run2'],
#         'metrics.test_r2': [0.8, 0.9]
#     })

#     with patch("mlflow.get_experiment_by_name", return_value=mock_experiment) as mock_get_experiment_by_name, \
#          patch("mlflow.search_runs", return_value=mock_runs) as mock_search_runs:
#         best_run_id = find_best_run_id_by_name()
#         assert best_run_id == "run2", "The best run ID is not correctly identified."

def test_get_experiment_id():
    """Test retrieving or creating an experiment ID."""
    exp_name = "test-experiment"

    # Case where experiment does not exist
    with patch("mlflow.get_experiment_by_name", return_value=None) as mock_get_experiment_by_name, \
         patch("mlflow.create_experiment", return_value="new_id") as mock_create_experiment:
        exp_id = get_experiment_id(exp_name)
        assert exp_id == "new_id", "Experiment ID creation did not return the expected ID."

    # Case where experiment exists
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "existing_id"
    with patch("mlflow.get_experiment_by_name", return_value=mock_experiment) as mock_get_experiment_by_name:
        exp_id = get_experiment_id(exp_name)
        assert exp_id == "existing_id", "Existing experiment ID was not correctly returned."
