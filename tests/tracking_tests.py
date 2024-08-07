import pandas as pd
from unittest.mock import MagicMock, patch
from src.tracking import get_experiment_id

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
