import mlflow

def mlflow_log_search(search):
  """Logs the results of a GridSearchCV to MLflow.

  Args:
    grid_search: A fitted GridSearchCV object.
  """

  # Log the best parameters
  mlflow.log_params(search.best_params_)

  # Log the best metric
  mlflow.log_metric("best_score", search.best_score_)

  # Log the best estimator
  mlflow.sklearn.log_model(search.best_estimator_, "model")


import mlflow
import pandas as pd

def find_best_run_id_by_name(experiment_name):
  """Finds the run ID with the best R2 score in the specified experiment.

  Args:
    experiment_name: The name of the MLflow experiment.

  Returns:
    The run ID of the run with the best R2 score.
  """

  # Get the experiment ID from the experiment name
  experiment = mlflow.get_experiment_by_name(experiment_name)
  experiment_id = experiment.experiment_id

  # Search for all runs in the experiment, sorted by R2 score descending
  df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.R2 DESC"])

  # Extract the run ID of the first (best) run
  best_run_id = df.iloc[0]['run_id']

  return best_run_id