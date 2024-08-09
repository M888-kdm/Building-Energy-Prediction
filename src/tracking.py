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


def find_best_run_id_by_name(experiment_name):
  """Finds the run ID with the best R2 score in the specified experiment.

  Args:
    experiment_name: The name of the MLflow experiment.

  Returns:
    The run ID of the run with the best R2 score.
  """

  # Search for all runs in the experiment, sorted by R2 score descending
  df = mlflow.search_runs(experiment_names=[experiment_name])
  df = df.sort_values('metrics.test_r2', ascending=False)

  # Extract the run ID of the first (best) run and return it
  return df.iloc[0]['run_id']

def get_experiment_id(exp_name):
  """Retrieves the experiment ID for a given experiment name.

  If the experiment doesn't exist, it creates a new experiment with the given name.

  Args:
    exp_name: The name of the experiment.

  Returns:
    The experiment ID.
  """

  experiment = mlflow.get_experiment_by_name(exp_name)
  if not experiment:
    experiment_id = mlflow.create_experiment(exp_name)
  else:
    experiment_id = experiment.experiment_id
  return experiment_id
