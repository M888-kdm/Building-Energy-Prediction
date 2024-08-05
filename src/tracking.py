import mlflow

def mlflow_log_grid_search(search):
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