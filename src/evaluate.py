import mlflow
import pandas as pd

from loguru import logger
from metrics import eval_metrics
from tracking import get_experiment_id
from utils import get_current_date, add_prefix_to_keys

def evaluate_models(estimators, x_train, x_test, y_train, y_test):

    """Evaluates and compares a set of machine learning estimators on a given dataset.

    This function trains each estimator on the training data, makes predictions on the testing data, 
    calculates various evaluation metrics, and logs them within an MLflow experiment. Additionally, 
    the function identifies the best performing estimator based on the R2 score and returns it.

    Args:
        estimators (dict): A dictionary containing the estimators to be evaluated.
        The dictionary keys are the estimator names, and the values are the actual scikit-learn estimator objects.
        x_train (pandas.DataFrame): The training data features.
        x_test (pandas.DataFrame): The testing data features.
        y_train (pandas.Series): The training data target variable.
        y_test (pandas.Series): The testing data target variable.

    Returns:
        tuple: A tuple containing the name of the best performing estimator and its R2 score.
    """

    exp_name = "building-energy-prediction-evaluation"
    experiment_id = get_experiment_id(exp_name)

    # Dict of R2 scores for the estimators
    r2_scores = {}

    with mlflow.start_run(run_name=f"Session-{get_current_date()}", experiment_id=experiment_id):
        for estimator_name, estimator in estimators.items():
            with mlflow.start_run(run_name=estimator_name, nested=True, experiment_id=experiment_id): 
                y_train_pred = estimator.predict(x_train)
                y_test_pred = estimator.predict(x_test)

                train_metrics = eval_metrics(y_train, y_train_pred)
                test_metrics = eval_metrics(y_test, y_test_pred)

                logger.info(f"""{estimator_name} performance \n{pd.DataFrame({'train': train_metrics, 'test': test_metrics}).T}""")

                # Add the R2 score of the model to the global dict
                r2_scores[estimator_name] = test_metrics['r2']

                train_metrics = add_prefix_to_keys(train_metrics, "train")
                test_metrics = add_prefix_to_keys(test_metrics, "test")

                # Log the regressor parameters
                mlflow.log_params(estimator.regressor.steps[-1][1].get_params())

                # Log the best metric
                mlflow.log_metrics(train_metrics)
                mlflow.log_metrics(test_metrics)

                # Log the model
                mlflow.sklearn.log_model(estimator, "model")

    mlflow.end_run()
    return max(r2_scores.items(), key=lambda item: item[1])