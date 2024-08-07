import mlflow
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from src.pipeline import define_pipeline
from src.tracking import get_experiment_id, mlflow_log_search
from src.utils import get_current_date


def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

scoring = {'r2': make_scorer(r2_score),
          'rmse': make_scorer(rmse, greater_is_better=False),
          'mae': make_scorer(mean_absolute_error, greater_is_better=False)}

def fine_tune_models(estimator_params, x_train, y_train):

    """Fine-tunes a set of machine learning estimators using GridSearchCV with MLflow tracking.

    This function performs hyperparameter tuning on multiple machine learning estimators using GridSearchCV. 
    The tuning process is tracked within an MLflow experiment, logging parameters, metrics, and the entire search object.

    Args:
        estimator_params (dict): A dictionary containing configuration for each estimator to be fine-tuned.
        The dictionary keys are the estimator names, and the values are dictionaries with the following keys:
            - "estimator": The scikit-learn estimator object.
            - "params": A dictionary containing the hyperparameter grid for GridSearchCV.
        x_train (pandas.DataFrame): The training data features.
        y_train (pandas.Series): The training data target variable.

    Returns:
        dict: A dictionary containing the fine-tuned GridSearchCV objects for each estimator, keyed by their names.
    """
    search_cvs = {}

    # Create an experiment if not exists
    exp_name = "building-energy-prediction-tuning-sklearn"
    experiment_id = get_experiment_id(exp_name)

    with mlflow.start_run(run_name=f"Session-{get_current_date()}", experiment_id=experiment_id):
        for estimator_name, settings in estimator_params.items():
            with mlflow.start_run(run_name=estimator_name, nested=True, experiment_id=experiment_id):  
                estimator = settings["estimator"]
                param_grid = settings["params"]
                pipeline = define_pipeline(numerical_transformer=[SimpleImputer(strategy="median"), FunctionTransformer(np.log1p), RobustScaler()],
                                categorical_transformer=[SimpleImputer(strategy="constant", fill_value="undefined"), OneHotEncoder(drop="if_binary", handle_unknown="ignore")],
                                target_transformer=True,
                                estimator=estimator
                            ) 
                grid_search = GridSearchCV(
                    estimator=pipeline,  # Instantiate the estimator
                    param_grid=param_grid,
                    scoring=scoring,
                    refit='r2',
                    cv=5,  # Adjust the number of cross-validation folds as needed
                    n_jobs=-1  # Use all available cores
                )
                grid_search.fit(x_train, y_train)
                search_cvs[estimator_name] = grid_search

                mlflow.log_param("Estimator", estimator_name)
                mlflow_log_search(grid_search)
    mlflow.end_run()
    return search_cvs