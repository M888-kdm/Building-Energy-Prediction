import numpy as np
import pandas as pd

from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, max_error
from typing import Union, Dict

def eval_metrics(y_actual: Union[pd.DataFrame, pd.Series, np.ndarray],
                 y_pred: Union[pd.DataFrame, pd.Series, np.ndarray]
                 ) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_actual: Ground truth (correct) target values
        y_pred: Estimated target values.

    Returns:
        Dict[str, float]: dictionary of evaluation metrics.
            Expected keys are: "rmse", "mae", "r2", "max_error"
    """
    metrics = dict()
    # Calculate Root mean squared error, named rmse
    metrics['rmse'] = mean_squared_error(y_actual, y_pred) ** 0.5
    # Calculate mean absolute error, named mae
    metrics['mae'] = mean_absolute_error(y_actual, y_pred)
    # Calculate R-squared: coefficient of determination, named r2
    metrics['r2'] = r2_score(y_actual, y_pred)
    # Calculate max error: maximum value of absolute error (y_actual - y_pred), named maxerror
    metrics['maxerror'] = max_error(y_actual, y_pred)
    # Return a dictionary
    return metrics