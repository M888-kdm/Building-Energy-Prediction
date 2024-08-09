from yellowbrick.regressor import PredictionError, ResidualsPlot


def prediction_error_plot(estimators, x_train, x_test, y_train, y_test):
    """
    Generates and displays prediction error plots for a set of fitted regressors.

    This function iterates through a dictionary of estimators, creates a
    `yellowbrick.regressor.PredictionError` visualizer for each, and displays
    the prediction error plot for the corresponding training and testing data.

    Args:
        estimators (dict): A dictionary where keys are estimator names and
                           values are fitted regressor objects.
        x_train (ndarray): The training data features.
        x_test (ndarray): The testing data features.
        y_train (ndarray): The training data target values.
        y_test (ndarray): The testing data target values.
    """

    for estimator_name, estimator in estimators.items():
        visualizer = PredictionError(
            estimator, is_fitted="auto", identity=True, bestfit=True
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        print(f"Prediction plot for estimator {estimator_name}")
        visualizer.show()


def residual_plot(estimators, x_train, x_test, y_train, y_test):
    """
    Generates and displays residual plots for a set of fitted regressors.

    This function iterates through a dictionary of estimators, creates a
    `yellowbrick.regressor.ResidualsPlot` visualizer for each, and displays
    the residual plot for the corresponding training and testing data.

    Args:
        estimators (dict): A dictionary where keys are estimator names and
                           values are fitted regressor objects.
        x_train (ndarray): The training data features.
        x_test (ndarray): The testing data features.
        y_train (ndarray): The training data target values.
        y_test (ndarray): The testing data target values.
    """

    for estimator_name, estimator in estimators.items():
        visualizer = ResidualsPlot(estimator, is_fitted="auto")
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        print(f"Residual plot for estimator {estimator_name}")
        visualizer.show()
