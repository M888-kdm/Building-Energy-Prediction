import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline import define_pipeline


def test_define_pipeline_no_target_transformer():
    """Test pipeline creation without target transformation."""
    numerical_transformer = [StandardScaler()]
    categorical_transformer = [OneHotEncoder()]
    estimator = Pipeline([("regressor", LinearRegression())])

    pipeline = define_pipeline(
        numerical_transformer, categorical_transformer, estimator
    )

    assert isinstance(
        pipeline, Pipeline
    ), "The returned object is not a Pipeline instance."
    assert isinstance(
        pipeline.named_steps["preprocessor"], ColumnTransformer
    ), "Preprocessor is not a ColumnTransformer."
    assert isinstance(
        pipeline.named_steps["estimator"], Pipeline
    ), "Estimator is not correctly wrapped in a Pipeline."


def test_define_pipeline_with_target_transformer():
    """Test pipeline creation with target transformation."""
    numerical_transformer = [StandardScaler()]
    categorical_transformer = [OneHotEncoder()]
    estimator = Pipeline([("regressor", LinearRegression())])

    pipeline = define_pipeline(
        numerical_transformer,
        categorical_transformer,
        estimator,
        target_transformer=True,
    )

    assert isinstance(
        pipeline, TransformedTargetRegressor
    ), "The returned object is not a TransformedTargetRegressor."
    assert isinstance(
        pipeline.regressor.named_steps["preprocessor"], ColumnTransformer
    ), "Preprocessor is not a ColumnTransformer."
    assert isinstance(
        pipeline.regressor.named_steps["estimator"], Pipeline
    ), "Estimator is not correctly wrapped in a Pipeline."


def test_define_pipeline_fit_transform():
    """Test that the pipeline can fit and transform data."""
    # Create synthetic data
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    # Convert to DataFrame for the pipeline
    X_df = pd.DataFrame(X, columns=["feature1", "feature2"])
    y_df = pd.Series(y, name="target")

    # Define the pipeline
    numerical_transformer = [StandardScaler()]
    categorical_transformer = [OneHotEncoder()]
    estimator = Pipeline([("regressor", LinearRegression())])
    pipeline = define_pipeline(
        numerical_transformer, categorical_transformer, estimator
    )

    # Fit the pipeline and make predictions
    pipeline.fit(X_df, y_df)
    predictions = pipeline.predict(X_df)

    assert (
        predictions.shape == y_df.shape
    ), "The shape of the predictions does not match the target variable."
