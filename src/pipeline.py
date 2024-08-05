import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor

def define_pipeline(numerical_transformer: list,
                    categorical_transformer: list,
                    estimator: Pipeline,
                    target_transformer: bool = False,
                    **kwargs: dict) -> Pipeline:
    """
    Defines a machine learning pipeline for pre-processing data and fitting a model.

    This function constructs a scikit-learn pipeline that performs the following steps:

    1. **Data Preprocessing:**
       * Applies separate transformations to numerical and categorical features:
         * `numerical_transformer`: List of transformers for numerical features.
         * `categorical_transformer`: List of transformers for categorical features.
       * Uses `ColumnTransformer` to combine these transformations.
       * Drops any remaining columns not explicitly specified.
    2. **Model Fitting:**
       * Appends the chosen `estimator` (a scikit-learn pipeline) to the preprocessor.
    3. **Target Transformation (Optional):**
       * If `target_transformer` is True, applies a logarithmic transformation 
         to the target variable using `TransformedTargetRegressor`.

    Args:
        numerical_transformer (list): List of transformers to apply to numerical features.
        categorical_transformer (list): List of transformers to apply to categorical features.
        estimator (Pipeline): A scikit-learn pipeline representing the model to be fit.
        target_transformer (bool, optional): Whether to apply a logarithmic transformation 
                                              to the target variable (default: False).
        **kwargs: Additional keyword arguments passed to the `Pipeline` constructor.

    Returns:
        Pipeline: A scikit-learn pipeline combining pre-processing and model fitting steps.
    """
    numerical_transformer = make_pipeline(*numerical_transformer)

    categorical_transformer = make_pipeline(*categorical_transformer)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, make_column_selector(dtype_include=["number"])),
            ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "bool"])),
        ],
        remainder="drop",  # non-specified columns are dropped
        verbose_feature_names_out=False,  # will not prefix any feature names with the name of the transformer
    )
    # Append regressor to preprocessing pipelineregreregffdffdffd.
    # Now we have a full prediction pipeline.
    if target_transformer:
        model_pipe1 = Pipeline(steps=[("preprocessor", preprocessor),
                                     ("estimator", estimator)])
        model_pipe = TransformedTargetRegressor(regressor=model_pipe1,
                                                func=np.log,
                                                inverse_func=np.exp)
    else:
        model_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        
    # logger.info(f"{model_pipe}")
    return model_pipe