"""Test dataset"""

from src.collect_data import load_dataset
from settings.params import MODEL_PARAMS

# load data
data = load_dataset("./datasets/raw_data.csv")
TARGET_NAME = MODEL_PARAMS["TARGET_NAME"]


def test_shape():
    """Test the shape of the dataset."""
    n_rows, n_cols = data.shape
    assert n_rows >= 6716
    assert n_cols == 45


def test_target_name():
    """Test target feature."""
    assert sum(data[TARGET_NAME] < 0) == 0

