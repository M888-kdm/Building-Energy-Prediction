"Module for collecting data from datasets"
import os

import pandas as pd
from loguru import logger


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from local machine
    Args:
        path (str): path to load dataset
    Returns:
       pd.DataFrame:   feature and target data
    """
    if not path:
        raise ValueError("The path like 'path' must be defined")
    # get dataset name
    dataset_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Loading {dataset_name} dataset...")
    data = pd.read_csv(path)
    logger.info(f"Loaded {dataset_name}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"{dataset_name} dataset fully lodded")
    return data
