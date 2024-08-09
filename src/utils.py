import os
import sys

import pendulum
from dotenv import load_dotenv
from logtail import LogtailHandler
from loguru import logger


def get_current_date():
    return pendulum.now().strftime("%Y%m%d_%H%m%S")


def add_prefix_to_keys(dict_, prefix):
    """Adds a prefix to all keys in a dictionary.

    Args:
      dict_: The input dictionary.
      prefix: The prefix to add.

    Returns:
      A new dictionary with prefixed keys.
    """

    return {f"{prefix}_{key}": value for key, value in dict_.items()}


def create_dir_if_not_exists(dir_path):
    """Creates a directory if it doesn't exist.

    Args:
      dir_path: The path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def configure_logger():
    load_dotenv()
    logtail_handler = LogtailHandler(
        source_token=os.getenv("BETTER_STACK_SOURCE_TOKEN")
    )
    log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
    logger.configure(
        handlers=[
            {"sink": sys.stderr, "format": log_fmt},
            {"sink": logtail_handler, "format": log_fmt, "level": "INFO"},
        ]
    )
