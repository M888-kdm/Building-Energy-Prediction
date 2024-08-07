import os
import pendulum
import sys

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from logtail import LogtailHandler
from loguru import logger

def get_current_date():
    return pendulum.now().strftime('%Y%m%d_%H%m%S')

def add_prefix_to_keys(dict_, prefix):
  """Adds a prefix to all keys in a dictionary.

  Args:
    dict_: The input dictionary.
    prefix: The prefix to add.

  Returns:
    A new dictionary with prefixed keys.
  """

  return {prefix + key: value for key, value in dict_.items()}

def configure_logger():
    load_dotenv()
    logtail_handler = LogtailHandler(source_token=os.getenv("BETTER_STACK_SOURCE_TOKEN"))
    log_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS!UTC}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_fmt}, {"sink": logtail_handler , "format": log_fmt, "level": "INFO"}])


def get_mlflow_tracking_uri():
    """
    Configure and Get Access to Azure workspace
    :return: mlflow_tracking_uri
    """
    load_dotenv()
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("AML_WORKSPACE_NAME")
    token_credential = DefaultAzureCredential()
    ml_client = MLClient(credential=token_credential,
                         subscription_id=subscription_id,
                         resource_group_name=resource_group,
                         workspace_name=workspace_name)
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    return mlflow_tracking_uri