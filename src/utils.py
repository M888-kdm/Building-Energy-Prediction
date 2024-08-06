import pendulum
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

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


if __name__ == "__main__":
    mlflow_trackingUri = get_mlflow_tracking_uri()
    print("mlfow tracking url")
    print(mlflow_trackingUri)
