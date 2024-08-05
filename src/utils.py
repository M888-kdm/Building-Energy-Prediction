import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

os.environ["AZURE_TENANT_ID"] = os.getenv("AZURE_TENANT_ID")
os.environ["AZURE_CLIENT_ID"] = os.getenv("AZURE_CLIENT_ID")
os.environ["AZURE_CLIENT_SECRET"] = os.getenv("AZURE_CLIENT_SECRET")
# Enter details of your AzureML workspace
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("AML_WORKSPACE_NAME")


def get_mlflow_tracking_uri():
    """
    Configure and Get Access to Azure workspace
    :return: mlflow_tracking_uri
    """
    token_credential = DefaultAzureCredential()
    ml_client = MLClient(credential=token_credential,
                         subscription_id=subscription_id,
                         resource_group_name=resource_group,
                         workspace_name=workspace_name)
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    return mlflow_tracking_uri


if __name__ == "__main__":
    print(workspace_name)
    mlflow_trackingUri = get_mlflow_tracking_uri()
    print("mlfow tracking url")
    print(mlflow_trackingUri)
