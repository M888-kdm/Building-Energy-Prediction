import os
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv
from tracking import find_best_run_id_by_name

load_dotenv()

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

def deploy_model_to_azure_ml_endpoint():
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
     
    model_name = "building-energy-usage"
    # Find the id of the best evaluation run and register its model artifact
    best_run_id = find_best_run_id_by_name("building-energy-prediction-evaluation")

    # Register the model
    model = ml_client.models.create_or_update(Model(
            path=f"azureml://jobs/{best_run_id}/outputs/artifacts/model",
            name=model_name,
            type=AssetTypes.MLFLOW_MODEL
        )
    )
     
    # Deploy model to an online endpoint
    endpoint_name = "seattle-energy"

    # Endpoint configuration
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="An online endpoint to generate predictions for energy usage of buildings in Seattle",
        auth_mode="key",
    )

    # Create the endpoint
    ml_client.begin_create_or_update(endpoint)

    # Configure the deployment
    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_F4s_v2",
        instance_count=1,
        readiness_probe=ProbeSettings(failure_threshold=115),
        liveness_probe=ProbeSettings(failure_threshold=115)
    )
    # Create the deployment
    ml_client.online_deployments.begin_create_or_update(blue_deployment)

    # # Assign traffic to deployment
    # endpoint.traffic = {"blue": 100}

    # # Update the endpoint configuration
    # ml_client.begin_create_or_update(endpoint).result()