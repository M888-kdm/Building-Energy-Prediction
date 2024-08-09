import mlflow

from src.azure_ml import deploy_model_to_azure_ml_endpoint, get_mlflow_tracking_uri


def deploy():
    uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(uri)
    deploy_model_to_azure_ml_endpoint()


if __name__ == "__main__":
    deploy()
