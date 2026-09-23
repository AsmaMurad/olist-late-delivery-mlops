import mlflow
import mlflow.xgboost
import json
import joblib

import os

mlflow_host = os.environ.get("MLFLOW_HOST", "localhost")
mlflow.set_tracking_uri(f"http://{mlflow_host}:5000")
mlflow.set_experiment("late_delivery_prediction")


def register_existing_model(model_path, config_path, model_name):
    """يسجل موديل XGBoost مدرب مسبقًا بـ MLflow، بدون إعادة تدريب."""
    with open(config_path) as f:
        config = json.load(f)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "threshold": config.get("threshold"),
                "weights": str(config.get("weights")),
            }
        )
        mlflow.log_metric("feature_count", config.get("feature_count", 0))

        model = joblib.load(model_path)
        model_info = mlflow.xgboost.log_model(
            model, artifact_path="model", registered_model_name=model_name
        )
    return model_info
