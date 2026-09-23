import mlflow
import mlflow.xgboost
import json

import os

mlflow_host = os.environ.get("MLFLOW_HOST", "localhost")
mlflow.set_tracking_uri(f"http://{mlflow_host}:5000")

MODEL_NAMES = {
    "best_f1": "late_delivery_ensemble_best_f1",
    "best_pr_auc": "late_delivery_ensemble_best_pr_auc",
    "weight_5_25": "late_delivery_ensemble_weight_5_25",
}


class LateDeliveryPredictor:
    def __init__(
        self, ensemble_config_path, candidate_features_path, model_version="1"
    ):
        self.models = {
            name: mlflow.xgboost.load_model(f"models:/{registry_name}/{model_version}")
            for name, registry_name in MODEL_NAMES.items()
        }

        with open(ensemble_config_path) as f:
            ensemble_config = json.load(f)
        self.weights = ensemble_config["weights"]
        self.threshold = ensemble_config["threshold"]

        with open(candidate_features_path) as f:
            self.candidate_features = json.load(f)["candidate_features"]

    def predict(self, processed_features):
        probability = 0.0
        for name, weight in self.weights.items():
            model = self.models[name]
            # نرتب الأعمدة بالضبط متل ما الموديل اتدرب عليها
            model_feature_order = model.get_booster().feature_names
            X = processed_features[model_feature_order]

            model_prob = model.predict_proba(X)[:, 1]
            probability = probability + weight * model_prob

        prediction = (probability >= self.threshold).astype(int)
        return probability, prediction
