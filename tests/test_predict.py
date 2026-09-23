from src.predict import LateDeliveryPredictor


def test_predictor_loads_all_three_models(monkeypatch):
    """Test predictor initialization without requiring a live MLflow server."""

    monkeypatch.setattr(
        "src.predict.mlflow.xgboost.load_model",
        lambda model_uri: object(),
    )

    predictor = LateDeliveryPredictor(
        "models/final_ensemble_configuration1.json",
        "models/candidate_features.json",
    )

    assert set(predictor.models.keys()) == {
        "best_f1",
        "best_pr_auc",
        "weight_5_25",
    }

    assert len(predictor.candidate_features) == 38
