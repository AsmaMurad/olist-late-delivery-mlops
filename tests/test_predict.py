from src.predict import LateDeliveryPredictor


def test_predictor_loads_all_three_models():
    predictor = LateDeliveryPredictor(
        "models/final_ensemble_configuration1.json", "models/candidate_features.json"
    )
    assert set(predictor.models.keys()) == {"best_f1", "best_pr_auc", "weight_5_25"}
    assert len(predictor.candidate_features) == 38
