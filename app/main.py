from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import time

from src.features import create_features, select_raw_features
from src.preprocess import load_preprocessor, transform_features
from src.predict import LateDeliveryPredictor
from src.logger import get_logger

metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "total_latency_seconds": 0.0,
}


app = FastAPI(title="Late Delivery Prediction Service")

logger = get_logger(__name__)


preprocessor = load_preprocessor("models/preprocessor1.joblib")
predictor = LateDeliveryPredictor(
    "models/final_ensemble_configuration1.json", "models/candidate_features.json"
)


class OrderInput(BaseModel):
    number_of_items: int
    number_of_sellers: int
    total_product_weight: float
    total_product_volume: float
    number_of_product_categories: int
    total_price: float
    total_freight: float
    number_of_payments: int
    total_payment: float
    average_installments: float
    max_installments: int
    main_payment_type: str
    customer_seller_distance_km: float
    customer_seller_same_state: bool
    customer_state: str
    customer_city: str
    customer_zip_code_prefix: int
    order_purchase_timestamp: str
    order_estimated_delivery_date: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    with open("models/final_ensemble_configuration1.json") as f:
        config = json.load(f)
    return {
        "model_type": config.get("model_type", "ensemble"),
        "threshold": config.get("threshold"),
        "feature_count": config.get("feature_count"),
    }


@app.post("/predict")
def predict(order: OrderInput):
    start_time = time.time()
    metrics["total_requests"] += 1

    try:
        df = pd.DataFrame([order.model_dump()])

        featured = create_features(df)
        raw_features = select_raw_features(featured)
        processed = transform_features(raw_features, preprocessor)

        probability, prediction = predictor.predict(processed)

        latency = time.time() - start_time
        metrics["total_latency_seconds"] += latency

        logger.info(
            f"Prediction request: input_items={order.number_of_items}, "
            f"output={'late' if prediction[0] == 1 else 'on_time'}, "
            f"probability={float(probability[0]):.3f}, "
            f"latency={latency:.3f}s, "
            f"model_version=1"
        )

        return {
            "prediction": "late" if prediction[0] == 1 else "on_time",
            "probability_late": float(probability[0]),
            "model_version": "1",
        }

    except Exception as e:
        metrics["total_errors"] += 1

        latency = time.time() - start_time
        metrics["total_latency_seconds"] += latency

        logger.error(
            f"Prediction failed: error={e}, "
            f"latency={latency:.3f}s, "
            f"model_version=1"
        )

        raise


@app.get("/metrics")
def get_metrics():
    total = metrics["total_requests"]
    avg_latency = metrics["total_latency_seconds"] / total if total > 0 else 0
    error_rate = metrics["total_errors"] / total if total > 0 else 0

    return {
        "total_requests": total,
        "total_errors": metrics["total_errors"],
        "error_rate": round(error_rate, 4),
        "average_latency_seconds": round(avg_latency, 4),
    }
