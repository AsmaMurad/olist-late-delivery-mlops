from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json

from src.features import create_features, select_raw_features
from src.preprocess import load_preprocessor, transform_features
from src.predict import LateDeliveryPredictor

app = FastAPI(title="Late Delivery Prediction Service")

# نحمّل الموديلات والـ preprocessor مرة وحدة بس، وقت بدء تشغيل السيرفر
# (مش كل مرة يجي طلب — هيك أسرع بكتير)
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
    df = pd.DataFrame([order.model_dump()])

    featured = create_features(df)
    raw_features = select_raw_features(featured)
    processed = transform_features(raw_features, preprocessor)

    probability, prediction = predictor.predict(processed)

    return {
        "prediction": "late" if prediction[0] == 1 else "on_time",
        "probability_late": float(probability[0]),
        "model_version": "1",
    }
