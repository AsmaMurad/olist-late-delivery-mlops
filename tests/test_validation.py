import pandas as pd
from src.validation import validate_dataframe


def test_validation_catches_invalid_state():
    df = pd.DataFrame(
        {
            "order_id": ["A1"],
            "customer_id": ["C1"],
            "customer_unique_id": ["U1"],
            "customer_state": ["XX"],  # ولاية غير موجودة - خطأ مقصود
            "order_status": ["delivered"],
            "main_payment_type": ["credit_card"],
            "customer_city": ["sao paulo"],
            "customer_zip_code_prefix": [12345],
            "number_of_items": [1],
            "number_of_sellers": [1],
            "number_of_payments": [1],
            "number_of_product_categories": [1],
            "total_price": [50.0],
            "total_payment": [50.0],
            "total_freight": [10.0],
            "total_product_weight": [500],
            "total_product_volume": [1000],
            "average_installments": [1],
            "max_installments": [1],
            "customer_seller_distance_km": [100.0],
            "customer_seller_same_state": [True],
            "order_purchase_timestamp": ["2024-01-01"],
            "order_estimated_delivery_date": ["2024-01-10"],
            "order_approved_at": ["2024-01-01"],
            "order_delivered_carrier_date": ["2024-01-02"],
            "order_delivered_customer_date": ["2024-01-08"],
        }
    )
    result = validate_dataframe(df)
    assert result.success is False  # لازم يفشل، لأنه "XX" مش ولاية حقيقية
