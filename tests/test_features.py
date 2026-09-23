import pandas as pd
from src.features import create_features


def test_create_features_extracts_purchase_year():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2024-03-15 10:00:00"],
            "order_estimated_delivery_date": ["2024-03-25 10:00:00"],
        }
    )
    result = create_features(df)
    assert result["purchase_year"].iloc[0] == 2024
    assert result["purchase_month"].iloc[0] == 3


def test_create_features_estimated_delivery_days_positive():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2024-03-15 10:00:00"],
            "order_estimated_delivery_date": ["2024-03-25 10:00:00"],
        }
    )
    result = create_features(df)
    # 10 أيام فرق بين الشراء والتسليم المتوقع
    assert result["estimated_delivery_days"].iloc[0] == 10
