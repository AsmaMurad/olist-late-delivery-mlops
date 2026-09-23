import pandas as pd
import holidays

RAW_FEATURE_COLUMNS = [
    "number_of_items",
    "number_of_sellers",
    "total_product_weight",
    "total_product_volume",
    "number_of_product_categories",
    "total_price",
    "total_freight",
    "number_of_payments",
    "total_payment",
    "average_installments",
    "max_installments",
    "main_payment_type",
    "customer_seller_distance_km",
    "customer_seller_same_state",
    "customer_state",
    "customer_city",
    "customer_zip_code_prefix",
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "purchase_dayofmonth",
    "is_weekend",
    "is_holiday",
    "estimated_delivery_days",
]


def create_features(df):
    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"], errors="coerce"
    )

    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_dayofmonth"] = df["order_purchase_timestamp"].dt.day
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)
    years = df["order_purchase_timestamp"].dt.year.dropna().astype(int).unique()
    brazil_holidays = holidays.Brazil(years=years)
    df["is_holiday"] = (
        df["order_purchase_timestamp"].dt.date.isin(brazil_holidays).astype(int)
    )
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    return df


def select_raw_features(df):
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw features: {missing}")
    return df[RAW_FEATURE_COLUMNS].copy()
