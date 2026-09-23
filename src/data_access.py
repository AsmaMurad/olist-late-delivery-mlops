from src.logger import get_logger
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os
from src.config import load_config

logger = get_logger(__name__)


def get_engine():
    config = load_config()
    db = config["database"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    url = f"postgresql+psycopg2://{user}:{password}@{db['host']}:{db['port']}/{db['name']}"
    return create_engine(url)


def read_orders(engine):
    return pd.read_sql("SELECT * FROM orders;", engine)


def read_customers(engine):
    return pd.read_sql("SELECT * FROM customers;", engine)


def read_order_items(engine):
    return pd.read_sql("SELECT * FROM order_items;", engine)


def read_products(engine):
    return pd.read_sql("SELECT * FROM products;", engine)


def read_sellers(engine):
    return pd.read_sql("SELECT * FROM sellers;", engine)


def read_geolocation(engine):
    return pd.read_sql("SELECT * FROM geolocation;", engine)


def read_order_payments(engine):
    return pd.read_sql("SELECT * FROM order_payments;", engine)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_items_agg(order_items, products):
    items_with_products = order_items.merge(
        products, on="product_id", how="left", validate="many_to_one"
    )
    items_with_products["product_volume_cm3"] = (
        items_with_products["product_length_cm"]
        * items_with_products["product_height_cm"]
        * items_with_products["product_width_cm"]
    )
    items_agg = (
        items_with_products.groupby("order_id")
        .agg(
            number_of_items=("order_item_id", "count"),
            number_of_sellers=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            total_product_weight=("product_weight_g", "sum"),
            total_product_volume=("product_volume_cm3", "sum"),
            number_of_product_categories=("product_category_name", "nunique"),
        )
        .reset_index()
    )
    return items_agg, items_with_products


def build_payments_agg(order_payments):
    payments_agg = (
        order_payments.groupby("order_id")
        .agg(
            number_of_payments=("payment_sequential", "count"),
            total_payment=("payment_value", "sum"),
            average_installments=("payment_installments", "mean"),
            max_installments=("payment_installments", "max"),
        )
        .reset_index()
    )
    payment_type_value = (
        order_payments.groupby(["order_id", "payment_type"])["payment_value"]
        .sum()
        .reset_index()
    )
    main_payment_type = (
        payment_type_value.sort_values(
            ["order_id", "payment_value"], ascending=[True, False]
        )
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "main_payment_type"})
    )
    return payments_agg.merge(
        main_payment_type, on="order_id", how="left", validate="one_to_one"
    )


def build_seller_geo_agg(orders, customers, sellers, geolocation, items_with_products):
    geo_by_zip = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg(
            latitude=("geolocation_lat", "mean"), longitude=("geolocation_lng", "mean")
        )
        .reset_index()
    )

    customers_geo = customers.merge(
        geo_by_zip,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"latitude": "customer_lat", "longitude": "customer_lng"})

    sellers_geo = sellers.merge(
        geo_by_zip,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"latitude": "seller_lat", "longitude": "seller_lng"})

    order_seller_map = (
        items_with_products[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(sellers_geo, on="seller_id", how="left")
    )
    order_customer_geo = orders[["order_id", "customer_id"]].merge(
        customers_geo[
            ["customer_id", "customer_state", "customer_lat", "customer_lng"]
        ],
        on="customer_id",
        how="left",
    )
    order_seller_geo = order_seller_map.merge(
        order_customer_geo, on="order_id", how="left"
    )

    order_seller_geo["seller_distance_km"] = order_seller_geo.apply(
        lambda row: (
            haversine_km(
                row["customer_lat"],
                row["customer_lng"],
                row["seller_lat"],
                row["seller_lng"],
            )
            if pd.notna(row["customer_lat"]) and pd.notna(row["seller_lat"])
            else pd.NA
        ),
        axis=1,
    )
    order_seller_geo["same_state"] = (
        order_seller_geo["customer_state"] == order_seller_geo["seller_state"]
    )

    return (
        order_seller_geo.groupby("order_id")
        .agg(
            customer_seller_distance_km=("seller_distance_km", "mean"),
            customer_seller_same_state=("same_state", "all"),
        )
        .reset_index()
    )


def build_ml_table(engine):
    logger.info("Starting to build ml_table from database")

    orders = read_orders(engine)
    customers = read_customers(engine)
    order_items = read_order_items(engine)
    products = read_products(engine)
    sellers = read_sellers(engine)
    geolocation = read_geolocation(engine)
    order_payments = read_order_payments(engine)

    items_agg, items_with_products = build_items_agg(order_items, products)
    seller_geo_agg = build_seller_geo_agg(
        orders, customers, sellers, geolocation, items_with_products
    )
    payments_agg = build_payments_agg(order_payments)

    ml_table = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(items_agg, on="order_id", how="left")
        .merge(seller_geo_agg, on="order_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
    )

    logger.info(
        f"Finished building ml_table: {ml_table.shape[0]} rows, {ml_table.shape[1]} columns"
    )

    logger.info("Validating ml_table against expectation suite")
    from src.validation import validate_dataframe

    result = validate_dataframe(ml_table)
    if result.success:
        logger.info("Validation passed")
    else:
        logger.warning("Validation FAILED — check the data before trusting predictions")

    return ml_table
