from src.data_access import haversine_km


def test_haversine_same_point_is_zero():
    distance = haversine_km(10.0, 20.0, 10.0, 20.0)
    assert distance == 0


def test_haversine_known_distance():
    # المسافة التقريبية بين القاهرة وباريس (~3200 كم)، بسماحية معقولة
    distance = haversine_km(30.04, 31.24, 48.86, 2.35)
    assert 3100 < distance < 3300


def test_build_items_agg_sums_correctly():
    import pandas as pd
    from src.data_access import build_items_agg

    order_items = pd.DataFrame(
        {
            "order_id": ["A1", "A1"],
            "order_item_id": [1, 2],
            "product_id": ["P1", "P2"],
            "seller_id": ["S1", "S1"],
            "price": [50.0, 30.0],
            "freight_value": [10.0, 5.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "product_category_name": ["cat1", "cat2"],
            "product_weight_g": [100, 200],
            "product_length_cm": [10, 10],
            "product_height_cm": [10, 10],
            "product_width_cm": [10, 10],
        }
    )

    items_agg, _ = build_items_agg(order_items, products)

    assert items_agg.loc[items_agg["order_id"] == "A1", "total_price"].iloc[0] == 80.0
    assert items_agg.loc[items_agg["order_id"] == "A1", "number_of_items"].iloc[0] == 2
