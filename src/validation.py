import great_expectations as gx

BRAZIL_STATES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]
ORDER_STATUSES = [
    "delivered",
    "shipped",
    "canceled",
    "unavailable",
    "invoiced",
    "processing",
    "created",
    "approved",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]


def _get_validator(context, df):
    data_source = context.sources.add_pandas("pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="ml_table_asset")
    batch_request = data_asset.build_batch_request(dataframe=df)
    return context.get_validator(
        batch_request=batch_request,
        create_expectation_suite_with_name="ml_table_suite",
    )


def _add_expectations(validator):
    for id_column in ["order_id", "customer_id", "customer_unique_id"]:
        validator.expect_column_values_to_not_be_null(column=id_column)

    validator.expect_column_values_to_be_in_set(
        column="customer_state", value_set=BRAZIL_STATES
    )
    validator.expect_column_values_to_be_in_set(
        column="order_status", value_set=ORDER_STATUSES
    )
    validator.expect_column_values_to_be_in_set(
        column="main_payment_type", value_set=PAYMENT_TYPES, mostly=0.95
    )

    validator.expect_column_values_to_not_be_null(column="customer_city")
    validator.expect_column_values_to_be_between(
        column="customer_zip_code_prefix", min_value=1000, max_value=99999
    )

    for count_column in ["number_of_items", "number_of_sellers", "number_of_payments"]:
        validator.expect_column_values_to_be_between(column=count_column, min_value=1)

    validator.expect_column_values_to_be_between(
        column="number_of_product_categories", min_value=0
    )

    for money_column in ["total_price", "total_payment"]:
        validator.expect_column_values_to_be_between(column=money_column, min_value=0)
    validator.expect_column_values_to_be_between(column="total_freight", min_value=0)

    for physical_column in ["total_product_weight", "total_product_volume"]:
        validator.expect_column_values_to_be_between(
            column=physical_column, min_value=0
        )

    for installment_column in ["average_installments", "max_installments"]:
        validator.expect_column_values_to_be_between(
            column=installment_column, min_value=0
        )

    validator.expect_column_values_to_be_between(
        column="customer_seller_distance_km", min_value=0, max_value=20000, mostly=0.95
    )
    validator.expect_column_values_to_not_be_null(
        column="customer_seller_distance_km", mostly=0.90
    )
    validator.expect_column_values_to_not_be_null(
        column="customer_seller_same_state", mostly=0.90
    )

    for required_timestamp in [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]:
        validator.expect_column_values_to_not_be_null(column=required_timestamp)

    for lifecycle_timestamp in [
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]:
        validator.expect_column_values_to_not_be_null(
            column=lifecycle_timestamp, mostly=0.85
        )

    return validator


def validate_dataframe(df):
    context = gx.get_context()
    validator = _get_validator(context, df)
    validator = _add_expectations(validator)
    return validator.validate()
