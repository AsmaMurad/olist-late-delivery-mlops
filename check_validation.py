from src.data_access import get_engine, build_ml_table
from src.validation import validate_ml_table


engine = get_engine()

ml_table = build_ml_table(engine)

result = validate_ml_table(ml_table)

print("\nRows with total_payment == 0:")
print(
    ml_table.loc[
        ml_table["total_payment"] == 0,
        [
            "order_id",
            "order_status",
            "number_of_payments",
            "total_payment",
            "main_payment_type",
        ],
    ].to_string(index=False)
)

print("\nRows with missing total_payment:")
print(
    ml_table.loc[
        ml_table["total_payment"].isna(),
        [
            "order_id",
            "order_status",
            "number_of_payments",
            "total_payment",
            "main_payment_type",
        ],
    ].to_string(index=False)
)

print("Overall success:", result.success)
print("Total expectations:", len(result.results))

for item in result.results:
    if not item.success:
        print("\nFAILED")
        print("Expectation:", item.expectation_config.type)
        print("Expectation config:", item.expectation_config)
        print("Result:", item.result)
