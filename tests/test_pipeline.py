import pandas as pd

from src.data_cleaning import clean_data


def test_clean_data_removes_duplicates_and_normalizes_text():
    data = pd.DataFrame(
        [
            {
                "customer_id": "1",
                "tenure_months": "3",
                "monthly_charges": 50,
                "support_tickets": 1,
                "contract_type": " Month-to-Month ",
                "payment_method": "Electronic-Check",
                "has_addons": "false",
                "churned": 1,
            },
            {
                "customer_id": "1",
                "tenure_months": "3",
                "monthly_charges": 50,
                "support_tickets": 1,
                "contract_type": " Month-to-Month ",
                "payment_method": "Electronic-Check",
                "has_addons": "false",
                "churned": 1,
            },
        ]
    )
    cleaned = clean_data(data)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "contract_type"] == "month-to-month"
    assert cleaned.loc[0, "has_addons"] == 0

