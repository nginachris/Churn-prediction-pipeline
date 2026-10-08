from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "payment_method",
    "has_addons",
    "churned",
}

NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "support_tickets"]
CATEGORICAL_COLUMNS = ["contract_type", "payment_method"]


def clean_data(data: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    required = REQUIRED_COLUMNS if include_target else REQUIRED_COLUMNS - {"churned"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates(subset=["customer_id"])

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        if (cleaned[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values")

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()
        cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0])

    cleaned["has_addons"] = cleaned["has_addons"].map(
        {True: 1, False: 0, "true": 1, "false": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
    )
    cleaned["has_addons"] = cleaned["has_addons"].fillna(0).astype(int)

    if include_target:
        cleaned["churned"] = pd.to_numeric(cleaned["churned"], errors="coerce")
        cleaned = cleaned.dropna(subset=["churned"])
        cleaned["churned"] = cleaned["churned"].astype(int)

    return cleaned.reset_index(drop=True)


def load_and_clean(path: str | Path, include_target: bool = True) -> pd.DataFrame:
    return clean_data(pd.read_csv(path), include_target=include_target)

