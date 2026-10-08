import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_cleaning import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_and_clean


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_customers.csv"
ARTIFACTS = ROOT / "artifacts"


def train():
    data = load_and_clean(DATA_PATH)
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["has_addons"]
    X = data[features]
    y = data["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS + ["has_addons"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": len(y_test),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "churn_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()

