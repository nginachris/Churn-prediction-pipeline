from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import Customer, Prediction


MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "churn_model.joblib"
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python -m src.train first.")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return Prediction(
        churn_probability=round(probability, 4),
        predicted_churn=probability >= 0.5,
    )

