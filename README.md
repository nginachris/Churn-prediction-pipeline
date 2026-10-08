# Customer Churn Prediction Pipeline

An end-to-end machine learning project for predicting whether a customer is likely to leave a subscription service.

## Project layout

```text
churn-prediction-pipeline/
├── data/sample_customers.csv
├── artifacts/                 # Generated model and evaluation files
├── src/
│   ├── api.py                # FastAPI prediction service
│   ├── data_cleaning.py      # Input validation and cleaning
│   ├── schemas.py            # API request and response models
│   └── train.py              # Training and evaluation pipeline
├── tests/
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Open `/docs` for the interactive Swagger page.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 4,
    "monthly_charges": 89.50,
    "support_tickets": 5,
    "contract_type": "month-to-month",
    "payment_method": "electronic-check",
    "has_addons": false
  }'
```

## Data hygiene

The cleaning step:

- checks that all required columns exist;
- removes duplicate records;
- normalizes text fields and contract labels;
- converts numeric fields safely;
- rejects impossible negative values;
- fills missing numeric values with the training median;
- fills missing categorical values with the most common category.

## Evaluation

`python -m src.train` creates `artifacts/metrics.json` with accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix. The model uses a stratified 80/20 train/test split and a preprocessing pipeline so the same transformations are applied during training and API inference.

This repository includes a small sample dataset for demonstrating the pipeline. For production use, replace it with a larger, representative historical dataset and add time-based validation where appropriate.

