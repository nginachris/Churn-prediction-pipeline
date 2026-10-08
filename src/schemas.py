from pydantic import BaseModel, Field


class Customer(BaseModel):
    tenure_months: float = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: float = Field(ge=0)
    contract_type: str
    payment_method: str
    has_addons: bool


class Prediction(BaseModel):
    churn_probability: float
    predicted_churn: bool

