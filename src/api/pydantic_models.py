from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Transaction(BaseModel):
    TransactionId: str
    CustomerId: str
    Amount: float = Field(..., gt=0)
    TransactionStartTime: datetime
    ProductCategory: Optional[str] = None
    FraudResult: Optional[int] = 0

class PredictionRequest(BaseModel):
    customer_id: str
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float = Field(..., ge=0, le=1)
    credit_score: int = Field(..., ge=300, le=850)
    risk_grade: str
    approval_recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool