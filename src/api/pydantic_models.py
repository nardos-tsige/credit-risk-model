"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class Transaction(BaseModel):
    """Individual transaction schema"""
    TransactionId: str = Field(..., description="Unique transaction identifier")
    CustomerId: str = Field(..., description="Customer identifier")
    Amount: float = Field(..., gt=0, description="Transaction amount")
    TransactionStartTime: datetime = Field(..., description="Transaction timestamp")
    ProductCategory: Optional[str] = Field(None, description="Product category")
    FraudResult: Optional[int] = Field(0, ge=0, le=1, description="Fraud flag (0/1)")
    
    @validator('Amount')
    def amount_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class PredictionRequest(BaseModel):
    """API request schema"""
    customer_id: str = Field(..., min_length=1, max_length=50, description="Unique customer identifier")
    transactions: List[Transaction] = Field(..., description="List of customer transactions")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "transactions": [
                    {
                        "TransactionId": "T001",
                        "CustomerId": "CUST12345",
                        "Amount": 150.50,
                        "TransactionStartTime": "2024-01-15T10:30:00",
                        "ProductCategory": "Electronics",
                        "FraudResult": 0
                    },
                    {
                        "TransactionId": "T002",
                        "CustomerId": "CUST12345",
                        "Amount": 75.25,
                        "TransactionStartTime": "2024-01-20T14:45:00",
                        "ProductCategory": "Clothing",
                        "FraudResult": 0
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """API response schema"""
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_grade: str = Field(..., description="Risk grade (A-E)")
    approval_recommendation: str = Field(..., description="Approve/Review/Decline")
    confidence_level: str = Field(..., description="Confidence level (High/Medium/Low)")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "risk_probability": 0.23,
                "credit_score": 723,
                "risk_grade": "B",
                "approval_recommendation": "Approve",
                "confidence_level": "Medium"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    engineer_loaded: bool