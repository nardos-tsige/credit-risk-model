from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    TransactionId: str
    CustomerId: str
    Amount: float
    TransactionStartTime: datetime
    ProductCategory: Optional[str] = None
    FraudResult: Optional[int] = 0

class PredictionRequest(BaseModel):
    customer_id: str
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    credit_score: int
    risk_grade: str
    approval_recommendation: str

model = None
feature_engineer = None

@app.on_event("startup")
async def load_models():
    global model, feature_engineer
    try:
        model = joblib.load('models/best_model.pkl')
        feature_engineer = joblib.load('models/feature_engineer.pkl')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([t.dict() for t in request.transactions])
        df['CustomerId'] = request.customer_id
        
        X, _, _ = feature_engineer.prepare_data(df, create_target=False)
        X = feature_engineer.clean_features(X)
        
        risk_probability = float(model.predict_proba(X)[0, 1])
        credit_score = int(850 - (risk_probability * 550))
        
        if risk_probability < 0.2:
            risk_grade, recommendation = "A", "Approve"
        elif risk_probability < 0.4:
            risk_grade, recommendation = "B", "Approve"
        elif risk_probability < 0.6:
            risk_grade, recommendation = "C", "Review"
        elif risk_probability < 0.8:
            risk_grade, recommendation = "D", "Review"
        else:
            risk_grade, recommendation = "E", "Decline"
        
        return PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=risk_probability,
            credit_score=credit_score,
            risk_grade=risk_grade,
            approval_recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))