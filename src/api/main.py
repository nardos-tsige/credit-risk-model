"""
FastAPI application for Credit Risk Scoring
Serves the trained model for real-time predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

# Import Pydantic models
from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Bati Bank Credit Risk Scoring API",
    description="Real-time credit risk prediction for BNPL service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
feature_engineer = None
feature_names = None


@app.on_event("startup")
async def load_models():
    """Load models on application startup"""
    global model, feature_engineer, feature_names
    
    try:
        # Load the trained model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load feature engineer
        engineer_path = 'models/feature_engineer.pkl'
        if os.path.exists(engineer_path):
            feature_engineer = joblib.load(engineer_path)
            logger.info(f"✓ Feature engineer loaded from {engineer_path}")
        else:
            logger.error(f"Feature engineer not found at {engineer_path}")
        
        # Load feature names
        features_path = 'models/feature_names.pkl'
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)
            logger.info(f"✓ Feature names loaded ({len(feature_names)} features)")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Bati Bank Credit Risk Scoring API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/health", "/predict", "/docs"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        engineer_loaded=feature_engineer is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer
    
    Args:
        request: Customer transaction history
        
    Returns:
        PredictionResponse with risk probability and credit score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing prediction request for customer: {request.customer_id}")
        
        # Convert request to DataFrame
        df = pd.DataFrame([t.dict() for t in request.transactions])
        df['CustomerId'] = request.customer_id
        
        # Process features
        if feature_engineer:
            # Get features
            X, _, _ = feature_engineer.prepare_data(df, create_target=False)
            
            # Ensure we have the right features
            if feature_names:
                # Reorder columns to match training
                missing_cols = set(feature_names) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[feature_names]
            
            # Clean features
            X = feature_engineer.clean_features(X)
        else:
            raise ValueError("Feature engineer not loaded")
        
        # Get prediction probability
        risk_probability = float(model.predict_proba(X)[0, 1])
        
        # Calculate credit score (300-850, higher is better)
        # Using standard mapping: 850 - (risk_probability * 550)
        credit_score = int(850 - (risk_probability * 550))
        
        # Determine risk grade and recommendation
        if risk_probability < 0.2:
            risk_grade = "A"
            recommendation = "Approve"
            confidence = "High"
        elif risk_probability < 0.4:
            risk_grade = "B"
            recommendation = "Approve"
            confidence = "Medium"
        elif risk_probability < 0.6:
            risk_grade = "C"
            recommendation = "Review"
            confidence = "Medium"
        elif risk_probability < 0.8:
            risk_grade = "D"
            recommendation = "Review"
            confidence = "Low"
        else:
            risk_grade = "E"
            recommendation = "Decline"
            confidence = "High"
        
        logger.info(f"Prediction for {request.customer_id}: risk={risk_probability:.3f}, score={credit_score}, grade={risk_grade}")
        
        return PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=risk_probability,
            credit_score=credit_score,
            risk_grade=risk_grade,
            approval_recommendation=recommendation,
            confidence_level=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))