# Credit Risk Probability Model for Alternative Data

[![CI/CD Pipeline](https://github.com/nardos-tsige/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/nardos-tsige/credit-risk-model/actions/workflows/ci.yml)

**Bati Bank - Buy Now Pay Later Credit Scoring System**

## Project Overview
Building an end-to-end credit risk scoring model using e-commerce transaction data for Bati Bank's BNPL service.

## Team Members
- Kerod
- Mahbubah  
- Feven

## Project Timeline
- Challenge Dates: 28 May – 03 Jun 2026
- Interim Submission: 31 May 2026
- Final Submission: 03 Jun 2026

## Credit Scoring Business Understanding

### How Basel II Influences Model Requirements

The Basel II Capital Accord mandates transparent risk measurement where models must be:
- **Explainable** to internal risk teams and regulators
- **Well-documented** with clear assumptions, data sources, and methodologies
- **Validated** for regulatory capital calculations

### Necessity and Risks of Proxy Variables

**Why necessary**: Without a direct "default" label, we must approximate default risk using behavioral patterns (RFM analysis).

**Key risks**:
- Label leakage/bias
- Overfitting to non-generalizable behaviors  
- Compliance risk

### Trade-offs: Interpretable vs. High-Performance Models

| Aspect | Logistic Regression | XGBoost |
|--------|---------------------|---------|
| Interpretability | High | Low |
| Regulatory Acceptance | Preferred | Needs SHAP |
| Performance | Moderate | High |

## Setup Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Download data to data/raw/

# Run EDA
jupyter notebook notebooks/eda.ipynb

# Train model
python src/train.py

# Run API
uvicorn src.api.main:app --reload
