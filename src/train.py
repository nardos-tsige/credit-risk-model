"""
Model Training Script for Credit Risk Prediction
Trains multiple models, logs experiments with MLflow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineer
from src.data_processing import FeatureEngineer


def load_and_prepare_data(data_path='data/raw/xente_transactions.csv'):
    """
    Load raw data and prepare features and target
    """
    print("Loading data...")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f" Data file not found at {data_path}")
        print("Creating sample data for testing...")
        
        # Create sample data if real data doesn't exist
        np.random.seed(42)
        n_transactions = 10000
        n_customers = 1000
        
        df = pd.DataFrame({
            'TransactionId': [f'T{i:06d}' for i in range(n_transactions)],
            'CustomerId': [f'C{np.random.randint(1, n_customers):04d}' for _ in range(n_transactions)],
            'Amount': np.random.exponential(100, n_transactions),
            'TransactionStartTime': pd.date_range('2023-01-01', periods=n_transactions, freq='H'),
            'FraudResult': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
        })
        print(f"✓ Created sample data with {len(df)} transactions")
    else:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} transactions from {data_path}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(random_state=42)
    
    # Prepare data with target
    X, y, customer_ids = engineer.prepare_data(df, create_target=True)
    
    # Apply WoE encoding to categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        X = engineer.apply_woe_encoding(X, y, categorical_cols)
    
    return X, y, engineer


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and track with MLflow
    """
    
    # Define models and hyperparameters
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    best_params = None
    
    # Set MLflow experiment
    mlflow.set_experiment("CreditRiskModeling")
    
    for model_name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print('='*50)
        
        with mlflow.start_run(run_name=model_name):
            # Log that we're starting
            mlflow.log_param("model_type", model_name)
            
            # Hyperparameter tuning
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                cv=3,
                scoring='roc_auc',
                n_iter=5,
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            
            # Best model from tuning
            best_estimator = search.best_estimator_
            
            # Predictions
            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            # Log parameters and metrics
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(best_estimator, model_name)
            
            # Print results
            print(f"\nBest parameters: {search.best_params_}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = best_estimator
                best_model_name = model_name
                best_params = search.best_params_
    
    print("\n" + "="*60)
    print(f" BEST MODEL: {best_model_name}")
    print(f"   ROC-AUC: {best_score:.4f}")
    print(f"   Parameters: {best_params}")
    print("="*60)
    
    return best_model, best_model_name


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load and prepare data
    X, y, engineer = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Target distribution - Train: {y_train.mean():.2%} high-risk")
    print(f"   Target distribution - Test: {y_test.mean():.2%} high-risk")
    
    # Train models
    best_model, model_name = train_models(X_train, X_test, y_train, y_test)
    
    # Save models
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    print("✓ Models saved to 'models/' directory")
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("\n Training complete!")
    print(f"   Best model: {model_name}")
    print(f"   Location: models/best_model.pkl")
    
    # MLflow UI command
    print("\n To view MLflow dashboard:")
    print("   Open another terminal and run: mlflow ui")
    print("   Then open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()