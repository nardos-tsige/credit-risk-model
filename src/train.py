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

from src.data_processing import FeatureEngineer

def load_and_prepare_data(data_path='data/raw/xente_transactions.csv'):
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")
    engineer = FeatureEngineer(random_state=42)
    X, y, customer_ids = engineer.prepare_data(df, create_target=True)
    return X, y, engineer

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    
    mlflow.set_experiment("CreditRiskModeling")
    
    for model_name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print('='*50)
        
        with mlflow.start_run(run_name=model_name):
            search = RandomizedSearchCV(
                config['model'], config['params'], cv=3, scoring='roc_auc', 
                n_iter=5, random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            
            y_pred = search.best_estimator_.predict(X_test)
            y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, model_name)
            
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = search.best_estimator_
                best_model_name = model_name
    
    print(f"\n BEST MODEL: {best_model_name} with ROC-AUC: {best_score:.4f}")
    return best_model, best_model_name

def main():
    print("="*60)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)
    
    X, y, engineer = load_and_prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    best_model, model_name = train_models(X_train, X_test, y_train, y_test)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    
    print("\n Training complete!")
    print(f"Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    main()