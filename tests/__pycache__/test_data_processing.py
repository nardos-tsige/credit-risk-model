import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import FeatureEngineer
from src.woe_encoder import ManualWOEEncoder


class TestFeatureEngineer:
    
    def setup_method(self):
        self.engineer = FeatureEngineer(random_state=42)
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'TransactionId': [f'T{i:06d}' for i in range(200)],
            'CustomerId': [f'C{np.random.randint(1, 30):04d}' for _ in range(200)],
            'Amount': np.random.exponential(100, 200),
            'TransactionStartTime': pd.date_range('2023-01-01', periods=200, freq='D'),
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 200),
            'FraudResult': np.random.choice([0, 1], 200, p=[0.95, 0.05])
        })
    
    def test_rfm_features_created(self):
        rfm = self.engineer.create_rfm_features(self.sample_df)
        assert 'CustomerId' in rfm.columns
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
        assert len(rfm) == len(self.sample_df['CustomerId'].unique())
    
    def test_rfm_values_positive(self):
        rfm = self.engineer.create_rfm_features(self.sample_df)
        assert (rfm['Frequency'] >= 0).all()
        assert (rfm['Monetary'] >= 0).all()
    
    def test_high_risk_proxy_binary(self):
        target_df = self.engineer.create_high_risk_proxy(self.sample_df, n_clusters=2)
        assert target_df['is_high_risk'].isin([0, 1]).all()
    
    def test_clean_features_handles_nan(self):
        df_nan = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
        cleaned = self.engineer.clean_features(df_nan)
        assert not cleaned.isna().any().any()
    
    def test_prepare_data_shapes(self):
        X, y, ids = self.engineer.prepare_data(self.sample_df, create_target=True)
        assert len(X) == len(y)
        assert len(X) == len(ids)
        assert X.shape[1] > 0
    
    def test_customer_features_count(self):
        features = self.engineer.create_customer_features(self.sample_df)
        unique_customers = self.sample_df['CustomerId'].nunique()
        assert len(features) == unique_customers


class TestWOEEncoder:
    
    def setup_method(self):
        self.woe = ManualWOEEncoder()
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'target': [1, 0, 1, 0, 1, 0]
        })
    
    def test_woe_fit_transform(self):
        X = self.df[['category']]
        y = self.df['target']
        X_transformed = self.woe.fit_transform(X, y)
        assert X_transformed.shape[0] == len(X)
    
    def test_iv_summary_returns_df(self):
        X = self.df[['category']]
        y = self.df['target']
        self.woe.fit(X, y)
        iv_df = self.woe.get_iv_summary()
        assert isinstance(iv_df, pd.DataFrame)
        assert 'Feature' in iv_df.columns
        assert 'IV' in iv_df.columns