import pytest
import pandas as pd
import numpy as np
from src.data_processing import FeatureEngineer

class TestFeatureEngineer:
    def setup_method(self):
        self.engineer = FeatureEngineer(random_state=42)
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'TransactionId': [f'T{i:06d}' for i in range(100)],
            'CustomerId': [f'C{np.random.randint(1, 20):04d}' for _ in range(100)],
            'Amount': np.random.exponential(100, 100),
            'TransactionStartTime': pd.date_range('2023-01-01', periods=100, freq='D'),
            'FraudResult': np.random.choice([0, 1], 100, p=[0.98, 0.02])
        })
    
    def test_create_rfm_features_returns_expected_columns(self):
        rfm = self.engineer.create_rfm_features(self.sample_df)
        expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'AvgAmount', 'StdAmount']
        for col in expected_columns:
            assert col in rfm.columns
    
    def test_high_risk_proxy_returns_binary_target(self):
        target_df = self.engineer.create_high_risk_proxy(self.sample_df, n_clusters=2)
        assert 'is_high_risk' in target_df.columns
        assert target_df['is_high_risk'].isin([0, 1]).all()
    
    def test_clean_features_handles_nan(self):
        df_nan = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
        cleaned = self.engineer.clean_features(df_nan)
        assert not cleaned.isna().any().any()