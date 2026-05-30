"""
Unit tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.engineer = FeatureEngineer(random_state=42)
        
        # Create sample data for testing
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'TransactionId': [f'T{i:06d}' for i in range(100)],
            'CustomerId': [f'C{np.random.randint(1, 20):04d}' for _ in range(100)],
            'Amount': np.random.exponential(100, 100),
            'TransactionStartTime': pd.date_range('2023-01-01', periods=100, freq='D'),
            'FraudResult': np.random.choice([0, 1], 100, p=[0.98, 0.02])
        })
    
    def test_create_rfm_features_returns_expected_columns(self):
        """Test that RFM features contains required columns"""
        rfm = self.engineer.create_rfm_features(self.sample_df)
        
        expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 
                           'AvgAmount', 'StdAmount', 'Monetary_log', 
                           'Frequency_log', 'Recency_log', 'AvgTransactionValue']
        
        for col in expected_columns:
            assert col in rfm.columns, f"Missing column: {col}"
    
    def test_rfm_values_are_positive(self):
        """Test that RFM values are non-negative"""
        rfm = self.engineer.create_rfm_features(self.sample_df)
        
        assert (rfm['Frequency'] >= 0).all()
        assert (rfm['Monetary'] >= 0).all()
        assert (rfm['Recency'] >= 0).all()
    
    def test_high_risk_proxy_returns_binary_target(self):
        """Test that proxy target is binary (0 or 1)"""
        target_df = self.engineer.create_high_risk_proxy(self.sample_df, n_clusters=2)
        
        assert 'is_high_risk' in target_df.columns
        assert target_df['is_high_risk'].isin([0, 1]).all()
        assert len(target_df) > 0
    
    def test_clean_features_handles_nan(self):
        """Test that clean_features removes NaN values"""
        # Create DataFrame with NaN
        df_nan = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        
        cleaned = self.engineer.clean_features(df_nan)
        
        assert not cleaned.isna().any().any(), "NaN values still present after cleaning"
    
    def test_prepare_data_returns_correct_shapes(self):
        """Test that prepare_data returns expected shapes"""
        X, y, ids = self.engineer.prepare_data(self.sample_df, create_target=True)
        
        assert len(X) == len(y)
        assert len(X) == len(ids)
        assert X.shape[1] > 0