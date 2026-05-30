"""
Manual Weight of Evidence (WoE) Implementation
No external dependencies needed - works with Python 3.13
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ManualWOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence Encoder - Manual implementation
    WoE = ln(% of non-events / % of events)
    """
    
    def __init__(self, target_col='target', min_samples=5):
        self.target_col = target_col
        self.min_samples = min_samples
        self.woe_mappings = {}
        self.iv_values = {}
        
    def fit(self, X, y=None):
        """Calculate WoE for each categorical feature"""
        
        # Handle target
        if y is not None:
            target = y
        elif self.target_col in X.columns:
            target = X[self.target_col]
        else:
            raise ValueError("Target column not found")
        
        # Get categorical columns (string or object type)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            woe_dict = {}
            iv_sum = 0
            
            # Calculate global distribution
            total_good = (target == 0).sum()
            total_bad = (target == 1).sum()
            
            # For each category
            for category in X[col].unique():
                mask = X[col] == category
                
                # Count goods (0) and bads (1) in this category
                n_good = ((target[mask] == 0).sum())
                n_bad = ((target[mask] == 1).sum())
                
                # Apply minimum samples to avoid division by zero
                if n_good < self.min_samples:
                    n_good = self.min_samples
                if n_bad < self.min_samples:
                    n_bad = self.min_samples
                
                # Calculate percentages
                good_pct = n_good / total_good
                bad_pct = n_bad / total_bad
                
                # Calculate WoE
                if bad_pct > 0 and good_pct > 0:
                    woe = np.log(good_pct / bad_pct)
                else:
                    woe = 0
                
                # Calculate IV contribution
                iv_contribution = (good_pct - bad_pct) * woe
                iv_sum += iv_contribution
                
                woe_dict[category] = woe
            
            self.woe_mappings[col] = woe_dict
            self.iv_values[col] = iv_sum
        
        return self
    
    def transform(self, X):
        """Apply WoE transformation"""
        X_copy = X.copy()
        
        for col, mapping in self.woe_mappings.items():
            if col in X_copy.columns:
                # Replace with WoE values
                X_copy[col] = X_copy[col].map(mapping)
                # Fill NaN with 0 (mean WoE)
                X_copy[col] = X_copy[col].fillna(0)
        
        return X_copy
    
    def get_iv_summary(self):
        """Get Information Value summary"""
        iv_df = pd.DataFrame({
            'Feature': list(self.iv_values.keys()),
            'IV': list(self.iv_values.values())
        })
        
        # Add interpretation
        def interpret(iv):
            if iv < 0.02:
                return 'Not useful'
            elif iv < 0.1:
                return 'Weak'
            elif iv < 0.3:
                return 'Medium'
            elif iv < 0.5:
                return 'Strong'
            else:
                return 'Suspicious'
        
        iv_df['Predictive_Power'] = iv_df['IV'].apply(interpret)
        return iv_df.sort_values('IV', ascending=False)


# Test the encoder
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
        'amount': [100, 200, 150, 300, 50, 75, 180, 250],
        'target': [1, 0, 1, 0, 1, 0, 1, 1]
    })
    
    X = df[['category', 'amount']]
    y = df['target']
    
    # Apply WoE
    encoder = ManualWOEEncoder()
    X_encoded = encoder.fit_transform(X, y)
    
    print("Original data:")
    print(X.head())
    print("\nAfter WoE encoding:")
    print(X_encoded.head())
    print("\nIV Summary:")
    print(encoder.get_iv_summary())