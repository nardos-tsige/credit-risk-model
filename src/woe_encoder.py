import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ManualWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='target', min_samples=5):
        self.target_col = target_col
        self.min_samples = min_samples
        self.woe_mappings = {}
        self.iv_values = {}
        
    def fit(self, X, y=None):
        if y is not None:
            target = y
        elif self.target_col in X.columns:
            target = X[self.target_col]
        else:
            raise ValueError("Target column not found")
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            woe_dict = {}
            iv_sum = 0
            total_good = (target == 0).sum()
            total_bad = (target == 1).sum()
            
            for category in X[col].unique():
                mask = X[col] == category
                n_good = ((target[mask] == 0).sum())
                n_bad = ((target[mask] == 1).sum())
                
                if n_good < 5:
                    n_good = 5
                if n_bad < 5:
                    n_bad = 5
                
                good_pct = n_good / total_good
                bad_pct = n_bad / total_bad
                
                if bad_pct > 0 and good_pct > 0:
                    woe = np.log(good_pct / bad_pct)
                else:
                    woe = 0
                
                iv_contribution = (good_pct - bad_pct) * woe
                iv_sum += iv_contribution
                woe_dict[category] = woe
            
            self.woe_mappings[col] = woe_dict
            self.iv_values[col] = iv_sum
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.woe_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(0)
        return X_copy
    
    def get_iv_summary(self):
        iv_df = pd.DataFrame({
            'Feature': list(self.iv_values.keys()),
            'IV': list(self.iv_values.values())
        })
        def interpret(iv):
            if iv < 0.02: return 'Not useful'
            elif iv < 0.1: return 'Weak'
            elif iv < 0.3: return 'Medium'
            elif iv < 0.5: return 'Strong'
            else: return 'Suspicious'
        iv_df['Predictive_Power'] = iv_df['IV'].apply(interpret)
        return iv_df.sort_values('IV', ascending=False)