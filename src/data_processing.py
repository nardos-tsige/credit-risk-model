import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.woe_encoder import ManualWOEEncoder

class FeatureEngineer:
    def __init__(self, snapshot_date=None, random_state=42):
        self.snapshot_date = snapshot_date or pd.Timestamp('2024-01-01')
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = None
        self.woe_encoder = None
        self.pipeline = None
        
    def clean_features(self, X):
        X_clean = X.copy()
        nan_cols = X_clean.columns[X_clean.isna().any()].tolist()
        if nan_cols:
            num_cols = X_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            X_clean[num_cols] = imputer.fit_transform(X_clean[num_cols])
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        if X_clean.isna().any().any():
            imputer = SimpleImputer(strategy='median')
            X_clean = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)
        return X_clean
    
    def build_pipeline(self, categorical_cols, numerical_cols):
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        return self.pipeline
    
    def apply_woe_encoding(self, X, y, categorical_cols):
        from src.woe_encoder import ManualWOEEncoder
        self.woe_encoder = ManualWOEEncoder()
        X_cat = X[categorical_cols].copy()
        X_encoded = self.woe_encoder.fit_transform(X_cat, y)
        X_result = X.drop(columns=categorical_cols)
        X_result = pd.concat([X_result, X_encoded], axis=1)
        print("WOE Encoding applied to:", categorical_cols)
        print(self.woe_encoder.get_iv_summary())
        return X_result
        
    def create_rfm_features(self, df):
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        if df['TransactionStartTime'].dt.tz is not None:
            df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)
        
        recency = df.groupby('CustomerId')['TransactionStartTime'].agg(
            lambda x: max(0, (self.snapshot_date - x.max()).days)).rename('Recency')
        frequency = df.groupby('CustomerId')['TransactionId'].count().rename('Frequency')
        monetary = df.groupby('CustomerId')['Amount'].sum().rename('Monetary')
        avg_amount = df.groupby('CustomerId')['Amount'].mean().rename('AvgAmount')
        std_amount = df.groupby('CustomerId')['Amount'].std().fillna(0).rename('StdAmount')
        
        rfm = pd.DataFrame({
            'Recency': recency, 
            'Frequency': frequency, 
            'Monetary': monetary, 
            'AvgAmount': avg_amount, 
            'StdAmount': std_amount
        })
        rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
        rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
        rfm['Recency_log'] = np.log1p(rfm['Recency'])
        rfm['AvgTransactionValue'] = rfm['Monetary'] / rfm['Frequency']
        return rfm.reset_index()
    
    def create_high_risk_proxy(self, df, n_clusters=3):
        rfm = self.create_rfm_features(df)
        cluster_features = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']].copy()
        for col in cluster_features.columns:
            if cluster_features[col].isna().any():
                cluster_features[col].fillna(cluster_features[col].median(), inplace=True)
        cluster_features = cluster_features.replace([np.inf, -np.inf], 0)
        
        self.scaler = StandardScaler()
        cluster_features_scaled = self.scaler.fit_transform(cluster_features)
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        rfm['Cluster'] = self.kmeans_model.fit_predict(cluster_features_scaled)
        
        cluster_analysis = rfm.groupby('Cluster').agg({
            'Recency': 'mean', 
            'Frequency': 'mean', 
            'Monetary': 'mean'
        })
        cluster_analysis['RiskScore'] = (
            cluster_analysis['Recency'].rank(ascending=False) + 
            cluster_analysis['Frequency'].rank(ascending=True) + 
            cluster_analysis['Monetary'].rank(ascending=True)
        )
        high_risk_cluster = cluster_analysis['RiskScore'].idxmin()
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        return rfm[['CustomerId', 'is_high_risk']]
    
    def create_customer_features(self, df):
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['IsWeekend'] = (df['TransactionStartTime'].dt.dayofweek >= 5).astype(int)
        
        customer_features = df.groupby('CustomerId').agg({
            'TransactionId': 'count', 
            'Amount': ['sum', 'mean', 'std'], 
            'FraudResult': 'mean'
        }).round(2)
        customer_features.columns = ['TransactionCount', 'TotalAmount', 'AvgAmount', 'StdAmount', 'FraudRate']
        customer_features['StdAmount'] = customer_features['StdAmount'].fillna(0)
        rfm = self.create_rfm_features(df)
        customer_features = customer_features.merge(rfm, on='CustomerId', how='left')
        return customer_features.reset_index()
    
    def prepare_data(self, df, create_target=True):
        customer_data = self.create_customer_features(df)
        if create_target:
            target_data = self.create_high_risk_proxy(df)
            customer_data = customer_data.merge(target_data, on='CustomerId', how='left')
            customer_data = customer_data.dropna(subset=['is_high_risk'])
            y = customer_data['is_high_risk'].astype(int)
        else:
            y = None
        exclude_cols = ['CustomerId']
        if create_target:
            exclude_cols.append('is_high_risk')
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]
        X = customer_data[feature_cols]
        X = self.clean_features(X)
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0 and create_target:
            X = self.apply_woe_encoding(X, y, categorical_cols)
        
        return X, y, customer_data['CustomerId']