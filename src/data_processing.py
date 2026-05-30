"""
Complete Feature Engineering Pipeline for Credit Risk Modeling
Handles: RFM analysis, feature extraction, WoE encoding, and target creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our manual WoE encoder
from src.woe_encoder import ManualWOEEncoder


class FeatureEngineer:
    """
    Main feature engineering class for credit risk modeling
    Creates features and proxy target variable using RFM and clustering
    """
    
    def __init__(self, snapshot_date=None, random_state=42):
        """
        Initialize the feature engineer
        
        Args:
            snapshot_date: Date to use for recency calculation (default: 2024-01-01)
            random_state: Random seed for reproducibility
        """
        self.snapshot_date = snapshot_date or pd.Timestamp('2024-01-01')
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = None
        self.woe_encoder = None
        
    def clean_features(self, X):
        """
        Clean features by handling NaN and infinite values
        
        Args:
            X: Feature DataFrame
            
        Returns:
            X_clean: Cleaned feature DataFrame
        """
        X_clean = X.copy()
        
        # Check for NaN values
        nan_cols = X_clean.columns[X_clean.isna().any()].tolist()
        if nan_cols:
            print(f"  Handling NaN in columns: {nan_cols}")
            
            # Impute numerical columns with median
            num_cols = X_clean.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='median')
            X_clean[num_cols] = imputer.fit_transform(X_clean[num_cols])
        
        # Replace infinite values with NaN then impute
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        if X_clean.isna().any().any():
            imputer = SimpleImputer(strategy='median')
            X_clean = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)
        
        return X_clean
        
    def create_rfm_features(self, df):
        """
        Create Recency, Frequency, Monetary features per customer
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with RFM features per customer
        """
        # Convert to datetime and remove timezone if present
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Remove timezone info if it exists
        if df['TransactionStartTime'].dt.tz is not None:
            df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)
        
        # Ensure snapshot_date has no timezone
        if hasattr(self.snapshot_date, 'tz') and self.snapshot_date.tz is not None:
            self.snapshot_date = self.snapshot_date.tz_localize(None)
        
        # Calculate RFM metrics - separate aggregations to avoid column issues
        recency = df.groupby('CustomerId')['TransactionStartTime'].agg(
            lambda x: max(0, (self.snapshot_date - x.max()).days)
        ).rename('Recency')
        
        frequency = df.groupby('CustomerId')['TransactionId'].count().rename('Frequency')
        
        monetary = df.groupby('CustomerId')['Amount'].sum().rename('Monetary')
        
        avg_amount = df.groupby('CustomerId')['Amount'].mean().rename('AvgAmount')
        
        std_amount = df.groupby('CustomerId')['Amount'].std().fillna(0).rename('StdAmount')
        
        # Combine all series into a DataFrame
        rfm = pd.DataFrame({
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'AvgAmount': avg_amount,
            'StdAmount': std_amount
        })
        
        # Add log transformations for skewed features
        rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
        rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
        rfm['Recency_log'] = np.log1p(rfm['Recency'])
        
        # Add average transaction value
        rfm['AvgTransactionValue'] = rfm['Monetary'] / rfm['Frequency']
        
        return rfm.reset_index()
    
    def create_high_risk_proxy(self, df, n_clusters=3):
        """
        Create proxy target variable using RFM and K-Means clustering
        High-risk customers = least engaged (low frequency, low monetary, high recency)
        
        Args:
            df: DataFrame with transaction data
            n_clusters: Number of clusters for segmentation
            
        Returns:
            DataFrame with CustomerId and is_high_risk target
        """
        print("Creating proxy target variable...")
        
        # Calculate RFM features
        rfm = self.create_rfm_features(df)
        
        # Prepare features for clustering (using log-transformed values)
        cluster_features = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']].copy()
        
        # Check for NaN values and handle them
        print(f"NaN check before imputation:")
        print(f"  Recency_log NaN count: {cluster_features['Recency_log'].isna().sum()}")
        print(f"  Frequency_log NaN count: {cluster_features['Frequency_log'].isna().sum()}")
        print(f"  Monetary_log NaN count: {cluster_features['Monetary_log'].isna().sum()}")
        
        # Fill NaN values with median
        for col in cluster_features.columns:
            if cluster_features[col].isna().any():
                median_val = cluster_features[col].median()
                cluster_features[col].fillna(median_val, inplace=True)
                print(f"  Filled NaN in {col} with median: {median_val}")
        
        # Also check for infinite values
        cluster_features = cluster_features.replace([np.inf, -np.inf], 0)
        
        # Standardize features
        self.scaler = StandardScaler()
        cluster_features_scaled = self.scaler.fit_transform(cluster_features)
        
        # Apply K-Means clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        rfm['Cluster'] = self.kmeans_model.fit_predict(cluster_features_scaled)
        
        # Analyze clusters to identify high-risk (least engaged)
        cluster_analysis = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        })
        
        print("\nCluster Profiles:")
        print(cluster_analysis)
        
        # High-risk cluster = highest recency (least recent) + lowest frequency + lowest monetary
        # Calculate risk score for each cluster
        cluster_analysis['RiskScore'] = (
            cluster_analysis['Recency'].rank(ascending=False) +  # Higher recency = higher risk
            cluster_analysis['Frequency'].rank(ascending=True) +   # Lower frequency = higher risk
            cluster_analysis['Monetary'].rank(ascending=True)      # Lower monetary = higher risk
        )
        
        high_risk_cluster = cluster_analysis['RiskScore'].idxmin()
        
        # Create binary target
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        
        print(f"\nHigh-risk cluster identified: {high_risk_cluster}")
        print(f"High-risk rate: {rfm['is_high_risk'].mean():.2%}")
        
        return rfm[['CustomerId', 'is_high_risk']]
    
    def create_customer_features(self, df):
        """
        Create customer-level features from transaction data
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with customer features
        """
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Remove timezone if exists
        if df['TransactionStartTime'].dt.tz is not None:
            df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)
        
        # Extract time features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['IsWeekend'] = (df['TransactionDayOfWeek'] >= 5).astype(int)
        
        # Aggregate per customer
        customer_features = df.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'FraudResult': 'mean',
            'TransactionHour': 'mean',
            'IsWeekend': 'mean'
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'TransactionCount', 'TotalAmount', 'AvgAmount', 'StdAmount', 
            'MinAmount', 'MaxAmount', 'FraudRate', 'AvgTransactionHour',
            'WeekendRatio'
        ]
        
        # Fill NaN values
        customer_features['StdAmount'] = customer_features['StdAmount'].fillna(0)
        
        # Add RFM features
        rfm = self.create_rfm_features(df)
        customer_features = customer_features.merge(rfm, on='CustomerId', how='left')
        
        return customer_features.reset_index()
    
    def prepare_data(self, df, create_target=True):
        """
        Main method to prepare data for modeling
        
        Args:
            df: Raw transaction DataFrame
            create_target: Whether to create proxy target variable
            
        Returns:
            X: Feature matrix
            y: Target variable (if create_target=True)
            customer_ids: Customer IDs
        """
        print("=" * 60)
        print("PREPARING DATA FOR CREDIT RISK MODELING")
        print("=" * 60)
        
        # Create customer features
        print("\n1. Creating customer features...")
        customer_data = self.create_customer_features(df)
        
        # Create target if requested
        if create_target:
            print("\n2. Creating proxy target variable...")
            target_data = self.create_high_risk_proxy(df)
            customer_data = customer_data.merge(target_data, on='CustomerId', how='left')
            
            # Remove any rows with missing target
            customer_data = customer_data.dropna(subset=['is_high_risk'])
            
            y = customer_data['is_high_risk'].astype(int)
            print(f"\nTarget distribution:\n{y.value_counts().to_dict()}")
        else:
            y = None
        
        # Select features for modeling (exclude CustomerId and target)
        exclude_cols = ['CustomerId']
        if create_target:
            exclude_cols.append('is_high_risk')
        
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]
        X = customer_data[feature_cols]
        
        # Clean features - handle NaN and infinite values
        print("\n3. Cleaning features for modeling...")
        X = self.clean_features(X)
        
        print(f"\n4. Feature set created:")
        print(f"   - Number of customers: {len(X)}")
        print(f"   - Number of features: {len(X.columns)}")
        print(f"   - Features: {list(X.columns)}")
        
        return X, y, customer_data['CustomerId']
    
    def apply_woe_encoding(self, X, y, categorical_cols=None):
        """
        Apply Weight of Evidence encoding to categorical features
        
        Args:
            X: Feature DataFrame
            y: Target variable
            categorical_cols: List of categorical columns (auto-detected if None)
            
        Returns:
            X_woe: DataFrame with WoE encoded features
        """
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) == 0:
            print("No categorical columns found for WoE encoding")
            return X
        
        print(f"\nApplying WoE encoding to: {categorical_cols}")
        
        # Initialize and fit WoE encoder
        self.woe_encoder = ManualWOEEncoder(min_samples=5)
        
        # Create a copy with only categorical columns for encoding
        X_cat = X[categorical_cols].copy()
        
        # Fit and transform
        X_encoded = self.woe_encoder.fit_transform(X_cat, y)
        
        # Drop original categorical columns and add encoded ones
        X_result = X.drop(columns=categorical_cols)
        X_result = pd.concat([X_result, X_encoded], axis=1)
        
        # Clean again after WoE encoding
        X_result = self.clean_features(X_result)
        
        # Display IV summary
        print("\nInformation Value Summary:")
        print(self.woe_encoder.get_iv_summary())
        
        return X_result


# Quick test
if __name__ == "__main__":
    # Create sample data for testing
    print("Testing FeatureEngineer with sample data...")
    
    # Generate sample transactions
    np.random.seed(42)
    n_transactions = 5000
    n_customers = 500
    
    sample_df = pd.DataFrame({
        'TransactionId': [f'T{i:06d}' for i in range(n_transactions)],
        'CustomerId': [f'C{np.random.randint(1, n_customers):04d}' for _ in range(n_transactions)],
        'Amount': np.random.exponential(100, n_transactions),
        'TransactionStartTime': pd.date_range('2023-01-01', periods=n_transactions, freq='H'),
        'FraudResult': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
    })
    
    # Initialize engineer
    engineer = FeatureEngineer()
    
    # Prepare data
    X, y, customer_ids = engineer.prepare_data(sample_df, create_target=True)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE - FeatureEngineer is working!")
    print("=" * 60)