"""
EDA for Credit Risk Modeling - Xente Transactions Data
Run with: python notebooks/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CREDIT RISK MODELING - EXPLORATORY DATA ANALYSIS")
print("Xente eCommerce Platform Transaction Data")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n" + "=" * 60)
print("1. DATA LOADING")
print("=" * 60)

df = pd.read_csv('data/raw/xente_transactions.csv')
print(f"✓ Dataset loaded successfully")
print(f"  - Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. DATA OVERVIEW
# ============================================================================
print("\n" + "=" * 60)
print("2. DATA OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics for numerical columns:")
print(df.describe())

# ============================================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("3. MISSING VALUES ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print("\nColumns with missing values:")
    print(missing_df)
else:
    print("\n✓ No missing values found in the dataset!")

# ============================================================================
# 4. NUMERICAL FEATURES DISTRIBUTION
# ============================================================================
print("\n" + "=" * 60)
print("4. NUMERICAL FEATURES DISTRIBUTION")
print("=" * 60)

numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumerical columns: {list(numerical_cols)}")

# Create histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount distribution
axes[0, 0].hist(df['Amount'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Amount')
axes[0, 0].set_ylabel('Frequency')

# Log Amount distribution
axes[0, 1].hist(np.log1p(df['Amount'].dropna()), bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Log(Amount) Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Log(Amount + 1)')
axes[0, 1].set_ylabel('Frequency')

# Fraud distribution
fraud_counts = df['FraudResult'].value_counts()
axes[1, 0].bar(['No Fraud (0)', 'Fraud (1)'], fraud_counts.values, color=['blue', 'red'], alpha=0.7)
axes[1, 0].set_title('Fraud Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Count')
for i, v in enumerate(fraud_counts.values):
    axes[1, 0].text(i, v + 100, str(v), ha='center', fontweight='bold')

# Value distribution
if 'Value' in df.columns:
    axes[1, 1].hist(df['Value'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_title('Transaction Value Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('notebooks/eda_numerical_distributions.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: notebooks/eda_numerical_distributions.png")

print(f"\nAmount Statistics:")
print(f"  - Mean: ${df['Amount'].mean():.2f}")
print(f"  - Median: ${df['Amount'].median():.2f}")
print(f"  - Std: ${df['Amount'].std():.2f}")
print(f"  - Skewness: {df['Amount'].skew():.2f}")
print(f"  - Kurtosis: {df['Amount'].kurtosis():.2f}")

# ============================================================================
# 5. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("5. CATEGORICAL FEATURES ANALYSIS")
print("=" * 60)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(categorical_cols)}")

# Product Category Analysis
if 'ProductCategory' in df.columns:
    print("\n Product Category Distribution:")
    cat_counts = df['ProductCategory'].value_counts()
    print(cat_counts.head(10))
    
    # Plot top categories
    fig, ax = plt.subplots(figsize=(12, 6))
    top_cats = cat_counts.head(10)
    ax.barh(range(len(top_cats)), top_cats.values, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_cats)))
    ax.set_yticklabels(top_cats.index)
    ax.set_xlabel('Transaction Count')
    ax.set_title('Top 10 Product Categories', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('notebooks/eda_top_categories.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: notebooks/eda_top_categories.png")

# ============================================================================
# 6. TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("6. TEMPORAL ANALYSIS")
print("=" * 60)

# Convert to datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Extract time features
df['Hour'] = df['TransactionStartTime'].dt.hour
df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
df['Month'] = df['TransactionStartTime'].dt.month

# Hourly pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

hourly = df['Hour'].value_counts().sort_index()
axes[0].plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=6)
axes[0].set_title('Transactions by Hour of Day', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hour (0-23)')
axes[0].set_ylabel('Number of Transactions')
axes[0].grid(True, alpha=0.3)

# Day of week pattern
dow = df['DayOfWeek'].value_counts().sort_index()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].bar(days, dow.values, color='coral', alpha=0.7)
axes[1].set_title('Transactions by Day of Week', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Number of Transactions')

plt.tight_layout()
plt.savefig('notebooks/eda_temporal_patterns.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: notebooks/eda_temporal_patterns.png")

print(f"\nPeak transaction hour: {hourly.idxmax()}:00 ({hourly.max():,} transactions)")
print(f"Busiest day: {days[dow.idxmax()]}")

# ============================================================================
# 7. CUSTOMER-LEVEL ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("7. CUSTOMER-LEVEL ANALYSIS")
print("=" * 60)

customer_metrics = df.groupby('CustomerId').agg({
    'TransactionId': 'count',
    'Amount': ['sum', 'mean', 'std']
}).round(2)

customer_metrics.columns = ['TransactionCount', 'TotalAmount', 'AvgAmount', 'StdAmount']
customer_metrics['StdAmount'] = customer_metrics['StdAmount'].fillna(0)

print(f"\nCustomer Metrics Summary:")
print(f"  - Total unique customers: {len(customer_metrics):,}")
print(f"  - Avg transactions per customer: {customer_metrics['TransactionCount'].mean():.2f}")
print(f"  - Median transactions per customer: {customer_metrics['TransactionCount'].median():.0f}")
print(f"  - Max transactions per customer: {customer_metrics['TransactionCount'].max():.0f}")
print(f"  - Avg total spend per customer: ${customer_metrics['TotalAmount'].mean():.2f}")

# Customer segmentation (for RFM)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Transactions per customer distribution
trans_counts = customer_metrics['TransactionCount'].clip(upper=50)
axes[0].hist(trans_counts, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Transactions per Customer (capped at 50)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Transactions')
axes[0].set_ylabel('Number of Customers')

# Total amount per customer (log scale)
axes[1].hist(np.log1p(customer_metrics['TotalAmount']), bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Log(Total Spend) per Customer', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Log(Total Amount + 1)')
axes[1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('notebooks/eda_customer_distributions.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: notebooks/eda_customer_distributions.png")

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("8. CORRELATION ANALYSIS")
print("=" * 60)

# Select numerical columns for correlation
corr_cols = ['Amount', 'FraudResult']
available_cols = [col for col in corr_cols if col in df.columns]
corr_matrix = df[available_cols].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# ============================================================================
# 9. KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("9. KEY INSIGHTS FOR MODELING")
print("=" * 60)

insights = [
    "1. TRANSACTION AMOUNT SKEWNESS",
    "   - Transaction amounts are highly right-skewed",
    "   - Recommendation: Use log transformation for monetary features",
    "",
    "2. CUSTOMER ACTIVITY VARIATION",
    "   - Wide variation in customer transaction frequency",
    "   - Median customers have few transactions",
    "   - Recommendation: Create RFM (Recency, Frequency, Monetary) features",
    "",
    "3. TEMPORAL PATTERNS",
    f"   - Peak activity at {hourly.idxmax()}:00",
    f"   - Busiest day: {days[dow.idxmax()]}",
    "   - Recommendation: Extract hour, day, month features",
    "",
    "4. PRODUCT CATEGORY CONCENTRATION",
    "   - Top categories dominate transaction volume",
    "   - Recommendation: One-hot encode top categories, group others as 'Other'",
    "",
    "5. PROXY TARGET STRATEGY",
    "   - No default label available",
    "   - Recommendation: Use RFM + K-Means clustering to identify high-risk customers",
    "   - High-risk = low frequency, low monetary, high recency"
]

for insight in insights:
    print(insight)

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("10. EDA SUMMARY")
print("=" * 60)

print("""
  Data Quality: Good - ready for modeling
  Features Available: Transaction amount, time, category, fraud flags
  Missing: Direct default labels (will use proxy target)
  Modeling Approach: 
   - RFM analysis for customer segmentation
   - K-Means clustering for proxy target creation
   - Random Forest for classification (based on high performance)
   - MLflow for experiment tracking
""")

print("\n" + "=" * 80)
print("EDA COMPLETE! Ready for Feature Engineering and Model Training")
print("=" * 80)