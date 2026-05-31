import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("EDA FOR CREDIT RISK MODELING")
print("=" * 60)

df = pd.read_csv('data/raw/xente_transactions.csv')
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# 1. Amount Distribution
plt.figure(figsize=(10, 5))
plt.hist(df['Amount'].clip(upper=df['Amount'].quantile(0.95)), bins=50, edgecolor='black', alpha=0.7)
plt.title('Transaction Amount Distribution (95th percentile capped)', fontsize=14)
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig('notebooks/amount_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/amount_distribution.png")

# 2. Log Amount Distribution
plt.figure(figsize=(10, 5))
plt.hist(np.log1p(df['Amount']), bins=50, edgecolor='black', alpha=0.7, color='green')
plt.title('Log(Amount) Distribution', fontsize=14)
plt.xlabel('Log(Amount + 1)')
plt.ylabel('Frequency')
plt.savefig('notebooks/log_amount_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/log_amount_distribution.png")

# 3. Fraud Distribution
plt.figure(figsize=(8, 6))
fraud_counts = df['FraudResult'].value_counts()
colors = ['green', 'red']
bars = plt.bar(['No Fraud', 'Fraud'], fraud_counts.values, color=colors, alpha=0.7)
plt.title(f'Fraud Distribution ({df["FraudResult"].mean()*100:.2f}% fraud rate)', fontsize=14)
plt.ylabel('Count')
for bar, count in zip(bars, fraud_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, str(count), ha='center', fontweight='bold')
plt.savefig('notebooks/fraud_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/fraud_distribution.png")

# 4. Top Product Categories
plt.figure(figsize=(12, 8))
top_cats = df['ProductCategory'].value_counts().head(10)
plt.barh(range(len(top_cats)), top_cats.values, color='steelblue', alpha=0.8)
plt.yticks(range(len(top_cats)), top_cats.index)
plt.xlabel('Transaction Count')
plt.title('Top 10 Product Categories', fontsize=14)
plt.gca().invert_yaxis()
plt.savefig('notebooks/top_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/top_categories.png")

# 5. Customer Transaction Distribution
customer_txn = df.groupby('CustomerId')['TransactionId'].count()
plt.figure(figsize=(10, 5))
plt.hist(customer_txn.clip(upper=50), bins=30, edgecolor='black', alpha=0.7, color='purple')
plt.title('Transactions per Customer (capped at 50)', fontsize=14)
plt.xlabel('Number of Transactions')
plt.ylabel('Number of Customers')
plt.savefig('notebooks/customer_transactions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/customer_transactions.png")

# 6. Temporal Patterns
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['Hour'] = df['TransactionStartTime'].dt.hour
hourly = df['Hour'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=8, color='orange')
plt.title('Transactions by Hour of Day', fontsize=14)
plt.xlabel('Hour (0-23)')
plt.ylabel('Number of Transactions')
plt.grid(True, alpha=0.3)
plt.savefig('notebooks/hourly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: notebooks/hourly_pattern.png")

print("\n=== ALL IMAGES GENERATED SUCCESSFULLY ===")
print("Check the 'notebooks' folder for your images.")