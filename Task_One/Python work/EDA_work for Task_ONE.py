
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer

# Load Data
df = pd.read_csv("C:\\Users\\sande\\Downloads\\Delinquency_prediction_dataset.csv")  # Replace with your file path

df

# ======================
# 1. DATA QUALITY CHECKS
# ======================

# Standardize categorical values
df['Employment_Status'] = df['Employment_Status'].str.title().replace('Emp', 'Employed')

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Handle missing data
imputer = IterativeImputer(max_iter=10, random_state=42)
num_cols = ['Income', 'Credit_Score', 'Loan_Balance']
df[num_cols] = imputer.fit_transform(df[num_cols])

# Fix outliers
df = df[df['Credit_Score'] >= 300]  # Remove invalid credit scores

# ======================
# 2. EXPLORATORY VISUALIZATION
# ======================

# Set style
sns.set(style="whitegrid")

# Plot 1: Delinquency by Employment Status
plt.figure(figsize=(10,6))
sns.barplot(x='Employment_Status', y='Delinquent_Account', data=df)
plt.title("Delinquency Rate by Employment Status")
plt.show()

# Plot 2: Credit Utilization Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Credit_Utilization'], bins=30, kde=True)
plt.axvline(x=0.8, color='r', linestyle='--', label='High Risk Threshold')
plt.title("Credit Utilization Distribution")
plt.show()

# Plot 3: Payment History Heatmap
payment_cols = ['Month_1','Month_2','Month_3','Month_4','Month_5','Month_6']
payment_df = df[payment_cols].apply(lambda x: x.map({'On-time':0, 'Late':1, 'Missed':2}))
plt.figure(figsize=(12,6))
sns.heatmap(payment_df.corr(), annot=True, cmap='coolwarm')
plt.title("Payment Behavior Correlation")
plt.show()

# ======================
# 3. RISK FACTOR ANALYSIS
# ======================

# Create risk flags
df['High_Risk_Flag'] = np.where(
    (df['Missed_Payments'] >= 3) | 
    (df['Credit_Utilization'] > 0.8), 1, 0)

# Risk segmentation
risk_groups = df.groupby('High_Risk_Flag').agg({
    'Delinquent_Account': 'mean',
    'Income': 'median',
    'Age': 'median'
}).rename(columns={'Delinquent_Account': 'Delinquency_Rate'})

print("\nRisk Group Analysis:\n", risk_groups)

# ======================
# 4. ADVANCED ANALYSIS
# ======================

# Payment consistency score
df['Payment_Score'] = payment_df.apply(
    lambda x: (x == 0).sum()/len(x), axis=1)  # % on-time payments

# Risk correlation matrix
corr_matrix = df[['Delinquent_Account','Credit_Utilization',
                 'Missed_Payments','Debt_to_Income_Ratio',
                 'Payment_Score']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Risk Factor Correlations")
plt.show()

# ======================
# 5. EXPORT INSIGHTS
# ======================

# Save cleaned data
df.to_csv('geldium_data_cleaned.csv', index=False)

# Generate summary report
with open('eda_summary.txt', 'w') as f:
    f.write("=== Key EDA Findings ===\n")
    f.write(f"High-risk customers: {df['High_Risk_Flag'].mean():.1%}\n")
    f.write(f"Top delinquency predictor: Missed Payments (r={corr_matrix.loc['Missed_Payments','Delinquent_Account']:.2f})\n")
    f.write("\n=== Data Quality Issues ===\n")
    f.write(f"Standardized employment categories: {df['Employment_Status'].unique()}\n")
    f.write(f"Imputed missing values: {num_cols}\n")

































































































































































































