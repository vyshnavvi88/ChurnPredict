import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Dataset shape
print("Dataset Shape:")
print(df.shape)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())