import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Original shape:", df.shape)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove missing rows
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Remove customerID
df = df.drop("customerID", axis=1)

print("\nCleaned shape:", df.shape)

# Save cleaned dataset
df.to_csv(
    "output/cleaned_telco_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")