import pandas as pd
import joblib

# --------------------------------------------------
# 1. Load original dataset
# --------------------------------------------------
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Original dataset:", df.shape)

# Keep customer IDs for final business report
customer_ids = df["customerID"]

# --------------------------------------------------
# 2. Clean TotalCharges
# --------------------------------------------------
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

print("After cleaning:", df.shape)

# --------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------
df["CustomerType"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Growing", "Established", "Loyal"]
)

# --------------------------------------------------
# 4. Prepare data for prediction
# --------------------------------------------------
X = df.drop(columns=["customerID", "Churn"])

# --------------------------------------------------
# 5. Load Decision Tree model
# --------------------------------------------------
model = joblib.load(
    "output/models/decision_tree.pkl"
)

print("Decision Tree model loaded successfully.")

# --------------------------------------------------
# 6. Predict churn probability
# --------------------------------------------------
churn_probability = model.predict_proba(X)[:, 1]

# Convert probability into percentage
churn_percentage = churn_probability * 100

# --------------------------------------------------
# 7. Create risk categories
# --------------------------------------------------
def risk_category(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


risk = [risk_category(p) for p in churn_probability]

# --------------------------------------------------
# 8. Create final results
# --------------------------------------------------
result = pd.DataFrame({
    "CustomerID": df["customerID"],
    "ChurnProbability": churn_percentage.round(2),
    "RiskLevel": risk,
    "ActualChurn": df["Churn"]
})

# --------------------------------------------------
# 9. Sort by highest risk
# --------------------------------------------------
result = result.sort_values(
    by="ChurnProbability",
    ascending=False
)

# --------------------------------------------------
# 10. Save results
# --------------------------------------------------
result.to_csv(
    "output/churn_risk_customers.csv",
    index=False
)

# --------------------------------------------------
# 11. Display results
# --------------------------------------------------
print("\nTop 10 High-Risk Customers:")
print(result.head(10))

print("\nRisk Distribution:")
print(result["RiskLevel"].value_counts())

print("\nRisk prediction completed successfully!")

print(
    "\nFile saved as: "
    "output/churn_risk_customers.csv"
)