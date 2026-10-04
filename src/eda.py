import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Dataset shape:", df.shape)

# 1. Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("output/churn_distribution.png")
plt.show()


# 2. Tenure distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", bins=30, kde=True)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure in Months")
plt.ylabel("Number of Customers")

plt.savefig("output/tenure_distribution.png")
plt.show()


# 3. Monthly charges distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyCharges", bins=30, kde=True)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.savefig("output/monthly_charges.png")
plt.show()