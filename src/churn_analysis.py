import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------

df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ---------------------------------------
# 2. Contract Type vs Churn
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("output/contract_vs_churn.png")
plt.show()


# ---------------------------------------
# 3. Tenure vs Churn
# ---------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.savefig("output/tenure_vs_churn.png")
plt.show()


# ---------------------------------------
# 4. Monthly Charges vs Churn
# ---------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.savefig("output/charges_vs_churn.png")
plt.show()


# ---------------------------------------
# 5. Internet Service vs Churn
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("output/internet_vs_churn.png")
plt.show()


# ---------------------------------------
# 6. Churn percentage by Contract
# ---------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract:")
print(contract_churn)


# ---------------------------------------
# 7. Churn percentage by Internet Service
# ---------------------------------------

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
print(internet_churn)


# ---------------------------------------
# 8. Average Tenure by Churn
# ---------------------------------------

average_tenure = df.groupby("Churn")["tenure"].mean()

print("\nAverage Tenure by Churn:")
print(average_tenure)


# ---------------------------------------
# 9. Average Monthly Charges by Churn
# ---------------------------------------

average_charges = df.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage Monthly Charges by Churn:")
print(average_charges)


# ---------------------------------------
# 10. Finish
# ---------------------------------------

print("\nDay 4 Churn Analysis completed successfully!")
print("Charts saved inside the output folder.")