
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 2. CONVERT CHURN INTO NUMERICAL VALUES
# ==========================================

# Keep a numerical version for correlation analysis
df["Churn_Numeric"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn Value Counts:")
print(df["Churn"].value_counts())


# ==========================================
# 3. CORRELATION HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap Including Churn")

plt.tight_layout()

plt.savefig("output/correlation_with_churn.png")

plt.show()


# ==========================================
# 4. ONLINE SECURITY VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OnlineSecurity",
    hue="Churn"
)

plt.title("Online Security vs Churn")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("output/online_security_vs_churn.png")

plt.show()


# ==========================================
# 5. TECH SUPPORT VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="TechSupport",
    hue="Churn"
)

plt.title("Tech Support vs Churn")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("output/tech_support_vs_churn.png")

plt.show()


# ==========================================
# 6. ONLINE BACKUP VS CHURN
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OnlineBackup",
    hue="Churn"
)

plt.title("Online Backup vs Churn")
plt.xlabel("Online Backup")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("output/online_backup_vs_churn.png")

plt.show()


# ==========================================
# 7. CHURN PERCENTAGE BY ONLINE SECURITY
# ==========================================

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Security:")
print(security_churn)


# ==========================================
# 8. CHURN PERCENTAGE BY TECH SUPPORT
# ==========================================

support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tech Support:")
print(support_churn)


# ==========================================
# 9. CHURN PERCENTAGE BY ONLINE BACKUP
# ==========================================

backup_churn = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Backup:")
print(backup_churn)


# ==========================================
# 10. CUSTOMER SEGMENTATION BY TENURE
# ==========================================

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]
)

tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tenure Group:")
print(tenure_churn)


# ==========================================
# 11. TENURE GROUP CHURN GRAPH
# ==========================================

tenure_churn_plot = tenure_churn.copy()

if "Yes" in tenure_churn_plot.columns:
    tenure_churn_plot["Yes"].plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Churn Percentage by Tenure Group")
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Percentage (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("output/tenure_group_churn.png")

    plt.show()


# ==========================================
# 12. OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    y="MonthlyCharges"
)

plt.title("Monthly Charges Outlier Analysis")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig("output/monthly_charges_outliers.png")

plt.show()


# ==========================================
# 13. AVERAGE TENURE BY CHURN
# ==========================================

average_tenure = df.groupby("Churn")["tenure"].mean()

print("\nAverage Tenure by Churn:")
print(average_tenure)


# ==========================================
# 14. AVERAGE MONTHLY CHARGES BY CHURN
# ==========================================

average_charges = df.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage Monthly Charges by Churn:")
print(average_charges)


# ==========================================
# 15. FINAL MESSAGE
# ==========================================

print("\n====================================")
print("Day 5 Advanced EDA Completed!")
print("Charts saved in the output folder.")
print("====================================")