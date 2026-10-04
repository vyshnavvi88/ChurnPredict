
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Original Dataset Shape:")
print(df.shape)


# ==========================================
# 2. REMOVE ANALYSIS-ONLY COLUMNS
# ==========================================

columns_to_remove = [
    "Churn_Numeric",
    "TenureGroup"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

df["CustomerType"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "New",
        "Growing",
        "Established",
        "Loyal"
    ]
)

print("\nCustomer Type Distribution:")
print(df["CustomerType"].value_counts())


# ==========================================
# 4. CONVERT TOTAL CHARGES
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()


# ==========================================
# 5. CONVERT TARGET INTO NUMBERS
# ==========================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Churn", axis=1)

y = df["Churn"]


# ==========================================
# 7. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==========================================
# 8. CREATE PREPROCESSING SYSTEM
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ==========================================
# 9. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ==========================================
# 10. PROCESS DATA
# ==========================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)


# ==========================================
# 11. FINAL MESSAGE
# ==========================================

print("\n====================================")
print("Day 6 Preprocessing Completed!")
print("Data is ready for Machine Learning.")
print("====================================")