
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. PREPARE DATA
# ==========================================

# Remove columns created only for EDA, if present
df = df.drop(
    columns=["Churn_Numeric", "TenureGroup"],
    errors="ignore"
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# Create CustomerType feature
df["CustomerType"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Growing", "Established", "Loyal"]
)

# Convert target into numbers
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Rows:", X_train.shape[0])
print("Testing Rows:", X_test.shape[0])


# ==========================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# ==========================================
# 6. CREATE PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==========================================
# 7. CREATE MACHINE LEARNING MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 8. TRAIN MODELS AND GENERATE PREDICTIONS
# ==========================================

os.makedirs("output/models", exist_ok=True)

for name, classifier in models.items():

    print("\n====================================")
    print("Training:", name)
    print("====================================")

    # Combine preprocessing and model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train using training data
    pipeline.fit(X_train, y_train)

    print(name, "trained successfully!")

    # Generate predictions on test features
    predictions = pipeline.predict(X_test)

    print("\nFirst 10 Predictions:")
    print(predictions[:10])

    # Save the complete pipeline
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        f"output/models/{filename}.pkl"
    )

    print("Model saved:", filename + ".pkl")


# ==========================================
# 9. COMPLETION MESSAGE
# ==========================================

print("\n====================================")
print("DAY 7 COMPLETED SUCCESSFULLY!")
print("All models trained and saved.")
print("====================================")