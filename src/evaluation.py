import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("output/cleaned_telco_churn.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Prepare the data exactly like Day 7
# --------------------------------------------------

# Remove EDA-only columns if they exist
df = df.drop(
    columns=["Churn_Numeric", "TenureGroup"],
    errors="ignore"
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()


# --------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Convert target variable
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"]


# --------------------------------------------------
# 6. Create the SAME train-test split as Day 7
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 7. Load trained models
# --------------------------------------------------

models = {
    "Logistic Regression":
        joblib.load(
            "output/models/logistic_regression.pkl"
        ),

    "Decision Tree":
        joblib.load(
            "output/models/decision_tree.pkl"
        ),

    "Random Forest":
        joblib.load(
            "output/models/random_forest.pkl"
        )
}


# --------------------------------------------------
# 8. Evaluate each model
# --------------------------------------------------

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of churn
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )


    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })


# --------------------------------------------------
# 9. Compare all models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# 10. Save evaluation results
# --------------------------------------------------

results_df.to_csv(
    "output/model_evaluation_results.csv",
    index=False
)

print(
    "\nEvaluation results saved to:"
    " output/model_evaluation_results.csv"
)


# --------------------------------------------------
# 11. Find best model based on F1-score
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model["Model"])
print(f"Accuracy : {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall   : {best_model['Recall']:.4f}")
print(f"F1-Score : {best_model['F1-Score']:.4f}")
print(f"ROC-AUC  : {best_model['ROC-AUC']:.4f}")