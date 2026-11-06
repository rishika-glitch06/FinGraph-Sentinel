import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score
)

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep known labels only
df = df[df["class"].isin(["licit", "illicit"])].copy()

df["target"] = (df["class"] == "illicit").astype(int)

# Use original 165 transaction features
features = [
    col for col in df.columns
    if col.startswith("feature_")
]

# Chronological train-validation-test split
train_df = df[df["time_step"] <= 35].copy()
val_df = df[
    (df["time_step"] >= 36) &
    (df["time_step"] <= 41)
].copy()
test_df = df[df["time_step"] >= 42].copy()

print("\nDATA SPLIT")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

def prepare_data(data):
    return data[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

X_train = prepare_data(train_df)
X_val = prepare_data(val_df)
X_test = prepare_data(test_df)

y_train = train_df["target"]
y_val = val_df["target"]
y_test = test_df["target"]

# Train model
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Validation probabilities
val_probabilities = model.predict_proba(X_val)[:, 1]

# Find best threshold using validation balanced accuracy
thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0.5
best_score = -1

print("\nOPTIMIZING THRESHOLD...")

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        y_val,
        val_predictions
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\nBEST VALIDATION THRESHOLD:", best_threshold)
print("Validation Balanced Accuracy:", best_score)

# Final test probabilities
test_probabilities = model.predict_proba(X_test)[:, 1]

# Compare default and optimized thresholds
for name, threshold in [
    ("Default Threshold", 0.50),
    ("Optimized Threshold", best_threshold)
]:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    print("\n" + "=" * 55)
    print(name, "| Threshold:", threshold)
    print("=" * 55)

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["licit", "illicit"],
        zero_division=0
    ))

    print("\nConfusion Matrix:")

    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ))

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_test, predictions)
    )

    print(
        "Macro F1:",
        f1_score(y_test, predictions, average="macro")
    )

    print(
        "Licit Recall:",
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )["0"]["recall"]
    )

    print(
        "Illicit Recall:",
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )["1"]["recall"]
    )

    print(
        "ROC-AUC:",
        roc_auc_score(y_test, test_probabilities)
    )

    print(
        "Licit PR-AUC:",
        average_precision_score(
            1 - y_test,
            1 - test_probabilities
        )
    )

# Save model and selected threshold
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/threshold_tuned_random_forest.pkl"
)

joblib.dump(
    best_threshold,
    "models/optimized_threshold.pkl"
)

print("\nTHRESHOLD OPTIMIZATION COMPLETED")
print("Model saved.")
print("Optimized threshold saved.")