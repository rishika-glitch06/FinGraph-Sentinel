import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score
)

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep only known labels
df = df[df["class"].isin(["licit", "illicit"])].copy()

df["target"] = (df["class"] == "illicit").astype(int)

# Chronological split
df = df.sort_values("time_step")

split_step = df["time_step"].quantile(0.80)

train_df = df[df["time_step"] <= split_step].copy()
test_df = df[df["time_step"] > split_step].copy()

# Original transaction features
original_features = [
    col for col in df.columns
    if col.startswith("feature_")
]

# Newly engineered temporal graph features
temporal_features = [
    "temporal_in_degree",
    "temporal_out_degree",
    "temporal_total_degree"
]

print("\nOriginal features:", len(original_features))
print("Temporal graph features:", len(temporal_features))

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

# Prepare target
y_train = train_df["target"]
y_test = test_df["target"]

# Same model configuration for both experiments
def train_and_evaluate(name, features):

    print(f"\n{'='*55}")
    print(f"EXPERIMENT: {name}")
    print(f"Number of features: {len(features)}")
    print(f"{'='*55}")

    X_train = train_df[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    X_test = test_df[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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

    results = {
        "Model": name,
        "Features": len(features),
        "Accuracy": (predictions == y_test).mean(),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro"
        ),
        "Licit Recall": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )["0"]["recall"],
        "Illicit Recall": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )["1"]["recall"]
    }

    if y_test.nunique() == 2:
        results["ROC-AUC"] = roc_auc_score(
            y_test, probabilities
        )

        results["Illicit PR-AUC"] = average_precision_score(
            y_test, probabilities
        )

        results["Licit PR-AUC"] = average_precision_score(
            1 - y_test, 1 - probabilities
        )

    return results


# Experiment A: Original features only
results_original = train_and_evaluate(
    "Original Features Only",
    original_features
)

# Experiment B: Original + temporal graph features
results_temporal = train_and_evaluate(
    "Original + Temporal Graph Features",
    original_features + temporal_features
)

# Compare results
comparison = pd.DataFrame([
    results_original,
    results_temporal
])

print("\n\nFINAL ABLATION COMPARISON")
print(comparison.to_string(index=False))

# Save results
Path("reports").mkdir(exist_ok=True)

comparison.to_csv(
    "reports/temporal_ablation_results.csv",
    index=False
)

print("\nAblation study completed.")
print("Saved: reports/temporal_ablation_results.csv")