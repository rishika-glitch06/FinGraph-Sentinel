import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep only labelled transactions
df = df[df["class"].isin(["licit", "illicit"])].copy()

# Binary target: illicit = 1, licit = 0
df["target"] = (df["class"] == "illicit").astype(int)

# Use original 165 features
feature_columns = [
    col for col in df.columns
    if col.startswith("feature_")
]

X = df[feature_columns]
y = df["target"]

# Test data: time steps 42–49
test_mask = df["time_step"] >= 42

test_df = df[test_mask].copy()

X_test = test_df[feature_columns]
y_test = test_df["target"]

print("Loading trained model...")

model = joblib.load(
    "models/threshold_tuned_random_forest.pkl"
)

threshold = joblib.load(
    "models/optimized_threshold.pkl"
)

print(f"Optimized threshold: {threshold}")

# Get probability of illicit class
probabilities = model.predict_proba(X_test)[:, 1]

test_df["probability"] = probabilities

results = []

print("\nPER-TIME-STEP PERFORMANCE")
print("=" * 65)

for timestep in sorted(test_df["time_step"].unique()):

    step_df = test_df[test_df["time_step"] == timestep]

    actual = step_df["target"].values
    probs = step_df["probability"].values

    predictions = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "time_step": int(timestep),
        "samples": len(step_df),
        "licit_count": int((actual == 0).sum()),
        "illicit_count": int((actual == 1).sum()),
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predictions, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predictions
        ),
        "licit_recall": tn / (tn + fp) if (tn + fp) else 0,
        "illicit_recall": tp / (tp + fn) if (tp + fn) else 0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    if len(np.unique(actual)) == 2:
        metrics["roc_auc"] = roc_auc_score(actual, probs)
        metrics["average_precision"] = average_precision_score(
            actual, probs
        )
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    results.append(metrics)

    print(
        f"Time {timestep}: "
        f"Samples={len(step_df)}, "
        f"Licit Recall={metrics['licit_recall']:.4f}, "
        f"Illicit Recall={metrics['illicit_recall']:.4f}, "
        f"Balanced Accuracy={metrics['balanced_accuracy']:.4f}"
    )

# Save results
results_df = pd.DataFrame(results)

os.makedirs("reports/figures", exist_ok=True)

results_df.to_csv(
    "reports/per_timestep_performance.csv",
    index=False
)

# Plot balanced accuracy
plt.figure(figsize=(10, 5))

plt.plot(
    results_df["time_step"],
    results_df["balanced_accuracy"],
    marker="o"
)

plt.xlabel("Time Step")
plt.ylabel("Balanced Accuracy")
plt.title("Model Performance Across Test Time Steps")
plt.xticks(results_df["time_step"])
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reports/figures/per_timestep_performance.png",
    dpi=300
)

plt.close()

print("\nRESULTS SAVED")
print("reports/per_timestep_performance.csv")
print("reports/figures/per_timestep_performance.png")

print("\nPER-TIME-STEP EVALUATION COMPLETED")