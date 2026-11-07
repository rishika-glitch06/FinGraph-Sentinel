import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep labelled transactions
df = df[df["class"].isin(["licit", "illicit"])].copy()

# Original 165 features
feature_columns = [
    col for col in df.columns
    if col.startswith("feature_")
]

# Test period: time steps 42–49
test_df = df[df["time_step"] >= 42].copy()

X_test = test_df[feature_columns]

print("Loading trained model...")

model = joblib.load(
    "models/threshold_tuned_random_forest.pkl"
)

# Predict probability of illicit class
test_df["illicit_probability"] = (
    model.predict_proba(X_test)[:, 1]
)

print("\nTEMPORAL PROBABILITY ANALYSIS")
print("=" * 65)

# Calculate probability statistics for each time step and class
summary = (
    test_df.groupby(["time_step", "class"])["illicit_probability"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

print(summary.round(4).to_string(index=False))

# Save results
os.makedirs("reports/figures", exist_ok=True)

summary.to_csv(
    "reports/temporal_probability_drift.csv",
    index=False
)

# Plot mean probability across time
plt.figure(figsize=(11, 6))

for class_name in ["licit", "illicit"]:

    class_data = summary[
        summary["class"] == class_name
    ]

    plt.plot(
        class_data["time_step"],
        class_data["mean"],
        marker="o",
        label=f"Actual {class_name}"
    )

plt.xlabel("Time Step")
plt.ylabel("Mean Predicted Illicit Probability")
plt.title("Temporal Probability Drift Across Test Time Steps")
plt.xticks(range(42, 50))
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "reports/figures/temporal_probability_drift.png",
    dpi=300
)

plt.close()

print("\nFILES SAVED")
print("reports/temporal_probability_drift.csv")
print("reports/figures/temporal_probability_drift.png")

print("\nTEMPORAL PROBABILITY DRIFT ANALYSIS COMPLETED")