import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

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

# Training and testing periods
train_df = df[df["time_step"] <= 35]
test_df = df[df["time_step"] >= 42]

print("\nDATA SPLIT")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Features:", len(feature_columns))

print("\nCalculating feature distribution drift...")

results = []

for feature in feature_columns:

    train_values = train_df[feature].dropna()
    test_values = test_df[feature].dropna()

    if len(train_values) == 0 or len(test_values) == 0:
        continue

    ks_stat, p_value = ks_2samp(
        train_values,
        test_values
    )

    train_mean = train_values.mean()
    test_mean = test_values.mean()

    results.append({
        "feature": feature,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "mean_difference": abs(test_mean - train_mean),
        "ks_statistic": ks_stat,
        "p_value": p_value
    })

results_df = pd.DataFrame(results)

# Rank features by KS statistic
results_df = results_df.sort_values(
    by="ks_statistic",
    ascending=False
)

print("\nTOP 20 FEATURES WITH DISTRIBUTION DRIFT")
print("=" * 75)

print(
    results_df.head(20).round(6).to_string(index=False)
)

# Summary
print("\nDRIFT SUMMARY")
print("=" * 50)

print("Total features analysed:", len(results_df))
print(
    "Features with KS statistic > 0.1:",
    (results_df["ks_statistic"] > 0.1).sum()
)
print(
    "Features with KS statistic > 0.2:",
    (results_df["ks_statistic"] > 0.2).sum()
)

# Save results
os.makedirs("reports/figures", exist_ok=True)

results_df.to_csv(
    "reports/feature_drift_results.csv",
    index=False
)

# Plot top 20 features
top_features = results_df.head(20).sort_values(
    "ks_statistic"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["ks_statistic"]
)

plt.xlabel("KS Statistic")
plt.ylabel("Feature")
plt.title("Top 20 Features with Temporal Distribution Drift")
plt.tight_layout()

plt.savefig(
    "reports/figures/feature_drift_analysis.png",
    dpi=300
)

plt.close()

print("\nFILES SAVED")
print("reports/feature_drift_results.csv")
print("reports/figures/feature_drift_analysis.png")

print("\nFEATURE DRIFT ANALYSIS COMPLETED")