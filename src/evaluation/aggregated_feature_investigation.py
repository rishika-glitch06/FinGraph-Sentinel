import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ks_2samp
from pathlib import Path

print("Loading dataset...")

DATA_PATH = Path("data/processed/transactions_temporal_features.csv")

df = pd.read_csv(DATA_PATH)

# Keep only labelled transactions
df = df[df["class"].isin(["licit", "illicit"])].copy()

# Aggregated features: feature_95 to feature_165
aggregated_features = [
    f"feature_{i}" for i in range(95, 166)
]

# Train and test periods
train = df[df["time_step"] <= 35]
test = df[df["time_step"] >= 42]

results = []

print("\nAGGREGATED FEATURE INVESTIGATION")
print("=" * 70)

for feature in aggregated_features:

    train_values = train[feature].dropna()
    test_values = test[feature].dropna()

    if len(train_values) == 0 or len(test_values) == 0:
        continue

    # Temporal correlation
    correlation, _ = spearmanr(
        df[feature],
        df["time_step"]
    )

    # Distribution shift
    ks_stat, p_value = ks_2samp(
        train_values,
        test_values
    )

    results.append({
        "feature": feature,
        "train_mean": train_values.mean(),
        "test_mean": test_values.mean(),
        "mean_absolute_change": abs(
            test_values.mean() - train_values.mean()
        ),
        "spearman_time_correlation": correlation,
        "ks_statistic": ks_stat,
        "p_value": p_value
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ks_statistic",
    ascending=False
)

# Save results
Path("reports").mkdir(exist_ok=True)

output_path = "reports/aggregated_feature_investigation.csv"
results_df.to_csv(output_path, index=False)

print("\nTOP 15 AGGREGATED FEATURES BY DISTRIBUTION SHIFT")
print("=" * 70)

print(
    results_df.head(15).to_string(index=False)
)

print("\nSUMMARY")
print("=" * 70)

print("Total aggregated features:", len(results_df))
print(
    "Features with KS > 0.1:",
    (results_df["ks_statistic"] > 0.1).sum()
)
print(
    "Features with KS > 0.2:",
    (results_df["ks_statistic"] > 0.2).sum()
)
print(
    "Features with |time correlation| > 0.5:",
    (
        results_df["spearman_time_correlation"].abs() > 0.5
    ).sum()
)

print("\nResults saved to:", output_path)