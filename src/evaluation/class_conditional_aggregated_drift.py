import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

aggregated_features = [
    f"feature_{i}" for i in range(95, 166)
]

train = df[df["time_step"] <= 35]
test = df[df["time_step"] >= 42]

results = []

print("\nCLASS-CONDITIONAL AGGREGATED FEATURE DRIFT")
print("=" * 75)

for label in ["licit", "illicit"]:

    train_class = train[train["class"] == label]
    test_class = test[test["class"] == label]

    print(f"\nAnalysing class: {label}")
    print("Training samples:", len(train_class))
    print("Testing samples:", len(test_class))

    for feature in aggregated_features:

        train_values = train_class[feature].dropna()
        test_values = test_class[feature].dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        ks_stat, p_value = ks_2samp(
            train_values,
            test_values
        )

        results.append({
            "class": label,
            "feature": feature,
            "train_mean": train_values.mean(),
            "test_mean": test_values.mean(),
            "mean_absolute_change": abs(
                test_values.mean() - train_values.mean()
            ),
            "ks_statistic": ks_stat,
            "p_value": p_value
        })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["class", "ks_statistic"],
    ascending=[True, False]
)

Path("reports").mkdir(exist_ok=True)

results_df.to_csv(
    "reports/class_conditional_aggregated_drift.csv",
    index=False
)

for label in ["licit", "illicit"]:

    class_results = results_df[
        results_df["class"] == label
    ]

    print(f"\nTOP 15 FEATURES — {label.upper()}")
    print("=" * 75)

    print(
        class_results.head(15).to_string(index=False)
    )

    print("\nFeatures with KS > 0.1:",
          (class_results["ks_statistic"] > 0.1).sum())

    print("Features with KS > 0.2:",
          (class_results["ks_statistic"] > 0.2).sum())

print("\nResults saved successfully.")