import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

feature_columns = [
    col for col in df.columns
    if col.startswith("feature_")
]

# Training period: 1–35
train_df = df[df["time_step"] <= 35]

# Testing period: 42–49
test_df = df[df["time_step"] >= 42]

results = []

print("\nCLASS-CONDITIONAL FEATURE DRIFT")
print("=" * 70)

for class_name in ["licit", "illicit"]:

    train_class = train_df[
        train_df["class"] == class_name
    ]

    test_class = test_df[
        test_df["class"] == class_name
    ]

    print(f"\nAnalysing class: {class_name}")
    print("Training samples:", len(train_class))
    print("Testing samples:", len(test_class))

    for feature in feature_columns:

        train_values = train_class[feature].dropna()
        test_values = test_class[feature].dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        ks_stat, p_value = ks_2samp(
            train_values,
            test_values
        )

        results.append({
            "class": class_name,
            "feature": feature,
            "train_mean": train_values.mean(),
            "test_mean": test_values.mean(),
            "ks_statistic": ks_stat,
            "p_value": p_value
        })

results_df = pd.DataFrame(results)

# Save complete results
os.makedirs("reports/figures", exist_ok=True)

results_df.to_csv(
    "reports/class_conditional_drift.csv",
    index=False
)

# Display top 15 features separately
for class_name in ["licit", "illicit"]:

    class_results = results_df[
        results_df["class"] == class_name
    ].sort_values(
        "ks_statistic",
        ascending=False
    )

    print(f"\nTOP 15 DRIFTING FEATURES: {class_name.upper()}")
    print("=" * 70)

    print(
        class_results.head(15).round(6).to_string(index=False)
    )

    print("\nDrift summary:")
    print(
        "KS > 0.1:",
        (class_results["ks_statistic"] > 0.1).sum()
    )

    print(
        "KS > 0.2:",
        (class_results["ks_statistic"] > 0.2).sum()
    )

# Plot class-wise drift
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

for ax, class_name in zip(axes, ["licit", "illicit"]):

    class_results = results_df[
        results_df["class"] == class_name
    ].nlargest(15, "ks_statistic")

    ax.barh(
        class_results["feature"],
        class_results["ks_statistic"]
    )

    ax.set_title(f"{class_name.capitalize()} Feature Drift")
    ax.set_xlabel("KS Statistic")
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig(
    "reports/figures/class_conditional_drift.png",
    dpi=300
)

plt.close()

print("\nFILES SAVED")
print("reports/class_conditional_drift.csv")
print("reports/figures/class_conditional_drift.png")

print("\nCLASS-CONDITIONAL DRIFT ANALYSIS COMPLETED")