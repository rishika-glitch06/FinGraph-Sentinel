import os
import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

# Load previously calculated drift results
drift_df = pd.read_csv(
    "reports/class_conditional_drift.csv"
)

# Select top 6 unique features using maximum KS statistic
top_features = (
    drift_df.groupby("feature")["ks_statistic"]
    .max()
    .sort_values(ascending=False)
    .head(6)
    .index.tolist()
)

print("\nSelected features:")
print(top_features)

print("\nCalculating feature values across 49 time steps...")

# Calculate mean, median and sample count
timeline = (
    df.groupby(["time_step", "class"])[top_features]
    .agg(["mean", "median", "count"])
)

# Convert grouped results into a flat table
timeline = timeline.stack(level=0).reset_index()

timeline.columns = [
    "time_step",
    "class",
    "feature",
    "mean",
    "median",
    "count"
]

os.makedirs("reports/figures", exist_ok=True)

# Save complete timeline
timeline.to_csv(
    "reports/feature_drift_timeline.csv",
    index=False
)

print("\nFEATURE TIMELINE SUMMARY")
print("=" * 65)

for feature in top_features:

    feature_data = timeline[
        timeline["feature"] == feature
    ]

    print(f"\n{feature}")
    print(
        feature_data[
            ["time_step", "class", "mean", "count"]
        ].round(4).to_string(index=False)
    )

# Create one plot per feature
for feature in top_features:

    feature_data = timeline[
        timeline["feature"] == feature
    ]

    plt.figure(figsize=(10, 5))

    for class_name in ["licit", "illicit"]:

        class_data = feature_data[
            feature_data["class"] == class_name
        ]

        plt.plot(
            class_data["time_step"],
            class_data["mean"],
            marker="o",
            markersize=3,
            label=f"Actual {class_name}"
        )

    plt.xlabel("Time Step")
    plt.ylabel("Mean Feature Value")
    plt.title(f"Temporal Drift: {feature}")
    plt.xticks(range(1, 50, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{feature}_timeline.png",
        dpi=300
    )

    plt.close()

print("\nFILES SAVED")
print("reports/feature_drift_timeline.csv")
print("reports/figures/")

print("\nFEATURE DRIFT TIMELINE ANALYSIS COMPLETED")