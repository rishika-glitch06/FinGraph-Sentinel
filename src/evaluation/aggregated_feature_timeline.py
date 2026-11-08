import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

# Six features showing the strongest class-conditional drift
selected_features = [
    "feature_100",
    "feature_101",
    "feature_103",
    "feature_136",
    "feature_137",
    "feature_139"
]

print("\nAGGREGATED FEATURE TIMELINE ANALYSIS")
print("=" * 70)

# Calculate statistics for every time step and class
timeline = (
    df.groupby(["time_step", "class"])[selected_features]
    .agg(["mean", "median", "count"])
)

timeline.columns = [
    f"{feature}_{stat}"
    for feature, stat in timeline.columns
]

timeline = timeline.reset_index()

# Save timeline data
Path("reports").mkdir(exist_ok=True)
Path("reports/figures").mkdir(exist_ok=True)

timeline.to_csv(
    "reports/aggregated_feature_timeline.csv",
    index=False
)

print("\nTimeline data shape:", timeline.shape)
print("Time steps analysed:", timeline["time_step"].nunique())

# Generate individual plots
for feature in selected_features:

    plt.figure(figsize=(12, 6))

    for label in ["licit", "illicit"]:

        class_data = timeline[
            timeline["class"] == label
        ]

        plt.plot(
            class_data["time_step"],
            class_data[f"{feature}_mean"],
            marker="o",
            label=label
        )

    plt.title(f"Temporal Trend: {feature}")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Feature Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{feature}_timeline.png",
        dpi=300
    )

    plt.close()

print("\nTIMELINE ANALYSIS COMPLETED")
print("=" * 70)

print("Generated plots:", len(selected_features))
print("CSV saved: reports/aggregated_feature_timeline.csv")
print("Plots saved in: reports/figures/")