import os
import pandas as pd
from scipy.stats import spearmanr

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

edges = pd.read_csv(
    "data/processed/edges.csv"
)

os.makedirs("reports", exist_ok=True)

# --------------------------------------------------
# 1. FEATURE GROUPS
# --------------------------------------------------

feature_columns = [
    col for col in df.columns
    if col.startswith("feature_")
]

local_features = [
    f"feature_{i}" for i in range(1, 95)
]

aggregated_features = [
    f"feature_{i}" for i in range(95, 166)
]

print("\nFEATURE GROUPS")
print("=" * 60)

print("Local features:", len(local_features))
print("Aggregated features:", len(aggregated_features))
print("Total original features:", len(feature_columns))


# --------------------------------------------------
# 2. FEATURE CORRELATION WITH TIME
# --------------------------------------------------

print("\n1. CHECKING CORRELATION WITH TIME")

correlation_results = []

for feature in feature_columns:

    corr, p_value = spearmanr(
        df["time_step"],
        df[feature]
    )

    correlation_results.append({
        "feature": feature,
        "time_correlation": corr,
        "p_value": p_value,
        "absolute_correlation": abs(corr)
    })

correlation_df = pd.DataFrame(correlation_results)

correlation_df = correlation_df.sort_values(
    "absolute_correlation",
    ascending=False
)

correlation_df.to_csv(
    "reports/feature_time_correlation.csv",
    index=False
)

print("\nTop 15 time-correlated features:")

print(
    correlation_df.head(15)
    .round(5)
    .to_string(index=False)
)


# --------------------------------------------------
# 3. LOCAL VS AGGREGATED FEATURE DRIFT
# --------------------------------------------------

print("\n2. COMPARING FEATURE GROUPS")

train = df[df["time_step"] <= 35]
test = df[df["time_step"] >= 42]

group_results = []

for group_name, features in [
    ("Local", local_features),
    ("Aggregated", aggregated_features)
]:

    values = []

    for feature in features:

        if feature not in df.columns:
            continue

        train_mean = train[feature].mean()
        test_mean = test[feature].mean()

        corr, _ = spearmanr(
            df["time_step"],
            df[feature]
        )

        values.append({
            "feature": feature,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "absolute_mean_change": abs(
                test_mean - train_mean
            ),
            "time_correlation": abs(corr)
        })

    group_df = pd.DataFrame(values)

    group_results.append({
        "feature_group": group_name,
        "feature_count": len(group_df),
        "average_absolute_mean_change":
            group_df["absolute_mean_change"].mean(),
        "average_absolute_time_correlation":
            group_df["time_correlation"].mean()
    })

group_summary = pd.DataFrame(group_results)

print("\nFeature group comparison:")

print(
    group_summary.round(5).to_string(index=False)
)

group_summary.to_csv(
    "reports/feature_group_audit.csv",
    index=False
)


# --------------------------------------------------
# 4. CROSS-TIME GRAPH EDGE ANALYSIS
# --------------------------------------------------

print("\n3. CHECKING CROSS-TIME GRAPH EDGES")

# Map transaction IDs to their time steps
time_map = df.set_index("txId")["time_step"]

# Edges CSV columns are source and target
edges["source_time"] = edges["source"].map(time_map)

edges["target_time"] = edges["target"].map(time_map)

valid_edges = edges.dropna(
    subset=["source_time", "target_time"]
).copy()

valid_edges["time_difference"] = (
    valid_edges["target_time"] -
    valid_edges["source_time"]
)

same_time_edges = valid_edges[
    valid_edges["time_difference"] == 0
]

cross_time_edges = valid_edges[
    valid_edges["time_difference"] != 0
]

future_edges = valid_edges[
    valid_edges["time_difference"] > 0
]

past_edges = valid_edges[
    valid_edges["time_difference"] < 0
]

print("\nEDGE ANALYSIS RESULTS")

print("Total edges:", len(edges))
print("Valid edges:", len(valid_edges))
print("Same-time edges:", len(same_time_edges))
print("Cross-time edges:", len(cross_time_edges))
print("Source-to-future edges:", len(future_edges))
print("Source-to-past edges:", len(past_edges))


# --------------------------------------------------
# 5. FINAL AUDIT SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "Total original features",
        "Local features",
        "Aggregated features",
        "Features with absolute time correlation > 0.5",
        "Features with absolute time correlation > 0.8",
        "Total graph edges",
        "Valid graph edges",
        "Same-time graph edges",
        "Cross-time graph edges",
        "Source-to-future edges",
        "Source-to-past edges"
    ],

    "value": [
        len(feature_columns),
        len(local_features),
        len(aggregated_features),
        (
            correlation_df["absolute_correlation"] > 0.5
        ).sum(),
        (
            correlation_df["absolute_correlation"] > 0.8
        ).sum(),
        len(edges),
        len(valid_edges),
        len(same_time_edges),
        len(cross_time_edges),
        len(future_edges),
        len(past_edges)
    ]
})

summary.to_csv(
    "reports/feature_leakage_audit_summary.csv",
    index=False
)

print("\nFINAL AUDIT SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))


# --------------------------------------------------
# 6. FINAL NOTES
# --------------------------------------------------

print("\nIMPORTANT OBSERVATIONS")
print("=" * 60)

print("1. High time correlation does not prove data leakage.")

print("2. Cross-time edges do not automatically prove leakage.")

print("3. Feature construction history must be checked separately.")

print("4. Same-time edges may introduce information availability concerns.")

print("\nFEATURE LEAKAGE AUDIT COMPLETED")