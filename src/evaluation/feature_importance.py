import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

print("Loading Temporal Model V2 and features...")

# Load updated model
model = joblib.load(
    "models/temporal_random_forest_v2.pkl"
)

# Load the exact dataset used for V2
df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Load exact feature columns used during training
feature_columns = joblib.load(
    "models/temporal_feature_columns_v2.pkl"
)

# Validate feature count
if len(feature_columns) != len(model.feature_importances_):
    raise ValueError("Feature count does not match model!")

# Extract feature importance
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# Display top 20
print("\nTOP 20 IMPORTANT FEATURES")
print(
    importance_df.head(20).to_string(index=False)
)

# Identify temporal graph features
temporal_features = [
    "temporal_in_degree",
    "temporal_out_degree",
    "temporal_total_degree"
]

print("\nTEMPORAL GRAPH FEATURE IMPORTANCE")

print(
    importance_df[
        importance_df["Feature"].isin(temporal_features)
    ].to_string(index=False)
)

# Calculate total importance of temporal features
temporal_importance = importance_df[
    importance_df["Feature"].isin(temporal_features)
]["Importance"].sum()

print(
    "\nCombined temporal feature importance:",
    round(temporal_importance, 6)
)

# Save results
Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

importance_df.to_csv(
    "reports/feature_importance_temporal_v2.csv",
    index=False
)

# Plot top 20 features
top_features = importance_df.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Temporal Random Forest V2 - Top 20 Features")

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance_temporal_v2.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed.")

print("\nFiles saved:")
print("- reports/feature_importance_temporal_v2.csv")
print("- reports/figures/feature_importance_temporal_v2.png")