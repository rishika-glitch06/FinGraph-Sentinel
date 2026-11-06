import pandas as pd
import numpy as np
import shap
import joblib

from pathlib import Path
from scipy.stats import spearmanr

print("Loading model and dataset...")

model = joblib.load("models/temporal_random_forest.pkl")

df = pd.read_csv(
    "data/processed/transactions_advanced_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

excluded = ["txId", "class", "target", "time_step"]

features = [
    col for col in df.columns if col not in excluded
]

# Chronological training and testing data
train_df = df[df["time_step"] <= 41]
test_df = df[df["time_step"] > 41]

# Keep the same test transactions for both experiments
X_test = test_df[features].sample(
    n=min(100, len(test_df)),
    random_state=42
)

# Two different background datasets
background_A = train_df[features].sample(
    n=50,
    random_state=42
)

background_B = train_df[features].sample(
    n=50,
    random_state=100
)

def clean_data(data):
    return data.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

X_test = clean_data(X_test)
background_A = clean_data(background_A)
background_B = clean_data(background_B)

print("Test samples:", len(X_test))
print("Background A:", len(background_A))
print("Background B:", len(background_B))

# Calculate SHAP values
def calculate_shap(background):

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw"
    )

    values = explainer.shap_values(X_test)

    if isinstance(values, list):
        values = values[1]

    elif isinstance(values, np.ndarray) and values.ndim == 3:
        values = values[:, :, 1]

    return values


print("\nCalculating SHAP with Background A...")
shap_A = calculate_shap(background_A)

print("\nCalculating SHAP with Background B...")
shap_B = calculate_shap(background_B)

# Compare feature importance
importance_A = np.abs(shap_A).mean(axis=0)
importance_B = np.abs(shap_B).mean(axis=0)

comparison = pd.DataFrame({
    "Feature": features,
    "Importance_A": importance_A,
    "Importance_B": importance_B
})

comparison["Difference"] = abs(
    comparison["Importance_A"] -
    comparison["Importance_B"]
)

comparison = comparison.sort_values(
    "Importance_A",
    ascending=False
)

# Rank correlation
rank_A = comparison["Importance_A"].rank(
    ascending=False
)

rank_B = comparison["Importance_B"].rank(
    ascending=False
)

correlation, p_value = spearmanr(rank_A, rank_B)

# Top 10 overlap
top_A = set(
    comparison.nlargest(10, "Importance_A")["Feature"]
)

top_B = set(
    comparison.nlargest(10, "Importance_B")["Feature"]
)

overlap = len(top_A.intersection(top_B))

# Results
print("\n========== SHAP STABILITY RESULTS ==========")

print("Spearman Rank Correlation:", correlation)
print("P-value:", p_value)

print("Top 10 Feature Overlap:", overlap, "/ 10")

print(
    "Mean Absolute Importance Difference:",
    comparison["Difference"].mean()
)

print("\nTOP 15 FEATURES")
print(comparison.head(15).to_string(index=False))

# Save results
Path("reports").mkdir(exist_ok=True)

comparison.to_csv(
    "reports/shap_stability_comparison.csv",
    index=False
)

print("\nSHAP STABILITY TEST COMPLETED")