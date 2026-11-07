import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep labelled transactions only
df = df[df["class"].isin(["licit", "illicit"])].copy()

df["target"] = (df["class"] == "illicit").astype(int)

# Original 165 features
feature_columns = [
    col for col in df.columns
    if col.startswith("feature_")
]

# Test data: time steps 42–49
test_df = df[df["time_step"] >= 42].copy()

X_test = test_df[feature_columns]

print("Loading trained model...")

model = joblib.load(
    "models/threshold_tuned_random_forest.pkl"
)

threshold = joblib.load(
    "models/optimized_threshold.pkl"
)

# Predict probabilities
test_df["illicit_probability"] = model.predict_proba(X_test)[:, 1]

licit = test_df[
    test_df["class"] == "licit"
]["illicit_probability"]

illicit = test_df[
    test_df["class"] == "illicit"
]["illicit_probability"]

print("\nPROBABILITY DISTRIBUTION")
print("=" * 60)

print("\nACTUAL LICIT TRANSACTIONS")
print(licit.describe().round(4))

print("\nACTUAL ILLICIT TRANSACTIONS")
print(illicit.describe().round(4))

# Prediction behaviour
test_df["predicted_class"] = (
    test_df["illicit_probability"] >= threshold
).astype(int)

print("\nPREDICTION BREAKDOWN")
print("=" * 60)

print(pd.crosstab(
    test_df["class"],
    test_df["predicted_class"],
    rownames=["Actual Class"],
    colnames=["Predicted Class"]
))

# Probability distribution plot
os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(10, 6))

plt.hist(
    licit,
    bins=40,
    alpha=0.6,
    density=True,
    label="Actual Licit"
)

plt.hist(
    illicit,
    bins=40,
    alpha=0.6,
    density=True,
    label="Actual Illicit"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.2f}"
)

plt.xlabel("Predicted Probability of Illicit Class")
plt.ylabel("Density")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/figures/score_distribution.png",
    dpi=300
)

plt.close()

# Save transaction-level probabilities
test_df[
    ["txId", "time_step", "class", "illicit_probability", "predicted_class"]
].to_csv(
    "reports/score_distribution.csv",
    index=False
)

print("\nFILES SAVED")
print("reports/score_distribution.csv")
print("reports/figures/score_distribution.png")

print("\nSCORE DISTRIBUTION ANALYSIS COMPLETED")