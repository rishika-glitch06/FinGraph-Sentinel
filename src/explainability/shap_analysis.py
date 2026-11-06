import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

print("Loading temporal Random Forest model...")

model = joblib.load("models/temporal_random_forest.pkl")

df = pd.read_csv(
    "data/processed/transactions_advanced_features.csv"
)

df = df[df["class"].isin(["licit", "illicit"])].copy()

excluded = ["txId", "class", "target", "time_step"]
feature_columns = [
    col for col in df.columns if col not in excluded
]

# Use earlier time steps as background data
train_df = df[df["time_step"] <= 41]
test_df = df[df["time_step"] > 41]

X_background = train_df[feature_columns].sample(
    n=min(100, len(train_df)),
    random_state=42
)

X_explain = test_df[feature_columns].sample(
    n=min(200, len(test_df)),
    random_state=42
)

X_background = X_background.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_explain = X_explain.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

print("Background samples:", len(X_background))
print("Explanation samples:", len(X_explain))

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

# Handle binary classification output
if isinstance(shap_values, list):
    values = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values

print("SHAP values shape:", values.shape)

Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

# SHAP summary plot
plt.figure()

shap.summary_plot(
    values,
    X_explain,
    show=False,
    max_display=20
)

plt.tight_layout()
plt.savefig(
    "reports/figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# SHAP feature importance
plt.figure()

shap.summary_plot(
    values,
    X_explain,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()
plt.savefig(
    "reports/figures/shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP ANALYSIS COMPLETED")
print("Saved:")
print("- reports/figures/shap_summary.png")
print("- reports/figures/shap_importance.png")