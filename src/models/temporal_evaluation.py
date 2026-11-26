import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

print("Loading temporal dataset...")

df = pd.read_csv(
    "data/processed/transactions_temporal_features.csv"
)

# Keep only known labels
df = df[df["class"].isin(["licit", "illicit"])].copy()

# Target: 1 = illicit, 0 = licit
df["target"] = (df["class"] == "illicit").astype(int)

# Sort chronologically
df = df.sort_values("time_step")

# Chronological 80:20 split
split_step = df["time_step"].quantile(0.80)

train_df = df[df["time_step"] <= split_step].copy()
test_df = df[df["time_step"] > split_step].copy()

# Exclude identifiers and target information
excluded = ["txId", "class", "target", "time_step"]

feature_columns = [
    col for col in df.columns
    if col not in excluded
]

X_train = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_test = test_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

y_train = train_df["target"]
y_test = test_df["target"]

print("\nTRAINING TIME STEPS:")
print(sorted(train_df["time_step"].unique()))

print("\nTESTING TIME STEPS:")
print(sorted(test_df["time_step"].unique()))

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Total features:", len(feature_columns))

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nTEMPORAL CLASSIFICATION REPORT")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["licit", "illicit"],
    zero_division=0
))

print("\nCONFUSION MATRIX")

print(confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
))

if y_test.nunique() == 2:

    print(
        "\nROC-AUC (illicit):",
        roc_auc_score(y_test, probabilities)
    )

    print(
        "PR-AUC (illicit):",
        average_precision_score(y_test, probabilities)
    )

    print(
        "PR-AUC (licit):",
        average_precision_score(
            1 - y_test,
            1 - probabilities
        )
    )

else:
    print("\nOnly one class present in test set.")

# Save model and feature columns separately
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/temporal_random_forest_v2.pkl"
)

joblib.dump(
    feature_columns,
    "models/temporal_feature_columns_v2.pkl"
)

print("\nTEMPORAL MODEL V2 TRAINED AND SAVED")