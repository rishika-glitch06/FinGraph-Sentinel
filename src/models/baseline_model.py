import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATA_DIR / "transactions_advanced_features.csv")

# Use only known labels
df = df[df["class"].isin(["licit", "illicit"])].copy()

df["target"] = (df["class"] == "illicit").astype(int)

# Exclude identifiers, labels and time to avoid leakage
excluded = ["txId", "class", "target", "time_step"]
feature_columns = [
    col for col in df.columns if col not in excluded
]

X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
y = df["target"]

print("Dataset shape:", X.shape)
print("Class distribution:")
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=["licit", "illicit"]
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("PR-AUC:", average_precision_score(y_test, probabilities))

# Save model and scaler
joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

print("\nBASELINE MODEL TRAINED AND SAVED")