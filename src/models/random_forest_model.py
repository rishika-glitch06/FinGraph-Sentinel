import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

df = df[df["class"].isin(["licit", "illicit"])].copy()
df["target"] = (df["class"] == "illicit").astype(int)

excluded = ["txId", "class", "target", "time_step"]
feature_columns = [c for c in df.columns if c not in excluded]

X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

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

joblib.dump(model, MODEL_DIR / "random_forest.pkl")
joblib.dump(feature_columns, MODEL_DIR / "rf_feature_columns.pkl")

print("\nRANDOM FOREST TRAINED AND SAVED")