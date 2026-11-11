import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"
OUTPUT_PATH = ROOT / "reports/rf_feature_ablation_results.csv"

FEATURES = [f"feature_{i}" for i in range(1, 166)]


def evaluate_model(name, model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "experiment": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "licit_recall": recall_score(y_test, predictions, pos_label=0),
        "illicit_recall": recall_score(y_test, predictions, pos_label=1),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "illicit_pr_auc": average_precision_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }


def main():

    print("Loading dataset and baseline model...")

    df = pd.read_csv(DATA_PATH)
    baseline_model = joblib.load(MODEL_PATH)

    df["target"] = -1
    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    labelled = df[df["target"] >= 0].copy()

    train = labelled[labelled["time_step"] <= 35]
    test = labelled[labelled["time_step"] >= 42]

    X_train = train[FEATURES]
    y_train = train["target"]

    X_test = test[FEATURES]
    y_test = test["target"]

    print("Training transactions:", len(train))
    print("Testing transactions:", len(test))

    # Rank features using training model importance
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": baseline_model.feature_importances_
    }).sort_values("importance", ascending=False)

    results = []

    print("\nExperiment 1: Baseline")

    results.append(
        evaluate_model(
            "Baseline - All 165 features",
            baseline_model,
            X_test,
            y_test
        )
    )

    # Remove top influential features and retrain
    for n in [5, 10, 20]:

        removed_features = importance.head(n)["feature"].tolist()

        selected_features = [
            feature for feature in FEATURES
            if feature not in removed_features
        ]

        print(f"\nExperiment 2: Removing top {n} features")
        print("Removed:", removed_features)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train[selected_features],
            y_train
        )

        result = evaluate_model(
            f"Without top {n} features",
            model,
            X_test[selected_features],
            y_test
        )

        result["removed_features"] = ", ".join(removed_features)

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFINAL ABLATION RESULTS:")
    print(
        results_df.drop(
            columns=["confusion_matrix", "removed_features"],
            errors="ignore"
        ).to_string(index=False)
    )

    print("\nCONFUSION MATRICES:")
    for result in results:
        print(
            result["experiment"],
            "->",
            result["confusion_matrix"]
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()