import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

FEATURES = [f"feature_{i}" for i in range(1, 166)]

THRESHOLD_REPORT = REPORT_DIR / "rf_threshold_validation.csv"
TEST_REPORT = REPORT_DIR / "rf_threshold_test_results.csv"
TIMESTEP_REPORT = REPORT_DIR / "rf_threshold_per_timestep.csv"


def evaluate(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "macro_f1": f1_score(
            y_true, predictions,
            average="macro",
            zero_division=0
        ),
        "licit_recall": np.mean(
            predictions[y_true == 0] == 0
        ),
        "illicit_recall": np.mean(
            predictions[y_true == 1] == 1
        )
    }


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["target"] = -1
    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    df = df[df["target"] >= 0].copy()

    train = df[df["time_step"] <= 35]
    validation = df[
        (df["time_step"] >= 36)
        & (df["time_step"] <= 41)
    ]
    test = df[df["time_step"] >= 42]

    X_train = train[FEATURES]
    y_train = train["target"].to_numpy()

    X_val = validation[FEATURES]
    y_val = validation["target"].to_numpy()

    X_test = test[FEATURES]
    y_test = test["target"].to_numpy()

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "rf_threshold_model.pkl"
    )

    # Validation probabilities
    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Search thresholds using validation only
    threshold_results = []

    thresholds = np.arange(0.10, 0.96, 0.01)

    for threshold in thresholds:

        result = evaluate(
            y_val,
            val_probabilities,
            threshold
        )

        threshold_results.append(result)

    threshold_df = pd.DataFrame(threshold_results)

    threshold_df.to_csv(
        THRESHOLD_REPORT,
        index=False
    )

    # Select threshold with highest validation balanced accuracy
    best_row = threshold_df.loc[
        threshold_df["balanced_accuracy"].idxmax()
    ]

    best_threshold = float(best_row["threshold"])

    print("\nBest validation threshold:", best_threshold)
    print("Validation balanced accuracy:",
          best_row["balanced_accuracy"])
    print("Validation macro F1:", best_row["macro_f1"])
    print("Validation licit recall:", best_row["licit_recall"])
    print("Validation illicit recall:", best_row["illicit_recall"])

    # Test evaluation
    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_results = []

    for threshold in [0.50, best_threshold]:

        predictions = (
            test_probabilities >= threshold
        ).astype(int)

        result = evaluate(
            y_test,
            test_probabilities,
            threshold
        )

        result["accuracy"] = np.mean(
            predictions == y_test
        )

        result["roc_auc"] = roc_auc_score(
            y_test,
            test_probabilities
        )

        result["illicit_pr_auc"] = average_precision_score(
            y_test,
            test_probabilities
        )

        test_results.append(result)

        print(f"\nTEST RESULTS - Threshold {threshold:.2f}")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Licit", "Illicit"],
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    test_df = pd.DataFrame(test_results)

    test_df.to_csv(
        TEST_REPORT,
        index=False
    )

    # Per-timestep evaluation using selected threshold
    predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    timestep_results = []

    for timestep in range(42, 50):

        mask = test["time_step"].to_numpy() == timestep

        actual = y_test[mask]
        predicted = predictions[mask]

        licit_mask = actual == 0
        illicit_mask = actual == 1

        licit_recall = (
            np.mean(predicted[licit_mask] == 0)
            if licit_mask.sum() > 0 else np.nan
        )

        illicit_recall = (
            np.mean(predicted[illicit_mask] == 1)
            if illicit_mask.sum() > 0 else np.nan
        )

        timestep_results.append({
            "timestep": timestep,
            "transactions": len(actual),
            "licit_support": int(licit_mask.sum()),
            "illicit_support": int(illicit_mask.sum()),
            "licit_recall": licit_recall,
            "illicit_recall": illicit_recall,
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            )
        })

    timestep_df = pd.DataFrame(timestep_results)

    timestep_df.to_csv(
        TIMESTEP_REPORT,
        index=False
    )

    print("\nPer-timestep results:")
    print(timestep_df.to_string(index=False))

    print("\nAll reports saved successfully.")


if __name__ == "__main__":
    main()