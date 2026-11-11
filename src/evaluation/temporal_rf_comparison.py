import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "temporal_rf_fair_comparison.pkl"
OUTPUT_PATH = REPORT_DIR / "rf_per_timestep_metrics.csv"

FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 166)
]


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["target"] = -1

    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    df = df[df["target"] >= 0].copy()

    # Same temporal split as GCN
    train = df[df["time_step"] <= 35].copy()

    validation = df[
        (df["time_step"] >= 36)
        & (df["time_step"] <= 41)
    ].copy()

    test = df[df["time_step"] >= 42].copy()

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    print("\nDataset split:")
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Testing:", len(test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

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

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved:", MODEL_PATH)

    # Overall test evaluation
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nOVERALL TEST RESULTS")

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

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_test, predictions)
    )

    print(
        "Macro F1:",
        f1_score(y_test, predictions, average="macro")
    )

    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    print(
        "Illicit PR-AUC:",
        average_precision_score(y_test, probabilities)
    )

    # Per-timestep evaluation
    results = []

    print("\nPER-TIMESTEP RESULTS")

    for timestep in range(42, 50):

        mask = test["time_step"].to_numpy() == timestep

        actual = y_test.to_numpy()[mask]
        predicted = predictions[mask]
        probability = probabilities[mask]

        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                actual,
                predicted,
                labels=[0, 1],
                zero_division=0
            )
        )

        result = {
            "timestep": timestep,
            "transactions": len(actual),
            "licit_support": support[0],
            "illicit_support": support[1],
            "licit_precision": precision[0],
            "licit_recall": recall[0],
            "illicit_precision": precision[1],
            "illicit_recall": recall[1],
            "macro_f1": f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                actual,
                predicted
            ),
            "roc_auc": (
                roc_auc_score(actual, probability)
                if len(np.unique(actual)) == 2
                else np.nan
            ),
            "illicit_pr_auc": average_precision_score(
                actual,
                probability
            )
        }

        results.append(result)

        print(
            f"Step {timestep} | "
            f"Licit Recall: {recall[0]:.4f} | "
            f"Illicit Recall: {recall[1]:.4f} | "
            f"Balanced Accuracy: {result['balanced_accuracy']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nPer-timestep results:")
    print(results_df.to_string(index=False))

    print("\nReport saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()