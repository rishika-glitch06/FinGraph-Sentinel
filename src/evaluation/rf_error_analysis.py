import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"
REPORT_DIR = ROOT / "reports"

FEATURE_REPORT = REPORT_DIR / "rf_error_feature_analysis.csv"
ERROR_REPORT = REPORT_DIR / "rf_misclassified_transactions.csv"
SUMMARY_REPORT = REPORT_DIR / "rf_error_summary.csv"

FEATURES = [f"feature_{i}" for i in range(1, 166)]

THRESHOLD = 0.50


def main():

    print("Loading dataset and model...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df["target"] = -1
    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    test = df[
        (df["time_step"] >= 42)
        & (df["target"] >= 0)
    ].copy()

    X_test = test[FEATURES]
    y_test = test["target"].to_numpy()

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    test["illicit_probability"] = probabilities
    test["predicted_class"] = predictions

    # Error categories
    test["error_type"] = "Correct"

    test.loc[
        (test["target"] == 0) & (predictions == 1),
        "error_type"
    ] = "False Positive"

    test.loc[
        (test["target"] == 1) & (predictions == 0),
        "error_type"
    ] = "False Negative"

    test.loc[
        (test["target"] == 0) & (predictions == 0),
        "error_type"
    ] = "True Negative"

    test.loc[
        (test["target"] == 1) & (predictions == 1),
        "error_type"
    ] = "True Positive"

    print("\nERROR DISTRIBUTION:")

    error_counts = (
        test["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    error_counts["percentage"] = (
        error_counts["count"] / len(test) * 100
    )

    print(error_counts.to_string(index=False))

    # Export misclassified transactions
    errors = test[
        test["error_type"].isin([
            "False Positive",
            "False Negative"
        ])
    ].copy()

    errors.to_csv(
        ERROR_REPORT,
        index=False
    )

    print("\nMisclassified transactions:", len(errors))

    # Per-timestep error summary
    summary = (
        test.groupby(["time_step", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_REPORT,
        index=False
    )

    print("\nPER-TIMESTEP ERROR SUMMARY:")
    print(summary.to_string(index=False))

    # Compare features for actual licit transactions
    # that were correctly and incorrectly classified.
    licit = test[test["target"] == 0]

    correct_licit = licit[
        licit["predicted_class"] == 0
    ]

    incorrect_licit = licit[
        licit["predicted_class"] == 1
    ]

    print("\nCorrectly classified licit:", len(correct_licit))
    print("Misclassified licit:", len(incorrect_licit))

    feature_results = []

    for feature in FEATURES:

        correct_values = correct_licit[feature].dropna()
        incorrect_values = incorrect_licit[feature].dropna()

        if len(correct_values) == 0 or len(incorrect_values) == 0:
            continue

        feature_results.append({
            "feature": feature,
            "correct_licit_mean": correct_values.mean(),
            "misclassified_licit_mean": incorrect_values.mean(),
            "correct_licit_median": correct_values.median(),
            "misclassified_licit_median": incorrect_values.median(),
            "mean_difference": (
                incorrect_values.mean() - correct_values.mean()
            ),
            "absolute_mean_difference": abs(
                incorrect_values.mean() - correct_values.mean()
            )
        })

    feature_df = pd.DataFrame(feature_results)

    feature_df = feature_df.sort_values(
        "absolute_mean_difference",
        ascending=False
    )

    feature_df.to_csv(
        FEATURE_REPORT,
        index=False
    )

    print("\nTOP 20 FEATURES DIFFERENTIATING LICIT ERRORS:")
    print(feature_df.head(20).to_string(index=False))

    # Per-timestep false positive rate among licit
    licit_summary = []

    for timestep, group in licit.groupby("time_step"):

        total = len(group)

        false_positives = (
            group["predicted_class"] == 1
        ).sum()

        licit_summary.append({
            "timestep": timestep,
            "licit_transactions": total,
            "false_positives": false_positives,
            "false_positive_rate": (
                false_positives / total
            )
        })

    licit_summary_df = pd.DataFrame(licit_summary)

    print("\nLICIT FALSE POSITIVE ANALYSIS:")
    print(licit_summary_df.to_string(index=False))

    print("\nSaved files:")
    print(ERROR_REPORT)
    print(SUMMARY_REPORT)
    print(FEATURE_REPORT)


if __name__ == "__main__":
    main()