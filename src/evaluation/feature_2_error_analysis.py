import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"

REPORT_PATH = ROOT / "reports/feature_2_error_analysis.csv"
FIGURE_PATH = ROOT / "reports/figures/feature_2_error_comparison.png"


def main():

    print("Loading dataset and Random Forest model...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Known labels only
    df["target"] = df["class"].map({
        "licit": 0,
        "illicit": 1
    })

    df = df[df["target"].notna()].copy()

    # Chronological test set: timesteps 42-49
    test = df[df["time_step"] >= 42].copy()

    features = [f"feature_{i}" for i in range(1, 166)]

    X_test = test[features]
    y_test = test["target"].astype(int)

    # Reproduce RF predictions at threshold 0.5
    probabilities = model.predict_proba(X_test)[:, 1]

    test["illicit_probability"] = probabilities
    test["predicted_class"] = (
        probabilities >= 0.5
    ).astype(int)

    # Correctly classified licit transactions
    correct_licit = test[
        (test["target"] == 0) &
        (test["predicted_class"] == 0)
    ]

    # False positive licit transactions
    incorrect_licit = test[
        (test["target"] == 0) &
        (test["predicted_class"] == 1)
    ]

    print("\nGROUP SIZES")
    print("Correctly classified licit:", len(correct_licit))
    print("Misclassified licit:", len(incorrect_licit))

    feature = "feature_2"

    correct_values = correct_licit[feature].dropna()
    incorrect_values = incorrect_licit[feature].dropna()

    print("\nFEATURE 2 STATISTICS")

    print("\nCorrectly classified licit:")
    print(correct_values.describe())

    print("\nMisclassified licit:")
    print(incorrect_values.describe())

    # Statistical tests
    ks_stat, ks_p = ks_2samp(
        correct_values,
        incorrect_values
    )

    mw_stat, mw_p = mannwhitneyu(
        correct_values,
        incorrect_values,
        alternative="two-sided"
    )

    mean_difference = (
        incorrect_values.mean() - correct_values.mean()
    )

    print("\nSTATISTICAL COMPARISON")

    print("Mean difference:", mean_difference)
    print("KS statistic:", ks_stat)
    print("KS p-value:", ks_p)
    print("Mann-Whitney statistic:", mw_stat)
    print("Mann-Whitney p-value:", mw_p)

    # Compare across timesteps
    print("\nPER-TIMESTEP COMPARISON")

    timestep_results = []

    for step in sorted(test["time_step"].unique()):

        correct_step = correct_licit[
            correct_licit["time_step"] == step
        ][feature]

        incorrect_step = incorrect_licit[
            incorrect_licit["time_step"] == step
        ][feature]

        timestep_results.append({
            "time_step": step,
            "correct_count": len(correct_step),
            "incorrect_count": len(incorrect_step),
            "correct_mean": correct_step.mean(),
            "incorrect_mean": incorrect_step.mean()
        })

    timestep_df = pd.DataFrame(timestep_results)

    print(timestep_df.to_string(index=False))

    timestep_df.to_csv(
        ROOT / "reports/feature_2_error_by_timestep.csv",
        index=False
    )

    # Save overall results
    results = pd.DataFrame([{
        "feature": feature,
        "correct_count": len(correct_values),
        "incorrect_count": len(incorrect_values),
        "correct_mean": correct_values.mean(),
        "incorrect_mean": incorrect_values.mean(),
        "mean_difference": mean_difference,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "mannwhitney_p_value": mw_p
    }])

    results.to_csv(REPORT_PATH, index=False)

    # Visualization
    plt.figure(figsize=(10, 6))

    plt.hist(
        correct_values,
        bins=50,
        density=True,
        alpha=0.5,
        label="Correctly classified licit"
    )

    plt.hist(
        incorrect_values,
        bins=50,
        density=True,
        alpha=0.5,
        label="False positive licit"
    )

    plt.xlabel("Feature 2 Value")
    plt.ylabel("Density")
    plt.title("Feature 2: Correct vs Misclassified Licit")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    print("\nINVESTIGATION COMPLETED")
    print("Saved:", REPORT_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()