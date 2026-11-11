import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"

REPORT_PATH = ROOT / "reports/feature_2_outlier_analysis.csv"
FIGURE_PATH = ROOT / "reports/figures/feature_2_outlier_analysis.png"


def main():

    print("Loading dataset and model...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df = df[df["class"].isin(["licit", "illicit"])].copy()

    df["target"] = df["class"].map({
        "licit": 0,
        "illicit": 1
    })

    # Focus on timestep 42
    test = df[df["time_step"] == 42].copy()

    features = [f"feature_{i}" for i in range(1, 166)]

    probabilities = model.predict_proba(
        test[features]
    )[:, 1]

    test["predicted_class"] = (
        probabilities >= 0.5
    ).astype(int)

    # Only actual licit transactions
    licit = test[test["target"] == 0].copy()

    correct = licit.loc[
        licit["predicted_class"] == 0,
        "feature_2"
    ].dropna()

    false_positive = licit.loc[
        licit["predicted_class"] == 1,
        "feature_2"
    ].dropna()

    print("\nGROUP SIZES")
    print("Correct:", len(correct))
    print("False Positive:", len(false_positive))

    # 1. Original statistics

    print("\nORIGINAL DISTRIBUTION")

    print("\nCorrect:")
    print(correct.describe())

    print("\nFalse Positive:")
    print(false_positive.describe())

    # 2. Calculate common IQR outlier limits

    combined = pd.concat(
        [correct, false_positive]
    )

    q1 = combined.quantile(0.25)
    q3 = combined.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    print("\nOUTLIER LIMITS")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)

    # 3. Count outliers

    correct_outliers = correct[
        (correct < lower_limit) |
        (correct > upper_limit)
    ]

    fp_outliers = false_positive[
        (false_positive < lower_limit) |
        (false_positive > upper_limit)
    ]

    print("\nOUTLIER COUNTS")
    print("Correct outliers:", len(correct_outliers))
    print("False positive outliers:", len(fp_outliers))

    print("\nCorrect extreme values:")
    print(correct_outliers.to_list())

    # 4. Remove common-IQR outliers from both groups

    correct_clean = correct[
        (correct >= lower_limit) &
        (correct <= upper_limit)
    ]

    fp_clean = false_positive[
        (false_positive >= lower_limit) &
        (false_positive <= upper_limit)
    ]

    print("\nAFTER OUTLIER REMOVAL")

    print("\nCorrect:")
    print(correct_clean.describe())

    print("\nFalse Positive:")
    print(fp_clean.describe())

    if len(correct_clean) > 0 and len(fp_clean) > 0:

        statistic, p_value = mannwhitneyu(
            correct_clean,
            fp_clean,
            alternative="two-sided"
        )

        print("\nMANN-WHITNEY TEST AFTER OUTLIER REMOVAL")
        print("Statistic:", statistic)
        print("P-value:", p_value)

    else:
        statistic = np.nan
        p_value = np.nan

    # 5. Compare original and cleaned distributions

    results = pd.DataFrame([
        {
            "comparison": "Original",
            "correct_count": len(correct),
            "fp_count": len(false_positive),
            "correct_mean": correct.mean(),
            "fp_mean": false_positive.mean(),
            "correct_median": correct.median(),
            "fp_median": false_positive.median(),
            "mean_difference": (
                false_positive.mean() - correct.mean()
            ),
            "mannwhitney_p": mannwhitneyu(
                correct,
                false_positive,
                alternative="two-sided"
            ).pvalue
        },
        {
            "comparison": "After IQR outlier removal",
            "correct_count": len(correct_clean),
            "fp_count": len(fp_clean),
            "correct_mean": correct_clean.mean(),
            "fp_mean": fp_clean.mean(),
            "correct_median": correct_clean.median(),
            "fp_median": fp_clean.median(),
            "mean_difference": (
                fp_clean.mean() - correct_clean.mean()
            ),
            "mannwhitney_p": p_value
        }
    ])

    results.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nFINAL COMPARISON")
    print(results.to_string(index=False))

    # 6. Visualization

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        [correct, false_positive],
        tick_labels=["Correct", "False Positive"],
        showfliers=True
    )

    plt.ylabel("Feature 2 Value")
    plt.title("Feature 2 Outlier Analysis - Timestep 42")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    print("\nINVESTIGATION COMPLETED")
    print("Saved:", REPORT_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()