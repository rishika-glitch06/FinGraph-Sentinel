import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import ks_2samp, mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"

OUTPUT_PATH = ROOT / "reports/feature_2_within_timestep.csv"


def main():

    print("Loading dataset and model...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Keep known labels
    df = df[df["class"].isin(["licit", "illicit"])].copy()

    df["target"] = df["class"].map({
        "licit": 0,
        "illicit": 1
    })

    # Chronological test period
    test = df[df["time_step"] >= 42].copy()

    features = [f"feature_{i}" for i in range(1, 166)]

    # Recreate RF predictions
    probabilities = model.predict_proba(
        test[features]
    )[:, 1]

    test["illicit_probability"] = probabilities
    test["predicted_class"] = (
        probabilities >= 0.5
    ).astype(int)

    # Only actual licit transactions
    licit = test[test["target"] == 0].copy()

    # Correct and incorrect predictions
    licit["prediction_status"] = np.where(
        licit["predicted_class"] == 0,
        "Correct",
        "False Positive"
    )

    print("\nWITHIN-TIMESTEP FEATURE 2 ANALYSIS")

    results = []

    for step in sorted(licit["time_step"].unique()):

        subset = licit[licit["time_step"] == step]

        correct = subset.loc[
            subset["prediction_status"] == "Correct",
            "feature_2"
        ].dropna()

        incorrect = subset.loc[
            subset["prediction_status"] == "False Positive",
            "feature_2"
        ].dropna()

        print(f"\nTIMESTEP {step}")
        print("Correct:", len(correct))
        print("False Positive:", len(incorrect))

        if len(correct) == 0 or len(incorrect) == 0:

            print("Insufficient groups for comparison.")

            continue

        ks_stat, ks_p = ks_2samp(
            correct,
            incorrect
        )

        mw_stat, mw_p = mannwhitneyu(
            correct,
            incorrect,
            alternative="two-sided"
        )

        difference = incorrect.mean() - correct.mean()

        results.append({
            "time_step": step,
            "correct_count": len(correct),
            "false_positive_count": len(incorrect),
            "correct_mean": correct.mean(),
            "false_positive_mean": incorrect.mean(),
            "correct_median": correct.median(),
            "false_positive_median": incorrect.median(),
            "mean_difference": difference,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
            "mannwhitney_p_value": mw_p
        })

        print("Correct mean:", correct.mean())
        print("False positive mean:", incorrect.mean())
        print("Mean difference:", difference)
        print("KS statistic:", ks_stat)
        print("KS p-value:", ks_p)
        print("Mann-Whitney p-value:", mw_p)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFINAL RESULTS")
    print(result_df.to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()