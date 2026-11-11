import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"

DRIFT_PATH = ROOT / "reports/feature_group_drift_details.csv"

ERROR_PATH = ROOT / "reports/rf_error_feature_analysis.csv"

OUTPUT_PATH = ROOT / "reports/drift_error_correlation.csv"


def main():

    print("Loading analysis reports and model...")

    model = joblib.load(MODEL_PATH)

    drift = pd.read_csv(DRIFT_PATH)
    errors = pd.read_csv(ERROR_PATH)

    features = [f"feature_{i}" for i in range(1, 166)]

    # 1. Model feature importance

    importance = pd.DataFrame({
        "feature": features,
        "model_importance": model.feature_importances_
    })

    # 2. Aggregate drift across licit and illicit classes

    drift_summary = (
        drift.groupby("feature")
        .agg(
            mean_ks=("ks_statistic", "mean"),
            max_ks=("ks_statistic", "max"),
            licit_ks=(
                "ks_statistic",
                lambda x: x.iloc[0]
            )
        )
        .reset_index()
    )

    # Explicitly calculate class-specific KS values

    class_drift = drift.pivot_table(
        index="feature",
        columns="class",
        values="ks_statistic"
    ).reset_index()

    class_drift = class_drift.rename(columns={
        "Licit": "licit_ks",
        "Illicit": "illicit_ks"
    })

    drift_summary = (
        drift.groupby("feature")["ks_statistic"]
        .mean()
        .reset_index(name="mean_ks")
    )

    drift_summary = drift_summary.merge(
        class_drift,
        on="feature",
        how="left"
    )

    # 3. Merge error analysis

    error_features = errors[[
        "feature",
        "absolute_mean_difference"
    ]].copy()

    result = importance.merge(
        drift_summary,
        on="feature",
        how="left"
    )

    result = result.merge(
        error_features,
        on="feature",
        how="left"
    )

    # 4. Calculate ranks

    result["importance_rank"] = (
        result["model_importance"].rank(
            ascending=False,
            method="average"
        )
    )

    result["drift_rank"] = (
        result["mean_ks"].rank(
            ascending=False,
            method="average"
        )
    )

    result["error_difference_rank"] = (
        result["absolute_mean_difference"].rank(
            ascending=False,
            method="average"
        )
    )

    result["combined_rank"] = (
        result["importance_rank"]
        + result["drift_rank"]
        + result["error_difference_rank"]
    )

    result = result.sort_values("combined_rank")

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 5. Correlation analysis

    print("\nSPEARMAN CORRELATIONS:")

    correlations = [
        ("Model importance vs Drift",
         "model_importance", "mean_ks"),

        ("Model importance vs Error difference",
         "model_importance", "absolute_mean_difference"),

        ("Drift vs Error difference",
         "mean_ks", "absolute_mean_difference")
    ]

    for name, col1, col2 in correlations:

        valid = result[[col1, col2]].dropna()

        correlation, p_value = spearmanr(
            valid[col1],
            valid[col2]
        )

        print(f"\n{name}")
        print(f"Spearman correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.6g}")

    # 6. Top drifted and error-associated features

    print("\nTOP 20 FEATURES BY COMBINED RANK:")

    print(
        result.head(20)[[
            "feature",
            "model_importance",
            "mean_ks",
            "licit_ks",
            "illicit_ks",
            "absolute_mean_difference",
            "combined_rank"
        ]].to_string(index=False)
    )

    # 7. Top 10 overlap

    top_importance = set(
        result.nlargest(20, "model_importance")["feature"]
    )

    top_drift = set(
        result.nlargest(20, "mean_ks")["feature"]
    )

    top_errors = set(
        result.nlargest(
            20, "absolute_mean_difference"
        )["feature"]
    )

    print("\nTOP 20 FEATURE OVERLAP:")

    print(
        "Importance and drift:",
        sorted(top_importance & top_drift)
    )

    print(
        "Drift and errors:",
        sorted(top_drift & top_errors)
    )

    print(
        "All three:",
        sorted(top_importance & top_drift & top_errors)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()