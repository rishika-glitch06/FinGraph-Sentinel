import pandas as pd
import joblib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"
ERROR_PATH = ROOT / "reports/rf_misclassified_transactions.csv"
FEATURE_PATH = ROOT / "reports/rf_error_feature_analysis.csv"

OUTPUT_PATH = ROOT / "reports/rf_error_importance_overlap.csv"

def main():

    print("Loading model and error analysis...")

    model = joblib.load(MODEL_PATH)
    errors = pd.read_csv(ERROR_PATH)
    feature_analysis = pd.read_csv(FEATURE_PATH)

    # Random Forest feature importance
    importance = pd.DataFrame({
        "feature": [f"feature_{i}" for i in range(1, 166)],
        "model_importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "model_importance",
        ascending=False
    )

    # Merge feature importance with error differences
    merged = importance.merge(
        feature_analysis,
        on="feature",
        how="inner"
    )

    merged["importance_rank"] = (
        merged["model_importance"].rank(
            ascending=False,
            method="min"
        )
    )

    merged["error_difference_rank"] = (
        merged["absolute_mean_difference"].rank(
            ascending=False,
            method="min"
        )
    )

    merged["combined_rank"] = (
        merged["importance_rank"]
        + merged["error_difference_rank"]
    )

    merged = merged.sort_values("combined_rank")

    merged.to_csv(OUTPUT_PATH, index=False)

    print("\nTOP 20 FEATURES BY MODEL IMPORTANCE:")
    print(
        importance.head(20).to_string(index=False)
    )

    print("\nTOP 20 FEATURES BY COMBINED RANK:")
    print(
        merged.head(20)[[
            "feature",
            "model_importance",
            "absolute_mean_difference",
            "importance_rank",
            "error_difference_rank",
            "combined_rank"
        ]].to_string(index=False)
    )

    # Top feature overlap
    top_model = set(importance.head(20)["feature"])
    top_error = set(
        feature_analysis.head(20)["feature"]
    )

    overlap = top_model.intersection(top_error)

    print("\nFEATURE OVERLAP:")
    print("Common features:", len(overlap))
    print("Features:", sorted(overlap))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()