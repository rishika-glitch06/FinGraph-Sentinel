import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, pearsonr


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"

REPORT_PATH = ROOT / "reports/feature_2_probability_analysis.csv"
FIGURE_PATH = ROOT / "reports/figures/feature_2_probability_relationship.png"


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

    # Generate model probabilities
    test["illicit_probability"] = model.predict_proba(
        test[features]
    )[:, 1]

    feature = "feature_2"

    print("\nDATASET")
    print("Transactions:", len(test))
    print("Licit:", (test["target"] == 0).sum())
    print("Illicit:", (test["target"] == 1).sum())

    # 1. Overall correlation

    print("\nOVERALL CORRELATION")

    rho, p_value = spearmanr(
        test[feature],
        test["illicit_probability"]
    )

    pearson, pearson_p = pearsonr(
        test[feature],
        test["illicit_probability"]
    )

    print("Spearman correlation:", rho)
    print("Spearman p-value:", p_value)
    print("Pearson correlation:", pearson)
    print("Pearson p-value:", pearson_p)

    results = []

    results.append({
        "group": "Overall",
        "sample_count": len(test),
        "spearman_correlation": rho,
        "spearman_p_value": p_value,
        "pearson_correlation": pearson,
        "pearson_p_value": pearson_p
    })

    # 2. Class-wise correlation

    print("\nCLASS-WISE CORRELATION")

    for label in ["licit", "illicit"]:

        subset = test[test["class"] == label]

        rho, p = spearmanr(
            subset[feature],
            subset["illicit_probability"]
        )

        pearson, pp = pearsonr(
            subset[feature],
            subset["illicit_probability"]
        )

        print(f"\n{label.upper()}")
        print("Transactions:", len(subset))
        print("Spearman correlation:", rho)
        print("Spearman p-value:", p)
        print("Pearson correlation:", pearson)

        results.append({
            "group": label,
            "sample_count": len(subset),
            "spearman_correlation": rho,
            "spearman_p_value": p,
            "pearson_correlation": pearson,
            "pearson_p_value": pp
        })

    # 3. Probability distribution by Feature 2 bins

    print("\nFEATURE 2 BIN ANALYSIS")

    test["feature_2_bin"] = pd.qcut(
        test[feature],
        q=5,
        duplicates="drop"
    )

    bin_summary = test.groupby(
        "feature_2_bin",
        observed=True
    ).agg(
        transaction_count=("target", "count"),
        mean_feature_2=(feature, "mean"),
        mean_illicit_probability=(
            "illicit_probability", "mean"
        ),
        median_illicit_probability=(
            "illicit_probability", "median"
        ),
        actual_illicit_rate=("target", "mean")
    ).reset_index()

    print(bin_summary.to_string(index=False))

    bin_summary.to_csv(
        ROOT / "reports/feature_2_probability_bins.csv",
        index=False
    )

    # 4. Visualization

    plt.figure(figsize=(10, 6))

    for label in ["licit", "illicit"]:

        subset = test[test["class"] == label]

        plt.scatter(
            subset[feature],
            subset["illicit_probability"],
            alpha=0.5,
            label=label,
            s=25
        )

    plt.xlabel("Feature 2 Value")
    plt.ylabel("Predicted Illicit Probability")
    plt.title("Feature 2 vs Random Forest Prediction Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    # 5. Save correlation report

    pd.DataFrame(results).to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nINVESTIGATION COMPLETED")
    print("Saved:", REPORT_PATH)
    print("Saved:", FIGURE_PATH)


if __name__ == "__main__":
    main()