import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions.csv"
ERROR_PATH = ROOT / "reports/rf_misclassified_transactions.csv"
REPORT_PATH = ROOT / "reports/feature_2_investigation.csv"
FIGURE_DIR = ROOT / "reports/figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading transaction data...")

    df = pd.read_csv(DATA_PATH)

    df["target"] = df["class"].map({
        "licit": 0,
        "illicit": 1
    })

    df = df[df["target"].notna()].copy()

    feature = "feature_2"

    print("\nDATASET INFORMATION")
    print("Transactions:", len(df))
    print("Feature:", feature)

    # 1. Feature distribution by class

    print("\nFEATURE 2 BY CLASS:")

    class_stats = df.groupby("class")[feature].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    print(class_stats)

    # 2. Temporal distribution

    temporal = df.groupby(
        ["time_step", "class"]
    )[feature].agg(
        ["count", "mean", "median", "std"]
    ).reset_index()

    temporal.to_csv(
        ROOT / "reports/feature_2_temporal_distribution.csv",
        index=False
    )

    # 3. Train vs test drift

    train = df[
        (df["time_step"] <= 35)
    ]

    test = df[
        (df["time_step"] >= 42)
    ]

    print("\nTRAIN VS TEST DRIFT:")

    drift_results = []

    for label in ["licit", "illicit"]:

        train_values = train.loc[
            train["class"] == label, feature
        ].dropna()

        test_values = test.loc[
            test["class"] == label, feature
        ].dropna()

        ks_stat, p_value = ks_2samp(
            train_values,
            test_values
        )

        drift_results.append({
            "class": label,
            "train_mean": train_values.mean(),
            "test_mean": test_values.mean(),
            "mean_difference": (
                test_values.mean() - train_values.mean()
            ),
            "ks_statistic": ks_stat,
            "p_value": p_value
        })

        print(f"\n{label.upper()}")
        print("Train mean:", train_values.mean())
        print("Test mean:", test_values.mean())
        print("KS statistic:", ks_stat)
        print("P-value:", p_value)

    # 4. Feature distribution over time

    plt.figure(figsize=(12, 6))

    for label in ["licit", "illicit"]:

        subset = temporal[
            temporal["class"] == label
        ]

        plt.plot(
            subset["time_step"],
            subset["mean"],
            marker="o",
            label=label
        )

    plt.xlabel("Time Step")
    plt.ylabel("Mean Feature 2")
    plt.title("Feature 2 Temporal Trend by Class")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_2_temporal_trend.png",
        dpi=300
    )

    plt.close()

    # 5. Compare class distributions

    plt.figure(figsize=(10, 6))

    for label in ["licit", "illicit"]:

        values = df.loc[
            df["class"] == label, feature
        ]

        plt.hist(
            values,
            bins=50,
            alpha=0.5,
            density=True,
            label=label
        )

    plt.xlabel("Feature 2 Value")
    plt.ylabel("Density")
    plt.title("Feature 2 Distribution by Class")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_2_class_distribution.png",
        dpi=300
    )

    plt.close()

    # 6. Save drift report

    pd.DataFrame(drift_results).to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nINVESTIGATION COMPLETED")
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()