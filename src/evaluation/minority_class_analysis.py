import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
MODEL_PATH = ROOT / "models/rf_threshold_model.pkl"
REPORT_DIR = ROOT / "reports"

CLASS_REPORT = REPORT_DIR / "minority_class_distribution.csv"
SCORE_REPORT = REPORT_DIR / "minority_class_scores.csv"


def main():

    print("Loading dataset and Random Forest model...")

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    df["target"] = -1
    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    # Only known labels
    df = df[df["target"] >= 0].copy()

    # Separate temporal periods
    train = df[df["time_step"] <= 35]
    validation = df[
        (df["time_step"] >= 36)
        & (df["time_step"] <= 41)
    ]
    test = df[df["time_step"] >= 42].copy()

    print("\nCLASS DISTRIBUTION")

    periods = {
        "Train": train,
        "Validation": validation,
        "Test": test
    }

    distribution = []

    for period_name, data in periods.items():

        counts = data["target"].value_counts()

        licit = int(counts.get(0, 0))
        illicit = int(counts.get(1, 0))
        total = licit + illicit

        distribution.append({
            "period": period_name,
            "total": total,
            "licit": licit,
            "illicit": illicit,
            "licit_percentage": licit / total * 100,
            "illicit_percentage": illicit / total * 100,
            "class_imbalance_ratio": (
                illicit / licit if licit else np.nan
            )
        })

    distribution_df = pd.DataFrame(distribution)

    print(distribution_df.to_string(index=False))

    # Per-timestep class distribution
    timestep_distribution = []

    for timestep, group in df.groupby("time_step"):

        counts = group["target"].value_counts()

        licit = int(counts.get(0, 0))
        illicit = int(counts.get(1, 0))
        total = licit + illicit

        timestep_distribution.append({
            "timestep": timestep,
            "total": total,
            "licit": licit,
            "illicit": illicit,
            "licit_percentage": licit / total * 100,
            "illicit_percentage": illicit / total * 100
        })

    timestep_df = pd.DataFrame(timestep_distribution)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestep_df.to_csv(
        CLASS_REPORT,
        index=False
    )

    # Test probability analysis
    features = [f"feature_{i}" for i in range(1, 166)]

    X_test = test[features]
    y_test = test["target"].to_numpy()

    probabilities = model.predict_proba(X_test)[:, 1]

    test["illicit_probability"] = probabilities

    test["predicted_class_050"] = (
        probabilities >= 0.50
    ).astype(int)

    test["predicted_class_060"] = (
        probabilities >= 0.60
    ).astype(int)

    score_results = []

    for timestep, group in test.groupby("time_step"):

        for actual_class, class_name in [
            (0, "Licit"),
            (1, "Illicit")
        ]:

            class_group = group[
                group["target"] == actual_class
            ]

            if len(class_group) == 0:
                continue

            scores = class_group["illicit_probability"]

            score_results.append({
                "timestep": timestep,
                "actual_class": class_name,
                "support": len(class_group),
                "mean_illicit_probability": scores.mean(),
                "median_illicit_probability": scores.median(),
                "minimum_probability": scores.min(),
                "maximum_probability": scores.max(),
                "predicted_licit_at_050": int(
                    (scores < 0.50).sum()
                ),
                "predicted_licit_at_060": int(
                    (scores < 0.60).sum()
                )
            })

    score_df = pd.DataFrame(score_results)

    score_df.to_csv(
        SCORE_REPORT,
        index=False
    )

    print("\nTEST CLASS DISTRIBUTION:")
    print(
        timestep_df[
            timestep_df["timestep"] >= 42
        ].to_string(index=False)
    )

    print("\nCLASS-WISE PROBABILITY ANALYSIS:")
    print(score_df.to_string(index=False))

    # Chart 1: Class distribution over time
    plt.figure(figsize=(12, 6))

    plt.plot(
        timestep_df["timestep"],
        timestep_df["licit_percentage"],
        marker="o",
        label="Licit"
    )

    plt.plot(
        timestep_df["timestep"],
        timestep_df["illicit_percentage"],
        marker="s",
        label="Illicit"
    )

    plt.axvline(
        x=41.5,
        linestyle="--",
        label="Train/Validation-Test Boundary"
    )

    plt.title("Class Distribution Across Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Percentage")
    plt.xticks(range(1, 50, 2))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "minority_class_distribution.png",
        dpi=300
    )

    plt.close()

    # Chart 2: Mean predicted illicit probability
    plt.figure(figsize=(12, 6))

    for class_name in ["Licit", "Illicit"]:

        subset = score_df[
            score_df["actual_class"] == class_name
        ]

        plt.plot(
            subset["timestep"],
            subset["mean_illicit_probability"],
            marker="o",
            label=f"Actual {class_name}"
        )

    plt.axhline(
        y=0.50,
        linestyle="--",
        label="Threshold 0.50"
    )

    plt.axhline(
        y=0.60,
        linestyle=":",
        label="Threshold 0.60"
    )

    plt.title("Mean Predicted Illicit Probability by Actual Class")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Illicit Probability")
    plt.xticks(range(42, 50))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "minority_class_probability.png",
        dpi=300
    )

    plt.close()

    print("\nAnalysis completed.")

    print("\nGenerated files:")
    print(CLASS_REPORT)
    print(SCORE_REPORT)
    print(REPORT_DIR / "minority_class_distribution.png")
    print(REPORT_DIR / "minority_class_probability.png")


if __name__ == "__main__":
    main()