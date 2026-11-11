import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
REPORT_DIR = ROOT / "reports"

DETAIL_PATH = REPORT_DIR / "feature_group_drift_details.csv"
SUMMARY_PATH = REPORT_DIR / "feature_group_drift_summary.csv"


# Feature groups
LOCAL_FEATURES = [
    f"feature_{i}" for i in range(1, 95)
]

AGGREGATED_FEATURES = [
    f"feature_{i}" for i in range(95, 166)
]


def calculate_drift(
    train,
    test,
    features,
    group_name,
    class_name
):

    results = []

    for feature in features:

        train_values = train[feature].dropna()
        test_values = test[feature].dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        ks_stat, p_value = ks_2samp(
            train_values,
            test_values
        )

        results.append({

            "feature_group": group_name,

            "class": class_name,

            "feature": feature,

            "train_mean": train_values.mean(),

            "test_mean": test_values.mean(),

            "mean_difference": (
                test_values.mean() - train_values.mean()
            ),

            "ks_statistic": ks_stat,

            "p_value": p_value

        })

    return results


def main():

    print("Loading dataset...")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load only required columns using float32
    # to reduce memory consumption.

    all_features = (
        LOCAL_FEATURES + AGGREGATED_FEATURES
    )

    required_columns = (
        ["class", "time_step"] + all_features
    )

    dtype_map = {
        feature: "float32"
        for feature in all_features
    }

    dtype_map["time_step"] = "int8"
    dtype_map["class"] = "category"

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        dtype=dtype_map,
        engine="c"
    )

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print(
        "Memory usage:",
        round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "MB"
    )

    # Target encoding

    df["target"] = -1

    df.loc[df["class"] == "licit", "target"] = 0

    df.loc[df["class"] == "illicit", "target"] = 1

    labelled = df[df["target"] >= 0]

    # Temporal split

    train = labelled[
        labelled["time_step"] <= 35
    ]

    test = labelled[
        labelled["time_step"] >= 42
    ]

    print("\nTRAINING DISTRIBUTION:")
    print(train["class"].value_counts())

    print("\nTEST DISTRIBUTION:")
    print(test["class"].value_counts())

    feature_groups = {

        "Local": LOCAL_FEATURES,

        "Aggregated": AGGREGATED_FEATURES

    }

    all_results = []

    # Class-conditional drift analysis

    for group_name, features in feature_groups.items():

        print(
            f"\nAnalyzing {group_name} features..."
        )

        for class_name, target in [
            ("Licit", 0),
            ("Illicit", 1)
        ]:

            train_class = train[
                train["target"] == target
            ]

            test_class = test[
                test["target"] == target
            ]

            print(
                f"{class_name}: "
                f"Train={len(train_class)}, "
                f"Test={len(test_class)}"
            )

            results = calculate_drift(
                train_class,
                test_class,
                features,
                group_name,
                class_name
            )

            all_results.extend(results)

    # Detailed results

    details = pd.DataFrame(all_results)

    details.to_csv(
        DETAIL_PATH,
        index=False
    )

    # Group-level summary

    summary = (

        details.groupby(
            ["feature_group", "class"]
        )

        .agg(

            feature_count=("feature", "count"),

            mean_ks=("ks_statistic", "mean"),

            median_ks=("ks_statistic", "median"),

            max_ks=("ks_statistic", "max"),

            features_ks_above_01=(
                "ks_statistic",
                lambda x: (x > 0.1).sum()
            ),

            features_ks_above_02=(
                "ks_statistic",
                lambda x: (x > 0.2).sum()
            ),

            mean_absolute_mean_difference=(
                "mean_difference",
                lambda x: x.abs().mean()
            )

        )

        .reset_index()

    )

    summary["percentage_ks_above_01"] = (
        summary["features_ks_above_01"]
        / summary["feature_count"] * 100
    )

    summary["percentage_ks_above_02"] = (
        summary["features_ks_above_02"]
        / summary["feature_count"] * 100
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # Print summary

    print("\n" + "=" * 65)

    print("FEATURE GROUP DRIFT SUMMARY")

    print("=" * 65)

    print(
        summary.to_string(index=False)
    )

    # Top 10 drifted features

    print("\n" + "=" * 65)

    print("TOP 10 DRIFTED FEATURES")

    print("=" * 65)

    for (group, class_name), subset in details.groupby(
        ["feature_group", "class"]
    ):

        print(
            f"\n{group} - {class_name}"
        )

        top = subset.sort_values(
            "ks_statistic",
            ascending=False
        ).head(10)

        print(
            top[[
                "feature",
                "ks_statistic",
                "train_mean",
                "test_mean",
                "p_value"
            ]].to_string(index=False)
        )

    print("\n" + "=" * 65)

    print("ANALYSIS COMPLETED")

    print("=" * 65)

    print("\nSaved reports:")

    print(DETAIL_PATH)

    print(SUMMARY_PATH)


if __name__ == "__main__":

    main()