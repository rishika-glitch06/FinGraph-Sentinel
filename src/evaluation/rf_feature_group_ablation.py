import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
OUTPUT_PATH = ROOT / "reports/rf_feature_group_ablation.csv"

LOCAL_FEATURES = [
    f"feature_{i}" for i in range(1, 95)
]

AGGREGATED_FEATURES = [
    f"feature_{i}" for i in range(95, 166)
]

ALL_FEATURES = LOCAL_FEATURES + AGGREGATED_FEATURES


def evaluate(name, features, train, test):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train[features], train["target"])

    probabilities = model.predict_proba(
        test[features]
    )[:, 1]

    predictions = (probabilities >= 0.50).astype(int)

    return {
        "experiment": name,
        "feature_count": len(features),
        "accuracy": accuracy_score(test["target"], predictions),
        "balanced_accuracy": balanced_accuracy_score(
            test["target"], predictions
        ),
        "macro_f1": f1_score(
            test["target"], predictions, average="macro"
        ),
        "licit_recall": recall_score(
            test["target"], predictions, pos_label=0
        ),
        "illicit_recall": recall_score(
            test["target"], predictions, pos_label=1
        ),
        "roc_auc": roc_auc_score(
            test["target"], probabilities
        ),
        "illicit_pr_auc": average_precision_score(
            test["target"], probabilities
        ),
        "confusion_matrix": confusion_matrix(
            test["target"], predictions
        ).tolist()
    }


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["target"] = -1
    df.loc[df["class"] == "licit", "target"] = 0
    df.loc[df["class"] == "illicit", "target"] = 1

    labelled = df[df["target"] >= 0].copy()

    train = labelled[labelled["time_step"] <= 35]
    test = labelled[labelled["time_step"] >= 42]

    print("Training samples:", len(train))
    print("Testing samples:", len(test))

    experiments = [
        ("Local features only", LOCAL_FEATURES),
        ("Aggregated features only", AGGREGATED_FEATURES),
        ("All 165 features", ALL_FEATURES)
    ]

    results = []

    for name, features in experiments:

        print(f"\nRunning: {name}")

        result = evaluate(
            name,
            features,
            train,
            test
        )

        results.append(result)

        print("Balanced accuracy:", result["balanced_accuracy"])
        print("Macro F1:", result["macro_f1"])
        print("Licit recall:", result["licit_recall"])
        print("Illicit recall:", result["illicit_recall"])
        print("ROC-AUC:", result["roc_auc"])
        print("Confusion matrix:", result["confusion_matrix"])

    results_df = pd.DataFrame(results)

    output_path = ROOT / "reports/rf_feature_group_ablation.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFINAL COMPARISON:")
    print(
        results_df.drop(
            columns=["confusion_matrix"]
        ).to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()