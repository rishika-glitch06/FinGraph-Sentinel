import os
import numpy as np
import pandas as pd
import joblib

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "transactions.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_threshold_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "feature_2_permutation_results.csv"
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
N_PERMUTATIONS = 20
THRESHOLD = 0.5
TARGET_TIMESTEP = 42

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading dataset and model...")

    feature_cols = [
        f"feature_{i}" for i in range(1, 166)
    ]

    required_cols = [
        "time_step",
        "class",
        *feature_cols
    ]

    dtype_map = {
        "time_step": "int8",
        "class": "category"
    }

    dtype_map.update({
        col: "float32"
        for col in feature_cols
    })

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_cols,
        dtype=dtype_map,
        engine="c"
    )

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # FILTER TEST DATA
    # --------------------------------------------------

    df = df[
        (df["time_step"] == TARGET_TIMESTEP) &
        (
            df["class"].astype(str).isin(
                ["licit", "illicit"]
            )
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No labelled transactions found for timestep "
            f"{TARGET_TIMESTEP}. Check class labels and time_step."
        )

    df["target"] = (
        df["class"].astype(str) == "illicit"
    ).astype(int)

    X = df[feature_cols].copy()
    y = df["target"].to_numpy()

    print("\nDATASET")
    print("Transactions:", len(df))
    print("Licit:", int((y == 0).sum()))
    print("Illicit:", int((y == 1).sum()))

    # --------------------------------------------------
    # BASELINE PREDICTIONS
    # --------------------------------------------------

    original_prob = model.predict_proba(X)[:, 1]

    original_pred = (
        original_prob >= THRESHOLD
    ).astype(int)

    original_auc = roc_auc_score(
        y, original_prob
    )

    original_balanced = balanced_accuracy_score(
        y, original_pred
    )

    original_f1 = f1_score(
        y,
        original_pred,
        average="macro"
    )

    print("\nBASELINE MODEL")
    print(f"ROC-AUC: {original_auc:.6f}")
    print(f"Balanced Accuracy: {original_balanced:.6f}")
    print(f"Macro F1: {original_f1:.6f}")

    # --------------------------------------------------
    # PERMUTATION EXPERIMENT
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_STATE)

    results = []

    print("\nFEATURE 2 PERMUTATION EXPERIMENT")

    for iteration in range(N_PERMUTATIONS):

        X_permuted = X.copy()

        shuffled_values = rng.permutation(
            X_permuted["feature_2"].to_numpy()
        )

        X_permuted["feature_2"] = shuffled_values

        permuted_prob = model.predict_proba(
            X_permuted
        )[:, 1]

        permuted_pred = (
            permuted_prob >= THRESHOLD
        ).astype(int)

        permuted_auc = roc_auc_score(
            y, permuted_prob
        )

        permuted_balanced = balanced_accuracy_score(
            y, permuted_pred
        )

        permuted_f1 = f1_score(
            y,
            permuted_pred,
            average="macro"
        )

        mean_probability_change = np.mean(
            np.abs(original_prob - permuted_prob)
        )

        correlation, _ = spearmanr(
            original_prob,
            permuted_prob
        )

        results.append({
            "iteration": iteration + 1,
            "permuted_roc_auc": permuted_auc,
            "roc_auc_drop": original_auc - permuted_auc,
            "permuted_balanced_accuracy": permuted_balanced,
            "balanced_accuracy_drop":
                original_balanced - permuted_balanced,
            "permuted_macro_f1": permuted_f1,
            "macro_f1_drop": original_f1 - permuted_f1,
            "mean_probability_change":
                mean_probability_change,
            "prediction_rank_correlation": correlation
        })

        print(
            f"Iteration {iteration + 1:02d} | "
            f"AUC: {permuted_auc:.4f} | "
            f"AUC Drop: "
            f"{original_auc - permuted_auc:.4f} | "
            f"Mean Probability Change: "
            f"{mean_probability_change:.6f}"
        )

    # --------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\nFINAL SUMMARY")

    summary_cols = [
        "roc_auc_drop",
        "balanced_accuracy_drop",
        "macro_f1_drop",
        "mean_probability_change",
        "prediction_rank_correlation"
    ]

    for col in summary_cols:

        print(
            f"{col}: "
            f"Mean = {results_df[col].mean():.6f}, "
            f"Std = {results_df[col].std():.6f}"
        )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nINVESTIGATION COMPLETED")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()