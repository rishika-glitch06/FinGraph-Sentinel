import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score
)

from scipy.stats import spearmanr


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "transactions.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "rf_threshold_model.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "feature_2_conditional_permutation.csv"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
N_PERMUTATIONS = 20
N_CLUSTERS = 10
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
        col: "float32" for col in feature_cols
    })

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_cols,
        dtype=dtype_map
    )

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # FILTER TIMESTEP 42
    # --------------------------------------------------

    df = df[
        (df["time_step"] == TARGET_TIMESTEP) &
        (df["class"].astype(str).isin(
            ["licit", "illicit"]
        ))
    ].copy()

    if df.empty:
        raise ValueError("No labelled transactions found.")

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
    # BASELINE
    # --------------------------------------------------

    original_prob = model.predict_proba(X)[:, 1]

    original_pred = (
        original_prob >= THRESHOLD
    ).astype(int)

    original_auc = roc_auc_score(y, original_prob)

    original_balanced = balanced_accuracy_score(
        y, original_pred
    )

    original_f1 = f1_score(
        y, original_pred, average="macro"
    )

    print("\nBASELINE")
    print("ROC-AUC:", original_auc)
    print("Balanced Accuracy:", original_balanced)
    print("Macro F1:", original_f1)

    # --------------------------------------------------
    # CLUSTER USING OTHER 164 FEATURES
    # --------------------------------------------------

    print("\nCreating transaction similarity groups...")

    other_features = [
        col for col in feature_cols
        if col != "feature_2"
    ]

    X_other = X[other_features].to_numpy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_other)

    clustering = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=512,
        n_init=10
    )

    cluster_labels = clustering.fit_predict(X_scaled)

    df["cluster"] = cluster_labels

    print("\nCLUSTER DISTRIBUTION")

    print(
        df["cluster"].value_counts().sort_index()
    )

    # --------------------------------------------------
    # CONDITIONAL PERMUTATION
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_STATE)

    results = []

    print("\nCONDITIONAL FEATURE 2 PERMUTATION")

    for iteration in range(N_PERMUTATIONS):

        X_permuted = X.copy()

        shuffled_feature = X["feature_2"].to_numpy().copy()

        for cluster_id in range(N_CLUSTERS):

            indices = np.where(
                cluster_labels == cluster_id
            )[0]

            if len(indices) > 1:

                shuffled_feature[indices] = rng.permutation(
                    shuffled_feature[indices]
                )

        X_permuted["feature_2"] = shuffled_feature

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

        probability_change = np.mean(
            np.abs(original_prob - permuted_prob)
        )

        rank_corr, _ = spearmanr(
            original_prob,
            permuted_prob
        )

        results.append({

            "iteration": iteration + 1,

            "permuted_roc_auc": permuted_auc,

            "roc_auc_drop":
                original_auc - permuted_auc,

            "permuted_balanced_accuracy":
                permuted_balanced,

            "balanced_accuracy_drop":
                original_balanced - permuted_balanced,

            "permuted_macro_f1": permuted_f1,

            "macro_f1_drop":
                original_f1 - permuted_f1,

            "mean_probability_change":
                probability_change,

            "prediction_rank_correlation":
                rank_corr

        })

        print(
            f"Iteration {iteration + 1:02d} | "
            f"AUC: {permuted_auc:.4f} | "
            f"AUC Drop: "
            f"{original_auc - permuted_auc:.4f} | "
            f"Probability Change: "
            f"{probability_change:.6f}"
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\nFINAL CONDITIONAL PERMUTATION SUMMARY")

    for col in [
        "roc_auc_drop",
        "balanced_accuracy_drop",
        "macro_f1_drop",
        "mean_probability_change",
        "prediction_rank_correlation"
    ]:

        print(
            f"{col}: "
            f"Mean = {results_df[col].mean():.6f}, "
            f"Std = {results_df[col].std():.6f}"
        )

    # --------------------------------------------------
    # SAVE
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