import pickle
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, kendalltau


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "temporal_random_forest.pkl"

DATA_PATH = (
    ROOT / "data" / "processed" /
    "transactions_advanced_features.csv"
)

GNN_PATH = (
    ROOT / "reports" / "gnn_explainability" /
    "gnn_feature_importance.csv"
)

OUTPUT_DIR = ROOT / "reports" / "explainability_comparison"
FIGURE_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading Random Forest model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

    print("Model loaded successfully.")
    print("Model type:", type(model).__name__)
    print("Expected features:", model.n_features_in_)

    return model


# ============================================================
# 2. LOAD AND FILTER DATA
# ============================================================

def load_data(model):

    print("\nLoading processed dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)
    print("Available columns:", list(df.columns[:12]))

    # Identify the timestep column.
    timestep_candidates = [
        "time_step",
        "Time step",
        "timestep",
        "TimeStep",
        "timeStep"
    ]

    timestep_col = next(
        (col for col in timestep_candidates if col in df.columns),
        None
    )

    if timestep_col is None:
        raise ValueError(
            "Could not identify timestep column. "
            f"Available columns: {list(df.columns)}"
        )

    df[timestep_col] = pd.to_numeric(
        df[timestep_col],
        errors="coerce"
    )

    print("\nTimestep column:", timestep_col)
    print(
        "Available timesteps:",
        sorted(df[timestep_col].dropna().unique())[:10],
        "...",
        sorted(df[timestep_col].dropna().unique())[-5:]
    )

    # Temporal test set: timesteps 42–49.
    df = df[
        (df[timestep_col] >= 42) &
        (df[timestep_col] <= 49)
    ].copy()

    print("Rows after temporal filtering:", len(df))

    if len(df) == 0:
        raise ValueError(
            "No records found for timesteps 42–49. "
            "Check the timestep values printed above."
        )

    # Use exactly the features used during RF training.
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not contain feature_names_in_. "
            "Cannot safely reconstruct its training feature order."
        )

    feature_names = list(model.feature_names_in_)

    missing = [
        col for col in feature_names
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing[:20]}"
        )

    X = df[feature_names].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    X = X.astype(np.float32)

    # Reproducible sample.
    if len(X) > 200:
        X = X.sample(
            n=200,
            random_state=42
        )

    if X.empty:
        raise ValueError("Test dataset is empty.")

    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: {X.shape[1]} vs "
            f"{model.n_features_in_}"
        )

    print("\nTest samples:", len(X))
    print("Input features:", X.shape[1])

    return X


# ============================================================
# 3. SHAP IMPORTANCE
# ============================================================

def calculate_shap_importance(model, X):

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X)

    if isinstance(values, list):

        if len(values) == 2:
            values = values[1]
        else:
            values = values[0]

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            values = values[:, :, 1]

        elif values.shape[0] == 2:
            values = values[1]

        else:
            raise ValueError(
                f"Unexpected SHAP shape: {values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"SHAP values must be 2D. Got {values.shape}"
        )

    if values.shape[0] == 0:
        raise ValueError("SHAP received zero samples.")

    importance = np.abs(values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X.columns.astype(str),
        "shap_importance": importance
    })

    shap_df = shap_df.sort_values(
        "shap_importance",
        ascending=False
    )

    shap_df.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("SHAP calculation completed.")
    print("SHAP matrix shape:", values.shape)

    return shap_df


# ============================================================
# 4. LOAD GNN IMPORTANCE
# ============================================================

def load_gnn_importance():

    print("\nLoading GNNExplainer results...")

    if not GNN_PATH.exists():
        raise FileNotFoundError(GNN_PATH)

    df = pd.read_csv(GNN_PATH)

    if "feature" not in df.columns:
        raise ValueError(
            "GNN CSV does not contain feature column."
        )

    if "importance" in df.columns:
        df = df.rename(
            columns={"importance": "gnn_importance"}
        )

    if "gnn_importance" not in df.columns:
        raise ValueError(
            "GNN CSV does not contain importance values."
        )

    df["feature"] = df["feature"].astype(str).str.strip()

    df["gnn_importance"] = pd.to_numeric(
        df["gnn_importance"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["gnn_importance"]
    )

    print("GNN features:", len(df))

    return df[["feature", "gnn_importance"]]


# ============================================================
# 5. COMPARE EXPLANATIONS
# ============================================================

def compare_methods(shap_df, gnn_df):

    print("\nComparing explanation methods...")

    comparison = pd.merge(
        shap_df,
        gnn_df,
        on="feature",
        how="inner"
    )

    if len(comparison) < 2:
        raise ValueError(
            "Insufficient common features between SHAP and GNN."
        )

    comparison["shap_rank"] = (
        comparison["shap_importance"]
        .rank(ascending=False, method="average")
    )

    comparison["gnn_rank"] = (
        comparison["gnn_importance"]
        .rank(ascending=False, method="average")
    )

    comparison["rank_difference"] = (
        comparison["shap_rank"] -
        comparison["gnn_rank"]
    ).abs()

    comparison = comparison.sort_values("shap_rank")

    comparison.to_csv(
        OUTPUT_DIR / "shap_gnn_feature_comparison.csv",
        index=False
    )

    spearman, spearman_p = spearmanr(
        comparison["shap_rank"],
        comparison["gnn_rank"]
    )

    kendall, kendall_p = kendalltau(
        comparison["shap_rank"],
        comparison["gnn_rank"]
    )

    if not np.isfinite(spearman):
        raise ValueError(
            "Spearman correlation is undefined. "
            "Check whether importance values are constant."
        )

    print("\n========== RANK CORRELATION ==========")
    print("Common features:", len(comparison))
    print(f"Spearman: {spearman:.6f}")
    print(f"Spearman p-value: {spearman_p:.6e}")
    print(f"Kendall: {kendall:.6f}")
    print(f"Kendall p-value: {kendall_p:.6e}")

    overlap_results = []

    for k in [5, 10, 20]:

        actual_k = min(k, len(comparison))

        shap_top = set(
            comparison.nsmallest(
                actual_k,
                "shap_rank"
            )["feature"]
        )

        gnn_top = set(
            comparison.nsmallest(
                actual_k,
                "gnn_rank"
            )["feature"]
        )

        common = shap_top.intersection(gnn_top)

        overlap_results.append({
            "top_k": actual_k,
            "overlap_count": len(common),
            "overlap_percentage": (
                len(common) / actual_k
            ) * 100,
            "common_features": ", ".join(sorted(common))
        })

    overlap_df = pd.DataFrame(overlap_results)

    overlap_df.to_csv(
        OUTPUT_DIR / "shap_gnn_topk_overlap.csv",
        index=False
    )

    print("\n========== TOP-K OVERLAP ==========")
    print(
        overlap_df[
            ["top_k", "overlap_count", "overlap_percentage"]
        ].to_string(index=False)
    )

    correlations = {
        "spearman": spearman,
        "spearman_p": spearman_p,
        "kendall": kendall,
        "kendall_p": kendall_p
    }

    return comparison, overlap_df, correlations


# ============================================================
# 6. VISUALIZATIONS
# ============================================================

def create_visualizations(comparison, overlap_df):

    print("\nGenerating visualizations...")

    # Feature rank comparison.
    plt.figure(figsize=(9, 7))

    plt.scatter(
        comparison["shap_rank"],
        comparison["gnn_rank"],
        alpha=0.7
    )

    plt.xlabel("SHAP Rank")
    plt.ylabel("GNNExplainer Rank")
    plt.title("SHAP vs GNNExplainer Feature Rankings")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_vs_gnn_rank.png",
        dpi=300
    )

    plt.close()

    # Importance comparison.
    plt.figure(figsize=(9, 7))

    plt.scatter(
        comparison["shap_importance"],
        comparison["gnn_importance"],
        alpha=0.7
    )

    plt.xlabel("Mean Absolute SHAP Importance")
    plt.ylabel("GNNExplainer Importance")
    plt.title("Feature Importance Comparison")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_vs_gnn_importance.png",
        dpi=300
    )

    plt.close()

    # Top-K overlap.
    plt.figure(figsize=(8, 5))

    plt.bar(
        overlap_df["top_k"].astype(str),
        overlap_df["overlap_percentage"]
    )

    plt.xlabel("Top-K Features")
    plt.ylabel("Overlap Percentage")
    plt.title("Top-K Feature Overlap")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_gnn_topk_overlap.png",
        dpi=300
    )

    plt.close()

    print("Visualizations saved.")


# ============================================================
# 7. RESEARCH REPORT
# ============================================================

def generate_report(comparison, overlap_df, correlations):

    report = f"""
# SHAP vs GNNExplainer Comparison

## Objective

This experiment compares feature-level explanations from two
different machine learning architectures:

- Random Forest explained using SHAP.
- Graph Convolutional Network explained using GNNExplainer.

## Experimental Setup

- Dataset: Elliptic Bitcoin transaction dataset.
- Temporal evaluation window: timesteps 42–49.
- SHAP aggregation: mean absolute SHAP values over 200 or fewer
  test samples.
- GNN explanation: one local node explanation.
- Common features compared: {len(comparison)}.

## Rank Correlation

- Spearman correlation: {correlations['spearman']:.6f}
- Spearman p-value: {correlations['spearman_p']:.6e}
- Kendall correlation: {correlations['kendall']:.6f}
- Kendall p-value: {correlations['kendall_p']:.6e}

## Top-K Feature Overlap

{overlap_df.to_string(index=False)}

## Top 15 Features

{comparison.nsmallest(15, 'shap_rank').to_string(index=False)}

## Interpretation

The rank correlation measures the degree of agreement between
the feature rankings produced by the two explanation methods.

Top-K overlap measures how many highly ranked features are shared.

A low agreement may indicate that the models rely on different
feature patterns, but it does not establish why the models differ.

## Limitations

1. SHAP explains a Random Forest, while GNNExplainer explains a GCN.
2. SHAP aggregates explanations across multiple test transactions.
3. The current GNNExplainer result represents one local node.
4. Raw importance magnitudes are not directly comparable.
5. Feature importance does not establish causality.
6. A single local GNN explanation cannot represent the entire graph.
7. These results do not establish which explanation is more correct.

## Conclusion

This experiment provides an initial comparison of tree-based
and graph-based explanations for illicit Bitcoin transaction
classification.

The results should be interpreted as model-specific explanation
agreement rather than evidence of explanation correctness.
"""

    path = OUTPUT_DIR / "shap_gnn_comparison_summary.md"

    path.write_text(report, encoding="utf-8")

    print("\nResearch report saved:", path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SHAP VS GNNEXPLAINER COMPARISON")
    print("=" * 60)

    model = load_model()

    X = load_data(model)

    shap_df = calculate_shap_importance(
        model,
        X
    )

    gnn_df = load_gnn_importance()

    comparison, overlap_df, correlations = compare_methods(
        shap_df,
        gnn_df
    )

    create_visualizations(
        comparison,
        overlap_df
    )

    generate_report(
        comparison,
        overlap_df,
        correlations
    )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated outputs:")
    print(OUTPUT_DIR)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
