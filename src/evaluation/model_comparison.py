import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports"

GCN_PATH = REPORT_DIR / "gnn_per_timestep_metrics.csv"
RF_PATH = REPORT_DIR / "rf_per_timestep_metrics.csv"

OUTPUT_CSV = REPORT_DIR / "gcn_vs_rf_comparison.csv"

# Overall metrics from the completed experiments
overall_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "ROC-AUC",
        "Licit Recall",
        "Illicit Recall"
    ],
    "GCN": [
        0.595,
        0.7028,
        0.4458,
        0.7299,
        0.8211,
        0.5846
    ],
    "Random Forest": [
        0.9734,
        0.7353,
        0.8039,
        0.8767,
        0.4730,
        0.9976
    ]
})


def main():

    print("Loading GCN and Random Forest results...")

    gcn = pd.read_csv(GCN_PATH)
    rf = pd.read_csv(RF_PATH)

    # Merge timestep results
    comparison = pd.merge(
        gcn,
        rf,
        on="timestep",
        suffixes=("_gcn", "_rf")
    )

    # Select important comparison columns
    selected_columns = [
        "timestep",
        "balanced_accuracy_gcn",
        "balanced_accuracy_rf",
        "macro_f1_gcn",
        "macro_f1_rf",
        "licit_recall_gcn",
        "licit_recall_rf",
        "illicit_recall_gcn",
        "illicit_recall_rf",
        "roc_auc_gcn",
        "roc_auc_rf"
    ]

    comparison = comparison[selected_columns]

    comparison.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nOverall Model Comparison:")
    print(overall_results.to_string(index=False))

    print("\nPer-Timestep Comparison:")
    print(comparison.to_string(index=False))

    # ----------------------------------
    # Chart 1: Overall metric comparison
    # ----------------------------------

    ax = overall_results.set_index("Metric").plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_title("GCN vs Random Forest - Overall Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylim(0, 1.1)

    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "overall_model_comparison.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------
    # Chart 2: Balanced accuracy by time
    # ----------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        comparison["timestep"],
        comparison["balanced_accuracy_gcn"],
        marker="o",
        label="GCN"
    )

    plt.plot(
        comparison["timestep"],
        comparison["balanced_accuracy_rf"],
        marker="s",
        label="Random Forest"
    )

    plt.title("Balanced Accuracy Across Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Balanced Accuracy")
    plt.xticks(range(42, 50))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "timestep_balanced_accuracy.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------
    # Chart 3: Class-wise recall
    # ----------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    axes[0].plot(
        comparison["timestep"],
        comparison["licit_recall_gcn"],
        marker="o",
        label="GCN"
    )

    axes[0].plot(
        comparison["timestep"],
        comparison["licit_recall_rf"],
        marker="s",
        label="Random Forest"
    )

    axes[0].set_title("Licit Transaction Recall")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Recall")
    axes[0].set_xticks(range(42, 50))
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        comparison["timestep"],
        comparison["illicit_recall_gcn"],
        marker="o",
        label="GCN"
    )

    axes[1].plot(
        comparison["timestep"],
        comparison["illicit_recall_rf"],
        marker="s",
        label="Random Forest"
    )

    axes[1].set_title("Illicit Transaction Recall")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Recall")
    axes[1].set_xticks(range(42, 50))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "classwise_recall_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nAll comparison files generated successfully!")

    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(REPORT_DIR / "overall_model_comparison.png")
    print(REPORT_DIR / "timestep_balanced_accuracy.png")
    print(REPORT_DIR / "classwise_recall_comparison.png")


if __name__ == "__main__":
    main()