import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score
)


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
EDGE_PATH = ROOT / "data/processed/edges.csv"
MODEL_PATH = ROOT / "models/temporal_gcn.pth"
OUTPUT_PATH = ROOT / "reports/gnn_per_timestep_metrics.csv"

DEVICE = torch.device("cpu")


class TemporalGCN(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels=64):

        super().__init__()

        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 2)
        self.dropout = 0.5

    def forward(self, x, edge_index):

        import torch.nn.functional as F

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.classifier(x)


def build_snapshot(transactions, edges, cutoff):

    snapshot = transactions[
        transactions["time_step"] <= cutoff
    ].copy().reset_index(drop=True)

    node_mapping = {
        tx_id: index
        for index, tx_id in enumerate(snapshot["txId"])
    }

    valid_edges = edges[
        edges["source"].isin(node_mapping)
        & edges["target"].isin(node_mapping)
    ]

    source = valid_edges["source"].map(node_mapping).to_numpy()
    target = valid_edges["target"].map(node_mapping).to_numpy()

    edge_index = torch.tensor(
        np.stack([source, target]),
        dtype=torch.long
    )

    feature_columns = [
        f"feature_{i}" for i in range(1, 166)
    ]

    x = torch.tensor(
        snapshot[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32
    )

    y = torch.tensor(
        snapshot["target"].to_numpy(),
        dtype=torch.long
    )

    time = snapshot["time_step"].to_numpy()

    return x, edge_index, y, time


def main():

    print("Loading dataset and trained GCN...")

    transactions = pd.read_csv(DATA_PATH)
    edges = pd.read_csv(EDGE_PATH)

    transactions["target"] = -1

    transactions.loc[
        transactions["class"] == "licit", "target"
    ] = 0

    transactions.loc[
        transactions["class"] == "illicit", "target"
    ] = 1

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = TemporalGCN(
        input_channels=checkpoint["input_channels"],
        hidden_channels=checkpoint["hidden_channels"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]

    results = []

    print("\nEvaluating individual timesteps...\n")

    for timestep in range(42, 50):

        x, edge_index, y, time = build_snapshot(
            transactions,
            edges,
            timestep
        )

        # Apply the same training scaler
        x = (
            x.numpy() - scaler_mean
        ) / scaler_scale

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        with torch.no_grad():

            logits = model(x, edge_index)

            probabilities = torch.softmax(
                logits, dim=1
            )[:, 1].numpy()

        mask = (
            (time == timestep)
            & (y.numpy() >= 0)
        )

        y_true = y.numpy()[mask]
        y_prob = probabilities[mask]
        y_pred = (y_prob >= 0.5).astype(int)

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=[0, 1],
                zero_division=0
            )
        )

        result = {
            "timestep": timestep,
            "transactions": len(y_true),
            "licit_support": support[0],
            "illicit_support": support[1],
            "licit_precision": precision[0],
            "licit_recall": recall[0],
            "illicit_precision": precision[1],
            "illicit_recall": recall[1],
            "macro_f1": f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "illicit_pr_auc": average_precision_score(
                y_true, y_prob
            )
        }

        results.append(result)

        print(
            f"Step {timestep} | "
            f"Transactions: {len(y_true)} | "
            f"Licit Recall: {recall[0]:.4f} | "
            f"Illicit Recall: {recall[1]:.4f} | "
            f"Balanced Accuracy: {result['balanced_accuracy']:.4f}"
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nComplete Results:")
    print(results_df.to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()