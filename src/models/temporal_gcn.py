import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pathlib import Path
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
EDGE_PATH = ROOT / "data/processed/edges.csv"
MODEL_DIR = ROOT / "models"


SEED = 42
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 64

DEVICE = torch.device("cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class TemporalGCN(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels=64):

        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = torch.nn.Linear(
            hidden_channels,
            2
        )

        self.dropout = 0.5

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

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

    time = torch.tensor(
        snapshot["time_step"].to_numpy(),
        dtype=torch.long
    )

    return x, edge_index, y, time


def evaluate(model, transactions, edges, scaler, feature_columns, timesteps):

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for timestep in timesteps:

            x, edge_index, y, time = build_snapshot(
                transactions, edges, timestep
            )

            x = torch.tensor(
                scaler.transform(
                    x.numpy()
                ),
                dtype=torch.float32
            )

            logits = model(x, edge_index)

            mask = (
                (time == timestep)
                & (y >= 0)
            )

            probabilities = torch.softmax(
                logits[mask],
                dim=1
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            all_probabilities.extend(
                probabilities.numpy()
            )

            all_predictions.extend(
                predictions.numpy()
            )

            all_targets.extend(
                y[mask].numpy()
            )

    return (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


def main():

    set_seed()

    print("Loading Elliptic dataset...")

    transactions = pd.read_csv(DATA_PATH)
    edges = pd.read_csv(EDGE_PATH)

    transactions["target"] = -1

    transactions.loc[
        transactions["class"] == "licit", "target"
    ] = 0

    transactions.loc[
        transactions["class"] == "illicit", "target"
    ] = 1

    feature_columns = [
        f"feature_{i}" for i in range(1, 166)
    ]

    # Training nodes: timesteps 1-35
    train_indices = (
        (transactions["time_step"] <= 35)
        & (transactions["target"] >= 0)
    )

    # Fit scaler only on labelled training nodes
    scaler = StandardScaler()

    scaler.fit(
        transactions.loc[
            train_indices, feature_columns
        ]
    )

    print("Building training graph...")

    x, edge_index, y, time = build_snapshot(
        transactions, edges, 35
    )

    x = torch.tensor(
        scaler.transform(x.numpy()),
        dtype=torch.float32
    )

    train_mask = (
        (time <= 35)
        & (y >= 0)
    )

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    y = y.to(DEVICE)
    train_mask = train_mask.to(DEVICE)

    # Class weights for imbalance
    counts = torch.bincount(y[train_mask])

    class_weights = (
        counts.sum() / (2.0 * counts.float())
    ).to(DEVICE)

    model = TemporalGCN(
        input_channels=len(feature_columns),
        hidden_channels=HIDDEN_CHANNELS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-4
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    print("\nStarting GCN training...")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        logits = model(x, edge_index)

        loss = F.cross_entropy(
            logits[train_mask],
            y[train_mask],
            weight=class_weights
        )

        loss.backward()

        optimizer.step()

        # Validation on newly arriving nodes
        val_targets, val_predictions, val_probabilities = evaluate(
            model,
            transactions,
            edges,
            scaler,
            feature_columns,
            range(36, 42)
        )

        val_loss = F.nll_loss(
            torch.log(
                torch.tensor(
                    np.column_stack([
                        1 - val_probabilities,
                        val_probabilities
                    ]),
                    dtype=torch.float32
                ).clamp(min=1e-8)
            ),
            torch.tensor(val_targets, dtype=torch.long)
        ).item()

        val_balanced = balanced_accuracy_score(
            val_targets,
            val_predictions
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Balanced Acc: {val_balanced:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a model.")

    model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "temporal_gcn.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": len(feature_columns),
            "hidden_channels": HIDDEN_CHANNELS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_
        },
        model_path
    )

    print("\nBest model saved:", model_path)

    print("\nEvaluating on untouched test period...")

    y_true, y_pred, y_prob = evaluate(
        model,
        transactions,
        edges,
        scaler,
        feature_columns,
        range(42, 50)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Licit", "Illicit"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_true, y_pred)
    )

    print(
        "Macro F1:",
        f1_score(y_true, y_pred, average="macro")
    )

    print(
        "ROC-AUC:",
        roc_auc_score(y_true, y_prob)
    )

    print(
        "PR-AUC (Illicit):",
        average_precision_score(y_true, y_prob)
    )


if __name__ == "__main__":
    main()