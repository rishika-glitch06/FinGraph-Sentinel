import pandas as pd
import numpy as np
import torch

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "transactions_temporal_features.csv"
EDGE_PATH = ROOT / "data" / "processed" / "edges.csv"


def load_gnn_data():

    print("Loading transaction data...")

    transactions = pd.read_csv(DATA_PATH)
    edges = pd.read_csv(EDGE_PATH)

    feature_columns = [
        f"feature_{i}" for i in range(1, 166)
    ]

    # Binary labels
    transactions["target"] = -1

    transactions.loc[
        transactions["class"] == "licit", "target"
    ] = 0

    transactions.loc[
        transactions["class"] == "illicit", "target"
    ] = 1

    # Node indexing
    transactions = transactions.reset_index(drop=True)

    node_mapping = {
        tx_id: index
        for index, tx_id in enumerate(transactions["txId"])
    }

    # Convert transaction IDs into node indices
    source = edges["source"].map(node_mapping)
    target = edges["target"].map(node_mapping)

    if source.isna().any() or target.isna().any():
        raise ValueError("Some edge endpoints are missing.")

    edge_index = torch.tensor(
        np.stack([
            source.to_numpy(),
            target.to_numpy()
        ]),
        dtype=torch.long
    )

    # Feature matrix
    x = torch.tensor(
        transactions[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32
    )

    # Labels (-1 represents unknown)
    y = torch.tensor(
        transactions["target"].to_numpy(),
        dtype=torch.long
    )

    # Chronological masks
    time = transactions["time_step"].to_numpy()

    train_mask = torch.tensor(time <= 35)
    val_mask = torch.tensor((time >= 36) & (time <= 41))
    test_mask = torch.tensor(time >= 42)

    print("\nGNN Dataset Summary")
    print("---------------------------")
    print("Nodes:", x.shape[0])
    print("Features:", x.shape[1])
    print("Edges:", edge_index.shape[1])

    print("\nSplit sizes:")
    print("Train:", train_mask.sum().item())
    print("Validation:", val_mask.sum().item())
    print("Test:", test_mask.sum().item())

    print("\nLabelled nodes:")
    print("Train:", (train_mask & (y >= 0)).sum().item())
    print("Validation:", (val_mask & (y >= 0)).sum().item())
    print("Test:", (test_mask & (y >= 0)).sum().item())

    print("\nTensor shapes:")
    print("X:", x.shape)
    print("Edge index:", edge_index.shape)
    print("Y:", y.shape)

    return (
        transactions,
        edge_index,
        x,
        y,
        train_mask,
        val_mask,
        test_mask
    )


if __name__ == "__main__":

    load_gnn_data()