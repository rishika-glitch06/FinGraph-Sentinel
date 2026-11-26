import pandas as pd
import numpy as np
import torch

from pathlib import Path
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "transactions_temporal_features.csv"
EDGE_PATH = ROOT / "data" / "processed" / "edges.csv"


FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 166)
]


def build_temporal_snapshot(transactions, edges, cutoff):

    snapshot = transactions[
        transactions["time_step"] <= cutoff
    ].copy().reset_index(drop=True)

    node_mapping = {
        tx_id: index
        for index, tx_id in enumerate(snapshot["txId"])
    }

    snapshot_edges = edges[
        edges["source"].isin(node_mapping)
        & edges["target"].isin(node_mapping)
    ]

    source = snapshot_edges["source"].map(node_mapping).to_numpy()
    target = snapshot_edges["target"].map(node_mapping).to_numpy()

    edge_array = np.stack([source, target])

    edge_index = torch.tensor(
        edge_array,
        dtype=torch.long
    )

    x = torch.tensor(
        snapshot[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32
    )

    y = torch.tensor(
        snapshot["target"].to_numpy(),
        dtype=torch.long
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.time_step = torch.tensor(
        snapshot["time_step"].to_numpy(),
        dtype=torch.long
    )

    return data


def main():

    print("Loading dataset...")

    transactions = pd.read_csv(DATA_PATH)
    edges = pd.read_csv(EDGE_PATH)

    transactions["target"] = -1

    transactions.loc[
        transactions["class"] == "licit", "target"
    ] = 0

    transactions.loc[
        transactions["class"] == "illicit", "target"
    ] = 1

    print("\nTRAINING GRAPH")

    train_graph = build_temporal_snapshot(
        transactions, edges, 35
    )

    train_mask = (
        (train_graph.time_step <= 35)
        & (train_graph.y >= 0)
    )

    print("Nodes:", train_graph.num_nodes)
    print("Edges:", train_graph.num_edges)
    print("Labelled training nodes:", train_mask.sum().item())

    print("\nVALIDATION SNAPSHOTS")

    for timestep in range(36, 42):

        graph = build_temporal_snapshot(
            transactions, edges, timestep
        )

        target_mask = (
            (graph.time_step == timestep)
            & (graph.y >= 0)
        )

        print(
            f"Step {timestep}: "
            f"Nodes={graph.num_nodes}, "
            f"Edges={graph.num_edges}, "
            f"Evaluation nodes={target_mask.sum().item()}"
        )

    print("\nTEST SNAPSHOTS")

    for timestep in range(42, 50):

        graph = build_temporal_snapshot(
            transactions, edges, timestep
        )

        target_mask = (
            (graph.time_step == timestep)
            & (graph.y >= 0)
        )

        print(
            f"Step {timestep}: "
            f"Nodes={graph.num_nodes}, "
            f"Edges={graph.num_edges}, "
            f"Evaluation nodes={target_mask.sum().item()}"
        )


if __name__ == "__main__":
    main()