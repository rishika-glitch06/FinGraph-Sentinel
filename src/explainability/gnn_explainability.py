from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/transactions_temporal_features.csv"
EDGE_PATH = ROOT / "data/processed/edges.csv"
CHECKPOINT_PATH = ROOT / "models/temporal_gcn.pth"

OUTPUT_DIR = ROOT / "reports/gnn_explainability"
FIGURE_DIR = ROOT / "reports/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

TARGET_TIMESTEP = 42
EXPLAINER_EPOCHS = 50

FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 166)
]


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading transaction data...")

    transactions = pd.read_csv(DATA_PATH)
    edges = pd.read_csv(EDGE_PATH)

    print("Transactions:", transactions.shape)
    print("Edges:", edges.shape)

    label_mapping = {
        "licit": 0,
        "illicit": 1
    }

    transactions["target"] = (
        transactions["class"]
        .map(label_mapping)
        .fillna(-1)
        .astype(int)
    )

    snapshot = (
        transactions[
            transactions["time_step"] <= TARGET_TIMESTEP
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("Snapshot nodes:", len(snapshot))

    node_mapping = {
        tx_id: index
        for index, tx_id in enumerate(snapshot["txId"])
    }

    snapshot_edges = edges[
        edges["source"].isin(node_mapping)
        & edges["target"].isin(node_mapping)
    ]

    source = (
        snapshot_edges["source"]
        .map(node_mapping)
        .to_numpy()
    )

    target = (
        snapshot_edges["target"]
        .map(node_mapping)
        .to_numpy()
    )

    edge_index = torch.tensor(
        np.stack([source, target]),
        dtype=torch.long
    )

    x = torch.tensor(
        snapshot[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    y = torch.tensor(
        snapshot["target"].to_numpy(),
        dtype=torch.long
    )

    time_steps = snapshot["time_step"].to_numpy()

    transaction_ids = snapshot["txId"].to_numpy()

    print("Feature matrix:", tuple(x.shape))
    print("Graph edges:", edge_index.shape[1])

    return (
        x,
        edge_index,
        y,
        time_steps,
        transaction_ids
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("\nLoading trained GCN checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = TemporalGCN(
        input_channels=checkpoint["input_channels"],
        hidden_channels=checkpoint["hidden_channels"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scaler_mean = torch.tensor(
        checkpoint["scaler_mean"],
        dtype=torch.float32
    )

    scaler_scale = torch.tensor(
        checkpoint["scaler_scale"],
        dtype=torch.float32
    )

    print("GCN model loaded successfully.")

    return model, scaler_mean, scaler_scale


# ============================================================
# SELECT A CONNECTED TEST NODE
# ============================================================

def select_target_node(
    model,
    x,
    edge_index,
    y,
    time_steps,
    transaction_ids
):

    print("\nGenerating predictions...")

    with torch.no_grad():

        logits = model(x, edge_index)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = logits.argmax(dim=1)

    timestep_mask = torch.tensor(
        time_steps == TARGET_TIMESTEP
    )

    known_mask = y >= 0

    candidate_mask = timestep_mask & known_mask

    candidates = torch.where(candidate_mask)[0]

    if len(candidates) == 0:

        raise RuntimeError(
            "No labelled nodes found at timestep 42."
        )

    # Incoming degree.
    incoming_degree = torch.bincount(
        edge_index[1],
        minlength=x.shape[0]
    )

    # Remove isolated nodes.
    candidates = candidates[
        incoming_degree[candidates] > 0
    ]

    if len(candidates) == 0:

        raise RuntimeError(
            "No connected labelled nodes found at timestep 42."
        )

    correct = candidates[
        predictions[candidates] == y[candidates]
    ]

    if len(correct) > 0:

        candidates = correct

        print(
            "Selecting a correctly classified connected node."
        )

    else:

        print(
            "No correctly classified candidate. "
            "Selecting a connected labelled node."
        )

    # Sort by illicit probability.
    sorted_candidates = candidates[
        torch.argsort(
            probabilities[candidates, 1],
            descending=True
        )
    ]

    selected = None

    print("\nSearching for a manageable 2-hop neighborhood...")

    for node in sorted_candidates[:200]:

        node_idx = int(node)

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=x.shape[0],
            flow="source_to_target"
        )

        if (
            len(subset) > 1
            and sub_edge_index.shape[1] > 0
            and len(subset) <= 1000
            and sub_edge_index.shape[1] <= 5000
        ):

            selected = (
                node_idx,
                subset,
                sub_edge_index,
                mapping,
                edge_mask
            )

            break

    if selected is None:

        raise RuntimeError(
            "No suitable connected neighborhood found. "
            "The neighborhood size limits may need adjustment."
        )

    (
        selected_node,
        selected_subset,
        selected_edge_index,
        selected_mapping,
        selected_edge_mask
    ) = selected

    actual = int(y[selected_node])

    predicted = int(predictions[selected_node])

    probability = float(
        probabilities[selected_node, 1]
    )

    print("\nSELECTED TRANSACTION")
    print("-" * 40)

    print(
        "Transaction ID:",
        transaction_ids[selected_node]
    )

    print(
        "Timestep:",
        time_steps[selected_node]
    )

    print(
        "Actual class:",
        "illicit" if actual == 1 else "licit"
    )

    print(
        "Predicted class:",
        "illicit" if predicted == 1 else "licit"
    )

    print(
        "Illicit probability:",
        round(probability, 6)
    )

    print(
        "Neighborhood nodes:",
        len(selected_subset)
    )

    print(
        "Neighborhood edges:",
        selected_edge_index.shape[1]
    )

    return (
        selected_node,
        selected_subset,
        selected_edge_index,
        selected_mapping,
        actual,
        predicted,
        probability
    )


# ============================================================
# RUN GNN EXPLAINER
# ============================================================

def explain_node(
    model,
    x,
    selected_subset,
    selected_edge_index,
    selected_mapping,
    transaction_ids
):

    print("\nPreparing local graph...")

    x_sub = x[selected_subset]

    local_node_index = int(
        selected_mapping[0]
    )

    print(
        "Local feature matrix:",
        tuple(x_sub.shape)
    )

    print(
        "Local edge matrix:",
        tuple(selected_edge_index.shape)
    )

    print("\nRunning GNNExplainer...")
    print("Epochs:", EXPLAINER_EPOCHS)

    explainer = Explainer(

        model=model,

        algorithm=GNNExplainer(
            epochs=EXPLAINER_EPOCHS
        ),

        explanation_type="model",

        node_mask_type="attributes",

        edge_mask_type="object",

        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
    )

    explanation = explainer(
        x_sub,
        selected_edge_index,
        index=local_node_index
    )

    print("\nGNNExplainer completed.")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    node_mask = explanation.node_mask

    if node_mask is None:

        raise RuntimeError(
            "Feature importance mask was not generated."
        )

    feature_importance = (
        node_mask.detach()
        .abs()
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    feature_df = pd.DataFrame({

        "feature": FEATURE_COLUMNS,

        "importance": feature_importance

    })

    feature_df = feature_df.sort_values(
        "importance",
        ascending=False
    )

    feature_path = (
        OUTPUT_DIR / "gnn_feature_importance.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False
    )

    print("\nTOP 15 IMPORTANT FEATURES")

    print(
        feature_df.head(15).to_string(index=False)
    )

    # --------------------------------------------------------
    # EDGE IMPORTANCE
    # --------------------------------------------------------

    edge_mask = explanation.edge_mask

    if edge_mask is None:

        raise RuntimeError(
            "Edge importance mask was not generated."
        )

    edge_importance = (
        edge_mask.detach()
        .cpu()
        .numpy()
    )

    original_edge_indices = selected_subset[
        selected_edge_index.cpu()
    ].numpy()

    edge_df = pd.DataFrame({

        "source_transaction": transaction_ids[
            original_edge_indices[0]
        ],

        "target_transaction": transaction_ids[
            original_edge_indices[1]
        ],

        "importance": edge_importance

    })

    edge_df = edge_df.sort_values(
        "importance",
        ascending=False
    )

    edge_path = (
        OUTPUT_DIR / "gnn_edge_importance.csv"
    )

    edge_df.to_csv(
        edge_path,
        index=False
    )

    print("\nTOP 15 IMPORTANT EDGES")

    print(
        edge_df.head(15).to_string(index=False)
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE PLOT
    # --------------------------------------------------------

    top_features = (
        feature_df.head(15)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("GNNExplainer Importance")
    plt.ylabel("Transaction Features")

    plt.title(
        "GNNExplainer: Top 15 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "gnn_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # EDGE IMPORTANCE PLOT
    # --------------------------------------------------------

    top_edges = edge_df.head(15).copy()

    top_edges["edge"] = (
        top_edges["source_transaction"].astype(str)
        + " → "
        + top_edges["target_transaction"].astype(str)
    )

    top_edges = top_edges.sort_values(
        "importance",
        ascending=True
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_edges["edge"],
        top_edges["importance"]
    )

    plt.xlabel("GNNExplainer Importance")
    plt.ylabel("Transaction Connections")

    plt.title(
        "GNNExplainer: Top 15 Important Edges"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "gnn_edge_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return feature_df, edge_df


# ============================================================
# SAVE EXPERIMENT SUMMARY
# ============================================================

def save_summary(
    transaction_id,
    actual,
    predicted,
    probability,
    feature_df,
    edge_df,
    neighborhood_nodes,
    neighborhood_edges
):

    summary_path = (
        OUTPUT_DIR / "gnn_explanation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FIN-GRAPH SENTINEL: GNN EXPLAINABILITY\n"
        )

        file.write("=" * 50 + "\n\n")

        file.write(
            f"Transaction ID: {transaction_id}\n"
        )

        file.write(
            f"Test timestep: {TARGET_TIMESTEP}\n"
        )

        file.write(
            f"Actual class: {'illicit' if actual == 1 else 'licit'}\n"
        )

        file.write(
            f"Predicted class: {'illicit' if predicted == 1 else 'licit'}\n"
        )

        file.write(
            f"Illicit probability: {probability:.6f}\n\n"
        )

        file.write(
            f"Neighborhood nodes: {neighborhood_nodes}\n"
        )

        file.write(
            f"Neighborhood edges: {neighborhood_edges}\n\n"
        )

        file.write("TOP 15 FEATURES\n")
        file.write("-" * 30 + "\n")

        file.write(
            feature_df.head(15).to_string(index=False)
        )

        file.write("\n\nTOP 15 EDGES\n")
        file.write("-" * 30 + "\n")

        file.write(
            edge_df.head(15).to_string(index=False)
        )

        file.write("\n\nLIMITATIONS\n")
        file.write("-" * 30 + "\n")

        file.write(
            "The explanation uses a two-hop local subgraph. "
            "Extracting a subgraph can change GCN degree normalization "
            "compared with the full graph. Therefore, the local "
            "explanation is an approximation of the full-graph "
            "prediction. Feature and edge importance do not establish "
            "causality.\n"
        )

    print("\nSummary saved:", summary_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FIN-GRAPH SENTINEL: GNN EXPLAINABILITY")
    print("=" * 60)

    (
        x,
        edge_index,
        y,
        time_steps,
        transaction_ids
    ) = load_data()

    model, scaler_mean, scaler_scale = load_model()

    print("\nApplying training scaler...")

    x = (x - scaler_mean) / scaler_scale

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    y = y.to(DEVICE)

    (
        selected_node,
        selected_subset,
        selected_edge_index,
        selected_mapping,
        actual,
        predicted,
        probability
    ) = select_target_node(
        model,
        x,
        edge_index,
        y,
        time_steps,
        transaction_ids
    )

    feature_df, edge_df = explain_node(
        model,
        x,
        selected_subset,
        selected_edge_index,
        selected_mapping,
        transaction_ids
    )

    save_summary(
        transaction_id=transaction_ids[selected_node],
        actual=actual,
        predicted=predicted,
        probability=probability,
        feature_df=feature_df,
        edge_df=edge_df,
        neighborhood_nodes=len(selected_subset),
        neighborhood_edges=selected_edge_index.shape[1]
    )

    print("\n" + "=" * 60)
    print("GNN EXPLAINABILITY COMPLETED")
    print("=" * 60)

    print("\nGENERATED FILES")

    print(
        OUTPUT_DIR / "gnn_feature_importance.csv"
    )

    print(
        OUTPUT_DIR / "gnn_edge_importance.csv"
    )

    print(
        OUTPUT_DIR / "gnn_explanation_summary.txt"
    )

    print(
        FIGURE_DIR / "gnn_feature_importance.png"
    )

    print(
        FIGURE_DIR / "gnn_edge_importance.png"
    )


if __name__ == "__main__":
    main()