import pandas as pd
import torch

from src.models.gnn_temporal_data import build_temporal_snapshot


def create_dummy_transactions():
    rows = []

    sample_data = [
        (101, 1, 0),
        (102, 2, 1),
        (103, 3, -1),
    ]

    for index, (tx_id, timestep, target) in enumerate(sample_data):
        row = {
            "txId": tx_id,
            "time_step": timestep,
            "target": target,
        }

        for feature in range(1, 166):
            row[f"feature_{feature}"] = float(feature + index)

        rows.append(row)

    return pd.DataFrame(rows)


def create_dummy_edges():
    return pd.DataFrame({
        "source": [101, 102, 999],
        "target": [102, 103, 101],
    })


def test_temporal_snapshot_filters_by_cutoff():
    transactions = create_dummy_transactions()
    edges = create_dummy_edges()

    graph = build_temporal_snapshot(
        transactions,
        edges,
        cutoff=2
    )

    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert graph.x.shape == (2, 165)

    assert torch.equal(
        graph.edge_index,
        torch.tensor([[0], [1]], dtype=torch.long)
    )


def test_temporal_snapshot_preserves_labels_and_timesteps():
    transactions = create_dummy_transactions()
    edges = create_dummy_edges()

    graph = build_temporal_snapshot(
        transactions,
        edges,
        cutoff=3
    )

    assert torch.equal(
        graph.y,
        torch.tensor([0, 1, -1], dtype=torch.long)
    )

    assert torch.equal(
        graph.time_step,
        torch.tensor([1, 2, 3], dtype=torch.long)
    )


def test_temporal_snapshot_has_correct_tensor_types():
    transactions = create_dummy_transactions()
    edges = create_dummy_edges()

    graph = build_temporal_snapshot(
        transactions,
        edges,
        cutoff=3
    )

    assert graph.x.dtype == torch.float32
    assert graph.edge_index.dtype == torch.long
    assert graph.y.dtype == torch.long
    assert graph.time_step.dtype == torch.long

    assert graph.x.shape[1] == 165
    assert graph.edge_index.shape[0] == 2
