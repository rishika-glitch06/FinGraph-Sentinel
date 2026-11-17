import numpy as np
import pandas as pd
import torch

from src.models.temporal_gcn import TemporalGCN, build_snapshot


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


def test_build_snapshot_filters_by_cutoff():
    transactions = create_dummy_transactions()
    edges = create_dummy_edges()

    x, edge_index, y, time = build_snapshot(
        transactions,
        edges,
        cutoff=2
    )

    assert x.shape == (2, 165)
    assert edge_index.shape == (2, 1)

    assert torch.equal(
        edge_index,
        torch.tensor([[0], [1]], dtype=torch.long)
    )

    assert torch.equal(
        y,
        torch.tensor([0, 1], dtype=torch.long)
    )

    assert torch.equal(
        time,
        torch.tensor([1, 2], dtype=torch.long)
    )


def test_build_snapshot_returns_correct_tensor_types():
    transactions = create_dummy_transactions()
    edges = create_dummy_edges()

    x, edge_index, y, time = build_snapshot(
        transactions,
        edges,
        cutoff=3
    )

    assert x.dtype == torch.float32
    assert edge_index.dtype == torch.long
    assert y.dtype == torch.long
    assert time.dtype == torch.long

    assert x.shape == (3, 165)
    assert edge_index.shape == (2, 2)


def test_temporal_gcn_forward_pass():
    model = TemporalGCN(
        input_channels=165,
        hidden_channels=8
    )

    model.eval()

    x = torch.randn(3, 165)

    edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long
    )

    with torch.no_grad():
        output = model(x, edge_index)

    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()


def test_temporal_gcn_output_has_two_classes():
    model = TemporalGCN(
        input_channels=165,
        hidden_channels=8
    )

    model.eval()

    x = torch.randn(4, 165)

    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long
    )

    with torch.no_grad():
        output = model(x, edge_index)

    probabilities = torch.softmax(output, dim=1)

    assert probabilities.shape == (4, 2)
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(4),
        atol=1e-6
    )
