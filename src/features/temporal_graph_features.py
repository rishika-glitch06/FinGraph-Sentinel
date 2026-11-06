import pandas as pd
from pathlib import Path

print("Loading transaction and edge data...")

transactions = pd.read_csv(
    "data/processed/transactions.csv"
)

edges = pd.read_csv(
    "data/processed/edges.csv"
)

# Map transaction IDs to time steps
time_map = transactions.set_index("txId")["time_step"]

edges["source_time"] = edges["source"].map(time_map)
edges["target_time"] = edges["target"].map(time_map)

# Remove edges with unknown transaction IDs
edges = edges.dropna(
    subset=["source_time", "target_time"]
).copy()

# An edge is available when both endpoints have appeared
edges["edge_time"] = edges[
    ["source_time", "target_time"]
].max(axis=1)

print("Valid edges:", len(edges))

# Calculate temporal in-degree
incoming = edges[
    edges["edge_time"] <= edges["target_time"]
]

in_degree = (
    incoming.groupby("target")
    .size()
    .rename("temporal_in_degree")
)

# Calculate temporal out-degree
outgoing = edges[
    edges["edge_time"] <= edges["source_time"]
]

out_degree = (
    outgoing.groupby("source")
    .size()
    .rename("temporal_out_degree")
)

# Merge features
result = transactions.copy()

result = result.merge(
    in_degree,
    left_on="txId",
    right_index=True,
    how="left"
)

result = result.merge(
    out_degree,
    left_on="txId",
    right_index=True,
    how="left"
)

result["temporal_in_degree"] = (
    result["temporal_in_degree"].fillna(0)
)

result["temporal_out_degree"] = (
    result["temporal_out_degree"].fillna(0)
)

result["temporal_total_degree"] = (
    result["temporal_in_degree"]
    + result["temporal_out_degree"]
)

# Save
output_path = Path(
    "data/processed/transactions_temporal_features.csv"
)

result.to_csv(output_path, index=False)

print("\nTEMPORAL GRAPH FEATURES CREATED")
print("Transactions:", len(result))
print("Output shape:", result.shape)

print("\nFeature summary:")
print(
    result[
        [
            "temporal_in_degree",
            "temporal_out_degree",
            "temporal_total_degree"
        ]
    ].describe()
)

print("\nSaved:", output_path)