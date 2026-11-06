import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("data/processed")

print("Loading graph data...")

transactions = pd.read_csv(DATA_DIR / "transactions.csv")
edges = pd.read_csv(DATA_DIR / "edges.csv")

# Create directed graph
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    create_using=nx.DiGraph()
)

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Calculate graph features
print("\nCalculating graph features...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

# Store features
graph_features = pd.DataFrame({
    "txId": list(G.nodes()),
    "in_degree": [in_degree.get(node, 0) for node in G.nodes()],
    "out_degree": [out_degree.get(node, 0) for node in G.nodes()],
    "pagerank": [pagerank.get(node, 0) for node in G.nodes()]
})

# Merge with transaction data
final_data = transactions.merge(graph_features, on="txId", how="left")

final_data.fillna(0, inplace=True)

# Save
output_path = DATA_DIR / "transactions_with_graph_features.csv"
final_data.to_csv(output_path, index=False)

print("\nGRAPH FEATURES CREATED")
print("Final dataset shape:", final_data.shape)
print("Saved to:", output_path)