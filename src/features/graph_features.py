import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("data/processed")

print("Loading datasets...")

transactions = pd.read_csv(DATA_DIR / "transactions.csv")
edges = pd.read_csv(DATA_DIR / "edges.csv")

print("Building directed transaction graph...")

G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    create_using=nx.DiGraph()
)

# Include isolated transactions
G.add_nodes_from(transactions["txId"])

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("Calculating graph features...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

graph_features = pd.DataFrame({
    "txId": list(G.nodes()),
    "in_degree": [in_degree[n] for n in G.nodes()],
    "out_degree": [out_degree[n] for n in G.nodes()],
    "pagerank": [pagerank[n] for n in G.nodes()]
})

final_data = transactions.merge(
    graph_features,
    on="txId",
    how="left"
)

output_path = DATA_DIR / "transactions_with_graph_features.csv"
final_data.to_csv(output_path, index=False)

print("\nGRAPH FEATURES COMPLETED")
print("Final shape:", final_data.shape)
print("Saved:", output_path)