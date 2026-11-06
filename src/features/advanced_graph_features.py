import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path("data/processed")

print("Loading datasets...")

transactions = pd.read_csv(
    DATA_DIR / "transactions_with_graph_features.csv"
)
edges = pd.read_csv(DATA_DIR / "edges.csv")

print("Building graph...")

G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    create_using=nx.DiGraph()
)

G.add_nodes_from(transactions["txId"])

U = G.to_undirected()

print("Calculating connected components...")

component_sizes = {}

for component in nx.connected_components(U):
    size = len(component)
    for node in component:
        component_sizes[node] = size

print("Calculating clustering coefficients...")

clustering = nx.clustering(U)

print("Calculating core numbers...")

core_numbers = nx.core_number(U)

print("Creating advanced features...")

advanced_features = pd.DataFrame({
    "txId": list(U.nodes()),
    "component_size": [
        component_sizes.get(n, 0) for n in U.nodes()
    ],
    "clustering_coefficient": [
        clustering.get(n, 0) for n in U.nodes()
    ],
    "core_number": [
        core_numbers.get(n, 0) for n in U.nodes()
    ]
})

final_data = transactions.merge(
    advanced_features,
    on="txId",
    how="left"
)

output = DATA_DIR / "transactions_advanced_features.csv"
final_data.to_csv(output, index=False)

print("\nADVANCED GRAPH FEATURES COMPLETED")
print("Final shape:", final_data.shape)
print("Saved:", output)