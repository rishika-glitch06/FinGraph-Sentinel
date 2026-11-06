import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
features = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)

# Assign column names
classes.columns = ["txId", "class"]
edges.columns = ["source", "target"]

feature_columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, 166)]
)
features.columns = feature_columns

# Clean labels
classes["class"] = classes["class"].replace({
    "1": "licit",
    "2": "illicit",
    "unknown": "unknown"
})

classes["txId"] = classes["txId"].astype("int64")
edges = edges.astype("int64")
features["txId"] = features["txId"].astype("int64")

# Merge features and labels
data = features.merge(classes, on="txId", how="left")

# Save processed datasets
data.to_csv(PROCESSED_DIR / "transactions.csv", index=False)
edges.to_csv(PROCESSED_DIR / "edges.csv", index=False)

print("DATA CLEANING COMPLETED")
print("Transactions:", data.shape)
print("Edges:", edges.shape)
print("\nClass distribution:")
print(data["class"].value_counts(dropna=False))
print("\nTime steps:")
print(sorted(data["time_step"].unique()))