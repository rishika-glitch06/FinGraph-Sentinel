import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

transactions = pd.read_csv(DATA_DIR / "transactions.csv")
edges = pd.read_csv(DATA_DIR / "edges.csv")

print("\n========== DATA VALIDATION ==========")

# 1. Duplicate transactions
print("Duplicate transactions:", transactions["txId"].duplicated().sum())

# 2. Duplicate edges
print("Duplicate edges:", edges.duplicated().sum())

# 3. Missing values
print("Missing transaction values:", transactions.isnull().sum().sum())
print("Missing edge values:", edges.isnull().sum().sum())

# 4. Invalid labels
print("Labels:", transactions["class"].value_counts().to_dict())

# 5. Graph integrity
transaction_ids = set(transactions["txId"])
source_ids = set(edges["source"])
target_ids = set(edges["target"])

print("Unique transactions:", len(transaction_ids))
print("Edges with unknown source:", len(source_ids - transaction_ids))
print("Edges with unknown target:", len(target_ids - transaction_ids))

# 6. Time-step validation
print("Number of time steps:", transactions["time_step"].nunique())

print("\nVALIDATION COMPLETED")