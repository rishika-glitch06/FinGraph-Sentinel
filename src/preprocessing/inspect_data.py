import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

files = {
    "classes": "elliptic_txs_classes.csv",
    "edges": "elliptic_txs_edgelist.csv",
    "features": "elliptic_txs_features.csv"
}

for name, filename in files.items():
    path = DATA_DIR / filename
    df = pd.read_csv(path, header=None)

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:", df.isnull().sum().sum())