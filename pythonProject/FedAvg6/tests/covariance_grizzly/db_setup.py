"""
Usage
-----
python generate_covariance_data.py --num-clients 4 --size-mb 100
"""

import argparse
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

def ensure_dir(path: Path) -> None:
    """Create directory path if it doesn’t exist (including parents)."""
    path.mkdir(parents=True, exist_ok=True)


def rows_for_size(size_mb: int, bytes_per_row: int = 8) -> int:
    """
    Roughly estimate number of rows that will yield a CSV of `size_mb`.
    Default `bytes_per_row` is an empirical average for two float32 values
    plus comma and newline in plain-text CSV.
    """
    return math.ceil(size_mb * 1024 ** 2 / bytes_per_row)


def generate_csv(csv_path: Path, size_mb: int = 100, seed: int = 42) -> None:
    """
    Create a synthetic two-column CSV of roughly `size_mb` megabytes.
    The columns are correlated so that covariance is non-zero.
    """
    np.random.seed(seed)
    n_rows = rows_for_size(size_mb)
    x = np.random.randn(n_rows).astype(np.float32)
    y = 3 * x + np.random.randn(n_rows).astype(np.float32) * 0.5  # correlated
    df = pd.DataFrame({"x": x, "y": y})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    actual = csv_path.stat().st_size / 1024 ** 2
    print(f"CSV saved to {csv_path}  ({actual:.2f} MB, {len(df):,} rows)")


def make_duckdb_files(csv_path: Path, output_dir: Path, num_clients: int = 2) -> None:
    """
    Split CSV rows evenly across `num_clients` and write DuckDB files:
    covariance_client1.duckdb, covariance_client2.duckdb, ...
    """
    df = pd.read_csv(csv_path)
    chunk = len(df) // num_clients

    for i in range(num_clients):
        start = i * chunk
        end = (i + 1) * chunk if i < num_clients - 1 else len(df)
        client_df = df.iloc[start:end]

        db_path = output_dir / f"covariance_client{i + 1}.duckdb"
        con = duckdb.connect(db_path, read_only=False)
        con.register("client_data", client_df)
        con.execute("CREATE OR REPLACE TABLE covariance AS SELECT * FROM client_data")
        con.close()

        print(f"Client {i + 1}: {db_path}  ({end - start:,} rows)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 100 MB CSV and split to DuckDBs.")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of client DBs to create.")
    parser.add_argument("--size-mb", type=int, default=100, help="Target CSV size in megabytes.")
    args = parser.parse_args()

    # Target directory: pythonProject/FedAvg6/data/grizzly_pandas_test
    base_dir = (
        Path(__file__).resolve().parent
        / ".."
        / ".."
        / "data"
        / "grizzly_pandas_test"
    )
    ensure_dir(base_dir)

    csv_file = base_dir / "covariance_100mb.csv"

    if not csv_file.exists():
        generate_csv(csv_file, size_mb=args.size_mb)
    else:
        size = csv_file.stat().st_size / 1024 ** 2
        print(f"ℹ CSV already exists at {csv_file} ({size:.2f} MB) — skipping generation.")

    make_duckdb_files(csv_file, base_dir, num_clients=args.num_clients)


if __name__ == "__main__":
    main()
