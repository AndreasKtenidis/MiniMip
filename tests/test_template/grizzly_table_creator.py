"""
Usage
-----
python generate_covariance_data.py --num-clients 4 --size-mb 100
"""

from pathlib import Path

import duckdb

from library.templates.partitioned_table import PartitionedPandasTable
from tests.help_datasets.blob import BlobDataset


def __find_project_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find pyproject.toml")

BASE_DIR = (__find_project_root()/ "data"/ "grizzly_tests")

def __ensure_dir(path: Path) -> None:
    """Create directory path if it doesn’t exist (including parents)."""
    path.mkdir(parents=True, exist_ok=True)

def create_db(dataset:PartitionedPandasTable, *, client_id,num_clients, base_dir=BASE_DIR) -> None:
    # Getting the local pandas Dataset
    local_df = dataset.get_local_dataset(client_id, num_clients)
    # Getting naming convenrions
    class_name = type(dataset).__name__
    n_rows, n_cols = local_df.shape
    db_path = base_dir / f"{class_name}"
    __ensure_dir(db_path)
    db_path = db_path/f"{n_rows}x{n_cols}_client{client_id}.duckdb"
    # Creating the Database
    con = duckdb.connect(db_path, read_only=False)
    con.register("client_data", local_df)
    con.execute(f"CREATE OR REPLACE TABLE {class_name} AS SELECT * FROM client_data")
    con.close()

if __name__ == "__main__":
    create_db(dataset=BlobDataset(), client_id = 0,num_clients=2)