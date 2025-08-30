"""
Usage
-----
python generate_covariance_data.py --num-clients 4 --size-mb 100
"""
import grizzly as grizzly
from pathlib import Path
from grizzly.sqlgenerator import SQLGenerator
import duckdb

from library.templates.partitioned_table import PartitionedPandasTable
from grizzly.relationaldbexecutor import RelationalExecutor

class GrizzlyFactory:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = (GrizzlyFactory.__find_project_root() / "data" / "grizzly_tests")
        else:
            self.base_dir = base_dir

    @staticmethod
    def __find_project_root():
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        raise FileNotFoundError("Could not find pyproject.toml")


    @staticmethod
    def __ensure_dir(path: Path) -> None:
        """Create directory path if it doesn’t exist (including parents)."""
        path.mkdir(parents=True, exist_ok=True)


    def create_db(self,dataset:PartitionedPandasTable, *, client_id,num_clients):
        # Getting the local pandas Dataset
        local_df = dataset.get_local_dataset(client_id, num_clients)
        # Getting naming convenrions
        class_name = type(dataset).__name__
        n_rows, n_cols = local_df.shape
        db_path = self.base_dir / f"{class_name}"
        GrizzlyFactory.__ensure_dir(db_path)
        db_path = db_path/f"{n_rows}x{n_cols}_client{client_id}.duckdb"
        # Creating Connection
        con = duckdb.connect(db_path, read_only=False)
        # Populating the Database
        con.register("client_data", local_df)
        con.execute(f"CREATE OR REPLACE TABLE {class_name} AS SELECT * FROM client_data")
        # Creating the Grizzly Object
        gen = SQLGenerator("duckdb")
        executor = RelationalExecutor(con, gen)
        grizzly.use(executor)
        return grizzly.read_table(f"{class_name}")


