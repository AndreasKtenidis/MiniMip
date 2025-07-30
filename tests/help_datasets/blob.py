from sklearn.datasets import load_diabetes
import pandas as pd
from library.templates.partitioned_table import PartitionedPandasTable
from sklearn.datasets import make_blobs


class BlobDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        records, _ = make_blobs(n_samples=500, centers=3, random_state=42)
        return pd.DataFrame(records, columns=["x", "y"])