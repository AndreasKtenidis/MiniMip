import pandas as pd

from library.core.partitioned_table import PartitionedPandasTable


class DummyDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        data = {
            'A': [10, 20, 30, 40],
            'B': [1.5, 2.5, 3.5, 4.5],
            'C': [100, 200, 300, 400]
        }
        # Create DataFrame
        df = pd.DataFrame(data)
        return df
