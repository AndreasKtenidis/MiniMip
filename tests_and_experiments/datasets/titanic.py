import pandas as pd

from library.core.partitioned_table import PartitionedPandasTable

class TitanicDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        return df
