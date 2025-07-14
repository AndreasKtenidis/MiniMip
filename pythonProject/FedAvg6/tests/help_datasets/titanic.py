import pandas as pd

from library.templates.partitioned_table import PartitionedPandasTable

class Titanic(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        return df
