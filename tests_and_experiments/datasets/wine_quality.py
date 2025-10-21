from library.core.partitioned_table import PartitionedPandasTable
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class WineQualityDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        data = pd.read_csv(url, sep=';')
        return data


