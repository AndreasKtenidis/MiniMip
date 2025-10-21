from sklearn.datasets import load_diabetes
import pandas as pd
from tests_and_experiments.core.partitioned_table import PartitionedPandasTable


class DiabetesDiseaseDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        diabetes = load_diabetes()
        x = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        y = pd.Series(diabetes.target, name='target')
        return pd.concat([x, y], axis=1)

