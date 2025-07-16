from library.templates.partitioned_table import PartitionedPandasTable
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

class IrisDataset(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        iris = load_iris()
        iris = load_iris()
        rng = np.random.RandomState(42)  # For reproducibility
        shuffled_indices = rng.permutation(len(iris.data))

        iris.data = iris.data[shuffled_indices]
        iris.target = iris.target[shuffled_indices]

        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Filter to only use two classes (0 and 1)
        df = df[df['target'] != 2]
        return df

