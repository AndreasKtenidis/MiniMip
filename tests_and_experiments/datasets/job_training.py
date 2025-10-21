from library.core.partitioned_table import PartitionedPandasTable
import pandas as pd
from library.core.partitioned_table import PartitionedPandasTable
import statsmodels.api as sm

class JobTrainingDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        dataset = sm.datasets.get_rdataset("lalonde", "MatchIt").data
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        race_dummies = pd.get_dummies(dataset["race"])
        dataset = dataset.drop(columns=["race"])
        dataset = pd.concat([dataset, race_dummies], axis=1)
        return dataset
