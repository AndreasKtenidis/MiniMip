from library.templates.partitioned_table import PartitionedPandasTable
import pandas as pd
from library.templates.partitioned_table import PartitionedPandasTable
import statsmodels.api as sm

class JobTrainingDataset(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        dataset = sm.datasets.get_rdataset("lalonde", "MatchIt").data
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        return dataset
