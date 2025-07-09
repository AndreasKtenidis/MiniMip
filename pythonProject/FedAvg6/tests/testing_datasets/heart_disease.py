import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from library.templates.partitioned_table import PartitionedPandasTable


class HeartDisease(PartitionedPandasTable):

    def get_dataset(self) -> pd.DataFrame:
        return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"fedesoriano/heart-failure-prediction","heart.csv")



