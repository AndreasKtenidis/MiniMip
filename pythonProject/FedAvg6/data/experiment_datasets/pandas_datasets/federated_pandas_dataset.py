

from data.abstract_table import AbstractTable
from abc import ABC,abstractmethod
import pandas as pd

class FederatedPandasDataset(AbstractTable,ABC):
    def __init__(self,partition_id, num_partitions):
        # Initialize the data only once
        dataset = self.get_dataset()
        n = len(dataset)
        _size = n // num_partitions  # number of rows per partition (ignores remainder)
        start = partition_id * _size
        end = (
                      partition_id + 1) * _size if partition_id < num_partitions - 1 else n  # last partition may include remainder
        self.dataset= dataset.iloc[start:end]

    @abstractmethod
    def get_dataset(self)-> pd.DataFrame:
        pass

    def get_attribute(self, attribute):
        return self.dataset[attribute].values

    def get_attribute_names(self):
        return list(self.dataset.keys())

    def get_attributes(self, *attributes: str):
        return self.dataset[list(attributes)].to_numpy()