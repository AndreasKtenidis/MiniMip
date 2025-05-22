from typing import List

import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from data.abstract_table import AbstractTable

class BlobDataset2(AbstractTable):

    def __init__(self, partition_id, num_partitions):
        self.partition_id = partition_id
        self.num_partitions = num_partitions

    def get_attributes(self, *attributes: str):
        ind = [BlobDataset2.attributes.index(attribute) for attribute in attributes]
        _part_size = np.ceil(len(BlobDataset2.records) / self.num_partitions).astype(int)
        _part_start = self.partition_id * _part_size.astype(int)
        if self.partition_id + 1 == self.num_partitions:
            return pd.DataFrame(BlobDataset2.records[_part_start:, ind],columns=attributes).values
        else:
            return pd.DataFrame(BlobDataset2.records[_part_start:_part_start + _part_size, ind][:_part_size],columns=attributes).values

    def get_attribute(self, attribute):
        ind = BlobDataset2.attributes.index(attribute)
        _part_size = np.ceil(len(BlobDataset2.records) / self.num_partitions).astype(int)
        _part_start = self.partition_id*_part_size.astype(int)
        if self.partition_id+1==self.num_partitions:
            return pd.DataFrame(BlobDataset2.records[_part_start:, ind]).values
        else:
            return pd.DataFrame(BlobDataset2.records[_part_start:_part_start + _part_size, ind][:_part_size]).values

    def get_attribute_names(self):
        return BlobDataset2.attributes

    attributes = ['x', 'y']
    records, _ = make_blobs(n_samples=300, centers=3, random_state=42)
