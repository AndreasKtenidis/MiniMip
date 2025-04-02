import numpy as np
from client.aggregation_client import NumpyAggregationClient


class MultiSet:
    def __init__(self,client:NumpyAggregationClient, records: np.ndarray):
        """
        Initializes a MultiSet class where the 0th dimension represents the elements.
        """
        self.client = client
        self.records = records

    def sum(self):
        _ans = self.client.__global_sum__(np.sum(self.records, axis=0))
        return _ans

    def count(self):
        _ans = self.client.__global_sum__(self.records.size[0])
        return _ans

    def avg(self):
        _ans = self.client.__global_sum__(np.stack([np.sum(self.records, axis=0), self.records.size[0]], axis=0))
        return _ans[0] / _ans[1]

    def min(self):
        _ans = self.client.__global_min__(np.min(self.records, axis=0))
        return _ans

    def max(self):
        _ans = self.client.__global_max__(np.max(self.records, axis=0))
        return _ans

