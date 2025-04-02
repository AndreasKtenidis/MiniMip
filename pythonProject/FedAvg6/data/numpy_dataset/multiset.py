import numpy as np

from client.aggregation_client import NumpyAggregationClient


class Multiset(np.ndarray):
    def __new__(cls, input_array, client:NumpyAggregationClient):
        # Convert input_array into an ndarray
        obj = np.asarray(input_array).view(cls)
        # Add client
        obj.client = client
        return obj

    def __array_finalize__(self, obj):
        """Ensures client is retained during slicing or operations"""
        if obj is None:
            return
        self.client = getattr(obj, 'client', None)

    def __repr__(self):
        return f"CustomArray({super().__repr__()}, client={self.client})"

    def fed_sum(self):
        _ans = self.client.__global_sum__(np.stack([np.sum(self, axis=0)], axis=0))
        return _ans[0]

    def fed_count(self):
        _ans = self.client.__global_sum__(np.stack([self.shape[0]], axis=0))
        return _ans[0]

    def fed_avg(self):
        _ans = self.client.__global_sum__(np.stack([np.sum(self, axis=0), self.shape[0]], axis=0))
        return _ans[0] / _ans[1]

    def fed_min(self):
        _ans = self.client.__global_min__(np.stack([np.min(self, axis=0)], axis=0))
        return _ans[0]

    def fed_max(self):
        _ans = self.client.__global_max__(np.stack([np.max(self, axis=0)], axis=0))
        return _ans[0]
