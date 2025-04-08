import numpy as np

from client.aggregation_client import NumpyAggregationClient
from data.numpy_dataset.np_fed_table import transform,inv_transform

class NumpyFedArray(np.ndarray):
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

    def fed_avg(self):
        _shape, _flattened = transform(np.sum(self, axis=0))
        _flattened = np.append(_flattened, self.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        return inv_transform(_shape, _ans[:-1]) / 1

