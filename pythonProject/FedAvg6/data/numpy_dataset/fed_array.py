import numpy as np

from client.aggregation_client import NumpyAggregationClient


class FedArray(np.ndarray):
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
        return self.client.fed_avg(self,1)
