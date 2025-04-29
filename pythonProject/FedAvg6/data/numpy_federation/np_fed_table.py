import numpy as np

from client.aggregation_client import AggregationClient
from data.fed_table import FedTable


class NumpyFedTable(np.ndarray):
    def __new__(cls, input_array, client:AggregationClient):
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
        _shape, _flattened = transform(np.sum(self, axis=0))
        _ans = self.client.__global_sum__(_flattened)
        return inv_transform(_shape, _ans)

    def fed_count(self):
        _ans = self.client.__global_sum__([self.shape[0]])
        return _ans[0]

    def fed_avg(self):
        _shape, _flattened = transform(np.sum(self, axis=0))
        _flattened = np.append(_flattened, self.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        return inv_transform(_shape, _ans[:-1]) / _ans[-1]


    def fed_min(self):
        _shape, _flattened = transform(np.min(self, axis=0))
        _ans = self.client.__global_min__(_flattened)
        return inv_transform(_shape, _ans)


    def fed_max(self):
        _shape, _flattened = transform(np.max(self, axis=0))
        _ans = self.client.__global_max__(_flattened)
        return inv_transform(_shape, _ans)

    def get_client(self)->AggregationClient:
        return self.client
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------


def transform(array:np.ndarray):
    out = array.flatten().astype(np.float64).tolist()
    return array.shape,out


def inv_transform(original_shape,answer):
    return np.asarray(answer).reshape(original_shape)

