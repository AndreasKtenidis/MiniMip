import pandas as pd
import numpy as np

from client.aggregation_client import NumpyAggregationClient
from data.numpy_dataset.np_fed_table import transform,inv_transform

class Multiset(pd.DataFrame):
    _metadata = ['client']

    def __init__(self, *args, client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client  # ← this ensures each instance has its own client

    @property
    def _constructor(self):
        # This ensures operations like df[['col']] return a Multiset, not a DataFrame
        return Multiset

    def __repr__(self):
        base = super().__repr__()
        return f"Multiset(client={self.client})\n{base}"

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

    def get_client(self) -> NumpyAggregationClient:
        return self.client