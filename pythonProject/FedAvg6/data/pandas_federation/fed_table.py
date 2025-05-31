import numpy as np
import pandas as pd

from system.client.aggregation_client import AggregationClient

class FedDataFrame(pd.DataFrame):
    _metadata = ['client']

    def __init__(self, *args, client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client  # ← this ensures each instance has its own client

    @property
    def _constructor(self):
        # This ensures operations like df[['col']] return a Multiset, not a DataFrame
        return FedDataFrame

    def __repr__(self):
        base = super().__repr__()
        return f"Multiset(client={self.client})\n{base}"

    def fed_sum(self):
        _agg = pd.DataFrame([self.sum()])
        _shape, _flattened = transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = FedDataFrame(inv_transform(_shape, _ans), client=self.client)
        rename_mapping = {old: new for old, new in zip(_ans.columns, self.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def fed_count(self):
        _agg = pd.DataFrame([self.count()])
        _shape, _flattened = transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = FedDataFrame(inv_transform(_shape, _ans), client=self.client)
        rename_mapping = {old: new for old, new in zip(_ans.columns, self.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def fed_min(self):
        _agg = pd.DataFrame([self.min()])
        _shape, _flattened = transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = FedDataFrame(inv_transform(_shape, _ans), client=self.client)
        rename_mapping = {old: new for old, new in zip(_ans.columns, self.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def fed_max(self):
        _agg = pd.DataFrame([self.max()])
        _shape, _flattened = transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = FedDataFrame(inv_transform(_shape, _ans), client=self.client)
        rename_mapping = {old: new for old, new in zip(_ans.columns, self.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def fed_avg(self):
        _shape, _flattened = transform(np.sum(self, axis=0))
        _flattened = np.append(_flattened, self.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        _ans =FedDataFrame(inv_transform(_shape, _ans[:-1]) / _ans[-1])
        rename_mapping = {old: new for old, new in zip(_ans.columns, self.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans





    def get_client(self) -> AggregationClient:
        return self.client

def transform(array):
    out = array.to_numpy().flatten().astype(np.float64).tolist()
    return array.shape,out


def inv_transform(original_shape,answer):
    return np.asarray(answer).reshape(original_shape)